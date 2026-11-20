from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

from comfy_profile_automation.config import load_config
from comfy_profile_automation.profiles import ProfileManager
from comfy_profile_automation.catalog import ProfileCatalog
from comfy_profile_automation.selection import AssetSelector
from comfy_profile_automation.database import UsageDatabase
from comfy_profile_automation.workflow import WorkflowBuilder
from comfy_profile_automation.comfy_api import ComfyUIClient
from comfy_profile_automation.observer import ProfileObserver


def _count(value, rng):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = int(value[0]), int(value[1])
        return rng.randint(min(low, high), max(low, high))
    return max(1, int(value))


def main() -> int:
    parser = argparse.ArgumentParser(description="ComfyUI profile automation")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--dry-run", action="store_true", help="Prepare one profile batch without sending")
    args = parser.parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("automation")
    rng = random.Random()
    data_path = Path(config["data_path"]).expanduser().resolve()
    manager = ProfileManager(config["profiles"], rng=rng)
    catalog = ProfileCatalog(data_path, config["profiles"], base_dir=Path(config.get("base_dir", ".")).resolve())
    selector = AssetSelector(rng=rng)
    builder = WorkflowBuilder(rng=rng)
    client = ComfyUIClient(**config["comfyui"], timeout=config["generation"].get("request_timeout", 30))
    databases = {}
    observer = None

    def reload_profile(profile_name, changed_path=None, event_type=None):
        log.info("Profile data changed; refreshed on next batch: %s (%s)", profile_name, changed_path)

    try:
        roots = {}
        for name, settings in config["profiles"].items():
            roots[name] = [data_path / name]
            for key in ("checkpoint_path", "character_path", "lora_path"):
                path = Path(settings.get(key, ""))
                if str(path):
                    roots[name].append(path if path.is_absolute() else Path(config.get("base_dir", ".")) / path)
            databases[name] = UsageDatabase(data_path / name / "database")
        if config.get("observer", {}).get("enabled", True):
            observer = ProfileObserver(roots, reload_profile)
            observer.start()

        while True:
            profile_name, batch_count = manager.next_batch()
            profile = catalog.load(profile_name)
            log.info("Profile selected: %s (%d images)", profile_name, batch_count)
            if not profile.checkpoints:
                log.warning("Profile %s has no checkpoint assets; skipping batch", profile_name)
                if args.dry_run:
                    return 2
                time.sleep(2)
                continue
            if not profile.workflow_api:
                log.error("Profile %s has no workflow_api.json content", profile_name)
                if args.dry_run:
                    return 2
                time.sleep(2)
                continue

            profile_cfg = config["profiles"][profile_name]
            gen = config["generation"]
            for index in range(batch_count):
                checkpoint_mode = profile_cfg.get("checkpoint_kind", gen.get("checkpoint_kind", "Weight"))
                character_mode = profile_cfg.get("character_kind", gen.get("character_kind", "Weight"))
                lora_mode = profile_cfg.get("lora_kind", gen.get("lora_kind", "Weight"))
                checkpoint = selector.choose(profile, "checkpoint", checkpoint_mode)[0]
                character = selector.choose(profile, "character", character_mode)[0] if profile.characters else None
                lora_count = _count(gen.get("lora_count", [1, 2]), rng)
                loras = selector.choose(profile, "lora", lora_mode, lora_count)
                workflow = builder.build(profile, checkpoint, character, loras)
                log.info("[%s %d/%d] checkpoint=%s character=%s loras=%s",
                         profile_name, index + 1, batch_count, checkpoint.relative_path,
                         character.relative_path if character else "-",
                         [item.relative_path for item in loras])
                if args.dry_run:
                    continue
                client.queue_prompt(workflow)
                if gen.get("wait_for_queue", True):
                    client.wait_until_idle(poll_interval=gen.get("poll_interval", 1),
                                           max_wait=gen.get("max_queue_wait"))
                databases[profile_name].record(
                    profile_name, checkpoint.relative_path,
                    character.relative_path if character else None,
                    [item.relative_path for item in loras],
                )
                interval = int(gen.get("xlsx_export_interval", 0))
                if interval and (index + 1) % interval == 0:
                    databases[profile_name].export_xlsx()
                time.sleep(float(gen.get("sleep_seconds", 0)))
            if args.dry_run:
                return 0
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        return 0
    finally:
        if observer:
            observer.stop()
        for database in databases.values():
            try:
                if config.get("generation", {}).get("export_xlsx_on_exit", True):
                    database.export_xlsx()
                database.close()
            except Exception:
                log.exception("Failed to close profile database")


if __name__ == "__main__":
    raise SystemExit(main())
