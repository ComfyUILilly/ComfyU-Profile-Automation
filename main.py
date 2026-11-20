from __future__ import annotations

import argparse
import logging
import random
import time
import json
from pathlib import Path

from comfy_profile_automation.config import load_config
from comfy_profile_automation.profiles import ProfileManager
from comfy_profile_automation.catalog import ProfileCatalog
from comfy_profile_automation.selection import AssetSelector
from comfy_profile_automation.database import UsageDatabase
from comfy_profile_automation.workflow import WorkflowBuilder
from comfy_profile_automation.comfy_api import ComfyUIClient
from comfy_profile_automation.observer import ProfileObserver
from comfy_profile_automation.from_image import FromImageWorkflow


def _count(value, rng):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return rng.randint(int(value[0]), int(value[1]))
    return max(1, int(value))


def _mode(value, rng):
    if isinstance(value, dict):
        names = list(value)
        weights = [max(0.0, float(value[name])) for name in names]
        return rng.choices(names, weights=weights if any(weights) else None, k=1)[0]
    return str(value)


def _span(generation, modern_key, legacy_key, default, rng):
    if modern_key in generation:
        return _count(generation[modern_key], rng)
    if legacy_key in generation:
        return _count(generation[legacy_key], rng) + 1
    return default


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
    databases = {name: UsageDatabase(data_path / name / "database") for name in config["profiles"]}
    selector = AssetSelector(rng=rng)
    builder = WorkflowBuilder(rng=rng)
    from_image = FromImageWorkflow(rng=rng)
    client = ComfyUIClient(**config["comfyui"], timeout=config["generation"].get("request_timeout", 30))
    observer = None

    def on_profile_change(profile_name, path, event_type):
        log.info("Profile %s file event: %s (%s); changes apply on next batch", profile_name, path, event_type)

    try:
        roots = {}
        for name, settings in config["profiles"].items():
            roots[name] = [data_path / name]
            for key in ("checkpoint_path", "character_path", "lora_path"):
                if settings.get(key):
                    roots[name].append(Path(settings[key]))
        if config.get("observer", {}).get("enabled", True):
            observer = ProfileObserver(roots, on_profile_change)
            observer.start()

        while True:
            profile_name, batch_count = manager.next_batch()
            profile = catalog.load(profile_name)
            selector.database = databases[profile_name]
            log.info("Profile selected: %s (%d images)", profile_name, batch_count)
            if not profile.checkpoints or not profile.workflow_api:
                log.warning("Profile %s has no checkpoint assets or workflow_api.json", profile_name)
                if args.dry_run:
                    return 2
                time.sleep(2)
                continue

            profile_cfg = config["profiles"][profile_name]
            gen = config["generation"]
            checkpoint = character = None
            loras = []
            checkpoint_remaining = character_remaining = lora_remaining = 0
            checkpoint_mode = character_mode = lora_mode = "Weight"
            for index in range(batch_count):
                imported = None
                if checkpoint is None or checkpoint_remaining <= 0:
                    checkpoint_mode = _mode(profile_cfg.get("checkpoint_kind", gen.get("checkpoint_kind", "Weight")), rng)
                    if checkpoint_mode == "fromImg":
                        image_dir = Path(profile_cfg.get("from_img_path", gen.get("from_img_path", "")))
                        if not image_dir.is_absolute():
                            image_dir = Path(config.get("base_dir", ".")) / image_dir
                        imported = from_image.load(profile, image_dir)
                    if imported:
                        prompt, checkpoint, character, loras = imported
                        workflow = builder.prepare_from_image(profile, prompt, checkpoint, character, loras)
                        checkpoint_remaining = character_remaining = lora_remaining = 1
                    else:
                        if checkpoint_mode == "fromImg":
                            log.warning("No compatible PNG workflow for %s; falling back to Weight", profile_name)
                            checkpoint_mode = "Weight"
                        checkpoint = selector.choose(profile, "checkpoint", checkpoint_mode)[0]
                        checkpoint_remaining = _span(gen, "images_per_checkpoint", "CheckpointLoop", 1, rng)
                        character = None
                        character_remaining = 0
                        loras = []
                        lora_remaining = 0
                elif checkpoint_mode == "fromImg":
                    image_dir = Path(profile_cfg.get("from_img_path", gen.get("from_img_path", "")))
                    if not image_dir.is_absolute():
                        image_dir = Path(config.get("base_dir", ".")) / image_dir
                    imported = from_image.load(profile, image_dir)
                    if imported:
                        prompt, checkpoint, character, loras = imported
                        workflow = builder.prepare_from_image(profile, prompt, checkpoint, character, loras)
                        checkpoint_remaining = character_remaining = lora_remaining = 1
                    else:
                        checkpoint_mode = "Weight"
                        checkpoint = selector.choose(profile, "checkpoint", checkpoint_mode)[0]
                        checkpoint_remaining = 1

                if imported is None:
                    if character is None or character_remaining <= 0:
                        character_mode = _mode(profile_cfg.get("character_kind", gen.get("character_kind", "Weight")), rng)
                        character = None if character_mode == "Wildcard" else (
                            selector.choose(profile, "character", character_mode)[0] if profile.characters else None)
                        character_remaining = _span(gen, "images_per_character", "CharLoop", 1, rng)
                        loras = []
                        lora_remaining = 0
                    if lora_remaining <= 0:
                        lora_mode = _mode(profile_cfg.get("lora_kind", gen.get("lora_kind", "Weight")), rng)
                        lora_count = _count(gen.get("lora_count", [1, 2]), rng)
                        loras = selector.choose(profile, "lora", lora_mode, lora_count)
                        lora_remaining = _span(gen, "images_per_lora", "queueLoop", 1, rng)
                    workflow = builder.build(profile, checkpoint, character, loras,
                                             weight_metadata=selector.last_group_metadata)

                log.info("[%s %d/%d] checkpoint=%s character=%s loras=%s",
                         profile_name, index + 1, batch_count, checkpoint.relative_path,
                         character.relative_path if character else "-",
                         [item.relative_path for item in loras])
                checkpoint_remaining -= 1
                character_remaining -= 1
                lora_remaining -= 1
                if gen.get("print_workflow", False):
                    log.info("Workflow: %s", json.dumps(workflow, ensure_ascii=False))
                if gen.get("save_workflow", False):
                    log_dir = Path("log")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    save_path = log_dir / f"{profile_name}-{checkpoint.name}-{index + 1}-{stamp}.json"
                    save_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
                if args.dry_run:
                    continue
                client.sync_model_names(workflow, gen.get("force_linux_path_nodes", []))
                if gen.get("queue_prompt", True):
                    client.queue_prompt(workflow)
                if gen.get("queue_prompt_wait", gen.get("wait_for_queue", True)):
                    client.wait_until_idle(poll_interval=gen.get("poll_interval", 1),
                                           max_wait=gen.get("max_queue_wait"))
                used_assets = [*([character] if character else []), *loras]
                tags = []
                for item in used_assets:
                    source = profile.metadata[item.category]
                    metadata = source.get(item.relative_path, source.get(item.name, {}))
                    raw_tags = metadata.get("tag", []) if isinstance(metadata, dict) else []
                    if isinstance(raw_tags, str):
                        tags.append(raw_tags)
                    elif isinstance(raw_tags, list):
                        tags.extend(raw_tags)
                databases[profile_name].record(
                    profile_name, checkpoint.relative_path,
                    character.relative_path if character else None,
                    [item.relative_path for item in loras], tags=tags,
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
