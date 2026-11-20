from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

from comfy_profile_automation.config import load_config
from comfy_profile_automation.profiles import ProfileManager
from comfy_profile_automation.catalog import ProfileCatalog
from comfy_profile_automation.workflow import WorkflowBuilder
from comfy_profile_automation.comfy_api import ComfyUIClient


def main() -> int:
    parser = argparse.ArgumentParser(description="ComfyUI profile automation")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--dry-run", action="store_true", help="Prepare prompts without sending")
    args = parser.parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    rng = random.Random()
    manager = ProfileManager(config["profiles"], rng=rng)
    catalog = ProfileCatalog(Path(config["data_path"]), config["profiles"])
    builder = WorkflowBuilder()
    client = ComfyUIClient(**config["comfyui"], timeout=config["generation"].get("request_timeout", 30))
    log = logging.getLogger("automation")

    while True:
        profile_name, batch_count = manager.next_batch()
        profile = catalog.load(profile_name)
        log.info("Profile selected: %s (%d images)", profile_name, batch_count)
        if not profile.checkpoints:
            log.warning("Profile %s has no checkpoint files; skipping batch", profile_name)
            time.sleep(1)
            continue
        for index in range(batch_count):
            checkpoint = rng.choice(profile.checkpoints)
            character = rng.choice(profile.characters) if profile.characters else None
            loras = rng.sample(profile.loras, min(len(profile.loras), rng.randint(*config["generation"]["lora_count"]))) if profile.loras else []
            workflow = builder.build(profile, checkpoint, character, loras)
            log.info("[%s %d/%d] checkpoint=%s character=%s loras=%s",
                     profile_name, index + 1, batch_count, checkpoint.relative_path,
                     character.relative_path if character else "-", [item.relative_path for item in loras])
            if args.dry_run:
                continue
            client.queue_prompt(workflow)
            if config["generation"].get("wait_for_queue", True):
                client.wait_until_idle(poll_interval=config["generation"].get("poll_interval", 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
