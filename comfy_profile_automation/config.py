from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class LegacySafeLoader(yaml.SafeLoader):
    pass


LegacySafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def _legacy_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    types = config.get("CheckpointTypes")
    if not isinstance(types, dict) or not types:
        return {}
    base = Path(config.get("base_dir", ".")).expanduser()
    checkpoint_root = Path(config.get("CheckpointPath", "ComfyUI/models/checkpoints"))
    unet_root = Path(config.get("unetPath", "ComfyUI/models/diffusion_models"))
    lora_root = Path(config.get("LoraPath", "ComfyUI/models/loras"))
    char_folder = config.get("LoraCharPath", "char")
    etc_folder = config.get("LoraEtcPath", "etc")
    profiles = {}
    for name, weight in types.items():
        if name == "Anime":
            checkpoint_path = unet_root / name
        else:
            checkpoint_path = checkpoint_root / name
        profiles[name] = {
            "weight": weight,
            "batch_images": config.get("profileBatchImages", 1),
            "checkpoint_path": str(checkpoint_path),
            "character_path": str(lora_root / name / char_folder),
            "lora_path": str(lora_root / name / etc_folder),
            "checkpoint_kind": config.get("GetCheckpointKind", "Weight"),
            "character_kind": config.get("GetCharKind", "Weight"),
            "lora_kind": config.get("GetLoraKind", "Weight"),
            "from_img_path": config.get("fromImgPath", ""),
        }
    return profiles


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}. Copy config.example.yml to config.yml first.")
    with config_path.open("r", encoding="utf-8-sig") as stream:
        config = yaml.load(stream, Loader=LegacySafeLoader) or {}

    if "data_path" not in config and "dataPath" in config:
        config["data_path"] = config["dataPath"]
    if "profiles" not in config:
        config["profiles"] = _legacy_profiles(config)

    generation = config.setdefault("generation", {})
    legacy_generation_keys = (
        "CheckpointLoop", "CharLoop", "queueLoop", "queue_prompt",
        "queue_prompt_wait", "sleep", "json_to_xlsx_interval",
        "WorkflowPrint", "WorkflowSave",
        "LoraDbCnt", "LoraRandomCnt", "LoraCycleCnt",
    )
    for key in legacy_generation_keys:
        if key in config and key not in generation:
            mapped = {
                "sleep": "sleep_seconds",
                "json_to_xlsx_interval": "xlsx_export_interval",
                "WorkflowPrint": "print_workflow",
                "WorkflowSave": "save_workflow",
                "LoraDbCnt": "lora_db_count",
                "LoraRandomCnt": "lora_random_count",
                "LoraCycleCnt": "lora_cycle_count",
            }.get(key, key)
            generation[mapped] = config[key]

    if "comfyui" not in config:
        prompt_url = config.get("url", "http://127.0.0.1:8188/prompt")
        queue_base = config.get("url2", prompt_url.rsplit("/", 1)[0])
        config["comfyui"] = {
            "prompt_url": prompt_url,
            "queue_url": queue_base.rstrip("/") + "/queue",
        }

    for key in ("data_path", "profiles", "generation", "comfyui"):
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    if not config["profiles"]:
        raise ValueError("At least one profile must be configured")

    project_root = config_path.parent
    for key in ("data_path", "base_dir"):
        if config.get(key):
            value = Path(config[key]).expanduser()
            config[key] = str((project_root / value).resolve() if not value.is_absolute() else value.resolve())

    for name, settings in config["profiles"].items():
        if not isinstance(settings, dict):
            raise ValueError(f"Profile settings must be a mapping: {name}")
        if float(settings.get("weight", 0)) < 0:
            raise ValueError(f"Profile weight cannot be negative: {name}")
        if not isinstance(settings.get("batch_images", 1), (int, list, tuple)):
            raise ValueError(f"Invalid batch_images for profile {name}")
        for key in ("checkpoint_path", "character_path", "lora_path", "from_img_path"):
            if settings.get(key):
                value = Path(settings[key]).expanduser()
                if not value.is_absolute():
                    value = Path(config.get("base_dir", project_root)) / value
                settings[key] = str(value.resolve())
    return config
