from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}. Copy config.example.yml to config.yml first.")
    with config_path.open("r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}
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
        if float(settings.get("weight", 0)) < 0:
            raise ValueError(f"Profile weight cannot be negative: {name}")
        if not isinstance(settings.get("batch_images", 1), (int, list, tuple)):
            raise ValueError(f"Invalid batch_images for profile {name}")
        for key in ("checkpoint_path", "character_path", "lora_path"):
            if settings.get(key):
                value = Path(settings[key]).expanduser()
                settings[key] = str((Path(config.get("base_dir", project_root)) / value).resolve()
                                    if not value.is_absolute() else value.resolve())
    return config
