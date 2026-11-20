from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}. Copy config.example.yml to config.yml first.")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    for key in ("data_path", "profiles", "generation", "comfyui"):
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    if not config["profiles"]:
        raise ValueError("At least one profile must be configured")
    for name, settings in config["profiles"].items():
        if float(settings.get("weight", 0)) < 0:
            raise ValueError(f"Profile weight cannot be negative: {name}")
    return config
