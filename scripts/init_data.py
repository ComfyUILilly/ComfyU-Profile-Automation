from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROFILES = ("IL", "Anime")


def initialize(data_path: Path) -> None:
    for profile in PROFILES:
        root = data_path / profile
        for folder in ("checkpoint", "lora/char", "lora/etc"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        defaults = {
            "WeightCheckpoint.yml": {},
            "WeightChar.yml": {},
            "WeightLora.yml": {},
            "setupWorkflow.yml": {},
            "setupWildcard.yml": {},
        }
        for filename, value in defaults.items():
            target = root / filename
            if not target.exists():
                target.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
        workflow = root / "workflow_api.json"
        if not workflow.exists():
            workflow.write_text("{}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the IL/Anime data folder skeleton")
    parser.add_argument("data_path", nargs="?", default="../ComfyU-auto-script_data")
    args = parser.parse_args()
    target = Path(args.data_path).expanduser()
    initialize(target)
    print(f"Created profile data skeleton: {target.resolve()}")


if __name__ == "__main__":
    main()
