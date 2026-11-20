from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelFile:
    profile: str
    category: str
    path: Path
    relative_path: str

    @property
    def name(self) -> str:
        return self.path.stem


@dataclass
class ProfileData:
    name: str
    root: Path
    checkpoints: list[ModelFile] = field(default_factory=list)
    characters: list[ModelFile] = field(default_factory=list)
    loras: list[ModelFile] = field(default_factory=list)
    weights: dict[str, Any] = field(default_factory=dict)
    setup_workflow: dict[str, Any] = field(default_factory=dict)
    setup_wildcard: dict[str, Any] = field(default_factory=dict)
    workflow_api: dict[str, Any] = field(default_factory=dict)


class ProfileCatalog:
    """Load one profile's YAML data and model files without crossing profiles."""

    MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth"}

    def __init__(self, data_path: Path, profiles: dict[str, dict[str, Any]],
                 base_dir: Path | None = None):
        self.data_path = data_path.expanduser().resolve()
        self.profiles = profiles
        self.profile_names = tuple(profiles)
        self.base_dir = (base_dir or Path.cwd()).expanduser().resolve()

    def _scan(self, profile: str, category: str, fallback_folder: str) -> list[ModelFile]:
        configured = self.profiles[profile].get(f"{category}_path")
        root = Path(configured).expanduser() if configured else self.data_path / profile / fallback_folder
        if not root.is_absolute():
            root = self.base_dir / root
        root = root.resolve()
        if not root.exists():
            return []
        items = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.MODEL_SUFFIXES:
                relative = path.relative_to(root).as_posix()
                items.append(ModelFile(profile, category, path, relative))
        return items

    @staticmethod
    def _yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8-sig") as stream:
            return yaml.safe_load(stream) or {}

    def load(self, profile: str) -> ProfileData:
        if profile not in self.profile_names:
            raise KeyError(f"Unknown profile: {profile}")
        root = self.data_path / profile
        settings = self.profiles[profile]
        checkpoints = self._scan(profile, "checkpoint", "checkpoint")
        characters = self._scan(profile, "character", "lora/char")
        loras = self._scan(profile, "lora", "lora/etc")
        workflow_path = root / settings.get("workflow_file", "workflow_api.json")
        workflow: dict[str, Any] = {}
        if workflow_path.exists():
            with workflow_path.open("r", encoding="utf-8-sig") as stream:
                workflow = json.load(stream)
        weights = {
            "checkpoint": self._yaml(root / "WeightCheckpoint.yml"),
            "character": self._yaml(root / "WeightChar.yml"),
            "lora": self._yaml(root / "WeightLora.yml"),
        }
        return ProfileData(
            name=profile, root=root, checkpoints=checkpoints, characters=characters,
            loras=loras, weights=weights,
            setup_workflow=self._yaml(root / "setupWorkflow.yml"),
            setup_wildcard=self._yaml(root / "setupWildcard.yml"),
            workflow_api=workflow,
        )
