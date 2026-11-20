from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from .catalog import ModelFile, ProfileData


class FromImageWorkflow:
    """Restore API prompts embedded in PNG metadata, using only active-profile assets."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def load(self, profile: ProfileData, image_dir: Path):
        if not image_dir.exists():
            return None
        images = list(image_dir.rglob("*.png"))
        self.rng.shuffle(images)
        for image_path in images:
            prompt = self._read_prompt(image_path)
            if not prompt:
                continue
            checkpoint_node, checkpoint_key, checkpoint = self._resolve_checkpoint(profile, prompt)
            if not checkpoint:
                continue
            character, loras, lora_nodes = self._resolve_loras(profile, prompt)
            if loras is None:
                continue
            prompt[checkpoint_node].setdefault("inputs", {})[checkpoint_key] = checkpoint.relative_path
            for node_id, item in lora_nodes:
                prompt[node_id].setdefault("inputs", {})["lora_name"] = item.relative_path
            return prompt, checkpoint, character, loras
        return None

    @staticmethod
    def _read_prompt(path: Path) -> dict[str, Any] | None:
        try:
            with Image.open(path) as image:
                raw = image.info.get("prompt")
                value = json.loads(raw) if isinstance(raw, str) else raw
                return value if isinstance(value, dict) else None
        except Exception:
            return None

    @staticmethod
    def _resolve_checkpoint(profile, prompt):
        node_id = "UNETLoader" if profile.name == "Anime" and "UNETLoader" in prompt else None
        if not node_id:
            node_id = "CheckpointLoaderSimple" if "CheckpointLoaderSimple" in prompt else "UNETLoader" if "UNETLoader" in prompt else None
        if not node_id:
            return None, None, None
        key = "unet_name" if node_id == "UNETLoader" else "ckpt_name"
        value = prompt.get(node_id, {}).get("inputs", {}).get(key)
        filename = Path(str(value or "").replace("\\", "/")).name
        stem = Path(filename).stem
        item = next((x for x in profile.checkpoints if x.name == stem or x.path.name == filename), None)
        return node_id, key, item

    @staticmethod
    def _resolve_loras(profile, prompt):
        available = profile.characters + profile.loras
        character, loras, nodes = None, [], []
        for node_id, node in prompt.items():
            if not isinstance(node, dict) or not str(node.get("class_type", "")).startswith("LoraLoader"):
                continue
            value = node.get("inputs", {}).get("lora_name")
            if not value:
                continue
            filename = Path(str(value).replace("\\", "/")).name
            item = next((x for x in available if x.path.name == filename), None)
            if item is None:
                item = next((x for x in available if x.name == Path(filename).stem), None)
            if item is None:
                return None, None, None
            nodes.append((node_id, item))
            if item.category == "character" and character is None:
                character = item
            elif item.category == "lora":
                loras.append(item)
        return character, loras, nodes
