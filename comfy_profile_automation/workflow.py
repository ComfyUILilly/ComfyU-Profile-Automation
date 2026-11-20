from __future__ import annotations

import copy
import random
from typing import Any

from .catalog import ModelFile, ProfileData
from .wildcards import WildcardEngine


class WorkflowBuilder:
    """Build profile-isolated ComfyUI API prompts, preserving node connections."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.wildcards = WildcardEngine(self.rng)

    def build(self, profile: ProfileData, checkpoint: ModelFile,
              character: ModelFile | None, loras: list[ModelFile]) -> dict[str, Any]:
        assets = [checkpoint, *([character] if character else []), *loras]
        if any(item.profile != profile.name for item in assets):
            raise ValueError("Cross-profile assets are not allowed")
        workflow = copy.deepcopy(profile.workflow_api)
        if not workflow:
            raise ValueError(f"Missing workflow_api.json for profile {profile.name}")
        self._apply_settings(workflow, profile.setup_workflow)
        self._set_checkpoint(workflow, profile.name, checkpoint.path.name)
        if character:
            self._set_first_input(workflow, ("LoraLoader",), "lora_name", character.path.name)
        self._attach_loras(workflow, loras, profile.name)
        self._randomize_seeds(workflow)
        self._apply_checkpoint_metadata(workflow, profile, checkpoint)
        self._apply_lora_metadata(workflow, profile, character, loras)
        self._apply_prompt_metadata(workflow, profile, checkpoint, character, loras)
        return self.wildcards.apply(workflow, profile.setup_wildcard)

    @staticmethod
    def _apply_settings(workflow: dict, settings: dict) -> None:
        workflow_settings = settings.get("workflow", settings)
        for node_id, values in workflow_settings.items():
            if node_id in ("charDefault", "loraDefault", "workflow_scale", "workflow_min", "workflow_max"):
                continue
            if node_id in workflow and isinstance(values, dict):
                for key, value in values.items():
                    workflow[node_id].setdefault("inputs", {})[key] = WorkflowBuilder._random_value(value)

    @staticmethod
    def _random_value(value):
        if isinstance(value, (list, tuple)) and value:
            if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                low, high = value
                if low > high:
                    low, high = high, low
                return random.uniform(low, high) if any(isinstance(v, float) for v in value) else random.randint(int(low), int(high))
            return random.choice(list(value))
        return value

    @staticmethod
    def _set_first_input(workflow, node_ids, input_name, value):
        for node_id in node_ids:
            node = workflow.get(node_id)
            if isinstance(node, dict):
                node.setdefault("inputs", {})[input_name] = value
                return True
        return False

    def _set_checkpoint(self, workflow, profile_name, filename):
        if profile_name == "Anime" and self._set_first_input(workflow, ("UNETLoader",), "unet_name", filename):
            return
        if not self._set_first_input(workflow, ("CheckpointLoaderSimple",), "ckpt_name", filename):
            self._set_first_input(workflow, ("UNETLoader",), "unet_name", filename)

    def _attach_loras(self, workflow, loras, profile_name):
        template = workflow.get("LoraLoader")
        if not isinstance(template, dict):
            return
        next_node = "LoraLoader"
        for index, lora in enumerate(loras, start=1):
            node_id = f"LoraLoader{index}"
            workflow[node_id] = copy.deepcopy(template)
            inputs = workflow[node_id].setdefault("inputs", {})
            inputs["lora_name"] = lora.path.name
            model = workflow.get(next_node, {}).get("inputs", {}).get("model")
            clip = workflow.get(next_node, {}).get("inputs", {}).get("clip")
            if model is not None:
                inputs["model"] = model
            if clip is not None:
                inputs["clip"] = clip
            if next_node in workflow:
                workflow[next_node].setdefault("inputs", {})["model"] = [node_id, 0]
                workflow[next_node].setdefault("inputs", {})["clip"] = [node_id, 1]
            next_node = node_id

    def _randomize_seeds(self, workflow):
        for node in workflow.values():
            if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
                if "seed" in node["inputs"]:
                    node["inputs"]["seed"] = self.rng.randint(0, 2**63 - 1)

    def _apply_checkpoint_metadata(self, workflow, profile, checkpoint):
        metadata = profile.metadata["checkpoint"].get(checkpoint.name, {})
        self._apply_node_metadata(workflow, metadata)

    def _apply_lora_metadata(self, workflow, profile, character, loras):
        for item in ([character] if character else []) + loras:
            metadata = profile.metadata["character" if item.category == "character" else "lora"].get(item.name, {})
            self._apply_node_metadata(workflow, metadata)

    @staticmethod
    def _apply_node_metadata(workflow, metadata):
        for node_id, values in metadata.items():
            if node_id in workflow and isinstance(values, dict):
                workflow[node_id].setdefault("inputs", {}).update(copy.deepcopy(values))

    @staticmethod
    def _apply_prompt_metadata(workflow, profile, checkpoint, character, loras):
        info = workflow.get("PrimitiveStringMultilineInfo")
        if isinstance(info, dict):
            lines = [profile.name, f"Checkpoint: {checkpoint.relative_path}"]
            if character:
                lines.append(f"Character: {character.relative_path}")
            lines.extend(f"LoRA: {item.relative_path}" for item in loras)
            info.setdefault("inputs", {})["value"] = "\n".join(lines) + "\n"
