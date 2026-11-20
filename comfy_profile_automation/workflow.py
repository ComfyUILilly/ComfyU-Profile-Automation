from __future__ import annotations

import copy
import random
from typing import Any

from .catalog import ModelFile, ProfileData
from .wildcards import WildcardEngine


class WorkflowBuilder:
    """Build profile-isolated ComfyUI API prompts, including YAML prompt metadata."""

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
        self._set_checkpoint(workflow, profile.name, checkpoint.relative_path)
        if character:
            self._set_first_input(workflow, ("LoraLoader",), "lora_name", character.relative_path)
        self._attach_loras(workflow, loras)
        self._randomize_seeds(workflow)
        metadata = [self._metadata(profile, checkpoint)]
        if character:
            metadata.append(self._metadata(profile, character))
        metadata.extend(self._metadata(profile, item) for item in loras)
        for item in assets:
            self._apply_node_metadata(workflow, self._metadata(profile, item))
        self._apply_character_defaults(workflow, profile, character)
        self._apply_lora_defaults(workflow, profile, loras)
        self._apply_prompt_metadata(workflow, profile, checkpoint, character, loras)
        return self.wildcards.apply(workflow, profile.setup_wildcard, metadata)

    @staticmethod
    def _metadata(profile, item):
        return profile.metadata[item.category].get(item.relative_path,
               profile.metadata[item.category].get(item.name, {}))

    def _apply_settings(self, workflow: dict, settings: dict) -> None:
        workflow_settings = settings.get("workflow", {})
        for node_id, values in workflow_settings.items():
            if node_id in workflow and isinstance(values, dict):
                for key, value in values.items():
                    workflow[node_id].setdefault("inputs", {})[key] = self._random_value(value)
        for meta in ("workflow_scale", "workflow_min", "workflow_max"):
            for node_id, values in settings.get(meta, {}).items():
                if node_id not in workflow or not isinstance(values, dict):
                    continue
                for key, raw in values.items():
                    if key not in workflow[node_id].get("inputs", {}):
                        continue
                    amount = self._random_value(raw)
                    current = workflow[node_id]["inputs"][key]
                    if meta == "workflow_scale":
                        workflow[node_id]["inputs"][key] = current * amount
                    elif meta == "workflow_min":
                        workflow[node_id]["inputs"][key] = max(current, amount)
                    else:
                        workflow[node_id]["inputs"][key] = min(current, amount)

    def _random_value(self, value):
        if isinstance(value, (list, tuple)) and value:
            if len(value) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                low, high = sorted(value)
                return self.rng.uniform(low, high) if any(isinstance(v, float) for v in value) else self.rng.randint(int(low), int(high))
            return self.rng.choice(list(value))
        return value

    @staticmethod
    def _set_first_input(workflow, node_ids, input_name, value):
        for node_id in node_ids:
            node = workflow.get(node_id)
            if isinstance(node, dict):
                node.setdefault("inputs", {})[input_name] = value
                return True
        return False

    def _set_checkpoint(self, workflow, profile_name, relative_path):
        if profile_name == "Anime" and self._set_first_input(workflow, ("UNETLoader",), "unet_name", relative_path):
            return
        if not self._set_first_input(workflow, ("CheckpointLoaderSimple",), "ckpt_name", relative_path):
            self._set_first_input(workflow, ("UNETLoader",), "unet_name", relative_path)

    def _attach_loras(self, workflow, loras):
        template = workflow.get("LoraLoader")
        if not isinstance(template, dict):
            return
        next_node = "LoraLoader"
        for index, lora in enumerate(loras, start=1):
            node_id = f"LoraLoader{index}"
            workflow[node_id] = copy.deepcopy(template)
            inputs = workflow[node_id].setdefault("inputs", {})
            inputs["lora_name"] = lora.relative_path
            prior_inputs = workflow.get(next_node, {}).get("inputs", {})
            if "model" in prior_inputs:
                inputs["model"] = prior_inputs["model"]
            if "clip" in prior_inputs:
                inputs["clip"] = prior_inputs["clip"]
            if next_node in workflow:
                workflow[next_node].setdefault("inputs", {})["model"] = [node_id, 0]
                workflow[next_node].setdefault("inputs", {})["clip"] = [node_id, 1]
            next_node = node_id

    def _randomize_seeds(self, workflow):
        for node in workflow.values():
            if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
                if "seed" in node["inputs"]:
                    node["inputs"]["seed"] = self.rng.randint(0, 2**63 - 1)

    @staticmethod
    def _apply_node_metadata(workflow, metadata):
        for node_id, values in metadata.items():
            if node_id in workflow and isinstance(values, dict):
                workflow[node_id].setdefault("inputs", {}).update(copy.deepcopy(values))

    def _apply_character_defaults(self, workflow, profile, character):
        node = workflow.get("LoraLoader", {})
        inputs = node.get("inputs", {})
        if not character:
            for key in ("strength_model", "strength_clip"):
                if key in inputs:
                    inputs[key] = 0.0
            return
        self._apply_strengths(node, self._metadata(profile, character),
                              profile.setup_workflow.get("charDefault", {}))

    def _apply_lora_defaults(self, workflow, profile, loras):
        defaults = profile.setup_workflow.get("loraDefault", {})
        for index, lora in enumerate(loras, start=1):
            self._apply_strengths(workflow.get(f"LoraLoader{index}", {}),
                                  self._metadata(profile, lora), defaults)

    def _apply_strengths(self, node, metadata, defaults):
        inputs = node.get("inputs", {})
        for key in ("strength_model", "strength_clip", "A", "B", "preset", "block_vector"):
            value = metadata.get(key, defaults.get(key))
            if value is not None:
                inputs[key] = self._random_value(value)

    @staticmethod
    def _apply_prompt_metadata(workflow, profile, checkpoint, character, loras):
        info = workflow.get("PrimitiveStringMultilineInfo")
        if isinstance(info, dict):
            lines = [profile.name, f"Checkpoint: {checkpoint.relative_path}"]
            if character:
                lines.append(f"Character: {character.relative_path}")
            lines.extend(f"LoRA: {item.relative_path}" for item in loras)
            info.setdefault("inputs", {})["value"] = "\n".join(lines) + "\n"
