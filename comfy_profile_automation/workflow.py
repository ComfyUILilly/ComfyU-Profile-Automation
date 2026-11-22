from __future__ import annotations

import copy
import random
from typing import Any

from .catalog import ModelFile, ProfileData
from .wildcards import WildcardEngine


class WorkflowBuilder:
    """Build profile-isolated ComfyUI API prompts, including YAML prompt metadata."""

    def __init__(self, rng: random.Random | None = None, settings: dict[str, Any] | None = None):
        self.rng = rng or random.Random()
        self.settings = settings or {}
        self.wildcards = WildcardEngine(self.rng)
        self._base_lora_node_id = "LoraLoader"
        self._last_lora_node_ids = []
        self._generation_counter = 0

    def build(self, profile: ProfileData, checkpoint: ModelFile,
              character: ModelFile | None, loras: list[ModelFile], weight_metadata: list[dict] | None = None) -> dict[str, Any]:
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
        checkpoint_metadata = self._metadata(profile, checkpoint)
        self._apply_checkpoint_sampler(workflow, checkpoint_metadata)
        self._attach_loras(workflow, loras, profile.name, checkpoint_metadata)
        self._randomize_seeds(workflow)
        character_metadata = self._metadata(profile, character) if character else profile.setup_wildcard.get("CharWildcard", {})
        lora_metadata = [self._metadata(profile, item) for item in loras]
        if not lora_metadata:
            lora_metadata = [profile.setup_wildcard.get("LoraWildcard", {})]
        source_groups = {
            "setup": [profile.setup_wildcard] if checkpoint_metadata.get("Setup", True) else [],
            "Checkpoint": [checkpoint_metadata],
            "Char": [character_metadata],
            "Weight": list(weight_metadata or []),
            "Lora": lora_metadata,
        }
        sort_order = self.settings.get(
            "SetWildcardSort",
            profile.setup_wildcard.get("SetWildcardSort", ["setup", "Checkpoint", "Char", "Weight", "Lora"]),
        )
        wildcard_sources = []
        for group_name in sort_order:
            wildcard_sources.extend(source_groups.get(group_name, []))
        for item in assets:
            self._apply_node_metadata(workflow, self._metadata(profile, item))
        self._apply_character_defaults(workflow, profile, character)
        self._apply_lora_defaults(workflow, profile, loras)
        self._apply_prompt_metadata(workflow, profile, checkpoint, character, loras)
        self._set_save_prefix(workflow, profile, checkpoint, character, loras)
        return self.wildcards.apply(
            workflow,
            profile.setup_wildcard,
            wildcard_sources,
            shuffle=self.settings.get(
                "shuffleWildcard", profile.setup_wildcard.get("shuffleWildcard", False)
            ),
        )

    @staticmethod
    def _metadata(profile, item):
        metadata = profile.metadata.get(item.category, {})
        if item.relative_path in metadata:
            return metadata[item.relative_path]
        collection_name = {"checkpoint": "checkpoints", "character": "characters", "lora": "loras"}[item.category]
        same_stem = [asset for asset in getattr(profile, collection_name) if asset.name == item.name]
        return metadata.get(item.name, {}) if len(same_stem) == 1 else {}

    def prepare_from_image(self, profile, prompt, checkpoint, character, loras):
        workflow = copy.deepcopy(prompt)
        self._apply_settings(workflow, profile.setup_workflow)
        checkpoint_metadata = self._metadata(profile, checkpoint)
        self._apply_checkpoint_sampler(workflow, checkpoint_metadata)
        self._apply_node_metadata(workflow, checkpoint_metadata)
        assets = [*([character] if character else []), *loras]
        for node in workflow.values():
            if not isinstance(node, dict) or not str(node.get("class_type", "")).startswith("LoraLoader"):
                continue
            filename = str(node.get("inputs", {}).get("lora_name", ""))
            stem = filename.replace("\\\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
            item = next((asset for asset in assets if asset.name == stem), None)
            if item:
                item_metadata = self._metadata(profile, item)
                self._apply_node_metadata(workflow, item_metadata)
                defaults = profile.setup_workflow.get("charDefault" if item.category == "character" else "loraDefault", {})
                self._apply_strengths(node, item_metadata, defaults)
        self._randomize_seeds(workflow)
        self._set_save_prefix(workflow, profile, checkpoint, character, loras)
        for node_id in ("PrimitiveStringMultilineP", "PrimitiveStringMultilineN"):
            if node_id in workflow:
                workflow[node_id].setdefault("inputs", {})["value"] = ""
        return workflow

    def _apply_settings(self, workflow: dict, settings: dict) -> None:
        workflow_settings = settings.get("workflow")
        if not isinstance(workflow_settings, dict):
            excluded = {"charDefault", "loraDefault", "workflow_scale", "workflow_min", "workflow_max"}
            workflow_settings = {key: value for key, value in settings.items()
                                 if key not in excluded and isinstance(value, dict)}
        for node_id, values in workflow_settings.items():
            if node_id in workflow and isinstance(values, dict):
                for key, value in values.items():
                    selected = self._random_value(value)
                    if selected is not None:
                        workflow[node_id].setdefault("inputs", {})[key] = selected
        for meta in ("workflow_scale", "workflow_min", "workflow_max"):
            for node_id, values in settings.get(meta, {}).items():
                if node_id not in workflow or not isinstance(values, dict):
                    continue
                for key, raw in values.items():
                    if key not in workflow[node_id].get("inputs", {}):
                        continue
                    amount = self._random_value(raw)
                    current = workflow[node_id]["inputs"][key]
                    if isinstance(current, bool) or not isinstance(current, (int, float)):
                        continue
                    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                        continue
                    if meta == "workflow_scale":
                        workflow[node_id]["inputs"][key] = current * amount
                    elif meta == "workflow_min":
                        workflow[node_id]["inputs"][key] = max(current, amount)
                    else:
                        workflow[node_id]["inputs"][key] = min(current, amount)

    def _random_value(self, value):
        if isinstance(value, dict):
            if not value:
                return None
            options = list(value)
            weights = []
            for weight in value.values():
                try:
                    weights.append(max(0.0, float(weight)))
                except (TypeError, ValueError):
                    weights.append(0.0)
            if not any(weights):
                return None
            return self.rng.choices(options, weights=weights, k=1)[0]
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
        preferred = ("UNETLoader", "CheckpointLoaderSimple") if profile_name == "Anime" else (
            "CheckpointLoaderSimple", "UNETLoader"
        )
        for loader_type in preferred:
            input_key = "unet_name" if loader_type == "UNETLoader" else "ckpt_name"
            direct = workflow.get(loader_type)
            if isinstance(direct, dict):
                direct.setdefault("inputs", {})[input_key] = relative_path
                return
            for node in workflow.values():
                if isinstance(node, dict) and node.get("class_type") == loader_type:
                    node.setdefault("inputs", {})[input_key] = relative_path
                    return

    def _attach_loras(self, workflow, loras, profile_name, checkpoint_metadata):
        self._last_lora_node_ids = []
        template_id = "LoraLoader" if isinstance(workflow.get("LoraLoader"), dict) else next(
            (node_id for node_id, node in workflow.items()
             if isinstance(node, dict) and str(node.get("class_type", "")).startswith("LoraLoader")),
            None
        )
        if template_id is None:
            return
        self._base_lora_node_id = template_id
        template = workflow[template_id]
        next_node = template_id
        for index, lora in enumerate(loras, start=1):
            node_id = f"LoraLoaderAuto{index}"
            while node_id in workflow:
                node_id += "_"
            self._last_lora_node_ids.append(node_id)
            workflow[node_id] = copy.deepcopy(template)
            inputs = workflow[node_id].setdefault("inputs", {})
            inputs["lora_name"] = lora.relative_path
            prior_inputs = workflow.get(next_node, {}).get("inputs", {})
            model_connection = prior_inputs.get("model")
            clip_connection = prior_inputs.get("clip")
            if profile_name != "Anime":
                if isinstance(checkpoint_metadata.get("ModelSamplingDiscrete"), dict):
                    model_connection = ["ModelSamplingDiscrete", 0]
                if isinstance(checkpoint_metadata.get("CLIPSetLastLayer"), dict):
                    clip_connection = ["CLIPSetLastLayer", 0]
            if model_connection is not None:
                inputs["model"] = model_connection
            if clip_connection is not None:
                inputs["clip"] = clip_connection
            if next_node in workflow:
                workflow[next_node].setdefault("inputs", {})["model"] = [node_id, 0]
                workflow[next_node].setdefault("inputs", {})["clip"] = [node_id, 1]
            next_node = node_id

    def _apply_checkpoint_sampler(self, workflow, metadata):
        sampler = workflow.get("KSampler")
        if not isinstance(sampler, dict):
            sampler = next((node for node in workflow.values()
                            if isinstance(node, dict) and node.get("class_type") == "KSampler"), None)
        if not isinstance(sampler, dict):
            return
        inputs = sampler.get("inputs", {})
        for key, value in metadata.items():
            if key in inputs:
                selected = self._random_value(value)
                if selected is not None:
                    inputs[key] = selected

    def _randomize_seeds(self, workflow):
        for node in workflow.values():
            if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
                if "seed" in node["inputs"]:
                    node["inputs"]["seed"] = self.rng.randint(0, 2**63 - 1)

    def _apply_node_metadata(self, workflow, metadata):
        for node_id, values in metadata.items():
            if node_id in workflow and isinstance(values, dict):
                inputs = workflow[node_id].setdefault("inputs", {})
                for key, value in values.items():
                    selected = self._random_value(value)
                    if selected is not None:
                        inputs[key] = selected

    def _set_save_prefix(self, workflow, profile, checkpoint, character, loras):
        from datetime import datetime

        checkpoint_metadata = self._metadata(profile, checkpoint)
        character_metadata = self._metadata(profile, character) if character else {}

        def flag_marker(metadata):
            if metadata.get("favorites") not in (False, None, 0):
                return "#"
            skip_value = metadata.get("skip")
            if skip_value == "auto":
                return "@"
            if skip_value not in (False, None, 0):
                return "="
            return "+"

        checkpoint_marker = flag_marker(checkpoint_metadata)
        character_marker = "%" if character is None else flag_marker(character_metadata)
        checkpoint_weight = checkpoint_metadata.get("weight", "")
        character_weight = character_metadata.get("weight", "") if character else ""
        character_name = character.name if character else "Wildcard"

        tags = []
        for item in loras:
            raw_tags = self._metadata(profile, item).get("tag", [])
            if isinstance(raw_tags, str):
                tags.append(raw_tags.lower())
            elif isinstance(raw_tags, (list, tuple, set)):
                tags.extend(str(tag).lower() for tag in raw_tags)
        style_dress = ("S" if "style" in tags else "") + ("D" if "dress" in tags else "")

        checkpoint_label = f"{checkpoint.name}{checkpoint_marker}{checkpoint_weight}"
        character_label = f"{character_name}{character_marker}{character_weight}"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._generation_counter += 1
        prefix = (
            f"{profile.name}/{checkpoint_label}/{character_label}/"
            f"{checkpoint_label}-{character_label}-{style_dress}-{len(loras)}-"
            f"{timestamp}-{self._generation_counter}"
        )
        for node_id, suffix in (("SaveImage1", "-1"), ("SaveImage2", "-2"), ("SaveVideo", "")):
            node = workflow.get(node_id)
            if isinstance(node, dict):
                node.setdefault("inputs", {})["filename_prefix"] = prefix + suffix
        if self.settings.get("noSaveImage1", profile.setup_workflow.get("noSaveImage1", False)):
            workflow.pop("SaveImage1", None)

    def _apply_character_defaults(self, workflow, profile, character):
        node = workflow.get(self._base_lora_node_id, {})
        if not isinstance(node, dict):
            node = next((value for value in workflow.values()
                         if isinstance(value, dict) and str(value.get("class_type", "")).startswith("LoraLoader")), {})
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
        for index, lora in enumerate(loras):
            node_id = self._last_lora_node_ids[index] if index < len(self._last_lora_node_ids) else f"LoraLoader{index + 1}"
            self._apply_strengths(workflow.get(node_id, {}),
                                  self._metadata(profile, lora), defaults)

    def _apply_strengths(self, node, metadata, defaults):
        inputs = node.get("inputs", {})
        for key in ("strength_model", "strength_clip", "A", "B", "preset", "block_vector"):
            value = metadata.get(key, defaults.get(key))
            if value is not None:
                selected = self._random_value(value)
                if selected is not None:
                    inputs[key] = selected

    @staticmethod
    def _apply_prompt_metadata(workflow, profile, checkpoint, character, loras):
        info = workflow.get("PrimitiveStringMultilineInfo")
        if not isinstance(info, dict):
            info = next((node for node in workflow.values()
                         if isinstance(node, dict) and node.get("class_type") == "PrimitiveStringMultilineInfo"), None)
        if isinstance(info, dict):
            lines = [profile.name, f"Checkpoint: {checkpoint.relative_path}"]
            if character:
                lines.append(f"Character: {character.relative_path}")
            lines.extend(f"LoRA: {item.relative_path}" for item in loras)
            info.setdefault("inputs", {})["value"] = "\n".join(lines) + "\n"
