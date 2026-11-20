from __future__ import annotations

import copy
from typing import Any

from .catalog import ModelFile, ProfileData


class WorkflowBuilder:
    """Apply selected profile-local assets to a ComfyUI API workflow."""

    def build(self, profile: ProfileData, checkpoint: ModelFile,
              character: ModelFile | None, loras: list[ModelFile]) -> dict[str, Any]:
        for item in [checkpoint, *([character] if character else []), *loras]:
            if item.profile != profile.name:
                raise ValueError(f"Cross-profile asset rejected: {item.profile} in {profile.name}")
        workflow = copy.deepcopy(profile.workflow_api)
        if not workflow:
            raise ValueError(f"Missing workflow_api.json for profile {profile.name}")
        self._apply_settings(workflow, profile.setup_workflow)
        if self._set_first_input(workflow, ("UNETLoader",), "unet_name", checkpoint.path.name):
            pass
        else:
            self._set_first_input(workflow, ("CheckpointLoaderSimple",), "ckpt_name", checkpoint.path.name)
        if character:
            self._set_first_input(workflow, ("LoraLoader", "LoraLoaderModelOnly"),
                                  "lora_name", character.path.name)
        for index, lora in enumerate(loras, start=1):
            node_id = f"LoraLoader{index}"
            if node_id in workflow:
                workflow[node_id].setdefault("inputs", {})["lora_name"] = lora.path.name
        return workflow

    @staticmethod
    def _apply_settings(workflow: dict[str, Any], settings: dict[str, Any]) -> None:
        for node_id, values in settings.items():
            if node_id in workflow and isinstance(values, dict):
                workflow[node_id].setdefault("inputs", {}).update(copy.deepcopy(values))

    @staticmethod
    def _set_first_input(workflow: dict[str, Any], node_ids: tuple[str, ...],
                         input_name: str, value: Any) -> bool:
        for node_id in node_ids:
            node = workflow.get(node_id)
            if isinstance(node, dict):
                node.setdefault("inputs", {})[input_name] = value
                return True
        return False
