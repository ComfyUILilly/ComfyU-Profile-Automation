import unittest
from pathlib import Path

from comfy_profile_automation.catalog import ModelFile, ProfileData
from comfy_profile_automation.from_image import FromImageWorkflow


class FromImageTests(unittest.TestCase):
    def test_from_img_node_setting_overrides_default_loader_name(self):
        checkpoint = ModelFile(
            "IL", "checkpoint", Path("base.safetensors"), "IL/base.safetensors"
        )
        profile = ProfileData(
            name="IL",
            root=Path("."),
            checkpoints=[checkpoint],
        )
        prompt = {
            "42": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "base.safetensors"},
            },
            "CheckpointLoaderSimple": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "not-selected.safetensors"},
            },
        }
        node_id, input_key, selected = FromImageWorkflow._resolve_checkpoint(
            profile, prompt, {"IL": "42"}
        )
        self.assertEqual(node_id, "42")
        self.assertEqual(input_key, "ckpt_name")
        self.assertIs(selected, checkpoint)


if __name__ == "__main__":
    unittest.main()
