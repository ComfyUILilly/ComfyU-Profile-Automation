import unittest
from pathlib import Path

from comfy_profile_automation.catalog import ModelFile, ProfileData
from comfy_profile_automation.workflow import WorkflowBuilder
from comfy_profile_automation.wildcards import WildcardEngine


class WorkflowTests(unittest.TestCase):
    def test_cross_profile_assets_are_rejected(self):
        profile = ProfileData(name="IL", root=Path("."),
                              workflow_api={"1": {"inputs": {}}})
        checkpoint = ModelFile("Anime", "checkpoint", Path("model.safetensors"),
                               "model.safetensors")
        with self.assertRaises(ValueError):
            WorkflowBuilder().build(profile, checkpoint, None, [])

    def test_setup_values_are_applied_without_mutating_source(self):
        source = {"1": {"inputs": {"steps": 20}}}
        profile = ProfileData(name="IL", root=Path("."),
                              workflow_api={"1": {"inputs": {"steps": 10}}},
                              setup_workflow={"workflow": {"1": {"steps": 30}}})
        checkpoint = ModelFile("IL", "checkpoint", Path("base.safetensors"),
                               "base.safetensors")
        result = WorkflowBuilder().build(profile, checkpoint, None, [])
        self.assertEqual(result["1"]["inputs"]["steps"], 30)
        self.assertEqual(source["1"]["inputs"]["steps"], 20)

    def test_checkpoint_loader_priority_is_profile_specific(self):
        workflow = {
            "CheckpointLoaderSimple": {"inputs": {"ckpt_name": "old"}},
            "UNETLoader": {"inputs": {"unet_name": "old"}},
        }
        builder = WorkflowBuilder()
        builder._set_checkpoint(workflow, "IL", "IL/new.safetensors")
        self.assertEqual(workflow["CheckpointLoaderSimple"]["inputs"]["ckpt_name"], "IL/new.safetensors")
        self.assertEqual(workflow["UNETLoader"]["inputs"]["unet_name"], "old")

        builder._set_checkpoint(workflow, "Anime", "Anime/new.safetensors")
        self.assertEqual(workflow["UNETLoader"]["inputs"]["unet_name"], "Anime/new.safetensors")

    def test_save_prefix_preserves_legacy_flag_markers_and_style_tags(self):
        checkpoint = ModelFile("IL", "checkpoint", Path("base.safetensors"), "IL/base.safetensors")
        character = ModelFile("IL", "character", Path("hero.safetensors"), "IL/char/hero.safetensors")
        lora = ModelFile("IL", "lora", Path("style.safetensors"), "IL/etc/style.safetensors")
        profile = ProfileData(
            name="IL",
            root=Path("."),
            checkpoints=[checkpoint],
            characters=[character],
            loras=[lora],
            metadata={
                "checkpoint": {"IL/base.safetensors": {"skip": "auto", "weight": 12}},
                "character": {"IL/char/hero.safetensors": {"favorites": True, "weight": 3}},
                "lora": {"IL/etc/style.safetensors": {"tag": ["style", "dress"]}},
            },
            setup_workflow={},
        )
        builder = WorkflowBuilder()
        workflow = {"SaveImage1": {"inputs": {}}}
        builder._set_save_prefix(workflow, profile, checkpoint, character, [lora])
        prefix = workflow["SaveImage1"]["inputs"]["filename_prefix"]
        self.assertIn("base@12", prefix)
        self.assertIn("hero#3", prefix)
        self.assertIn("-SD-1-", prefix)

    def test_wildcard_merge_is_recursive(self):
        engine = WildcardEngine()
        prompts = {"positive": {"group": {"first": "a"}}}
        engine._merge(prompts, {"positive": {"group": {"second": "b"}}})
        self.assertEqual(prompts["positive"]["group"], {"first": "a", "second": "b"})

    def test_zero_weight_wildcard_mapping_returns_empty(self):
        engine = WildcardEngine()
        self.assertEqual(engine._pick({"a": 0, "b": -1}), "")


if __name__ == "__main__":
    unittest.main()
