import unittest
from pathlib import Path

from comfy_profile_automation.catalog import ModelFile, ProfileData
from comfy_profile_automation.workflow import WorkflowBuilder
from comfy_profile_automation.wildcards import WildcardEngine


class WorkflowTests(unittest.TestCase):
    def test_from_image_workflow_does_not_require_profile_template(self):
        profile = ProfileData(name="IL", root=Path("."), workflow_api={})
        checkpoint = ModelFile("IL", "checkpoint", Path("base.safetensors"), "IL/base.safetensors")
        prompt = {"CheckpointLoaderSimple": {"inputs": {"ckpt_name": "old.safetensors"}}}
        result = WorkflowBuilder().prepare_from_image(profile, prompt, checkpoint, None, [])
        self.assertEqual(
            result["CheckpointLoaderSimple"]["inputs"]["ckpt_name"],
            "old.safetensors",
        )

    def test_weighted_dictionary_setting_uses_legacy_weight_choice(self):
        builder = WorkflowBuilder()
        builder.rng = __import__("random").Random(5)
        self.assertEqual(builder._random_value({"first": 0, "second": 5}), "second")
        self.assertIsNone(builder._random_value({"first": 0, "second": -1}))

    def test_workflow_scale_does_not_multiply_non_numeric_values(self):
        workflow = {
            "1": {"inputs": {"text": "keep", "steps": 20}}
        }
        builder = WorkflowBuilder()
        builder._apply_settings(
            workflow,
            {"workflow": {}, "workflow_scale": {"1": {"text": 2, "steps": 2}}},
        )
        self.assertEqual(workflow["1"]["inputs"]["text"], "keep")
        self.assertEqual(workflow["1"]["inputs"]["steps"], 40)

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


    def test_empty_weighted_settings_preserve_existing_inputs(self):
        workflow = {
            "1": {"inputs": {"sampler_name": "euler", "steps": 20}},
        }
        builder = WorkflowBuilder()
        builder._apply_settings(workflow, {
            "workflow": {"1": {"sampler_name": {"invalid": 0, "disabled": -1}}},
        })
        builder._apply_node_metadata(workflow, {
            "1": {"steps": {"zero": 0, "none": -1}},
        })
        self.assertEqual(workflow["1"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(workflow["1"]["inputs"]["steps"], 20)

    def test_saveimage_nodes_have_distinct_legacy_suffixes(self):
        checkpoint = ModelFile("IL", "checkpoint", Path("base.safetensors"), "IL/base.safetensors")
        profile = ProfileData(name="IL", root=Path("."), checkpoints=[checkpoint])
        workflow = {
            "SaveImage1": {"inputs": {}},
            "SaveImage2": {"inputs": {}},
            "SaveVideo": {"inputs": {}},
        }
        builder = WorkflowBuilder()
        builder._set_save_prefix(workflow, profile, checkpoint, None, [])
        prefix = workflow["SaveVideo"]["inputs"]["filename_prefix"]
        self.assertEqual(workflow["SaveImage1"]["inputs"]["filename_prefix"], prefix + "-1")
        self.assertEqual(workflow["SaveImage2"]["inputs"]["filename_prefix"], prefix + "-2")

    def test_wildcard_character_mode_loads_a_valid_disabled_lora(self):
        checkpoint = ModelFile("IL", "checkpoint", Path("base.safetensors"), "IL/base.safetensors")
        chars = [
            ModelFile("IL", "character", Path("hero1.safetensors"), "IL/char/hero1.safetensors"),
            ModelFile("IL", "character", Path("hero2.safetensors"), "IL/char/hero2.safetensors"),
        ]
        profile = ProfileData(
            name="IL",
            root=Path("."),
            checkpoints=[checkpoint],
            characters=chars,
            workflow_api={
                "CheckpointLoaderSimple": {"inputs": {"ckpt_name": "old"}},
                "LoraLoader": {"class_type": "LoraLoader", "inputs": {
                    "lora_name": "stale.safetensors", "strength_model": 1.0,
                    "strength_clip": 1.0, "model": ["CheckpointLoaderSimple", 0],
                    "clip": ["CheckpointLoaderSimple", 1],
                }},
            },
        )
        workflow = WorkflowBuilder(rng=__import__("random").Random(1)).build(
            profile, checkpoint, None, []
        )
        self.assertIn(workflow["LoraLoader"]["inputs"]["lora_name"],
                      {item.relative_path for item in chars})
        self.assertEqual(workflow["LoraLoader"]["inputs"]["strength_model"], 0.0)
        self.assertEqual(workflow["LoraLoader"]["inputs"]["strength_clip"], 0.0)

    def test_lora_wildcard_prompt_is_only_added_in_wildcard_mode(self):
        checkpoint = ModelFile("IL", "checkpoint", Path("base.safetensors"), "IL/base.safetensors")
        profile = ProfileData(
            name="IL", root=Path("."), checkpoints=[checkpoint],
            setup_wildcard={"LoraWildcard": {"positive": {"lora": "wildcard-token"}}},
            workflow_api={
                "CheckpointLoaderSimple": {"inputs": {"ckpt_name": "old"}},
                "PrimitiveStringMultilineP": {"inputs": {"value": ""}},
                "PrimitiveStringMultilineN": {"inputs": {"value": ""}},
            },
        )
        normal = WorkflowBuilder().build(profile, checkpoint, None, [], lora_kind="Weight")
        self.assertNotIn("wildcard-token", normal["PrimitiveStringMultilineP"]["inputs"]["value"])
        wildcard = WorkflowBuilder().build(profile, checkpoint, None, [], lora_kind="Wildcard")
        self.assertIn("wildcard-token", wildcard["PrimitiveStringMultilineP"]["inputs"]["value"])

    def test_excluded_workflow_nodes_are_not_overridden(self):
        workflow = {"1": {"inputs": {"steps": 20}}}
        builder = WorkflowBuilder(settings={"excludeNode": ["1"]})
        builder._apply_settings(workflow, {"workflow": {"1": {"steps": 40}}})
        self.assertEqual(workflow["1"]["inputs"]["steps"], 20)

    def test_prompt_metadata_preserves_template_info(self):
        checkpoint = ModelFile("IL", "checkpoint", Path("base.safetensors"), "IL/base.safetensors")
        profile = ProfileData(
            name="IL", root=Path("."), checkpoints=[checkpoint],
            workflow_api={
                "CheckpointLoaderSimple": {"inputs": {"ckpt_name": "old"}},
                "PrimitiveStringMultilineInfo": {"inputs": {"value": "Template note"}},
            },
        )
        result = WorkflowBuilder().build(profile, checkpoint, None, [])
        info = result["PrimitiveStringMultilineInfo"]["inputs"]["value"]
        self.assertIn("Template note", info)
        self.assertIn("Checkpoint: IL/base.safetensors", info)

if __name__ == "__main__":
    unittest.main()
