import unittest
from pathlib import Path

from comfy_profile_automation.catalog import ModelFile, ProfileData
from comfy_profile_automation.workflow import WorkflowBuilder


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
        profile = ProfileData(name="IL", root=Path("."), workflow_api=source,
                              setup_workflow={"1": {"steps": 30}})
        checkpoint = ModelFile("IL", "checkpoint", Path("base.safetensors"),
                               "base.safetensors")
        result = WorkflowBuilder().build(profile, checkpoint, None, [])
        self.assertEqual(result["1"]["inputs"]["steps"], 30)
        self.assertEqual(source["1"]["inputs"]["steps"], 20)


if __name__ == "__main__":
    unittest.main()
