import tempfile
import unittest
from pathlib import Path

from comfy_profile_automation.catalog import ModelFile, ProfileCatalog


class CatalogTests(unittest.TestCase):
    def test_duplicate_stems_keep_distinct_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("IL/checkpoint/a", "IL/checkpoint/b"):
                path = root / folder
                path.mkdir(parents=True)
                (path / "same.safetensors").write_text("x")
            assets = ProfileCatalog(root, {"IL": {}, "Anime": {}}).load("IL").checkpoints
            self.assertEqual(len(assets), 2)
            self.assertEqual({x.relative_path for x in assets},
                             {"a/same.safetensors", "b/same.safetensors"})

    def test_anime_does_not_fallback_to_shared_parent_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "models" / "diffusion_models"
            (shared / "IL").mkdir(parents=True)
            (shared / "IL" / "not_anime.safetensors").write_text("x")
            profiles = {
                "Anime": {"checkpoint_path": str(shared / "Anime")},
                "IL": {"checkpoint_path": str(shared / "IL")},
            }
            catalog = ProfileCatalog(root / "data", profiles, base_dir=root)
            self.assertEqual(catalog.load("Anime").checkpoints, [])
            self.assertEqual([x.name for x in catalog.load("IL").checkpoints], ["not_anime"])

    def test_workflow_json_accepts_trailing_commas_in_objects_and_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow_api.json"
            path.write_text(
                '{"1": {"class_type": "Test", "inputs": {"text": "comma, } in string", "items": [1, 2,],},},}',
                encoding="utf-8",
            )
            loaded = ProfileCatalog._load_workflow_json(path)
            self.assertEqual(loaded["1"]["inputs"]["items"], [1, 2])
            self.assertEqual(loaded["1"]["inputs"]["text"], "comma, } in string")

    def test_invalid_workflow_json_reports_file_and_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow_api.json"
            path.write_text('{"1": {"inputs": [1, 2, }', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, str(path)):
                ProfileCatalog._load_workflow_json(path)

    def test_workflow_json_must_be_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow_api.json"
            path.write_text('[1, 2, 3]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                ProfileCatalog._load_workflow_json(path)

    def test_yaml_errors_include_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.yml"
            path.write_text("key: [unterminated", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, str(path)):
                ProfileCatalog._yaml(path)

    def test_yaml_root_must_be_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.yml"
            path.write_text("- one\n- two\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root must be a mapping"):
                ProfileCatalog._yaml(path)

    def test_profile_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for profile in ("IL", "Anime"):
                folder = root / profile / "checkpoint"
                folder.mkdir(parents=True)
                (folder / f"{profile}.safetensors").write_text("x")
            catalog = ProfileCatalog(root, {"IL": {}, "Anime": {}})
            self.assertEqual([x.name for x in catalog.load("IL").checkpoints], ["IL"])
            self.assertEqual([x.name for x in catalog.load("Anime").checkpoints], ["Anime"])


    def test_asset_metadata_zero_weight_overrides_configured_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = ModelFile(
                "IL", "checkpoint", root / "base.safetensors", "IL/base.safetensors"
            )
            weights = ProfileCatalog(root, {"IL": {}})._effective_asset_weights(
                "checkpoint", [model], {"IL/base.safetensors": {"weight": 0}},
                {"IL/base.safetensors": 9},
            )
            self.assertEqual(weights[model.identity], 0)
            self.assertEqual(weights[model.relative_path], 0)

if __name__ == "__main__":
    unittest.main()
