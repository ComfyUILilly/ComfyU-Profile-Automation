import tempfile
import unittest
from pathlib import Path

from comfy_profile_automation.catalog import ProfileCatalog


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


if __name__ == "__main__":
    unittest.main()
