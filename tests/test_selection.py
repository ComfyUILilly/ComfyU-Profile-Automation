import random
import tempfile
import unittest
from pathlib import Path

import yaml

from comfy_profile_automation.catalog import ProfileCatalog, ModelFile, ProfileData
from comfy_profile_automation.selection import AssetSelector


class SelectionTests(unittest.TestCase):
    def make_profile(self):
        assets = [
            ModelFile("IL", "character", Path("a.safetensors"), "a.safetensors"),
            ModelFile("IL", "character", Path("b.safetensors"), "b.safetensors"),
        ]
        return ProfileData(
            name="IL",
            root=Path("."),
            characters=assets,
            metadata={"character": {
                "a.safetensors": {"skip": True, "weight": 9},
                "b.safetensors": {"skip": False, "weight": 1},
            }},
            weights={"character": {
                assets[0].identity: 9,
                assets[1].identity: 1,
            }},
        )

    def test_skip_true_is_candidate_only_in_skip_mode(self):
        profile = self.make_profile()
        selector = AssetSelector(rng=random.Random(3))
        selected = selector.choose(profile, "character", "Skip")
        self.assertEqual([item.name for item in selected], ["a"])

    def test_skip_true_is_not_global_exclusion(self):
        profile = self.make_profile()
        selector = AssetSelector(rng=random.Random(4))
        selected = selector.choose(profile, "character", "Random", count=2)
        self.assertEqual({item.name for item in selected}, {"a", "b"})

    def test_skip_weight_uses_flagged_candidates(self):
        profile = self.make_profile()
        selector = AssetSelector(rng=random.Random(5))
        selected = selector.choose(profile, "character", "Skip_Weight")
        self.assertEqual([item.name for item in selected], ["a"])

    def test_skip_mode_without_flags_uses_legacy_fallback(self):
        profile = self.make_profile()
        profile.metadata["character"] = {
            "a.safetensors": {"skip": False},
            "b.safetensors": {"skip": False},
        }
        selector = AssetSelector(rng=random.Random(1), settings={"CharWeightPer": 1})
        selected = selector.choose(profile, "character", "Skip")
        self.assertEqual(len(selected), 1)
        self.assertIn(selected[0].name, {"a", "b"})

    def test_group_lora_mapping_selects_one_weighted_name(self):
        selector = AssetSelector(rng=random.Random(7))
        # A mapping is one weighted choice, matching the legacy random_weight helper.
        chosen = selector._pick_group_lora({"a": 0, "b": 1})
        self.assertEqual(chosen, ["b"])


class CatalogTests(unittest.TestCase):
    def test_asset_yaml_weight_overrides_weight_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "IL" / "char"
            model_dir.mkdir(parents=True)
            (model_dir / "hero.safetensors").write_text("x", encoding="utf-8")
            yml_dir = root / "IL" / "lora" / "char"
            yml_dir.mkdir(parents=True)
            (yml_dir / "hero.yml").write_text(
                yaml.safe_dump({"hero": {"skip": True, "weight": 9}}), encoding="utf-8"
            )
            (root / "IL" / "WeightChar.yml").write_text("hero: 2\n", encoding="utf-8")
            catalog = ProfileCatalog(
                root,
                {"IL": {"character_path": str(model_dir)}},
                defaults={"CharWeightDefault": 3},
            )
            profile = catalog.load("IL")
            self.assertEqual(profile.weights["character"]["IL/character/IL/hero.safetensors"], 9)
            self.assertTrue(profile.metadata["character"]["hero"]["skip"])


if __name__ == "__main__":
    unittest.main()
