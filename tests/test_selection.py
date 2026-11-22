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

    def test_weight_mode_always_uses_weight_map(self):
        profile = self.make_profile()
        profile.weights["character"] = {
            profile.characters[0].identity: 100,
            profile.characters[1].identity: 0,
        }
        selector = AssetSelector(rng=random.Random(1), settings={"CharWeightPer": 0})
        selected = selector.choose(profile, "character", "Weight")
        self.assertEqual([item.name for item in selected], ["a"])

    def test_skip_true_is_candidate_only_in_skip_mode(self):
        profile = self.make_profile()
        selector = AssetSelector(rng=random.Random(3))
        selected = selector.choose(profile, "character", "Skip")
        self.assertEqual([item.name for item in selected], ["a"])

    def test_skip_auto_is_also_a_skip_candidate(self):
        profile = self.make_profile()
        profile.metadata["character"]["a.safetensors"]["skip"] = "auto"
        selector = AssetSelector(rng=random.Random(3))
        self.assertEqual(selector.choose(profile, "character", "Skip")[0].name, "a")

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

    def test_db_lora_selection_draws_with_replacement_then_deduplicates(self):
        class RepeatFirst(random.Random):
            def choices(self, population, weights=None, *, cum_weights=None, k=1):
                return [population[0]] * k

        class FakeDatabase:
            def counts(self, profile, category):
                return {}

        loras = [
            ModelFile("IL", "lora", Path("a.safetensors"), "IL/etc/a.safetensors"),
            ModelFile("IL", "lora", Path("b.safetensors"), "IL/etc/b.safetensors"),
        ]
        profile = ProfileData(name="IL", root=Path("."), loras=loras)
        selector = AssetSelector(rng=RepeatFirst(2), database=FakeDatabase())
        selected = selector.choose(profile, "lora", "DB", count=2)
        self.assertEqual(selected, [loras[0]])

    def test_group_lora_mapping_selects_one_weighted_name(self):
        selector = AssetSelector(rng=random.Random(7))
        chosen = selector._pick_group_lora({"a": 0, "b": 1})
        self.assertEqual(chosen, ["b"])

    def test_group_lora_mapping_with_no_positive_weights_is_empty(self):
        selector = AssetSelector(rng=random.Random(7))
        self.assertEqual(selector._pick_group_lora({"a": 0, "b": -1}), [])


class CatalogTests(unittest.TestCase):
    def test_asset_yaml_weight_overrides_weight_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            model_dir = root / "models" / "IL" / "char"
            model_dir.mkdir(parents=True)
            (model_dir / "hero.safetensors").write_text("x", encoding="utf-8")

            yml_dir = data_root / "IL" / "lora" / "char"
            yml_dir.mkdir(parents=True)
            (yml_dir / "hero.yml").write_text(
                yaml.safe_dump({"hero": {"skip": True, "weight": 9}}), encoding="utf-8"
            )
            (data_root / "IL" / "WeightChar.yml").write_text("hero: 2\n", encoding="utf-8")

            catalog = ProfileCatalog(
                data_root,
                {"IL": {"character_path": str(model_dir)}},
                defaults={"CharWeightDefault": 3},
            )
            profile = catalog.load("IL")
            item = profile.characters[0]
            self.assertEqual(profile.weights["character"][item.identity], 9)
            self.assertTrue(profile.metadata["character"][item.relative_path]["skip"])

    def test_configured_default_weight_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            model_dir = root / "models" / "IL" / "char"
            model_dir.mkdir(parents=True)
            (model_dir / "hero.safetensors").write_text("x", encoding="utf-8")
            profile = ProfileCatalog(
                data_root,
                {"IL": {"character_path": str(model_dir)}},
                defaults={"CharWeightDefault": 17},
            ).load("IL")
            item = profile.characters[0]
            self.assertEqual(profile.weights["character"][item.identity], 17)


if __name__ == "__main__":
    unittest.main()
