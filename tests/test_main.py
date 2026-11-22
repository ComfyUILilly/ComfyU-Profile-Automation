import random
import tempfile
from pathlib import Path
import unittest

from main import _count, _mode


class MainHelperTests(unittest.TestCase):
    def test_count_range_is_inclusive(self):
        rng = random.Random(1)
        self.assertIn(_count([2, 4], rng), {2, 3, 4})

    def test_count_rejects_reversed_or_negative_ranges(self):
        rng = random.Random(1)
        with self.assertRaisesRegex(ValueError, "Invalid count range"):
            _count([4, 2], rng)
        with self.assertRaisesRegex(ValueError, "Invalid count range"):
            _count([-1, 2], rng)

    def test_empty_mode_mapping_falls_back_to_weight(self):
        self.assertEqual(_mode({}, random.Random(1)), "Weight")


    def test_resolve_start_checkpoint_supports_windows_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "IL" / "checkpoint"
            models.mkdir(parents=True)
            (models / "base.safetensors").write_text("model", encoding="utf-8")
            from comfy_profile_automation.catalog import ProfileCatalog
            catalog = ProfileCatalog(root / "data", {
                "IL": {"checkpoint_path": str(models)},
            }, base_dir=root)
            profile_name, checkpoint = __import__("main")._resolve_start_checkpoint(
                r"IL\checkpoint\base.safetensors", catalog
            )
            self.assertEqual(profile_name, "IL")
            self.assertEqual(checkpoint.name, "base")

    def test_sleep_duration_supports_legacy_ranges(self):
        from main import _sleep_duration
        rng = random.Random(1)
        self.assertIn(_sleep_duration([1, 3], rng), {1, 2, 3})
        self.assertGreaterEqual(_sleep_duration([0.1, 0.2], rng), 0.1)
        self.assertLessEqual(_sleep_duration([0.1, 0.2], rng), 0.2)

    def test_sleep_duration_rejects_negative_values(self):
        from main import _sleep_duration
        with self.assertRaisesRegex(ValueError, "Sleep duration"):
            _sleep_duration(-1, random.Random(1))
        with self.assertRaisesRegex(ValueError, "Invalid sleep range"):
            _sleep_duration([2, 1], random.Random(1))

if __name__ == "__main__":
    unittest.main()
