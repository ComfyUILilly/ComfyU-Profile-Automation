import tempfile
import unittest
from pathlib import Path

import yaml

from comfy_profile_automation.config import load_config


class ConfigTests(unittest.TestCase):
    def test_legacy_lora_count_settings_are_mapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "config.yml"
            config = {
                "base_dir": ".",
                "dataPath": "../data",
                "CheckpointTypes": {"IL": 1},
                "url": "http://127.0.0.1:8188/prompt",
                "url2": "http://127.0.0.1:8188",
                "LoraDbCnt": [2, 4],
                "LoraRandomCnt": [3, 5],
                "LoraCycleCnt": [1, 6],
            }
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = load_config(path)
            self.assertEqual(result["generation"]["lora_db_count"], [2, 4])
            self.assertEqual(result["generation"]["lora_random_count"], [3, 5])
            self.assertEqual(result["generation"]["lora_cycle_count"], [1, 6])

    def test_relative_config_paths_resolve_from_config_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "config.yml"
            path.write_text(
                yaml.safe_dump({
                    "base_dir": "../portable",
                    "data_path": "../data",
                    "profiles": {"IL": {"weight": 1}},
                    "generation": {},
                    "comfyui": {
                        "prompt_url": "http://127.0.0.1:8188/prompt",
                        "queue_url": "http://127.0.0.1:8188/queue",
                    },
                }),
                encoding="utf-8",
            )
            result = load_config(path)
            self.assertEqual(result["base_dir"], str((root.parent / "portable").resolve()))
            self.assertEqual(result["data_path"], str((root.parent / "data").resolve()))


    def test_malformed_config_yaml_reports_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yml"
            path.write_text("profiles: [bad", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, str(path)):
                load_config(path)

    def test_config_root_must_be_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yml"
            path.write_text("- one\n- two\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root must be a mapping"):
                load_config(path)

    def test_invalid_profile_batch_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yml"
            path.write_text(yaml.safe_dump({
                "data_path": "../data",
                "profiles": {"IL": {"weight": 1, "batch_images": [4, 2]}},
                "generation": {},
                "comfyui": {"prompt_url": "http://127.0.0.1:8188/prompt",
                             "queue_url": "http://127.0.0.1:8188/queue"},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch_images range"):
                load_config(path)

    def test_non_finite_profile_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yml"
            path.write_text(yaml.safe_dump({
                "data_path": "../data",
                "profiles": {"IL": {"weight": float("nan")}},
                "generation": {},
                "comfyui": {"prompt_url": "http://127.0.0.1:8188/prompt",
                             "queue_url": "http://127.0.0.1:8188/queue"},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                load_config(path)

if __name__ == "__main__":
    unittest.main()
