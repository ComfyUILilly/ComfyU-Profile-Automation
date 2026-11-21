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


if __name__ == "__main__":
    unittest.main()
