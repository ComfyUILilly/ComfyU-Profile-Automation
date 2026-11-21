import unittest
from unittest.mock import Mock

from comfy_profile_automation.comfy_api import ComfyUIClient


class ComfyApiTests(unittest.TestCase):
    def test_sync_model_names_resolves_unique_basename(self):
        client = ComfyUIClient(
            "http://127.0.0.1:8188/prompt",
            "http://127.0.0.1:8188/queue",
        )
        response = Mock()
        response.json.return_value = ["subfolder/model.safetensors"]
        response.raise_for_status.return_value = None
        client.session.get = Mock(return_value=response)

        workflow = {
            "CheckpointLoaderSimple": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": r"old\\folder\\model.safetensors"},
            }
        }
        client.sync_model_names(workflow)
        self.assertEqual(
            workflow["CheckpointLoaderSimple"]["inputs"]["ckpt_name"],
            "subfolder/model.safetensors",
        )

    def test_force_linux_path_nodes_normalizes_nested_strings(self):
        client = ComfyUIClient(
            "http://127.0.0.1:8188/prompt",
            "http://127.0.0.1:8188/queue",
        )
        workflow = {
            "Detector": {
                "class_type": "UltralyticsDetectorProvider",
                "inputs": {"model_name": r"bbox\\face_yolov9c.pt"},
            }
        }
        client.sync_model_names(workflow, ["UltralyticsDetectorProvider"])
        self.assertEqual(
            workflow["Detector"]["inputs"]["model_name"],
            "bbox/face_yolov9c.pt",
        )


if __name__ == "__main__":
    unittest.main()
