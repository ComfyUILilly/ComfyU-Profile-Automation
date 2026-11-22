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

    def test_repeated_windows_separators_are_collapsed(self):
        self.assertEqual(
            ComfyUIClient._posix(r"folder\\\\nested\\\\model.safetensors"),
            "folder/nested/model.safetensors",
        )

    def test_legacy_custom_model_nodes_are_synchronized(self):
        client = ComfyUIClient(
            "http://127.0.0.1:8188/prompt",
            "http://127.0.0.1:8188/queue",
        )
        response = Mock()
        response.json.return_value = ["controlnet/control.safetensors"]
        response.raise_for_status.return_value = None
        client.session.get = Mock(return_value=response)
        workflow = {
            "1": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": "missing/control.safetensors"},
            }
        }
        client.sync_model_names(workflow)
        self.assertEqual(
            workflow["1"]["inputs"]["control_net_name"],
            "controlnet/control.safetensors",
        )
        self.assertEqual(
            client.session.get.call_args.args[0],
            "http://127.0.0.1:8188/models/controlnet",
        )

    def test_from_img_fallback_is_used_for_missing_model(self):
        client = ComfyUIClient(
            "http://127.0.0.1:8188/prompt",
            "http://127.0.0.1:8188/queue",
        )
        response = Mock()
        response.json.return_value = []
        response.raise_for_status.return_value = None
        client.session.get = Mock(return_value=response)
        workflow = {
            "1": {
                "class_type": "UltralyticsDetectorProvider",
                "inputs": {"model_name": "missing/model.pt"},
            }
        }
        client.sync_model_names(
            workflow,
            from_img_fallbacks={
                "UltralyticsDetectorProvider": {"model_name": "bbox/face_yolov9c.pt"}
            },
        )
        self.assertEqual(
            workflow["1"]["inputs"]["model_name"],
            "bbox/face_yolov9c.pt",
        )

    def test_lora_loader_block_weight_uses_lora_endpoint(self):
        client = ComfyUIClient(
            "http://127.0.0.1:8188/prompt",
            "http://127.0.0.1:8188/queue",
        )
        response = Mock()
        response.json.return_value = ["IL/etc/sample.safetensors"]
        response.raise_for_status.return_value = None
        client.session.get = Mock(return_value=response)
        workflow = {
            "1": {
                "class_type": "LoraLoaderBlockWeight",
                "inputs": {"lora_name": "sample.safetensors"},
            }
        }
        client.sync_model_names(workflow)
        self.assertEqual(
            workflow["1"]["inputs"]["lora_name"],
            "IL/etc/sample.safetensors",
        )
        self.assertEqual(
            client.session.get.call_args.args[0],
            "http://127.0.0.1:8188/models/loras",
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


    def test_wait_until_idle_rejects_invalid_poll_interval(self):
        client = ComfyUIClient(
            "http://127.0.0.1:8188/prompt",
            "http://127.0.0.1:8188/queue",
        )
        with self.assertRaisesRegex(ValueError, "poll_interval"):
            client.wait_until_idle(poll_interval=0)

    def test_wait_until_idle_rejects_negative_timeout(self):
        client = ComfyUIClient(
            "http://127.0.0.1:8188/prompt",
            "http://127.0.0.1:8188/queue",
        )
        with self.assertRaisesRegex(ValueError, "max_wait"):
            client.wait_until_idle(max_wait=-1)

if __name__ == "__main__":
    unittest.main()
