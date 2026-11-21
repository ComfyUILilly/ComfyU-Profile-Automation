from __future__ import annotations

import logging
import re
import time
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests


log = logging.getLogger(__name__)


class ComfyUIClient:
    """Small ComfyUI API client with tolerant model-name/path synchronization."""

    MODEL_INPUTS = {
        "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
        "UNETLoader": ("unet_name", "diffusion_models"),
        "VAELoader": ("vae_name", "vae"),
    }

    def __init__(self, prompt_url: str, queue_url: str, timeout: int = 30):
        self.prompt_url, self.queue_url, self.timeout = prompt_url, queue_url, timeout
        self.session = requests.Session()
        self._model_cache: dict[str, tuple[float, set[str] | None]] = {}
        parts = urlsplit(prompt_url)
        path = parts.path.rsplit("/", 1)[0] if "/" in parts.path else ""
        self.server_url = urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", "")).rstrip("/")

    @staticmethod
    def _normalize_path(value: str) -> str:
        # ComfyUI model names are relative paths; collapse both Windows
        # separators and repeated separators into a single POSIX separator.
        return re.sub(r"/+", "/", str(value).strip().replace("\\", "/"))

    @staticmethod
    def _posix(value: str) -> str:
        return ComfyUIClient._normalize_path(value)

    @staticmethod
    def _flatten_names(value: Any) -> set[str]:
        names: set[str] = set()

        def visit(item: Any) -> None:
            if isinstance(item, str):
                cleaned = ComfyUIClient._normalize_path(item)
                if cleaned:
                    names.add(cleaned)
            elif isinstance(item, dict):
                for key in ("name", "filename", "path"):
                    if isinstance(item.get(key), str):
                        visit(item[key])
                        return
                for child in item.values():
                    visit(child)
            elif isinstance(item, (list, tuple, set)):
                for child in item:
                    visit(child)

        if isinstance(value, dict):
            value = value.get("models", value.get("files", value))
        visit(value)
        return names

    def _available_models(self, folder: str) -> set[str] | None:
        now = time.monotonic()
        cached = self._model_cache.get(folder)
        if cached and now - cached[0] < 10:
            return cached[1]

        try:
            response = self.session.get(
                f"{self.server_url}/models/{folder}",
                timeout=self.timeout,
            )
            response.raise_for_status()
            names = self._flatten_names(response.json())
        except (requests.RequestException, ValueError) as exc:
            # Model listing is advisory: keep the selected local name if this
            # endpoint is unavailable (older ComfyUI builds/custom servers).
            log.debug("Unable to refresh ComfyUI model list for %s: %s", folder, exc)
            names = None

        self._model_cache[folder] = (now, names)
        return names

    @staticmethod
    def _replace_separators(value: Any) -> Any:
        if isinstance(value, str):
            return ComfyUIClient._normalize_path(value)
        if isinstance(value, list):
            return [ComfyUIClient._replace_separators(item) for item in value]
        if isinstance(value, tuple):
            return tuple(ComfyUIClient._replace_separators(item) for item in value)
        if isinstance(value, dict):
            return {key: ComfyUIClient._replace_separators(item) for key, item in value.items()}
        return value

    def sync_model_names(self, workflow: dict[str, Any],
                         force_linux_path_nodes: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
        """Normalize model paths and resolve unique filename aliases against ComfyUI.

        The server model-list endpoints are used when available. If a list cannot
        be fetched, or a name is not present, the workflow's original selection
        is retained rather than blocking generation.
        """
        forced = {str(value) for value in (force_linux_path_nodes or [])}
        lora_names: set[str] | None = None
        lora_names_loaded = False

        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type", ""))
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue

            if node_id in forced or class_type in forced:
                normalized_inputs = self._replace_separators(inputs)
                inputs.clear()
                inputs.update(normalized_inputs)

            model_spec = self.MODEL_INPUTS.get(class_type)
            if model_spec is None and class_type.startswith("LoraLoader"):
                model_spec = ("lora_name", "loras")
            if model_spec is None:
                continue

            input_key, folder = model_spec
            selected = inputs.get(input_key)
            if not isinstance(selected, str) or not selected.strip():
                continue

            normalized = self._posix(selected)
            if folder == "loras":
                if not lora_names_loaded:
                    lora_names = self._available_models(folder)
                    lora_names_loaded = True
                available = lora_names
            else:
                available = self._available_models(folder)
            if not available:
                inputs[input_key] = normalized
                continue

            exact = {name.casefold(): name for name in available}
            canonical = exact.get(normalized.casefold())
            if canonical is None:
                basename = PurePosixPath(normalized).name.casefold()
                matches = [name for name in available if PurePosixPath(name).name.casefold() == basename]
                if len(matches) == 1:
                    canonical = matches[0]
            inputs[input_key] = canonical or normalized

        return workflow

    def queue_prompt(self, workflow: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(self.prompt_url, json={"prompt": workflow}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def wait_until_idle(self, poll_interval: float = 1.0,
                        max_wait: float | None = None) -> None:
        started = time.monotonic()
        while True:
            response = self.session.get(self.queue_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if not data.get("queue_running") and not data.get("queue_pending"):
                return
            if max_wait is not None and time.monotonic() - started > max_wait:
                raise TimeoutError("Timed out waiting for ComfyUI queue")
            time.sleep(poll_interval)
