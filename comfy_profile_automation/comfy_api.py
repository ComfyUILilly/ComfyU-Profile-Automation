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

    # Keep the endpoint/input mapping used by the original automation.
    # In particular, legacy ComfyUI exposes UNETLoader choices through
    # /models/checkpoints on installations used by this project.
    MODEL_INPUTS = {
        "UltralyticsDetectorProvider": ("model_name", "ultralytics"),
        "SAMLoader": ("model_name", "sams"),
        "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
        "UNETLoader": ("unet_name", "checkpoints"),
        "LoraLoaderBlockWeight": ("lora_name", "loras"),
        "LoraLoader": ("lora_name", "loras"),
        "ControlNetLoader": ("control_net_name", "controlnet"),
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

    @staticmethod
    def _from_img_fallback(from_img_fallbacks: dict[str, Any] | None,
                           class_type: str, input_key: str) -> str | None:
        if not isinstance(from_img_fallbacks, dict):
            return None
        direct = from_img_fallbacks.get(class_type)
        if isinstance(direct, dict):
            value = direct.get(input_key)
            if isinstance(value, str) and value:
                return value
        for prefix, settings in from_img_fallbacks.items():
            if class_type.startswith(str(prefix)) and isinstance(settings, dict):
                value = settings.get(input_key)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _model_spec(class_type: str) -> tuple[str, str] | None:
        exact = ComfyUIClient.MODEL_INPUTS.get(class_type)
        if exact:
            return exact
        # Legacy custom node families commonly append a suffix to their name.
        for prefix in sorted(ComfyUIClient.MODEL_INPUTS, key=len, reverse=True):
            if class_type.startswith(prefix):
                return ComfyUIClient.MODEL_INPUTS[prefix]
        return None

    def sync_model_names(
        self,
        workflow: dict[str, Any],
        force_linux_path_nodes: list[str] | tuple[str, ...] | None = None,
        from_img_fallbacks: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synchronize legacy model-loader nodes with ComfyUI's model lists.

        Exact relative paths win. A basename is used only when it identifies
        one model, preventing accidental cross-folder selection. When the
        current model is missing, the old fromImgIf fallback is applied.
        Unavailable endpoints never block prompt submission.
        """
        forced = {str(value) for value in (force_linux_path_nodes or [])}
        model_cache: dict[str, set[str] | None] = {}

        def available(folder: str) -> set[str] | None:
            if folder not in model_cache:
                model_cache[folder] = self._available_models(folder)
            return model_cache[folder]

        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type", ""))
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue

            force_linux = node_id in forced or class_type in forced
            if force_linux:
                normalized_inputs = self._replace_separators(inputs)
                inputs.clear()
                inputs.update(normalized_inputs)

            model_spec = self._model_spec(class_type)
            if model_spec is None:
                continue

            input_key, folder = model_spec
            selected = inputs.get(input_key)
            if not isinstance(selected, str) or not selected.strip():
                continue

            model_names = available(folder)
            selected_normalized = self._normalize_path(selected)
            canonical = None
            if model_names:
                exact = {
                    self._normalize_path(name).casefold(): name
                    for name in model_names
                }
                canonical = exact.get(selected_normalized.casefold())
                if canonical is None:
                    basename = PurePosixPath(selected_normalized).name.casefold()
                    matches = [
                        name for name in model_names
                        if PurePosixPath(self._normalize_path(name)).name.casefold() == basename
                    ]
                    if len(matches) == 1:
                        canonical = matches[0]

            if canonical is not None:
                inputs[input_key] = self._posix(canonical) if force_linux else canonical
                continue

            fallback = self._from_img_fallback(from_img_fallbacks, class_type, input_key)
            if fallback:
                fallback_normalized = self._normalize_path(fallback)
                fallback_match = None
                if model_names:
                    exact = {
                        self._normalize_path(name).casefold(): name
                        for name in model_names
                    }
                    fallback_match = exact.get(fallback_normalized.casefold())
                    if fallback_match is None:
                        fallback_basename = PurePosixPath(fallback_normalized).name.casefold()
                        matches = [
                            name for name in model_names
                            if PurePosixPath(self._normalize_path(name)).name.casefold() == fallback_basename
                        ]
                        if len(matches) == 1:
                            fallback_match = matches[0]
                resolved_fallback = fallback_match or fallback
                inputs[input_key] = self._posix(resolved_fallback) if force_linux else resolved_fallback
                log.warning(
                    "[%s] Model not found: %s. Fallback applied: %s",
                    class_type, selected, inputs[input_key],
                )
            elif force_linux:
                # The forced node was normalized above; no guessed replacement.
                inputs[input_key] = self._posix(selected)

        return workflow

    def queue_prompt(self, workflow: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(self.prompt_url, json={"prompt": workflow}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def wait_until_idle(self, poll_interval: float = 1.0,
                        max_wait: float | None = None) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")
        if max_wait is not None and max_wait < 0:
            raise ValueError("max_wait must be non-negative or None")
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
