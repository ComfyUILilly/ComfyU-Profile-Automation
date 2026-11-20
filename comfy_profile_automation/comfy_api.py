from __future__ import annotations

import time
from typing import Any

import requests


class ComfyUIClient:
    def __init__(self, prompt_url: str, queue_url: str, timeout: int = 30):
        self.prompt_url, self.queue_url, self.timeout = prompt_url, queue_url, timeout
        self.session = requests.Session()

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
