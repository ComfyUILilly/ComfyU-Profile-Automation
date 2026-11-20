from __future__ import annotations

import random
import re
from typing import Any


_TOKEN = re.compile(r"__([A-Za-z0-9_.-]+)__")


class WildcardEngine:
    """Expand profile-local wildcard dictionaries and shuffle weighted alternatives."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def expand(self, text: str, definitions: dict[str, Any], max_depth: int = 12) -> str:
        result = str(text)
        for _ in range(max_depth):
            matches = list(_TOKEN.finditer(result))
            if not matches:
                break
            changed = False
            for match in reversed(matches):
                value = definitions.get(match.group(1))
                if value is None:
                    continue
                result = result[:match.start()] + self._pick(value) + result[match.end():]
                changed = True
            if not changed:
                break
        return result

    def _pick(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return str(self.rng.choice(value)) if value else ""
        if isinstance(value, dict):
            options = list(value.items())
            if not options:
                return ""
            weights = []
            for _, weight in options:
                try:
                    weights.append(max(0.0, float(weight)))
                except (TypeError, ValueError):
                    weights.append(1.0)
            return str(self.rng.choices([k for k, _ in options], weights=weights if any(weights) else None, k=1)[0])
        return str(value)

    def apply(self, workflow: dict, profile_wildcards: dict[str, Any]) -> dict:
        positive = profile_wildcards.get("positive", {})
        negative = profile_wildcards.get("negative", {})
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if "text" in inputs and isinstance(inputs["text"], str):
                inputs["text"] = self.expand(inputs["text"], positive)
            if "value" in inputs and isinstance(inputs["value"], str):
                inputs["value"] = self.expand(inputs["value"], positive)
        return workflow
