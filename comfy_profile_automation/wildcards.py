from __future__ import annotations

import random
import re
from typing import Any


_TOKEN = re.compile(r"__([A-Za-z0-9_.-]+)__")


class WildcardEngine:
    """Expand nested weighted wildcard tokens using only the active profile's definitions."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def _pick(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return str(self.rng.choice(value)) if value else ""
        if isinstance(value, dict):
            if "options" in value:
                return self._pick(value["options"])
            options, weights = [], []
            for key, weight in value.items():
                options.append(str(key))
                try:
                    weights.append(max(0.0, float(weight)))
                except (TypeError, ValueError):
                    weights.append(1.0)
            return self.rng.choices(options, weights=weights if any(weights) else None, k=1)[0] if options else ""
        return str(value)

    def expand(self, text: str, definitions: dict[str, Any], max_depth: int = 12) -> str:
        result = str(text)
        for _ in range(max_depth):
            matches = list(_TOKEN.finditer(result))
            if not matches:
                break
            changed = False
            for match in reversed(matches):
                key = match.group(1)
                value = definitions.get(key)
                if value is None:
                    continue
                result = result[:match.start()] + self._pick(value) + result[match.end():]
                changed = True
            if not changed:
                break
        return result

    def apply(self, workflow: dict, profile_wildcards: dict[str, Any],
              metadata: list[dict[str, Any]] | None = None) -> dict:
        definitions = {}
        for section in ("positive", "negative"):
            definitions.update(profile_wildcards.get(section, {}) or {})
        definitions.update(profile_wildcards.get("wildcards", {}) or {})
        prompt_parts = {"positive": [], "negative": []}
        for source in metadata or []:
            for section in prompt_parts:
                values = source.get(section, {})
                if isinstance(values, dict):
                    prompt_parts[section].extend(str(v) for v in values.values() if v)
                elif isinstance(values, str) and values:
                    prompt_parts[section].append(values)
        for section in prompt_parts:
            for key, value in (profile_wildcards.get(section, {}) or {}).items():
                if isinstance(value, str) and value:
                    prompt_parts[section].append(value)
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type", ""))
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            if "text" not in inputs or not isinstance(inputs["text"], str):
                continue
            section = "negative" if node_id.endswith("N") or "negative" in node_id.lower() else "positive"
            original = inputs["text"]
            additions = prompt_parts[section]
            if additions:
                original = ", ".join([original, *additions]).strip(", ")
            inputs["text"] = self.expand(original, definitions)
        return workflow
