from __future__ import annotations

import random
import re
from typing import Any


_TOKEN = re.compile(r"__([A-Za-z0-9_.-]+)__")


class WildcardEngine:
    """Expand nested weighted wildcard tokens using only active-profile definitions."""

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
                value = definitions.get(match.group(1))
                if value is not None:
                    result = result[:match.start()] + self._pick(value) + result[match.end():]
                    changed = True
            if not changed:
                break
        return result

    @staticmethod
    def _flatten_section(container, section):
        value = container.get(section, {}) if isinstance(container, dict) else {}
        if isinstance(value, dict):
            return [str(item) for item in value.values() if isinstance(item, str) and item]
        if isinstance(value, str) and value:
            return [value]
        return []

    def apply(self, workflow: dict, profile_wildcards: dict[str, Any],
              metadata: list[dict[str, Any]] | None = None) -> dict:
        definitions = {}
        for section in ("positive", "negative", "CharWildcard", "LoraWildcard", "wildcards"):
            values = profile_wildcards.get(section, {}) or {}
            if isinstance(values, dict):
                definitions.update(values)
        prompt_parts = {"positive": [], "negative": []}
        for source in metadata or []:
            for section in prompt_parts:
                prompt_parts[section].extend(self._flatten_section(source, section))
        for section in prompt_parts:
            prompt_parts[section].extend(self._flatten_section(profile_wildcards, section))
            for wildcard_key in ("CharWildcard", "LoraWildcard"):
                prompt_parts[section].extend(self._flatten_section(profile_wildcards.get(wildcard_key, {}), section))
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict) or not isinstance(inputs.get("text"), str):
                continue
            section = "negative" if node_id.endswith("N") or "negative" in node_id.lower() else "positive"
            text_value = inputs["text"]
            if prompt_parts[section]:
                text_value = ", ".join([text_value, *prompt_parts[section]]).strip(", ")
            inputs["text"] = self.expand(text_value, definitions)
        return workflow
