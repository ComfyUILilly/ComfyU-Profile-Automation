from __future__ import annotations

import random
import re
import yaml
from typing import Any


_TOKEN = re.compile(r"__([A-Za-z0-9_.-]+)__")


class WildcardEngine:
    """Compose ordered positive/negative wildcard dictionaries and expand nested tokens."""

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
            if not options or not any(weight > 0 for weight in weights):
                return ""
            return self.rng.choices(options, weights=weights, k=1)[0]
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
    def _deep_update(target: dict, source: dict) -> None:
        for key, value in source.items():
            if isinstance(value, dict):
                current = target.get(key)
                if not isinstance(current, dict):
                    current = {}
                    target[key] = current
                WildcardEngine._deep_update(current, value)
            else:
                target[key] = value

    @staticmethod
    def _merge(target: dict, source: Any) -> None:
        if not isinstance(source, dict):
            return
        for section in ("positive", "negative"):
            values = source.get(section, {})
            if isinstance(values, dict):
                destination = target.setdefault(section, {})
                WildcardEngine._deep_update(destination, values)
            elif isinstance(values, str) and values:
                target[section][f"item{len(target[section]) + 1}"] = values

    def apply(self, workflow: dict, profile_wildcards: dict[str, Any],
              metadata: list[dict[str, Any]] | None = None, shuffle: Any = False) -> dict:
        prompts = {"positive": {}, "negative": {}}
        for source in metadata or []:
            self._merge(prompts, source)

        definitions = {}
        for section in ("positive", "negative", "CharWildcard", "LoraWildcard", "wildcards"):
            values = profile_wildcards.get(section, {}) or {}
            if isinstance(values, dict):
                definitions.update(values)

        for section, suffix in (("positive", "P"), ("negative", "N")):
            values = list(prompts[section].values())
            if isinstance(shuffle, (list, tuple)) and len(shuffle) == 2:
                should_shuffle = bool(self.rng.choice(list(shuffle)))
            else:
                should_shuffle = bool(shuffle)
            if should_shuffle:
                self.rng.shuffle(values)
            yaml_node = workflow.get(f"PrimitiveStringMultiline{suffix}")
            if isinstance(yaml_node, dict):
                yaml_node.setdefault("inputs", {})["value"] = yaml.safe_dump(
                    prompts[section], allow_unicode=True, sort_keys=False
                )
            wildcard_node = workflow.get(f"{section}Wildcard")
            if isinstance(wildcard_node, dict) and "wildcard_text" in wildcard_node.get("inputs", {}):
                wildcard_node["inputs"]["wildcard_text"] = f",,,,{','.join(str(v) for v in values)},,,,"
            else:
                target_id = "CLIPTextEncodeN" if section == "negative" else "CLIPTextEncodeP"
                target = workflow.get(target_id)
                if isinstance(target, dict) and "text" in target.get("inputs", {}) and values:
                    original = target["inputs"]["text"]
                    target["inputs"]["text"] = ", ".join([original, *map(str, values)]).strip(", ")

        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict) and isinstance(inputs.get("text"), str):
                inputs["text"] = self.expand(inputs["text"], definitions)
        return workflow
