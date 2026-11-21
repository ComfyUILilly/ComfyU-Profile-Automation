from __future__ import annotations

import random
from typing import Any

from .catalog import ModelFile, ProfileData


def _weighted(items: list[ModelFile], weights: dict[str, Any], rng: random.Random) -> ModelFile:
    if not items:
        raise ValueError("No selectable assets")
    values = []
    for item in items:
        value = weights.get(item.identity, weights.get(item.relative_path, weights.get(item.name, 1)))
        try:
            values.append(max(0.0, float(value)))
        except (TypeError, ValueError):
            values.append(1.0)
    return rng.choices(items, weights=values if any(values) else None, k=1)[0]


class AssetSelector:
    """Profile-local Weight, Random, Cycle, DB, Favorites and Skip selection."""

    def __init__(self, rng: random.Random | None = None, database=None,
                 settings: dict[str, Any] | None = None):
        self.rng = rng or random.Random()
        self.database = database
        self.settings = settings or {}
        self.pools: dict[tuple[str, str], list[str]] = {}
        self.last_group_metadata: list[dict[str, Any]] = []

    def _probability(self, category: str) -> float:
        key = {"checkpoint": "CheckpointWeightPer", "character": "CharWeightPer"}.get(category)
        if key is None:
            return 1.0
        try:
            return min(1.0, max(0.0, float(self.settings.get(key, 0.5))))
        except (TypeError, ValueError):
            return 0.5

    def _flag_candidates(self, profile: ProfileData, category: str,
                         source: list[ModelFile], flag: str) -> list[ModelFile]:
        return [item for item in source
                if self._metadata(profile, category, item).get(flag) not in (False, None)]

    def _fallback_flag_mode(self, profile: ProfileData, category: str,
                            source: list[ModelFile], count: int) -> list[ModelFile]:
        """Legacy fallback when Favorites/Skip has no flagged rows."""
        weights = profile.weights.get(category, {})
        if weights and self.rng.random() < self._probability(category):
            return self._weighted_sample(source, weights, count)
        return self.rng.sample(source, min(count, len(source)))

    def choose(self, profile: ProfileData, category: str, kind: str = "Weight",
               count: int = 1) -> list[ModelFile]:
        self.last_group_metadata = []
        source = {"checkpoint": profile.checkpoints, "character": profile.characters,
                  "lora": profile.loras}[category]
        if not source:
            return []
        mode = kind or "Weight"
        if mode == "Wildcard" and category == "lora":
            return []

        amount = max(1, min(int(count), len(source)))

        if mode.startswith(("Favorites", "Skip")):
            flag = "favorites" if mode.startswith("Favorites") else "skip"
            candidates = self._flag_candidates(profile, category, source, flag)
            if not candidates:
                return self._fallback_flag_mode(profile, category, source, amount)
            amount = min(amount, len(candidates))
            if mode.endswith("_Weight"):
                return self._weighted_sample(candidates, profile.weights.get(category, {}), amount)
            return self.rng.sample(candidates, amount)

        if mode == "Random":
            return self.rng.sample(source, amount)

        if mode == "Cycle":
            key = (profile.name, category)
            pool = self.pools.setdefault(key, [item.identity for item in source])
            # Refresh after a model-folder change; never return a removed asset.
            valid = {item.identity for item in source}
            pool[:] = [identity for identity in pool if identity in valid]
            if not pool:
                pool.extend(item.identity for item in source)
            chosen = []
            for _ in range(amount):
                if not pool:
                    pool.extend(item.identity for item in source)
                identity = self.rng.choice(pool)
                pool.remove(identity)
                chosen.append(next(item for item in source if item.identity == identity))
            return chosen

        if mode == "DB" and self.database:
            counts = self.database.counts(profile.name, category)
            maximum = int(self.settings.get(
                {"checkpoint": "CheckpointDbWeightMax", "character": "CharDbWeightMax",
                 "lora": "LoraDbWeightMax"}.get(category, ""), 100))
            minimum = int(self.settings.get(
                {"checkpoint": "CheckpointDbWeightMin", "character": "CharDbWeightMin",
                 "lora": "LoraDbWeightMin"}.get(category, ""), 1))
            weights = {item.identity: max(minimum, min(maximum - counts.get(
                item.relative_path, counts.get(item.name, 0)), maximum)) for item in source}
            return self._weighted_sample(source, weights, amount)

        if category in ("checkpoint", "character") and mode == "Weight":
            weights = profile.weights.get(category, {})
            if weights and self.rng.random() < self._probability(category):
                return self._weighted_sample(source, weights, amount)
            return self.rng.sample(source, amount)

        if category == "lora" and mode == "Weight" and self._has_group_weights(profile):
            return self._choose_group_loras(profile, source)

        weights = profile.weights.get(category, {})
        if isinstance(weights, dict) and weights:
            return self._weighted_sample(source, weights, amount)
        return self.rng.sample(source, amount)

    @staticmethod
    def _metadata(profile, category, item):
        metadata = profile.metadata.get(category, {})
        if item.relative_path in metadata:
            return metadata[item.relative_path]
        collection = {"checkpoint": profile.checkpoints, "character": profile.characters,
                      "lora": profile.loras}[category]
        if sum(1 for candidate in collection if candidate.name == item.name) == 1:
            return metadata.get(item.name, {})
        return {}

    @staticmethod
    def _has_group_weights(profile):
        return any(isinstance(group, dict) and isinstance(group.get("dic"), dict)
                   for group in profile.weights.get("lora", {}).values())

    def _pick_group_lora(self, value):
        if isinstance(value, dict):
            names, weights = [], []
            for name, weight in value.items():
                try:
                    numeric = max(0.0, float(weight))
                except (TypeError, ValueError):
                    numeric = 0.0
                names.append(name)
                weights.append(numeric)
            if not names or not any(weight > 0 for weight in weights):
                return []
            return [self.rng.choices(names, weights=weights, k=1)[0]]
        if isinstance(value, (list, tuple)):
            return [self.rng.choice(value)] if value else []
        if isinstance(value, str):
            return [value]
        return []

    def _choose_group_loras(self, profile, candidates):
        by_identity = {item.identity: item for item in candidates}
        by_relative = {item.relative_path: item for item in candidates}
        stem_groups = {}
        for item in candidates:
            stem_groups.setdefault(item.name, []).append(item)
        by_unique_stem = {name: values[0] for name, values in stem_groups.items() if len(values) == 1}
        aliases = {**by_unique_stem, **by_relative, **by_identity}
        selected = {}

        for group_name, group in profile.weights.get("lora", {}).items():
            if not isinstance(group, dict):
                continue
            entries = group.get("dic", {})
            if not isinstance(entries, dict):
                continue

            total_max = self._random_count(group.get("totalMax", 0))
            if group.get("total") and total_max <= 0:
                continue

            pool = {}
            if group.get("per"):
                limit = self._random_count(group.get("perMax", 0))
                per_candidates = {}
                for key, entry in entries.items():
                    if not isinstance(entry, dict):
                        continue
                    try:
                        chance = float(entry.get("per", 0))
                    except (TypeError, ValueError):
                        chance = 0.0
                    if self.rng.random() < chance:
                        for name in self._resolve_group_names(group_name, key, entry):
                            if name in aliases:
                                per_candidates[aliases[name].identity] = entry
                per_ids = list(per_candidates)
                if limit and per_ids:
                    for identity in self.rng.sample(per_ids, min(limit, len(per_ids))):
                        pool[identity] = per_candidates[identity]

            if group.get("weight"):
                limit = self._random_count(group.get("weightMax", 0))
                weighted_entries = []
                for key, entry in entries.items():
                    if isinstance(entry, dict):
                        try:
                            weight = max(0.0, float(entry.get("weight", 0)))
                        except (TypeError, ValueError):
                            weight = 0.0
                        if weight:
                            weighted_entries.append((key, entry, weight))
                if weighted_entries and limit:
                    selected_keys = self.rng.choices(
                        weighted_entries,
                        weights=[item[2] for item in weighted_entries],
                        k=limit,
                    )
                    for key, entry, _weight in selected_keys:
                        for name in self._resolve_group_names(group_name, key, entry):
                            if name in aliases:
                                pool[aliases[name].identity] = entry

            chosen = list(pool)
            if group.get("total"):
                chosen = self.rng.sample(chosen, min(total_max, len(chosen)))
            selected.update({identity: pool.get(identity, {}) for identity in chosen})

        if not selected:
            return []
        self.last_group_metadata = list(selected.values())
        return [by_identity[identity] for identity in selected if identity in by_identity]

    def _resolve_group_names(self, group, key, entry):
        values = entry.get("loras")
        if values:
            return self._pick_group_lora(values)
        return [f"{group}-{key}"]

    def _random_count(self, value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return self.rng.randint(int(value[0]), int(value[1]))
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _weighted_sample(self, items, weights, count):
        remaining = list(items)
        selected = []
        for _ in range(min(count, len(remaining))):
            selected_item = _weighted(remaining, weights, self.rng)
            selected.append(selected_item)
            remaining.remove(selected_item)
        return selected
