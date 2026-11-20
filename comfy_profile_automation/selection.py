from __future__ import annotations

import random
from typing import Any

from .catalog import ModelFile, ProfileData


def _weighted(items: list[ModelFile], weights: dict[str, Any], rng: random.Random) -> ModelFile:
    if not items:
        raise ValueError("No selectable assets")
    values = []
    for item in items:
        value = weights.get(item.relative_path, weights.get(item.name, 1))
        try:
            values.append(max(0.0, float(value)))
        except (TypeError, ValueError):
            values.append(1.0)
    return rng.choices(items, weights=values if any(values) else None, k=1)[0]


class AssetSelector:
    """Profile-local Weight, Random, Cycle, DB, Favorites and Skip selection."""

    def __init__(self, rng: random.Random | None = None, database=None):
        self.rng = rng or random.Random()
        self.database = database
        self.pools: dict[tuple[str, str], list[str]] = {}

    def choose(self, profile: ProfileData, category: str, kind: str = "Weight",
               count: int = 1) -> list[ModelFile]:
        source = {"checkpoint": profile.checkpoints, "character": profile.characters, "lora": profile.loras}[category]
        if not source:
            return []
        mode = kind or "Weight"
        if mode == "Wildcard" and category == "lora":
            return []
        candidates = source
        if mode.startswith(("Favorites", "Skip")):
            flag = "favorites" if mode.startswith("Favorites") else "skip"
            candidates = [x for x in source if self._metadata(profile, category, x).get(flag) not in (None, False, 0)]
            if not candidates:
                candidates = source
        amount = max(1, min(int(count), len(candidates)))
        if mode == "Random":
            return self.rng.sample(candidates, amount)
        if mode == "Cycle":
            key = (profile.name, category)
            pool = self.pools.setdefault(key, [x.identity for x in candidates])
            chosen = []
            for _ in range(amount):
                if not pool:
                    pool.extend(x.identity for x in candidates)
                identity = self.rng.choice(pool)
                pool.remove(identity)
                chosen.append(next(x for x in candidates if x.identity == identity))
            return chosen
        if mode in ("Favorites", "Skip"):
            return self.rng.sample(candidates, amount)
        if mode == "DB" and self.database:
            counts = self.database.counts(profile.name, category)
            weights = {x.identity: max(1, 100 - counts.get(x.relative_path, counts.get(x.name, 0))) for x in candidates}
            return self._weighted_sample(candidates, weights, amount)
        if category == "lora" and mode == "Weight" and self._has_group_weights(profile):
            return self._choose_group_loras(profile, candidates)
        weights = profile.weights.get(category, {})
        if mode in ("Favorites_Weight", "Skip_Weight", "Weight") or weights:
            return self._weighted_sample(candidates, weights, amount)
        return self.rng.sample(candidates, amount)

    @staticmethod
    def _metadata(profile, category, item):
        return profile.metadata.get(category, {}).get(item.relative_path,
               profile.metadata.get(category, {}).get(item.name, {}))

    @staticmethod
    def _has_group_weights(profile):
        return any(isinstance(group, dict) and isinstance(group.get("dic"), dict)
                   for group in profile.weights.get("lora", {}).values())

    def _choose_group_loras(self, profile, candidates):
        by_name = {item.name: item for item in candidates}
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
                for key, entry in entries.items():
                    if not isinstance(entry, dict):
                        continue
                    try:
                        chance = float(entry.get("per", 0))
                    except (TypeError, ValueError):
                        chance = 0
                    if self.rng.random() < chance:
                        for name in self._resolve_group_names(group_name, key, entry):
                            if name in by_name:
                                pool[name] = entry
            if group.get("weight"):
                limit = self._random_count(group.get("weightMax", 0))
                weighted_entries = []
                for key, entry in entries.items():
                    if isinstance(entry, dict):
                        try:
                            weight = max(0.0, float(entry.get("weight", 0)))
                        except (TypeError, ValueError):
                            weight = 0
                        if weight:
                            weighted_entries.append((key, entry, weight))
                for key, entry, weight in self.rng.choices(
                    weighted_entries, weights=[x[2] for x in weighted_entries],
                    k=min(limit, len(weighted_entries))
                ) if weighted_entries and limit else []:
                    for name in self._resolve_group_names(group_name, key, entry):
                        if name in by_name:
                            pool[name] = entry
            chosen = list(pool)
            if group.get("total"):
                chosen = self.rng.sample(chosen, min(total_max, len(chosen)))
            selected.update({name: by_name[name] for name in chosen})
        if not selected:
            return self._weighted_sample(candidates, profile.weights.get("lora", {}), min(1, len(candidates)))
        return list(selected.values())

    def _resolve_group_names(self, group, key, entry):
        values = entry.get("loras")
        if isinstance(values, dict):
            return list(values)
        if isinstance(values, list):
            return values
        if isinstance(values, str):
            return [values]
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
