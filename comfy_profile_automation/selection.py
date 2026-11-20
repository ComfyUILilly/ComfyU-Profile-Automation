from __future__ import annotations

import random
from typing import Any, Iterable

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
    """Profile-local implementation of Weight, Random, Cycle, DB and flag modes."""

    def __init__(self, rng: random.Random | None = None, database=None):
        self.rng = rng or random.Random()
        self.database = database
        self.pools: dict[tuple[str, str], list[str]] = {}

    def choose(self, profile: ProfileData, category: str, kind: str = "Weight",
               count: int = 1, fallback_weight: float = 1.0) -> list[ModelFile]:
        source = {
            "checkpoint": profile.checkpoints,
            "character": profile.characters,
            "lora": profile.loras,
        }[category]
        if not source:
            return []
        mode = kind or "Weight"
        if mode == "Wildcard" and category == "lora":
            return []
        candidates = source
        if mode.startswith("Favorites") or mode.startswith("Skip"):
            flag = "favorites" if mode.startswith("Favorites") else "skip"
            candidates = [x for x in source if self._metadata(profile, category, x).get(flag) not in (None, False, 0)]
            if not candidates:
                candidates = source
        weights = profile.weights.get(category, {})
        if mode == "DB" and self.database:
            counts = self.database.counts(profile.name, category)
            maximum = 100
            weight_map = {x.relative_path: max(1, maximum - counts.get(x.relative_path, counts.get(x.name, 0))) for x in candidates}
        else:
            weight_map = weights
        amount = max(1, min(int(count), len(candidates)))
        if mode == "Random":
            return self.rng.sample(candidates, amount)
        if mode == "Cycle":
            key = (profile.name, category)
            pool = self.pools.setdefault(key, [x.relative_path for x in candidates])
            chosen = []
            for _ in range(amount):
                if not pool:
                    pool.extend(x.relative_path for x in candidates)
                selected_id = self.rng.choice(pool)
                pool.remove(selected_id)
                chosen.append(next(x for x in candidates if x.relative_path == selected_id))
            return chosen
        if mode in ("Favorites", "Skip"):
            return self.rng.sample(candidates, amount)
        if mode in ("Favorites_Weight", "Skip_Weight", "DB", "Weight"):
            return self._weighted_sample(candidates, weight_map, amount)
        return self._weighted_sample(candidates, weight_map, amount) if weight_map else self.rng.sample(candidates, amount)

    @staticmethod
    def _metadata(profile: ProfileData, category: str, item: ModelFile) -> dict[str, Any]:
        return profile.metadata.get(category, {}).get(item.relative_path,
               profile.metadata.get(category, {}).get(item.name, {}))

    def _weighted_sample(self, items: list[ModelFile], weights: dict[str, Any], count: int) -> list[ModelFile]:
        remaining = list(items)
        selected = []
        for _ in range(min(count, len(remaining))):
            chosen = _weighted(remaining, weights, self.rng)
            selected.append(chosen)
            remaining.remove(chosen)
        return selected
