from __future__ import annotations

import random
from typing import Any


class ProfileManager:
    """Choose a profile, then keep it for the entire selected image batch."""

    def __init__(self, profiles: dict[str, dict[str, Any]], rng: random.Random | None = None):
        if not profiles:
            raise ValueError("profiles cannot be empty")
        self.profiles = profiles
        self.rng = rng or random.Random()

    def next_batch(self) -> tuple[str, int]:
        names = list(self.profiles)
        weights = [float(self.profiles[name].get("weight", 1)) for name in names]
        if not any(weight > 0 for weight in weights):
            raise ValueError("At least one profile must have a positive weight")
        name = self.rng.choices(names, weights=weights, k=1)[0]
        setting = self.profiles[name].get("batch_images", 1)
        if isinstance(setting, int):
            count = setting
        elif isinstance(setting, (list, tuple)) and len(setting) == 2:
            low, high = map(int, setting)
            if low < 1 or high < low:
                raise ValueError(f"Invalid batch_images range for {name}: {setting}")
            count = self.rng.randint(low, high)
        else:
            raise ValueError(f"Invalid batch_images for {name}: {setting}")
        return name, count
