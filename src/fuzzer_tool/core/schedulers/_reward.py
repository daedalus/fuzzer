"""Shared reward clamp for schedulers whose updates assume rewards in [0, 1]."""

from __future__ import annotations

import math


def unit_reward(success: bool, weight: float) -> float:
    """Reward in [0, 1]: 0 on failure or NaN weight, else *weight* clamped.

    ``record()`` receives cost-adjusted surprisal weights that are not
    guaranteed bounded; +inf clamps to 1, -inf and negatives to 0.
    """
    if not success or math.isnan(weight):
        return 0.0
    return min(1.0, max(0.0, weight))
