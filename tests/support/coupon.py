"""Coupon-collector draw budgets for "every value appears" tests.

No finite draw count guarantees full coverage, so derive it from a target
failure probability ``delta`` (union bound over the n coupons)::

    P(some coupon unseen after T draws) <= n * (1 - p_min)^T <= delta
    T = ceil((ln n + ln 1/delta) / -ln(1 - p_min))

Uniform draws (p_min = 1/n) give ~ n (ln n + ln 1/delta); e.g. n = 16 at
delta = 1e-9 is 376 draws, not an arbitrary 2000.
"""

import math

DELTA = 1e-9


def coverage_budget(n: int, delta: float = DELTA) -> int:
    """Uniform draws needed so P(any of n values unseen) <= delta."""
    return math.ceil(n * (math.log(n) + math.log(1.0 / delta)))


def skewed_budget(n: int, p_min: float, delta: float = DELTA) -> int:
    """Draws when the rarest of n coupons has probability ``p_min`` per draw."""
    if n < 1 or not 0.0 < p_min < 1.0 or n * p_min > 1.0 + 1e-12:
        raise ValueError(f"need n >= 1 and 0 < p_min <= 1/n, got n={n}, p_min={p_min}")
    return math.ceil((math.log(n) + math.log(1.0 / delta)) / -math.log1p(-p_min))
