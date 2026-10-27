"""Inverse lottery (Waldspurger & Weihl 1994, section 5.2): randomized eviction.

A lottery picks a *winner* with probability proportional to tickets. The
inverse lottery picks a *victim* with probability proportional to
``1 / tickets``: low-value holders lose the resource more often, but never
deterministically, so a cull cannot systematically empty one lineage the way a
fixed top-K cut does.

``inverse_lottery_victims`` draws *k* victims without replacement. Sequential
sampling without replacement with weights ``w = 1 / t`` is exactly "sort by
``E / w``" with ``E ~ Exp(1)`` (Efraimidis & Spirakis 2006), i.e. take the *k*
smallest ``E * t``. One pass, O(n log n), no per-draw renormalisation::

    tickets a=9 b=1:  P(b evicted first) = 0.9

The tickets are a keep-score. Garbage (NaN, inf) and non-positive tickets are
floored at a tiny fraction of the largest valid ticket: a worthless seed is the
likeliest victim, never a divide-by-zero and never a tie at zero. A single
huge ticket needs no cap: its key is linear in the ticket, so it simply is
almost never evicted.

Equal tickets reduce to a uniform random cull -- the falsification condition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

#: Floor for non-positive tickets, relative to the largest valid ticket.
FLOOR_FRACTION = 1e-9
#: Floor when no ticket is valid at all.
ABSOLUTE_FLOOR = 1e-12


class _Random(Protocol):
    def random(self) -> float: ...


def _clean(tickets: Sequence[float]) -> list[float]:
    valid = [float(t) for t in tickets if math.isfinite(t) and t > 0]
    floor = max(valid) * FLOOR_FRACTION if valid else ABSOLUTE_FLOOR
    return [float(t) if math.isfinite(t) and t > 0 else floor for t in tickets]


def inverse_lottery_victims(tickets: Sequence[float], k: int, rng: _Random) -> list[int]:
    """Indices of *k* victims drawn without replacement, P(victim) ~ 1 / ticket.

    ``k <= 0`` returns no victims; ``k >= len(tickets)`` returns every index.
    The result is in draw order (first victim first).
    """
    n = len(tickets)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))
    clean = _clean(tickets)
    # E = -ln(1 - u) with u in [0, 1): the argument stays in (0, 1], never log(0).
    keys = [(-math.log(1.0 - rng.random()) * t, i) for i, t in enumerate(clean)]
    keys.sort()
    return [i for _, i in keys[:k]]
