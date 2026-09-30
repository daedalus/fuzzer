"""Non-repeating m-permutation draws for ``_swap_tuple`` (combinadic-backed).

Independent uniform draws repeat: a 16-byte seed has 16!/(16-4)! = 43680
ordered 4-tuples, so ``_swap_tuple`` re-tries one after ~200 calls. This walks
the whole space once before repeating, in O(1) memory per (n, m):

    idx_k = (start + step * k) mod total,   gcd(step, total) = 1

An affine map with a unit step is a bijection on ``[0, total)``, so ``idx``
visits every index exactly once per period. ``unrank_perm`` maps it to the
tuple. Random ``start``/``step`` decorrelate seeds and campaigns; consecutive
tuples are arithmetic-progression neighbours in lexicographic order, so they
are not independent draws. Whether that pays is unmeasured (handover
combinatorics §1).
"""

from __future__ import annotations

import math
from typing import Any

from fuzzer_tool.core.combinadic import perm_count, sample_indices, unrank_perm

# Distinct (n, m) shapes tracked; oldest evicted first (Hard Rule 54).
MAX_KEYS = 256

_MIN_M = 2
_TOTAL, _START, _STEP, _COUNT = range(4)


class TupleWalk:
    """Per-(n, m) cursor over all ordered m-tuples of ``range(n)``."""

    def __init__(self) -> None:
        self._state: dict[tuple[int, int], list[int]] = {}

    def __len__(self) -> int:
        return len(self._state)

    def _new(self, n: int, m: int, rng: Any) -> list[int]:
        total = perm_count(n, m)
        start, step = sample_indices(total, 2, rng)

        # Nudge to the next unit; total-1 is always coprime, so this ends.
        step = step or 1
        while math.gcd(step, total) != 1:
            step = step % total + 1
        return [total, start, step, 0]

    def draw(self, n: int, m: int, rng: Any) -> tuple[int, ...] | None:
        """Next unseen ordered *m*-tuple of ``range(n)``; ``None`` if n < m."""
        if m < _MIN_M or n < m:
            return None

        key = (n, m)
        st = self._state.get(key)
        if st is None:
            if len(self._state) >= MAX_KEYS:
                del self._state[next(iter(self._state))]
            st = self._state[key] = self._new(n, m, rng)

        idx = (st[_START] + st[_STEP] * st[_COUNT]) % st[_TOTAL]
        st[_COUNT] = (st[_COUNT] + 1) % st[_TOTAL]
        return unrank_perm(idx, n, m)
