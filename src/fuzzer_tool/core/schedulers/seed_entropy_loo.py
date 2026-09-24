"""Entropy leave-one-out seed strategy: pick seeds the pooled entropy leans on.

Plan: ``docs/handover/handover_entropy_seed_schedulers_2026-09-19.md`` §5,
first step. Score each seed by how much pooled corpus byte-entropy drops
without it::

    Δ(s) = H(pool) - H(pool \\ s)

Not Shapley: the doc's order is LOO first, permutation-sampled Shapley
only if LOO's known misprice matters. That misprice is near-duplicates --
two copies of a pattern each score ~0 (dropping either alone costs
nothing), while Shapley credits both (pinned in
``tests/test_seed_entropy_loo.py``). Δ can be negative: a flat seed drags
pooled entropy down, so removing it raises H. Weights floor at 0.

All seeds are scored in one pass over a per-seed count slab. With pool
counts ``C`` (total ``N``) and a seed's counts ``c`` (total ``n``)::

    R = C - c        H(R) = log2(N - n) - sum(R log2 R) / (N - n)

``R log2 max(R, 1)`` makes empty bins contribute 0 without a mask.
Rescored lazily, on the first pick after the live corpus changed: ~5 ms at
2000 seeds (``log2`` over the n x 256 slab dominates), 0.5 ms otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fuzzer_tool.core.byte_entropy import ENTROPY_SAMPLE_CAP, byte_histogram

#: Floor added to every weight so a seed the pool does not lean on (Δ <= 0)
#: is rare rather than unreachable. Matches seed_entropy_kl.
MIN_WEIGHT = 1e-6

#: Rows the count slab starts with; it doubles from there.
SLAB_INITIAL_ROWS = 256


def _entropy(xlogx_sum: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Shannon bits from ``sum(c log2 c)`` and ``N``; 0 where ``N == 0``."""
    safe = np.maximum(total, 1)
    return np.where(total > 0, np.log2(safe) - xlogx_sum / safe, 0.0)


class EntropyLOOSeedStrategy:
    """Elo-arbitrated ``entropy_loo`` seed arm (``--entropy-loo``)."""

    def __init__(self, rng: Any, cap: int = ENTROPY_SAMPLE_CAP) -> None:
        self._rng = rng
        self._cap = cap

        # Per-seed byte counts, one row each, parallel to _keys.
        self._counts = np.zeros((SLAB_INITIAL_ROWS, 256), dtype=np.int64)
        self._sizes = np.zeros(SLAB_INITIAL_ROWS, dtype=np.int64)
        self._keys: list[bytes] = []
        self._index: dict[bytes, int] = {}

        # Pooled counts of every live seed.
        self._pool = np.zeros(256, dtype=np.int64)
        self._total = 0

        self._loo: dict[bytes, float] = {}
        self._dirty = True
        self._selected = 0

    def scores(self, seeds: list[bytes]) -> list[float]:
        """Leave-one-out entropy drop per seed, aligned with *seeds*."""
        self._sync(seeds)
        if self._dirty:
            self._refresh()
        return [self._loo.get(s, 0.0) for s in seeds]

    def select(self, seeds: list[bytes]) -> bytes | None:
        """Draw proportional to ``max(Δ, 0) + MIN_WEIGHT``; None on empty."""
        if not seeds:
            return None

        weights = [max(v, 0.0) + MIN_WEIGHT for v in self.scores(seeds)]
        self._selected += 1
        chosen: bytes = self._rng.weighted_choice(seeds, weights)
        return chosen

    # ── pool maintenance ────────────────────────────────────────────

    def _sync(self, seeds: list[bytes]) -> None:
        """Make the slab the live corpus: fold admissions, unfold evictions."""
        live = set(seeds)
        for seed in live.difference(self._index):
            self._fold(seed)
        for seed in [s for s in self._keys if s not in live]:
            self._unfold(seed)

    def _fold(self, seed: bytes) -> None:
        counts, total = byte_histogram(seed, self._cap)
        row = len(self._keys)
        self._reserve(row + 1)
        self._counts[row] = counts
        self._sizes[row] = total
        self._index[seed] = row
        self._keys.append(seed)
        self._pool += self._counts[row]
        self._total += total
        self._dirty = True

    def _unfold(self, seed: bytes) -> None:
        # Fill the hole from the tail rather than shifting the slab down.
        row, last = self._index.pop(seed), len(self._keys) - 1
        self._pool -= self._counts[row]
        self._total -= int(self._sizes[row])
        if row != last:
            self._counts[row] = self._counts[last]
            self._sizes[row] = self._sizes[last]
            self._keys[row] = self._keys[last]
            self._index[self._keys[row]] = row
        self._keys.pop()
        self._loo.pop(seed, None)
        self._dirty = True

    def _reserve(self, size: int) -> None:
        """Grow the slab to hold ``size`` rows, doubling to stay amortized."""
        capacity = self._counts.shape[0]
        if size <= capacity:
            return

        used = len(self._keys)
        capacity = max(size, capacity * 2)
        counts = np.zeros((capacity, 256), dtype=np.int64)
        counts[:used] = self._counts[:used]
        sizes = np.zeros(capacity, dtype=np.int64)
        sizes[:used] = self._sizes[:used]
        self._counts, self._sizes = counts, sizes

    def _refresh(self) -> None:
        """Rescore every seed against the current pool in one pass."""
        self._dirty = False
        used = len(self._keys)
        if not used:
            self._loo = {}
            return

        pool = self._pool.astype(np.float64)
        h_all = _entropy(
            np.array((pool * np.log2(np.maximum(pool, 1))).sum()), np.array(self._total)
        )

        rest = pool[None, :] - self._counts[:used]
        rest_xlogx = (rest * np.log2(np.maximum(rest, 1))).sum(axis=1)
        h_rest = _entropy(rest_xlogx, self._total - self._sizes[:used])

        self._loo = dict(zip(self._keys, (h_all - h_rest).tolist(), strict=True))

    def stats(self) -> dict[str, Any]:
        live = list(self._loo.values())
        return {
            "selected": self._selected,
            "pooled": len(self._keys),
            "mean_loo": (sum(live) / len(live)) if live else 0.0,
        }
