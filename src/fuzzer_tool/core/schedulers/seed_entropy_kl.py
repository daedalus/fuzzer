"""Entropy-KL seed strategy: pick seeds whose bytes diverge from the corpus.

Plan: ``docs/handover/handover_entropy_seed_schedulers_2026-09-19.md`` §2.

Score a seed by the relative entropy of its own byte distribution against
the pooled distribution of the corpus it sits in::

    score(s) = KL(P_s || Q) = sum_b P_s(b) * log2(P_s(b) / Q(b))

Not the same criterion as a gap between two entropy *numbers*: two seeds
uniform over two different byte pairs carry identical entropy and opposite
divergence, so a scalar comparison ranks them equal and this does not.

KL is evaluated in its cross-entropy form, which is what makes the whole
corpus one matrix-vector product::

    KL = sum_b p log2 p  -  sum_b p log2 q  =  (-H_s) + P_s . (-log2 Q)

``-H_s`` is a static per-seed number, so only the dot product moves when the
pool does. The per-seed distributions live in one slab that grows
amortized and fills evictions from the tail, so a corpus admission costs a
matvec and not a rebuild: measured on a 2000-seed corpus, the pick after an
admission is 1.0 ms against 6.6 ms when the slab was re-stacked each time,
and a pick that changes nothing is 0.33 ms. A prune is the expensive
direction -- each evicted seed unfolds from the pool at ~50 us -- but it
runs once per prune, not once per pick.

The pool is owned here rather than read off ``Fuzzer._corpus_entropy``:
that tracker is rebuilt once per ``load_corpus()`` and never sees a seed
discovered mid-campaign, so scoring against it would measure every seed
found during the run against a distribution that predates it -- and it has
no way to drop a seed the corpus pruned. This one folds on admission and
unfolds on eviction, so ``Q`` is exactly the live corpus.

``Q`` is smoothed (``byte_entropy.POOL_SMOOTHING``) so a byte value the pool
has never seen keeps positive probability. Every seed passed to
:meth:`scores` is folded before it is scored, so its support is already in
the pool and the floor only matters at the margin -- it is there because an
unsmoothed ``log2(p/0)`` is undefined, not because it tunes anything.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fuzzer_tool.core.byte_entropy import (
    ENTROPY_SAMPLE_CAP,
    CumulativeByteEntropy,
    byte_histogram,
    entropy_bits_from_counts,
)

#: Floor added to every weight so a seed matching the pool exactly (KL 0)
#: is rare rather than unreachable. Matches seed_kruskal_count.
MIN_WEIGHT = 1e-6

#: Rows the distribution slab starts with; it doubles from there.
SLAB_INITIAL_ROWS = 256
STATE_VERSION = 1

_COUNTERS = ("scored", "selected")


def _valid_state(data: Any) -> bool:
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        return False

    return all(type(data.get(key, 0)) is int and data.get(key, 0) >= 0 for key in _COUNTERS)


class EntropyKLSeedStrategy:
    """Elo-arbitrated ``entropy_kl`` seed arm (``--entropy-kl``)."""

    def __init__(self, rng: Any, cap: int = ENTROPY_SAMPLE_CAP) -> None:
        self._rng = rng
        self._cap = cap
        self._pool = CumulativeByteEntropy()
        # Slab of per-seed byte distributions, one row each, parallel to
        # _keys; _index maps a seed to its row.
        self._probs = np.zeros((SLAB_INITIAL_ROWS, 256), dtype=np.float64)
        self._neg_ent = np.zeros(SLAB_INITIAL_ROWS, dtype=np.float64)
        self._keys: list[bytes] = []
        self._index: dict[bytes, int] = {}
        self._kl: dict[bytes, float] = {}
        # Bumped on every fold/unfold; _kl is stale while it disagrees with
        # _scored_at, which is cheaper than diffing 256 pooled bins.
        self._pool_version = 0
        self._scored_at = -1
        self._scored = 0
        self._selected = 0

    def scores(self, seeds: list[bytes]) -> list[float]:
        """KL of every seed against the pooled corpus distribution."""
        self._sync(seeds)
        if self._scored_at != self._pool_version:
            self._refresh()
        return [self._kl.get(s, 0.0) for s in seeds]

    def select(self, seeds: list[bytes]) -> bytes | None:
        """Draw proportional to ``KL + MIN_WEIGHT``; None on an empty corpus."""
        if not seeds:
            return None

        weights = [w + MIN_WEIGHT for w in self.scores(seeds)]
        self._selected += 1
        chosen: bytes = self._rng.weighted_choice(seeds, weights)
        return chosen

    def _sync(self, seeds: list[bytes]) -> None:
        """Make the pool the live corpus: fold admissions, unfold evictions."""
        live = set(seeds)
        for seed in live.difference(self._index):
            self._fold(seed)

        for seed in [s for s in self._keys if s not in live]:
            self._unfold(seed)

    def _fold(self, seed: bytes) -> None:
        counts, total = byte_histogram(seed, self._cap)
        row = np.asarray(counts, dtype=np.float64)
        if total:
            row /= total

        used = len(self._keys)
        self._reserve(used + 1)
        self._probs[used] = row
        self._neg_ent[used] = -entropy_bits_from_counts(counts, total)
        self._index[seed] = used
        self._keys.append(seed)
        self._scored += 1
        if not total:
            return  # an empty seed is a zero row and contributes no bytes

        self._pool.add(seed, self._cap)
        self._pool_version += 1

    def _unfold(self, seed: bytes) -> None:
        # Fill the hole from the tail rather than shifting the slab down.
        row, last = self._index.pop(seed), len(self._keys) - 1
        if row != last:
            self._probs[row] = self._probs[last]
            self._neg_ent[row] = self._neg_ent[last]
            self._keys[row] = self._keys[last]
            self._index[self._keys[row]] = row

        self._keys.pop()
        self._kl.pop(seed, None)
        if not seed[: self._cap]:
            return

        self._pool.remove(seed, self._cap)
        self._pool_version += 1

    def _reserve(self, size: int) -> None:
        """Grow the slab to hold ``size`` rows, doubling to stay amortized."""
        capacity = self._probs.shape[0]
        if size <= capacity:
            return

        used = len(self._keys)
        capacity = max(size, capacity * 2)
        probs = np.zeros((capacity, 256), dtype=np.float64)
        probs[:used] = self._probs[:used]
        neg_ent = np.zeros(capacity, dtype=np.float64)
        neg_ent[:used] = self._neg_ent[:used]
        self._probs, self._neg_ent = probs, neg_ent

    def _refresh(self) -> None:
        """Recompute every seed's KL against the current pool, in one matvec."""
        self._scored_at = self._pool_version
        used = len(self._keys)
        if not used:
            self._kl = {}
            return

        neg_log_q = -np.log2(np.asarray(self._pool.freq_dist(), dtype=np.float64))
        # KL >= 0 analytically; the subtraction can land a hair below zero.
        kl = np.maximum(self._neg_ent[:used] + self._probs[:used] @ neg_log_q, 0.0)
        self._kl = dict(zip(self._keys, kl.tolist(), strict=True))

    def stats(self) -> dict[str, Any]:
        live = self._kl.values()
        return {
            "scored": self._scored,
            "selected": self._selected,
            "pooled": len(self._keys),
            "pool_bytes": len(self._pool),
            "mean_kl": math.fsum(live) / len(live) if live else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        """Counters only; scores are a pure function of the corpus bytes."""
        return {
            "version": STATE_VERSION,
            "scored": self._scored,
            "selected": self._selected,
        }

    @classmethod
    def from_dict(
        cls, data: Any, rng: Any, cap: int = ENTROPY_SAMPLE_CAP
    ) -> EntropyKLSeedStrategy:
        """Restore counters; a malformed or unversioned payload is ignored whole."""
        out = cls(rng, cap)
        if not _valid_state(data):
            return out

        out._scored = data.get("scored", 0)
        out._selected = data.get("selected", 0)
        return out
