"""Kruskal-count seed strategy: score seeds by how fast byte-driven walkers couple.

Plan: ``docs/handover/kruskal_count_seed_strategy.md``.

A seed is read as a jump table. ``WALKER_COUNT`` walkers start at distinct,
evenly spaced offsets and step synchronously::

    seed  = 08 07 06 05 04 03 02 01        byte b at p jumps to (p + max(1, b)) % n
    start   w0    w1    w2    w3
    step 1  all four land on 0             -> 6/6 pairs coupled at step 1

The jump is a function of position alone, so a coupled pair stays coupled:
the walk is a functional graph and coupling means two starts drain into the
same cycle in phase. ``score = coupling_rate * (1 - mean_step / MAX_STEPS)``
lies in ``[0, 1)``; a seed with no coupled pair scores 0.

When the target profile carries a ``format_signature``, a walker jumps to the
next boundary-marker occurrence instead, falling back to the byte rule past
the last one. Caveat: every walker that starts before the first marker lands
on that same marker at step 1, so on marker-dense formats the score saturates
near its maximum and stops discriminating. The fallback path does not.

``generate`` reuses the walk: the trajectory the first coupled pair shares
after coupling becomes a mask, and donor bytes are copied into the anchor at
those offsets (modulo the donor length), leaving a detected magic prefix intact.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

WALKER_COUNT = 4
MAX_STEPS = 256
MIN_WEIGHT = 1e-6
STATE_VERSION = 1

_COUNTERS = ("scored", "coupled_pairs", "generated")
# Seeds per numpy batch: bounds the padded (batch x max_len) byte matrix.
_BATCH = 256


@dataclass
class WalkTrace:
    """First coupling step and position per walker pair ``(i, j)``, ``i < j``."""

    couple_steps: dict[tuple[int, int], int] = field(default_factory=dict)
    couple_pos: dict[tuple[int, int], int] = field(default_factory=dict)


def walker_starts(n: int) -> list[int]:
    """Distinct evenly spaced starts; fewer walkers than WALKER_COUNT on short seeds."""
    k = min(n, WALKER_COUNT)
    return [i * n // k for i in range(k)]


def boundaries(seed: bytes, profile) -> list[int] | None:
    """Sorted marker offsets in *seed*, or None when no format hint applies."""
    if not getattr(profile, "format_signature", None):
        return None

    found: set[int] = set()
    for marker in getattr(profile, "boundary_markers", None) or ():
        if not marker:
            continue
        i = seed.find(marker)
        while i >= 0:
            found.add(i)
            i = seed.find(marker, i + 1)
    return sorted(found) or None


def jump_position(seed: bytes, position: int, bounds: list[int] | None) -> int:
    """Next boundary strictly after *position*, else the wrapped byte-value jump.

    A zero byte jumps by one, so only a one-byte seed can self-loop from zero.
    """
    if bounds:
        i = bisect.bisect_right(bounds, position)
        if i < len(bounds):
            return bounds[i]
    return (position + max(1, seed[position])) % len(seed)


def trace(seed: bytes, bounds: list[int] | None) -> WalkTrace:
    """Advance all walkers MAX_STEPS steps, recording each pair's first meeting."""
    out = WalkTrace()
    pos = walker_starts(len(seed))
    k = len(pos)
    total = k * (k - 1) // 2

    for step in range(1, MAX_STEPS + 1):
        if len(out.couple_steps) == total:
            break
        pos = [jump_position(seed, p, bounds) for p in pos]

        # Record newly coupled pairs; coupled pairs never separate.
        for i in range(k):
            for j in range(i + 1, k):
                if pos[i] != pos[j] or (i, j) in out.couple_steps:
                    continue
                out.couple_steps[(i, j)] = step
                out.couple_pos[(i, j)] = pos[i]
    return out


def _score(t: WalkTrace, walkers: int) -> float:
    total = walkers * (walkers - 1) // 2
    if not total or not t.couple_steps:
        return 0.0
    rate = len(t.couple_steps) / total
    speed = 1.0 - sum(t.couple_steps.values()) / len(t.couple_steps) / MAX_STEPS
    return rate * speed


def batch_scores(seeds: list[bytes]) -> list[tuple[float, int]]:
    """Vectorized ``(score, coupled_pairs)`` for the byte-fallback walk.

    Requires ``len(seed) >= WALKER_COUNT`` and no boundary hints; ``trace`` is
    the scalar reference. Measured 12.6x over it on 500 seeds of 8-4096 bytes.
    """
    out: list[tuple[float, int]] = []
    for lo in range(0, len(seeds), _BATCH):
        out.extend(_batch(seeds[lo : lo + _BATCH]))
    return out


def _batch(seeds: list[bytes]) -> list[tuple[float, int]]:
    k = WALKER_COUNT
    lens = np.fromiter((len(s) for s in seeds), np.int64, len(seeds))
    table = np.zeros((len(seeds), int(lens.max())), np.int64)
    for i, s in enumerate(seeds):
        table[i, : len(s)] = np.frombuffer(s, np.uint8)

    # (seeds x walkers) positions, and first coupling step per (seeds x pairs).
    pos = np.arange(k)[None, :] * lens[:, None] // k
    rows = np.arange(len(seeds))[:, None]
    left, right = np.triu_indices(k, 1)
    first = np.zeros((len(seeds), left.size), np.int64)

    for step in range(1, MAX_STEPS + 1):
        pos = (pos + np.maximum(1, table[rows, pos])) % lens[:, None]
        first[(pos[:, left] == pos[:, right]) & (first == 0)] = step

    coupled = (first > 0).sum(1)
    mean_step = first.sum(1) / np.maximum(coupled, 1)
    score = np.where(coupled > 0, coupled / left.size * (1.0 - mean_step / MAX_STEPS), 0.0)
    return list(zip(score.tolist(), coupled.tolist(), strict=True))


def _valid_state(data) -> bool:
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        return False

    for key in _COUNTERS:
        v = data.get(key, 0)
        if type(v) is not int or v < 0:
            return False

    s = data.get("score_sum", 0.0)
    return type(s) in (int, float) and math.isfinite(s) and s >= 0


class KruskalCountSeedStrategy:
    """Elo-arbitrated ``kruskal_count`` seed arm (``--kruskal-count``)."""

    def __init__(self, rng, profile) -> None:
        self._rng = rng
        self._profile = profile
        self._cache: dict[bytes, float] = {}
        self._masks: dict[bytes, list[int]] = {}
        self._scored = 0
        self._coupled_pairs = 0
        self._generated = 0
        self._score_sum = 0.0

    def score_seed(self, seed: bytes) -> float:
        """Coupling score of *seed*, memoized on content."""
        cached = self._cache.get(seed)
        if cached is not None:
            return cached

        t = trace(seed, boundaries(seed, self._profile))
        v = _score(t, len(walker_starts(len(seed))))
        self._record(seed, v, len(t.couple_steps))
        return v

    def _record(self, seed: bytes, v: float, pairs: int) -> None:
        self._cache[seed] = v
        self._scored += 1
        self._coupled_pairs += pairs
        self._score_sum += v

    def scores(self, seeds: list[bytes]) -> list[float]:
        """Score a corpus and evict cache entries for seeds no longer in it."""
        self._fill(seeds)
        out = [self.score_seed(s) for s in seeds]
        if self._masks or len(self._cache) > len(seeds):
            live = set(seeds)
            self._cache = {k: v for k, v in self._cache.items() if k in live}
            self._masks = {k: v for k, v in self._masks.items() if k in live}
        return out

    def _fill(self, seeds: list[bytes]) -> None:
        """Batch-score uncached seeds the vectorized walk covers."""
        if getattr(self._profile, "format_signature", None):
            return
        miss = list(
            dict.fromkeys(s for s in seeds if len(s) >= WALKER_COUNT and s not in self._cache)
        )
        for seed, (v, pairs) in zip(miss, batch_scores(miss), strict=True):
            self._record(seed, v, pairs)

    def select(self, seeds: list[bytes]) -> bytes | None:
        """Anchor draw proportional to ``score + MIN_WEIGHT``."""
        if not seeds:
            return None
        weights = [w + MIN_WEIGHT for w in self.scores(seeds)]
        return self._rng.weighted_choice(seeds, weights)

    def generate(self, anchor: bytes, donors: list[bytes]) -> bytes | None:
        """Copy donor bytes into *anchor* along its post-coupling trajectory."""
        if len(anchor) < 2:
            return None

        mask = self._mask(anchor)
        if not mask:
            return None

        pool = [d for d in donors if d and d != anchor]
        if not pool:
            return None
        donor = self._donor(pool)

        # Recombine at masked offsets; modulo handles shorter/longer donors.
        out = bytearray(anchor)
        for p in mask:
            out[p] = donor[p % len(donor)]

        # Identical result: one deterministic change instead of retrying.
        if out == anchor:
            p = self._rng.choice(mask)
            out[p] ^= self._rng.randint(1, 255)

        self._generated += 1
        return bytes(out)

    def _mask(self, anchor: bytes) -> list[int]:
        """Offsets the earliest-coupled pair visits from coupling to MAX_STEPS."""
        cached = self._masks.get(anchor)
        if cached is None:
            cached = self._masks[anchor] = self._walk_mask(anchor)
        return cached

    def _walk_mask(self, anchor: bytes) -> list[int]:
        bounds = boundaries(anchor, self._profile)
        t = trace(anchor, bounds)
        if not t.couple_steps:
            return []

        pair = min(t.couple_steps, key=lambda q: (t.couple_steps[q], q))
        p = t.couple_pos[pair]
        seen: dict[int, None] = {}

        # Functional graph: the first revisit closes the cycle.
        for _ in range(t.couple_steps[pair], MAX_STEPS + 1):
            if p in seen:
                break
            seen[p] = None
            p = jump_position(anchor, p, bounds)

        prefix = self._magic_len(anchor)
        return [q for q in seen if q >= prefix]

    def _magic_len(self, anchor: bytes) -> int:
        magic = getattr(self._profile, "magic_bytes", None) or ()
        return max((len(m) for m in magic if m and anchor.startswith(m)), default=0)

    def _donor(self, pool: list[bytes]) -> bytes:
        if len(pool) == 1:
            return pool[0]
        weights = [self.score_seed(d) + MIN_WEIGHT for d in pool]
        return self._rng.weighted_choice(pool, weights)

    def stats(self) -> dict:
        return {
            "scored": self._scored,
            "coupled_pairs": self._coupled_pairs,
            "generated": self._generated,
            "mean_score": self._score_sum / self._scored if self._scored else 0.0,
            "cached": len(self._cache),
        }

    def to_dict(self) -> dict:
        """Primitive counters only; scores are a pure function of bytes and profile."""
        return {
            "version": STATE_VERSION,
            "scored": self._scored,
            "coupled_pairs": self._coupled_pairs,
            "generated": self._generated,
            "score_sum": self._score_sum,
        }

    @classmethod
    def from_dict(cls, data, rng, profile) -> KruskalCountSeedStrategy:
        """Restore counters; a malformed or unversioned payload is ignored whole."""
        out = cls(rng, profile)
        if not _valid_state(data):
            return out

        out._scored = data.get("scored", 0)
        out._coupled_pairs = data.get("coupled_pairs", 0)
        out._generated = data.get("generated", 0)
        out._score_sum = float(data.get("score_sum", 0.0))
        return out
