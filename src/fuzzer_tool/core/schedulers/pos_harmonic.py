"""PositionHarmonicScheduler: a learned angular-harmonic density over record phase.

Idea carried over from the universal-pattern field (``core/universal_pattern.py``):
a periodic structure is a sum of angular harmonics ``cos(n*theta + phi_n)``. Here
the angle is the record phase ``theta = 2*pi*(offset mod L)/L`` (``L`` from
``periodicity.estimate_record_size``) and the harmonics are *fitted* to where
coverage gains landed instead of being chosen by hand::

    c_n   = sum_k w_k * exp(+i n theta_k)          n = 1..K  (running, decayed)
    p(th) = FLOOR + max(0, 1 + 2 * sum_n |c_n|/W * cos(n*(th - arg c_n)))

This is a truncated Fourier-series density estimate on the circle. K = 1 is
exactly the von-Mises-style lock that ``circular_stats.concentration`` tests
for (one preferred field); higher n resolves several fields per record. The
record index is then drawn uniformly, so the scheduler learns *which field*
inside a record is productive and spreads across *all* records.

The Bessel radial envelope of the original equation has no 1-D analogue here
(a byte offset has one angular coordinate and one record index), so it is
deliberately unused: the field and the scheduler share the angular math only.

Prototype: not wired into ``position_arena``/CLI and not persisted. Declines
(``None``) until a seed has a stride and ``MIN_GAINS`` gain rounds.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import xxhash

from fuzzer_tool.core.circular_stats import MIN_STRIDE, fold_offsets
from fuzzer_tool.core.periodicity import estimate_record_size
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

MAX_SEEDS = 256
HARMONICS = 6  # K: highest harmonic fitted
MIN_GAINS = 3  # gain rounds before the density is trusted
DECAY = 0.98  # forgetting per gain round, keeps the density adaptive
FLOOR = 0.15  # uniform mass added to every phase bin: never starve a field
SPARK_RATE = 0.05  # uniform escapes


@dataclass
class _State:
    stride: int | None
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros(HARMONICS, dtype=np.complex128))
    total: float = 0.0
    gains: int = 0


def phase_density(coeffs: np.ndarray, total: float, stride: int) -> np.ndarray:
    """Normalised per-byte-phase probability vector of length *stride*."""
    th = 2.0 * math.pi * np.arange(stride) / stride
    p = np.ones(stride, dtype=np.float64)
    if total > 0.0:
        for n in range(1, len(coeffs) + 1):
            c = coeffs[n - 1] / total
            p += 2.0 * abs(c) * np.cos(n * th - np.angle(c))
    p = np.maximum(p, 0.0) + FLOOR
    return p / p.sum()


class PositionHarmonicScheduler:
    name = "harmonic"

    def __init__(self, rng: RandPool) -> None:
        self._rng = rng
        self._seeds: OrderedDict[int, _State] = OrderedDict()

    def propose(self, data: bytes, buf_len: int) -> int | None:
        st = self._seeds.get(self._key(data))
        if buf_len <= 0 or st is None or st.stride is None or st.gains < MIN_GAINS:
            return None
        last = buf_len - 1
        if self._rng.random() < SPARK_RATE:
            return self._rng.randint(0, last)

        p = phase_density(st.coeffs, st.total, st.stride)
        u = self._rng.random()
        phase = int(min(np.searchsorted(np.cumsum(p), u, side="right"), st.stride - 1))
        records = max(1, (buf_len + st.stride - 1) // st.stride)
        rec = self._rng.randint(0, records - 1)
        pos = rec * st.stride + phase
        # A short final record may not contain this phase: fall back inside it.
        return pos if pos <= last else rec * st.stride + (phase % max(1, buf_len - rec * st.stride))

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        offsets = [o for o in offsets if o >= 0]
        if outcome is not Outcome.GAIN or not offsets or not data:
            return
        st = self._state_for(data)
        if st.stride is None:
            return
        th = fold_offsets(offsets, st.stride)
        w = weight / len(offsets)
        st.coeffs *= DECAY
        st.total *= DECAY
        for n in range(1, HARMONICS + 1):
            st.coeffs[n - 1] += w * np.exp(1j * n * th).sum()
        st.total += weight
        st.gains += 1

    def stride(self, data: bytes) -> int | None:
        st = self._seeds.get(self._key(data))
        return st.stride if st else None

    def seed_count(self) -> int:
        return len(self._seeds)

    @staticmethod
    def _key(data: bytes) -> int:
        return xxhash.xxh3_64_intdigest(data)

    def _state_for(self, data: bytes) -> _State:
        key = self._key(data)
        st = self._seeds.get(key)
        if st is None:
            stride = estimate_record_size(data)
            st = self._seeds[key] = _State(stride if stride and stride >= MIN_STRIDE else None)
            while len(self._seeds) > MAX_SEEDS:
                self._seeds.popitem(last=False)
        self._seeds.move_to_end(key)
        return st
