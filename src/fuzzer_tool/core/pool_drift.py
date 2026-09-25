"""Byte-distribution drift of the live corpus away from its seed set.

A campaign health readout, not a scheduling signal. The seed snapshot is
frozen on the first :meth:`PoolDrift.sync` (startup: the corpus as loaded,
so on ``--resume`` it is the corpus at resume -- nothing on disk records
which inputs were the user's). Later syncs fold admissions and unfold
evictions, so the pool is exactly the live corpus, the same lazy diff
``EntropyKLSeedStrategy._sync`` uses.

    seeds (frozen) ----\
                        +--> dH, JS, novel mass
    corpus --sync--> pool --> I(seed; byte)

- ``delta_bits``: H(pool) - H(seeds). > 0 drifting toward noise, < 0
  concentrating (dictionary tokens, or collapse onto padding / repeats).
- ``js_bits``: how far the distribution moved, whatever the direction.
  Bounded [0, 1]; KL(pool || seeds) is infinite on any new byte value.
- ``novel_mass``: pool mass on byte values the seeds never held.
- ``mutual_info``: H(pool) - sum(n_i/N * H(seed_i)). Inter-seed
  heterogeneity: 1000 copies of one seed read 0, ``[AAAA, BBBB]`` reads 1.

All order-0 and over the first ``ENTROPY_SAMPLE_CAP`` bytes of each input.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

from fuzzer_tool.core.byte_entropy import ENTROPY_SAMPLE_CAP, CumulativeByteEntropy


class DriftReading(NamedTuple):
    """One drift readout; all values in bits except ``novel_mass``."""

    seed_bits: float
    pool_bits: float
    delta_bits: float
    js_bits: float
    mutual_info: float
    novel_mass: float


class PoolDrift:
    """Frozen seed distribution vs the live corpus pool."""

    __slots__ = ("_pool", "_seeds", "_within", "_within_sum")

    def __init__(self) -> None:
        self._pool = CumulativeByteEntropy()
        self._seeds: CumulativeByteEntropy | None = None
        # n_i * H(seed_i) per folded seed; the sum is I(seed; byte)'s second term.
        self._within: dict[bytes, float] = {}
        self._within_sum = 0.0

    def sync(self, corpus: Iterable[bytes]) -> None:
        """Bring the pool in line with ``corpus``; the first call freezes the seeds."""
        live = set(corpus)
        within = self._within

        for seed in live.difference(within):
            bits = self._pool.add(seed)
            within[seed] = min(len(seed), ENTROPY_SAMPLE_CAP) * bits
            self._within_sum += within[seed]

        for seed in [s for s in within if s not in live]:
            self._pool.remove(seed)
            self._within_sum -= within.pop(seed)

        if self._seeds is None:
            self._seeds = self._pool.copy()

    def reading(self) -> DriftReading | None:
        """Current drift, or None before a sync or with an empty side."""
        seeds, pool = self._seeds, self._pool
        if seeds is None or not len(seeds) or not len(pool):
            return None

        seed_bits, pool_bits = seeds.bits(), pool.bits()
        # Clamp: the running sum drifts by float error over long churn.
        mutual_info = max(0.0, pool_bits - self._within_sum / len(pool))
        return DriftReading(
            seed_bits=seed_bits,
            pool_bits=pool_bits,
            delta_bits=pool_bits - seed_bits,
            js_bits=pool.js_bits(seeds),
            mutual_info=mutual_info,
            novel_mass=pool.novel_mass(seeds),
        )
