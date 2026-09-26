"""PositionFibonacciScheduler: golden-ratio offset sweep.

Stateless sibling of ``core/schedulers/pos_round_robin.py``. Same binning,
but over the live buffer (``width = ceil(buf_len / MAX_BINS)``), not the
parent seed: earlier operators may have resized it, and seed-sized bins
clamped to a shrunk buffer piled every overshoot onto its last byte. Bins
are visited in Fibonacci-hashing order instead of ``index % num_bins``::

    bin_n = floor(frac(n / phi) * num_bins)       n = global counter

    n      0     1     2     3     4     5
    frac   0    .618  .236  .854  .472  .090     (x num_bins)

Why: any prefix of the sequence spreads evenly over the input (three-gap
theorem), so the tail is reached on the second pick, not after
``num_bins`` picks. And there is no per-seed state: round-robin keeps its
cycle in an LRU of ``MAX_SEEDS`` seeds, so on a corpus larger than that a
seed's cycle is evicted between visits and restarts at bin 0, pinning
mutations to the head of the file. One counter, O(1) memory, no eviction.

``record()`` is a no-op kept for arena parity, like round-robin's.
"""

from __future__ import annotations

from collections.abc import Sequence

from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.schedulers.pos_burn_front import MAX_BINS

GOLDEN_64 = 0x9E3779B97F4A7C15  # floor(2^64 / phi): Fibonacci hashing multiplier
MASK_64 = (1 << 64) - 1
WORD_BITS = 64


class PositionFibonacciScheduler:
    """Golden-ratio sweep over a seed's offset bins.

    A deterministic, signal-free baseline for the position arena that,
    unlike round-robin, covers the whole input from the first picks and
    keeps no per-seed state.
    """

    name = "fibonacci"

    #: No meaningful priors for a fixed sweep, mirrors PositionRoundRobinScheduler.
    supports_priors = False

    def __init__(self) -> None:
        self._n = 0

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """First byte of the next golden-ratio bin; never declines on a live buffer."""
        if buf_len <= 0:
            return None

        # Live buffer, not the seed: see the module docstring.
        n = buf_len
        width = max(1, -(-n // MAX_BINS))
        num_bins = max(1, -(-n // width))

        # Fibonacci hashing: top bits of n * 2^64/phi, scaled to num_bins.
        frac = (self._n * GOLDEN_64) & MASK_64
        self._n = (self._n + 1) & MASK_64
        b = (frac * num_bins) >> WORD_BITS

        return min(b * width, buf_len - 1)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """No-op: the sweep ignores the outcome signal, same as round-robin."""
