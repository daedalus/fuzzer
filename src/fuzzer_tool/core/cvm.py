"""CVM algorithm — streaming distinct-elements (F₀) estimator.

Ported and cleaned from AIscripts/CVM.py for fuzzer-tool.
Implements the textbook algorithm from:

  "Distinct Elements in Streams: An Algorithm for the (Text) Book"
  (arXiv:2301.10191)

Useful for approximate cardinality of edges, paths, comparison sites,
or seeds without materializing the full set. Complements Chao2,
Renyi spectrum, rate-distortion and edge-tracker modules.
"""

from __future__ import annotations

import math
from collections.abc import Hashable

from fuzzer_tool.core.rand_pool import RandPool


class F0Estimator:
    """Streaming approximate distinct-count (F₀) estimator.

    Parameters
    ----------
    eps:
        Relative approximation error (ε).
    delta:
        Failure probability (δ).
    m:
        Upper bound on stream length (used to size the threshold).
    """

    def __init__(
        self, eps: float = 0.1, delta: float = 1e-6, m: int = 1_000_000, rng: RandPool | None = None
    ) -> None:
        if not (0 < eps < 1):
            raise ValueError("eps must be in (0, 1)")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if m < 1:
            raise ValueError("m must be >= 1")

        self.eps = eps
        self.delta = delta
        self.m = m
        self._rng = rng or RandPool()
        # thresh ≈ (2/ε²) · ln(8m/δ)
        self.thresh = math.ceil((2.0 / (eps**2)) * math.log((8.0 * m) / delta))
        self.X: set[Hashable] = set()
        self.p: float = 1.0  # current sampling probability

    def update(self, a: Hashable) -> bool | None:
        """Process one stream element.

        Returns
        -------
        True
            Update succeeded.
        None
            Algorithm returned ⊥ (failure) — the set is still at capacity
            after a down-sample. Caller may choose to restart or ignore.
        """
        # Remove any previous occurrence (the algorithm treats the stream
        # as a sequence; duplicates are handled by the sampling).
        if a in self.X:
            self.X.remove(a)

        # Re-sample with current probability p
        if self._rng.random() < self.p:
            self.X.add(a)

        # Down-sample when the working set reaches the threshold
        if len(self.X) == self.thresh:
            new_X: set[Hashable] = set()
            for x in self.X:
                if self._rng.random() < 0.5:
                    new_X.add(x)
            self.X = new_X
            self.p /= 2.0

            # Still full after down-sample → ⊥
            if len(self.X) == self.thresh:
                return None

        return True

    def estimate(self) -> float:
        """Return the current F₀ estimate = |X| / p."""
        if self.p <= 0:
            return float("inf")
        return len(self.X) / self.p

    def clear(self) -> None:
        """Reset state (keeps the original ε, δ, m parameters)."""
        self.X.clear()
        self.p = 1.0

    @property
    def size(self) -> int:
        """Current number of items held in the working set."""
        return len(self.X)
