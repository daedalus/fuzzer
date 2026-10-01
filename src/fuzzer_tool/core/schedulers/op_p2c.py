"""OpP2CScheduler: power of two choices over operators (Mitzenmacher 2001).

Operator-arena twin of ``seed_p2c``: draw two candidate operators uniformly
and keep the one with the higher Beta(1, 1) posterior mean. O(1) per pick,
unlike Thompson/UCB arms that score every operator::

    draws (flip, havoc), mean flip=0.2 havoc=0.6  ->  havoc

Equal means keep the first draw, so with no signal it is uniform random
selection. Experimental, Elo-only (absent from ``_FALLBACK_PRECEDENCE``).
"""

from __future__ import annotations

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts


class OpP2CScheduler(ArmCounts):
    """Two uniform draws, higher posterior mean wins."""

    #: Posterior starts at Beta(1, 1) for every operator; no prior override.
    supports_priors = False

    def __init__(self, rng) -> None:
        if rng is None:
            raise ValueError("OpP2CScheduler requires a RandPool (Hard Rule 16)")
        super().__init__()
        self._rng = rng

    def select_op(self, ops: list[str]) -> str:
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        first = self._rng.choice(ops)
        second = self._rng.choice(ops)
        return second if self.mean(second) > self.mean(first) else first
