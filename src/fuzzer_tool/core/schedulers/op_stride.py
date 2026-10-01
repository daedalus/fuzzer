"""OpStrideScheduler: stride scheduling over operators (Waldspurger 1995).

Tickets are each operator's Beta(1, 1) posterior mean of success, so the
share follows the evidence deterministically -- a Thompson share without
the sampling variance::

    3 wins vs 3 losses:  means 0.8 vs 0.2  ->  4:1 picks, interleaved

No evidence gives every operator 0.5 tickets: plain round robin in
candidate order (falsification). Unlike an argmax arm no operator's share
reaches zero. Experimental, Elo-only (absent from ``_FALLBACK_PRECEDENCE``).
"""

from __future__ import annotations

from fuzzer_tool.core.fair_queue import Stride
from fuzzer_tool.core.schedulers._arm_counts import ArmCounts


class OpStrideScheduler(ArmCounts):
    """Stride over operators with posterior-mean tickets."""

    #: Tickets start at Beta(1, 1) for every operator; no prior override.
    supports_priors = False

    def __init__(self) -> None:
        super().__init__()
        self._stride = Stride()

    def select_op(self, ops: list[str]) -> str:
        return self._stride.pick(ops, self.mean)
