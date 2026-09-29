"""Per-execution target choice in multi-target mode (``--target-schedule``)."""

from __future__ import annotations

from enum import Enum


class TargetSchedule(Enum):
    """How each execution picks its target.

    WEIGHTED:    round robin for the first 100 execs, then a random draw
                 weighted by 1 / cumulative edges (least covered first).
    ROUND_ROBIN: exec i -> target i mod V, always; exactly equal shares.
    WRR:         smooth weighted round robin on 1 / cumulative edges: same
                 long-run share as WEIGHTED, but deterministic (no RNG, no
                 variance). Shares are of iterations, not of time.
    WFQ:         weighted fair queuing on 1 / cumulative edges: shares are of
                 wall time, so a slow target cannot take more than its share.
    """

    WEIGHTED = "weighted"
    ROUND_ROBIN = "round-robin"
    WRR = "wrr"
    WFQ = "wfq"
