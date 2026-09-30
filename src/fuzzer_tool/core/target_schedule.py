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
    PHI:         gated first-passage / isoperimetric schedule (Module 5 + P2-1).
                 Weights ∝ estimate_time_to_next_discovery per target so lagging
                 / harder-looking targets get more share. Uses a cached Φ
                 profile when one has been supplied; otherwise the regime prior
                 inside estimate_time_to_next_discovery. Off unless chosen —
                 default schedule is unchanged.
    """

    WEIGHTED = "weighted"
    ROUND_ROBIN = "round-robin"
    WRR = "wrr"
    WFQ = "wfq"
    PHI = "phi"
