"""Per-execution target choice in multi-target mode (``--target-schedule``)."""

from __future__ import annotations

from enum import Enum


class TargetSchedule(Enum):
    """How each execution picks its target.

    WEIGHTED:    round robin for the first 100 execs, then a random draw
                 weighted by 1 / cumulative edges (least covered first).
    ROUND_ROBIN: exec i -> target i mod V, always; exactly equal shares.
    """

    WEIGHTED = "weighted"
    ROUND_ROBIN = "round-robin"
