"""Moved to ``op_consolidated_v1.py``; kept so pre-v2 imports still resolve."""

from __future__ import annotations

from fuzzer_tool.core.schedulers.op_consolidated_v1 import ConsolidatedV1Scheduler


class ConsolidatedScheduler(ConsolidatedV1Scheduler):
    """Pre-v2 name of ConsolidatedV1Scheduler; reports the pre-v2 stats keys."""

    _STATS_PREFIX = "consolidated"


__all__ = ["ConsolidatedScheduler"]
