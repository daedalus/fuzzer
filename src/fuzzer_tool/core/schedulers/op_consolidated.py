"""Moved to ``op_consolidated_v1.py``; kept so pre-v2 imports still resolve."""

from __future__ import annotations

from fuzzer_tool.core.schedulers.op_consolidated_v1 import ConsolidatedV1Scheduler

#: Pre-v2 name of :class:`ConsolidatedV1Scheduler`.
ConsolidatedScheduler = ConsolidatedV1Scheduler

__all__ = ["ConsolidatedScheduler"]
