"""Pre-fuzzing crash ETA estimation.

Combines static analysis (TargetProfiler) with calibrated execution
statistics to estimate how many edges/executions until the first crash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fuzzer_tool.core.target_profiler import TargetProfile

# Matches error-related keywords in function names and rodata strings.
_ERROR_RE = re.compile(
    r"error|invalid|overflow|underflow|corrupt|malformed|bad|failed|unable",
    re.IGNORECASE,
)


@dataclass
class CrashETA:
    """Estimated executions to first crash."""

    point_est: int
    low: int
    high: int
    confidence: str  # "low", "medium", "high"
    reasoning: str


def estimate_risky_density(profile: TargetProfile) -> float:
    """Estimate fraction of control flow that's a potential crash site.

    Uses ERROR_KEYWORDS matches in function names and rodata strings as a
    proxy for defensive/error-handling paths in the binary.

    Returns a value in [0.0, 1.0].
    """
    if not profile.functions:
        return 0.0

    risky = 0

    # Count functions whose names suggest error handling / defensive paths.
    for func_name in profile.functions:
        if _ERROR_RE.search(func_name):
            risky += 1

    # Also count rodata error strings as additional risky-signal contributions.
    for _offset, s in profile.rodata_strings:
        if _ERROR_RE.search(s):
            risky += 1

    # Clamp to [0, 1].
    density = min(1.0, risky / max(1, len(profile.functions)))
    return density
