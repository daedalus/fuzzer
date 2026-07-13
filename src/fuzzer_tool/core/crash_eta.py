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
    r"error|invalid|overflow|underflow|corrupt|malformed|bad |failed|unable",
    re.IGNORECASE,
)


@dataclass
class CrashETA:
    """Estimated executions to first crash."""

    point_est: int
    low: int
    high: int
    edges_to_crash: int  # estimated risky edges needed before first crash
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


def estimate_execs_to_first_crash(
    profile: TargetProfile,
    gt_result: dict,
    discovery_rate: float,
    calibration_execs: int = 0,
) -> CrashETA:
    """Estimate executions needed to reach the first crash.

    Combines:
    - Static risky density (rho) from TargetProfiler
    - Good-Turing total edge estimate from EdgeTracker
    - Calibrated discovery rate (edges per 1000 execs)
    - Calibration execs count for confidence interval scaling

    More calibration execs tighten the interval via 1/sqrt(N) scaling
    relative to a 1000-exec baseline.
    """
    import math

    rho = estimate_risky_density(profile)
    e_total = gt_result.get("n", 0) + gt_result.get("estimated_undiscovered", 0)
    confidence = gt_result.get("confidence", "low")

    if rho <= 0 or e_total <= 0 or discovery_rate <= 0:
        return CrashETA(
            point_est=10_000_000,
            low=1_000_000,
            high=100_000_000,
            edges_to_crash=0,
            confidence="low",
            reasoning="Insufficient data: zero density, edges, or discovery rate",
        )

    risky_edges_needed = 1.0 / rho
    edges_to_crash = int(risky_edges_needed)
    execs = (risky_edges_needed / discovery_rate) * 1000

    # Base multiplier ranges by GT confidence
    if confidence == "high":
        base_low, base_high = 0.5, 2.0
    elif confidence == "medium":
        base_low, base_high = 0.2, 5.0
    else:
        base_low, base_high = 0.1, 10.0

    # Scale by calibration execs: more execs → tighter interval
    # Reference baseline is 1000 execs; scale via 1/sqrt(N/N_ref)
    if calibration_execs > 0:
        scale = math.sqrt(1000.0 / max(1, calibration_execs))
        # Clamp scale to [0.3, 3.0] so very few execs don't blow out
        # and very many don't collapse to a point
        scale = max(0.3, min(3.0, scale))
    else:
        scale = 3.0  # no calibration → widest interval

    multiplier_low = base_low * scale
    multiplier_high = base_high * scale

    low = max(100, int(execs * multiplier_low))
    high = int(execs * multiplier_high)
    point = int(execs)

    return CrashETA(
        point_est=point,
        low=low,
        high=high,
        edges_to_crash=edges_to_crash,
        confidence=confidence,
        reasoning=(
            f"rho={rho:.3f} (risky density), "
            f"E_total={e_total}, "
            f"discovery_rate={discovery_rate:.1f}/1k, "
            f"calib_execs={calibration_execs}, "
            f"CI_scale={scale:.2f}, "
            f"geometric ETA={execs:.0f}"
        ),
    )
