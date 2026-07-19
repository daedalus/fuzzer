"""Pre-fuzzing crash ETA estimation.

Combines static analysis (TargetProfiler) with calibrated execution
statistics to estimate how many edges/executions until the first crash.
Also provides dynamic crash prediction via mutual information between
input bytes and crash outcomes.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass

from fuzzer_tool.core.target_profiler import TargetProfile

# Matches error-related keywords in function names and rodata strings.
# Note: "bad" without trailing space to match C identifiers like bad_alloc.
_ERROR_RE = re.compile(
    r"error|invalid|overflow|underflow|corrupt|malformed|bad|failed|unable",
    re.IGNORECASE,
)


class CrashMITracker:
    """Track mutual information between input bytes and crash outcomes.

    Computes I(X_i; C) where X_i is byte position i and C is the
    binary crash/no-crash outcome. High-MI bytes are the ones that
    actually control whether the program crashes — mutating them is
    more likely to trigger crashes.

    This is distinct from MutualInformationTracker (mi.py) which
    tracks I(X_i; coverage_edges) for coverage guidance.

    Args:
        max_positions: Maximum byte positions to track.
        min_observations: Minimum observations before computing MI.
    """

    def __init__(self, max_positions: int = 4096, min_observations: int = 20):
        self.max_positions = max_positions
        self.min_observations = min_observations
        # Per-position: byte_value -> crash_count
        self.joint_crash: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # Per-position: byte_value -> total_count
        self.byte_total: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # Per-position: total observations
        self.position_counts: dict[int, int] = defaultdict(int)
        # Global: crash count, total count
        self.total_crashes: int = 0
        self.total_execs: int = 0
        # Cached MI values
        self._mi_cache: dict[int, float] = {}
        self._cache_valid: bool = False

    def record(self, input_bytes: bytes, is_crash: bool) -> None:
        """Record one input-crash pair.

        Args:
            input_bytes: The input that was executed.
            is_crash: Whether this execution crashed.
        """
        self.total_execs += 1
        if is_crash:
            self.total_crashes += 1
            self._cache_valid = False

        # For crashes: track all positions. For non-crashes: sample 20%.
        n = min(len(input_bytes), self.max_positions)
        if is_crash:
            positions = range(n)
        else:
            # Sample ~20% of positions for non-crashes to reduce overhead
            step = max(1, n // max(1, n // 5))
            positions = range(0, n, step)

        for pos in positions:
            byte_val = input_bytes[pos]
            self.position_counts[pos] += 1
            self.byte_total[pos][byte_val] += 1
            if is_crash:
                self.joint_crash[pos][byte_val] += 1

        # Prune per-position byte values to cap memory growth
        if self.total_execs % 500 == 0:
            self._prune()

    def _prune(self, max_values_per_pos: int = 32) -> None:
        """Keep only the top N most frequent byte values per position."""
        for pos in list(self.byte_total):
            bv = self.byte_total[pos]
            if len(bv) > max_values_per_pos + 8:
                top = sorted(bv.items(), key=lambda x: x[1], reverse=True)[:max_values_per_pos]
                self.byte_total[pos] = defaultdict(int, dict(top))
                if pos in self.joint_crash:
                    self.joint_crash[pos] = defaultdict(
                        int, {k: v for k, v in self.joint_crash[pos].items()
                              if k in self.byte_total[pos]}
                    )
        # Only invalidate cache periodically to avoid recomputing MI on every call
        if self.total_execs % 50 == 0:
            self._cache_valid = False

    def mi(self, position: int) -> float:
        """Compute I(X_pos; C) in bits.

        I(X; C) = sum_{x,c} P(x,c) * log2(P(x,c) / (P(x) * P(c)))

        Returns 0.0 if insufficient data or position not observed.
        """
        if self.position_counts.get(position, 0) < self.min_observations:
            return 0.0
        if self.total_execs == 0 or self.total_crashes == 0:
            return 0.0

        n = self.total_execs
        p_crash = self.total_crashes / n
        p_no_crash = 1.0 - p_crash

        if p_crash == 0 or p_no_crash == 0:
            return 0.0

        mi_val = 0.0
        pos_total = self.position_counts[position]

        for byte_val, bc in self.byte_total[position].items():
            p_x = bc / n
            crash_count = self.joint_crash[position].get(byte_val, 0)
            no_crash_count = bc - crash_count

            if crash_count > 0:
                p_xy = crash_count / n
                mi_val += p_xy * math.log2(p_xy / (p_x * p_crash))
            if no_crash_count > 0:
                p_x_no_crash = no_crash_count / n
                mi_val += p_x_no_crash * math.log2(p_x_no_crash / (p_x * p_no_crash))

        return max(0.0, mi_val)

    def all_mi(self) -> dict[int, float]:
        """Compute MI for all observed positions."""
        if self._cache_valid:
            return self._mi_cache
        self._mi_cache = {
            pos: self.mi(pos)
            for pos in self.position_counts
            if self.position_counts[pos] >= self.min_observations
        }
        # Cache sorted positions and weights for weighted_position
        self._cached_positions = sorted(self._mi_cache.keys())
        self._cached_weights = [max(self._mi_cache[p], 0.01) for p in self._cached_positions]
        self._cached_total = sum(self._cached_weights)
        self._cache_valid = True
        return self._mi_cache

    def top_positions(self, k: int = 10) -> list[tuple[int, float]]:
        """Return the k byte positions with highest MI (most crash-predictive)."""
        mi_vals = self.all_mi()
        if not mi_vals:
            return []
        sorted_mi = sorted(mi_vals.items(), key=lambda x: x[1], reverse=True)
        return sorted_mi[:k]

    def weighted_position(self, input_length: int) -> int:
        """Sample a byte position weighted by crash MI."""
        if not self._cache_valid:
            self.all_mi()
        if not self._cached_positions:
            return 0
        # Binary search for cutoff index
        import bisect
        idx = bisect.bisect_left(self._cached_positions, input_length)
        if idx == 0:
            return 0
        # Sample using precomputed weights (first idx elements)
        r = random.random() * self._cached_total
        cumulative = 0.0
        for i in range(idx):
            cumulative += self._cached_weights[i]
            if r <= cumulative:
                return self._cached_positions[i]
        return self._cached_positions[idx - 1]

    def top_values(self, position: int, k: int = 5) -> list[int]:
        """Return the k byte values at position with highest crash count."""
        if position not in self.joint_crash:
            return []
        crash_vals = sorted(
            self.joint_crash[position].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [bv for bv, _ in crash_vals[:k]]

    def crash_density_estimate(self) -> float:
        """Estimate crash probability from MI profile.

        Uses the average MI across positions as a proxy for how much
        the input controls crash outcomes. Higher average MI means
        crashes are more input-dependent (easier to find via mutation).

        Returns a value in [0.0, 1.0] where higher = more crash-predictive.
        """
        mi_vals = self.all_mi()
        if not mi_vals:
            return 0.0
        # Average MI across positions, normalized by max possible (1 bit for binary outcome)
        avg_mi = sum(mi_vals.values()) / len(mi_vals)
        # Clamp to [0, 1] — MI of a binary variable is at most 1 bit
        return min(1.0, avg_mi)

    def save(self) -> dict:
        """Serialize tracker state."""
        return {
            "max_positions": self.max_positions,
            "min_observations": self.min_observations,
            "total_crashes": self.total_crashes,
            "total_execs": self.total_execs,
            "position_counts": dict(self.position_counts),
            "joint_crash": {
                str(pos): {str(bv): c for bv, c in bv_map.items()}
                for pos, bv_map in self.joint_crash.items()
            },
            "byte_total": {
                str(pos): {str(bv): c for bv, c in bv_map.items()}
                for pos, bv_map in self.byte_total.items()
            },
        }

    def load(self, data: dict) -> None:
        """Deserialize tracker state."""
        self.max_positions = data.get("max_positions", self.max_positions)
        self.min_observations = data.get("min_observations", self.min_observations)
        self.total_crashes = data.get("total_crashes", 0)
        self.total_execs = data.get("total_execs", 0)
        self.position_counts = defaultdict(int, {
            int(k): v for k, v in data.get("position_counts", {}).items()
        })
        self.joint_crash = defaultdict(lambda: defaultdict(int))
        for pos_str, bv_map in data.get("joint_crash", {}).items():
            for bv_str, c in bv_map.items():
                self.joint_crash[int(pos_str)][int(bv_str)] = c
        self.byte_total = defaultdict(lambda: defaultdict(int))
        for pos_str, bv_map in data.get("byte_total", {}).items():
            for bv_str, c in bv_map.items():
                self.byte_total[int(pos_str)][int(bv_str)] = c
        self._cache_valid = False


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

    Normalizes function-risk and string-risk separately against their own
    totals, then combines via weighted average (60% function, 40% string).

    Returns a value in [0.0, 1.0].
    """
    if not profile.functions:
        return 0.0

    # Function-risk: fraction of functions with error-handling names
    risky_funcs = sum(1 for f in profile.functions if _ERROR_RE.search(f))
    func_density = risky_funcs / len(profile.functions)

    # String-risk: fraction of rodata strings that are error messages
    if profile.rodata_strings:
        risky_strings = sum(1 for _, s in profile.rodata_strings if _ERROR_RE.search(s))
        string_density = risky_strings / len(profile.rodata_strings)
    else:
        string_density = 0.0

    # Weighted combination: functions are more indicative of code structure
    density = 0.6 * func_density + 0.4 * string_density
    return min(1.0, density)


def estimate_execs_to_first_crash(
    profile: TargetProfile,
    gt_result: dict,
    discovery_rate: float,
    calibration_execs: int = 0,
    crash_mi: CrashMITracker | None = None,
) -> CrashETA:
    """Estimate executions needed to reach the first crash.

    Combines:
    - Static risky density (rho) from TargetProfiler
    - Dynamic crash MI density from CrashMITracker (if available)
    - Good-Turing total edge estimate from EdgeTracker
    - Calibrated discovery rate (edges per 1000 execs)
    - Calibration execs count for confidence interval scaling

    When crash_mi has enough data, its dynamic estimate blends with
    the static keyword heuristic (70% dynamic, 30% static) to produce
    a live, evidence-updating risk estimate.
    """
    rho_static = estimate_risky_density(profile)

    # Blend static and dynamic if MI data is available
    if crash_mi and crash_mi.total_execs >= crash_mi.min_observations:
        rho_dynamic = crash_mi.crash_density_estimate()
        # Dynamic data gets more weight as observations accumulate
        dynamic_weight = min(0.7, crash_mi.total_execs / (crash_mi.total_execs + 1000))
        rho = dynamic_weight * rho_dynamic + (1.0 - dynamic_weight) * rho_static
    else:
        rho = rho_static
    e_total = gt_result.get("n", 0) + gt_result.get("estimated_undiscovered", 0)
    confidence = gt_result.get("confidence", "low")

    if rho <= 0 or e_total <= 0:
        return CrashETA(
            point_est=10_000_000,
            low=1_000_000,
            high=100_000_000,
            edges_to_crash=0,
            confidence="low",
            reasoning="Insufficient data: zero density or edges",
        )

    risky_edges_needed = 1.0 / rho
    edges_to_crash = int(risky_edges_needed)

    saturated = discovery_rate <= 0 and calibration_execs > 0

    if discovery_rate > 0:
        execs = (risky_edges_needed / discovery_rate) * 1000
    elif saturated:
        # Coverage saturated: crash is hard to find despite full coverage.
        # The crash likely requires specific data values, not just reaching
        # new code. Conservative estimate: more risky edges → more likely
        # to find crash in existing coverage, but still needs many execs.
        base_multiplier = 10.0 + (1.0 - rho) * 90.0  # 10x (high rho) to 100x (low rho)
        execs = calibration_execs * base_multiplier
    else:
        execs = 10_000_000

    # Confidence interval
    if saturated:
        # Saturated case: use fixed 0.3x-3x range around the conservative estimate
        low = max(100, int(execs * 0.3))
        high = int(execs * 3.0)
    else:
        # Normal case: CI scaling based on GT confidence and calibration execs
        if confidence == "high":
            base_low, base_high = 0.5, 2.0
        elif confidence == "medium":
            base_low, base_high = 0.2, 5.0
        else:
            base_low, base_high = 0.1, 10.0

        if calibration_execs > 0:
            scale = math.sqrt(1000.0 / max(1, calibration_execs))
            scale = max(0.3, min(3.0, scale))
        else:
            scale = 3.0

        low = max(100, int(execs * base_low * scale))
        high = int(execs * base_high * scale)

    point = int(execs)

    mode = "saturated" if saturated else "geometric"
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
            f"mode={mode}, "
            f"ETA={execs:.0f}"
        ),
    )
