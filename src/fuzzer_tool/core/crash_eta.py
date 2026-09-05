"""Pre-fuzzing crash ETA estimation.

Combines static analysis (TargetProfiler) with calibrated execution
statistics to estimate how many edges/executions until the first crash.
Also provides dynamic crash prediction via mutual information between
input bytes and crash outcomes.
"""

from __future__ import annotations

import bisect
import math
import random
import re
from dataclasses import dataclass

import numpy as _np

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

    #: Positions past this are not tracked even if ``max_positions`` is
    #: larger. A dense row is 256 uint32 = 1 KiB, so this bounds the
    #: tracker at 128 MiB in the pathological case. MI over byte positions
    #: that far out is not estimable anyway -- each one would need
    #: ``min_observations`` samples of its own.
    DENSE_POSITION_CAP = 65536

    def __init__(self, max_positions: int = 4096, min_observations: int = 20):
        self.max_positions = max_positions
        self.min_observations = min_observations
        # Per-position byte histograms, dense.
        #
        # These were dict[int, dict[int, int]] and record() walked every
        # byte of every executed input updating three nested defaultdicts.
        # That put the tracker at 14% of a campaign's runtime -- measured
        # ~300 ns/byte, so 1.4 ms for a 4 KiB input, more than executing
        # the target in process. A dense (positions, 256) uint32 pair lets
        # one execution be a single vectorised scatter-add.
        #
        # Rows are allocated lazily up to the largest input seen, so the
        # default 4096-position tracker costs 8 MiB only if inputs
        # actually reach that length.
        self._rows = 0
        self._byte_total_arr = _np.zeros((0, 256), dtype=_np.uint32)
        self._joint_crash_arr = _np.zeros((0, 256), dtype=_np.uint32)
        self._position_counts_arr = _np.zeros(0, dtype=_np.uint64)
        # Global: crash count, total count
        self.total_crashes: int = 0
        self.total_execs: int = 0
        # Cached MI values
        self._mi_cache: dict[int, float] = {}
        self._cache_valid: bool = False
        # Sampling caches, rebuilt by all_mi()
        self._cached_positions: list[int] = []
        self._cached_weights: list[float] = []
        self._cached_prefix: list[float] = []
        self._cached_total: float = 0.0

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

        # Track every execution's byte values, crash or not — MI needs the
        # non-crash outcomes as the contrasting class, otherwise
        # byte_total == joint_crash and every byte looks perfectly
        # crash-predictive.
        n = min(len(input_bytes), self.max_positions, self.DENSE_POSITION_CAP)
        if n:
            self._grow(n)
            values = _np.frombuffer(bytes(input_bytes), dtype=_np.uint8, count=n)
            rows = _np.arange(n)
            # Positions are distinct, so plain fancy-index += is safe here.
            self._byte_total_arr[rows, values] += 1
            self._position_counts_arr[:n] += 1
            if is_crash:
                self._joint_crash_arr[rows, values] += 1

        # The dense representation is already bounded, so there is nothing
        # to prune. The old _prune() kept only the 32 most frequent byte
        # values per position, which discarded evidence the MI estimate
        # then could not see; it also accounted for 17,724 of a 4000-exec
        # campaign's 36,140 sorted() calls.
        if self.total_execs % 50 == 0:
            self._cache_valid = False

    def _grow(self, n: int) -> None:
        """Make sure at least *n* position rows are allocated."""
        if n <= self._rows:
            return
        cap = min(self.max_positions, self.DENSE_POSITION_CAP)
        new_rows = min(cap, max(64, 1 << (n - 1).bit_length()))
        if new_rows < n:  # pragma: no cover - n is already clamped to cap
            new_rows = n
        bt = _np.zeros((new_rows, 256), dtype=_np.uint32)
        jc = _np.zeros((new_rows, 256), dtype=_np.uint32)
        pc = _np.zeros(new_rows, dtype=_np.uint64)
        if self._rows:
            bt[: self._rows] = self._byte_total_arr
            jc[: self._rows] = self._joint_crash_arr
            pc[: self._rows] = self._position_counts_arr
        self._byte_total_arr = bt
        self._joint_crash_arr = jc
        self._position_counts_arr = pc
        self._rows = new_rows

    # ------------------------------------------------------------------
    # Dict views over the dense store
    #
    # The histograms are read from tests, from save(), and from report
    # code as dict[int, dict[int, int]]. These properties materialise that
    # shape on demand, so the representation change stays inside the
    # class. They are not on any hot path -- record() and mi() go straight
    # to the arrays.
    # ------------------------------------------------------------------

    @property
    def position_counts(self) -> dict[int, int]:
        if not self._rows:
            return {}
        nz = _np.flatnonzero(self._position_counts_arr)
        return {int(p): int(self._position_counts_arr[p]) for p in nz}

    @property
    def byte_total(self) -> dict[int, dict[int, int]]:
        return self._as_dict(self._byte_total_arr)

    @property
    def joint_crash(self) -> dict[int, dict[int, int]]:
        return self._as_dict(self._joint_crash_arr)

    def _as_dict(self, arr) -> dict[int, dict[int, int]]:
        if not self._rows:
            return {}
        out: dict[int, dict[int, int]] = {}
        rows, cols = _np.nonzero(arr)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            out.setdefault(r, {})[c] = int(arr[r, c])
        return out

    def mi(self, position: int) -> float:
        """Compute I(X_pos; C) in bits.

        I(X; C) = sum_{x,c} P(x,c) * log2(P(x,c) / (P(x) * P(c)))

        Returns 0.0 if insufficient data or position not observed.
        """
        if position < 0 or position >= self._rows:
            return 0.0
        if int(self._position_counts_arr[position]) < self.min_observations:
            return 0.0
        if self.total_execs == 0 or self.total_crashes == 0:
            return 0.0

        n = self.total_execs
        p_crash = self.total_crashes / n
        p_no_crash = 1.0 - p_crash

        if p_crash == 0 or p_no_crash == 0:
            return 0.0

        # Probabilities are normalised by total execs (n), not by the
        # per-position count, so position_counts is not needed here.
        mi_val = 0.0

        total_row = self._byte_total_arr[position]
        crash_row = self._joint_crash_arr[position]
        for byte_val in _np.flatnonzero(total_row).tolist():
            bc = int(total_row[byte_val])
            p_x = bc / n
            crash_count = int(crash_row[byte_val])
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
        counts = self._position_counts_arr
        observed = (
            _np.flatnonzero(counts >= self.min_observations).tolist() if self._rows else []
        )
        self._mi_cache = {pos: self.mi(pos) for pos in observed}
        # Cache sorted positions and weights for weighted_position
        self._cached_positions = sorted(self._mi_cache.keys())
        self._cached_weights = [max(self._mi_cache[p], 0.01) for p in self._cached_positions]
        self._cached_total = sum(self._cached_weights)
        # Prefix sums, so a draw restricted to the positions below some
        # input length can be normalised by that prefix's mass rather than
        # by the total, and located by bisection instead of a scan.
        self._cached_prefix = []
        running = 0.0
        for w in self._cached_weights:
            running += w
            self._cached_prefix.append(running)
        self._cache_valid = True
        return self._mi_cache

    def top_positions(self, k: int = 10) -> list[tuple[int, float]]:
        """Return the k byte positions with highest MI (most crash-predictive)."""
        mi_vals = self.all_mi()
        if not mi_vals:
            return []
        sorted_mi = sorted(mi_vals.items(), key=lambda x: x[1], reverse=True)
        return sorted_mi[:k]

    def weighted_position(self, input_length: int) -> int | None:
        """Sample a byte position weighted by crash MI.

        Only positions below *input_length* are eligible, so the draw is
        normalised by that prefix's weight, not by the total over every
        tracked position. Scaling by the total was a defect: the scan
        could only ever accumulate the prefix, so whenever the prefix was
        the smaller share the draw fell off the end of the loop and
        returned the last eligible position. Measured with 200 tracked
        positions and input_length=50, 75% of draws landed on that one
        position, whose correct share was 2.1% -- the MI weighting was
        inert for any input shorter than the tracked range, which is the
        normal case since positions accumulate up to max_positions.
        """
        if not self._cache_valid:
            self.all_mi()
        if not self._cached_positions:
            return None
        # Binary search for cutoff index
        idx = bisect.bisect_left(self._cached_positions, input_length)
        if idx == 0:
            return None
        prefix = self._cached_prefix
        eligible_mass = prefix[idx - 1]
        if eligible_mass <= 0.0:
            return self._cached_positions[random.randrange(idx)]
        r = random.random() * eligible_mass
        # prefix is non-decreasing, so the first entry >= r is the pick.
        i = bisect.bisect_left(prefix, r, 0, idx)
        if i >= idx:
            i = idx - 1
        return self._cached_positions[i]

    def top_values(self, position: int, k: int = 5) -> list[int]:
        """Return the k byte values at position with highest crash count."""
        if position < 0 or position >= self._rows:
            return []
        row = self._joint_crash_arr[position]
        nz = _np.flatnonzero(row)
        if nz.size == 0:
            return []
        # Descending by count, ties broken by ascending byte value, which
        # is what sorted(..., reverse=True) on an insertion-ordered dict
        # gave for the counts that mattered.
        order = sorted(nz.tolist(), key=lambda bv: (-int(row[bv]), bv))
        return order[:k]

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
        """Deserialize tracker state.

        The on-disk shape is unchanged -- nested dicts keyed by stringified
        position and byte value -- so state written before the dense
        representation still restores.
        """
        self.max_positions = data.get("max_positions", self.max_positions)
        self.min_observations = data.get("min_observations", self.min_observations)
        self.total_crashes = data.get("total_crashes", 0)
        self.total_execs = data.get("total_execs", 0)

        counts = {int(k): int(v) for k, v in data.get("position_counts", {}).items()}
        joint = {
            int(p): {int(bv): int(c) for bv, c in m.items()}
            for p, m in data.get("joint_crash", {}).items()
        }
        totals = {
            int(p): {int(bv): int(c) for bv, c in m.items()}
            for p, m in data.get("byte_total", {}).items()
        }

        cap = min(self.max_positions, self.DENSE_POSITION_CAP)
        highest = -1
        for source in (counts, joint, totals):
            for pos in source:
                if 0 <= pos < cap and pos > highest:
                    highest = pos

        self._rows = 0
        self._byte_total_arr = _np.zeros((0, 256), dtype=_np.uint32)
        self._joint_crash_arr = _np.zeros((0, 256), dtype=_np.uint32)
        self._position_counts_arr = _np.zeros(0, dtype=_np.uint64)
        if highest >= 0:
            self._grow(highest + 1)
            for pos, c in counts.items():
                if 0 <= pos < self._rows:
                    self._position_counts_arr[pos] = c
            for pos, m in totals.items():
                if 0 <= pos < self._rows:
                    for bv, c in m.items():
                        if 0 <= bv < 256:
                            self._byte_total_arr[pos, bv] = c
            for pos, m in joint.items():
                if 0 <= pos < self._rows:
                    for bv, c in m.items():
                        if 0 <= bv < 256:
                            self._joint_crash_arr[pos, bv] = c
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
