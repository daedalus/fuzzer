"""Overlapping Allan variance for noise-type identification.

This module implements the **true overlapping Allan deviation** on the
incremental edge-discovery rate, and retains the second-order structure
function (variogram) as a secondary diagnostic.

Allan variance is a first difference of *block averages* — equivalently a
second difference of the cumulative series.  For rate samples ``x`` with
cumulative sum ``S`` (``S[0]=0``, ``S[k]=sum(x[:k])``) and averaging factor
``m = τ``:

    σ²_y(τ) = 1 / (2 m² (N - 2m))  Σ_i (S[i+2m] - 2 S[i+m] + S[i])²
    adev(τ) = √σ²_y(τ)

This separates white noise (log-log slope ≈ −0.5) from flicker (≈ 0) and
random-walk / trending discovery rates (slope ≳ +0.4), which the structure
function cannot do.

Classification (tailored to edge-discovery-rate signals):

  - **Active** (healthy random exploration):  adev(2) > stall, slope < fatigue
  - **Fatiguing** (approaching saturation):   adev(2) > stall, slope ≥ fatigue
  - **Stalled** (effectively zero discovery): adev(2) ≤ stall

Fatigue threshold is calibrated against true Allan slopes: white ≈ −0.5,
flicker ≈ 0, random-walk ≈ +0.45, strong downtrend ≈ +0.8.  Threshold 0.15
separates stationary / white from RW and trends (handover audit 2026-09-13).

The structure-function deviation ``sdev`` is kept for diagnostics and for
callers that explicitly want the variogram.

This module also provides :class:`DispersionIndex` — a sliding-window
Index of Dispersion (Fano factor, D = σ²/μ).  D complements Allan variance
by resolving a key ambiguity: a buffer full of zeros and a buffer with rare
bursts both produce low Allan deviation, but D discriminates them.

Rather than compare D against fixed constants (which are only correctly
calibrated at one particular sample size), dispersion significance is
decided with the standard Poisson dispersion test: under the null
hypothesis that the signal is Poisson-distributed (D = 1, i.i.d.),
``T = (n-1) * D`` follows a chi-squared distribution with ``n-1`` degrees
of freedom. This gives a threshold that adapts to how many samples are
actually available, instead of a single magic number applied regardless
of window fill level. See :func:`chi2_sf` / :func:`chi2_cdf` and the
``is_overdispersed`` / ``is_underdispersed`` methods below.
"""

from __future__ import annotations

import collections
import math

from fuzzer_tool.core.chi_squared import chi_squared_pvalue
from fuzzer_tool.core.running_stats import RunningMoments

# True overlapping Allan deviation thresholds (edge-discovery rate signals).
# Calibrated against synthetic series (N=256, 50–80 replicates):
#   white/Poisson  slope ≈ −0.53, adev(2) high
#   flicker-ish    slope ≈ 0–0.2
#   random-walk    slope ≈ +0.45
#   strong down    slope ≈ +0.81
# Fatigue at 0.15 separates stationary/white from RW and trends.
_ADEV_ACTIVE_THRESHOLD = 0.1   # adev(2) above this → signal has meaningful variance
_ADEV_STALL_THRESHOLD = 0.01   # adev(2) below this → signal is effectively constant
_FATIGUE_SLOPE_THRESHOLD = 0.15  # Allan log-log slope above this → fatiguing

# Default significance level for the chi-squared dispersion test.
_DISPERSION_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Chi-squared distribution (pure Python — project deliberately has no scipy
# dependency; see core/running_stats.py and the scipy-removal fix in
# edge_tracker.py for the same rationale).
#
# The survival function is canonical in core/chi_squared.py;
# chi2_sf delegates there, keeping only the k<=0 ValueError contract.
# Verified against reference chi-squared critical values in
# tests/test_structure_function.py.
# ---------------------------------------------------------------------------


def chi2_sf(x: float, k: int) -> float:
    """Survival function (1 - CDF) of the chi-squared distribution.

    ``x``: test statistic. ``k``: degrees of freedom (must be > 0).
    Returns the probability of observing a value >= x under the
    chi-squared(k) distribution — i.e. the one-sided upper-tail p-value.

    Delegates to :func:`fuzzer_tool.core.chi_squared.chi_squared_pvalue`,
    the canonical implementation.
    """
    if k <= 0:
        raise ValueError("degrees of freedom must be positive")
    return chi_squared_pvalue(x, k)


def chi2_cdf(x: float, k: int) -> float:
    """CDF of the chi-squared distribution. See :func:`chi2_sf`."""
    return 1.0 - chi2_sf(x, k)


class StructureFunctionDetector:
    """Overlapping Allan-variance detector for fuzzing stall analysis.

    Maintains a fixed-size buffer of incremental edge counts and computes the
    true overlapping Allan deviation at power-of-two averaging times.
    Classification is tailored to edge-discovery-rate signals.

    The structure-function deviation (:meth:`sdev`) is retained as a secondary
    diagnostic; :meth:`noise_type` / :meth:`noise_slope` use :meth:`adev`.

    Args:
        max_buffer_pow: Buffer capacity = 2**max_buffer_pow. Default 8 → 256.
        min_samples: Minimum samples before noise_type() returns a result.
    """

    def __init__(self, max_buffer_pow: int = 8, min_samples: int = 8):
        self._maxlen = 2**max_buffer_pow
        self._min_samples = min_samples
        self._buf: collections.deque[float] = collections.deque(maxlen=self._maxlen)
        # Running cumulative sum for O(n) overlapping Allan (phase data).
        # _cum[0] = 0; _cum[k] = sum of first k rate samples.
        self._cum: collections.deque[float] = collections.deque([0.0], maxlen=self._maxlen + 1)
        # Dispersion index tracks the same sliding window (window=maxlen
        # keeps its count ≡ len(_buf) = min(total, maxlen)).
        self._disp = DispersionIndex(window=self._maxlen)

    # ── Public API ────────────────────────────────────────────────────

    def update(self, value: float) -> None:
        """Record a new observation (incremental edge count)."""
        dropped = self._buf[0] if len(self._buf) == self._maxlen else None
        self._buf.append(value)
        if dropped is None:
            self._cum.append(self._cum[-1] + value)
        else:
            # Window slid: rebuild cumulative sum over the current buffer
            # so _cum[0] == 0 and len(_cum) == len(_buf) + 1.
            self._cum = collections.deque([0.0], maxlen=self._maxlen + 1)
            running = 0.0
            for v in self._buf:
                running += v
                self._cum.append(running)
        self._disp.update(value)

    def adev(self, tau: int) -> float:
        """Overlapping Allan deviation at averaging time *tau*.

        For rate samples ``x`` with cumulative sum ``S``:

            σ²_y(τ) = 1/(2 τ² (N-2τ)) Σ_i (S[i+2τ] - 2 S[i+τ] + S[i])²
            adev(τ) = √σ²_y(τ)

        Returns NaN if fewer than ``2*tau+1`` samples are available.
        """
        n = len(self._buf)
        m = tau
        if m < 1 or n < 2 * m + 1:
            return float("nan")
        # Prefer the maintained cumulative buffer when it is in sync.
        if len(self._cum) == n + 1:
            S = self._cum
        else:
            S_list = [0.0]
            for v in self._buf:
                S_list.append(S_list[-1] + v)
            S = S_list
        sq_sum = 0.0
        count = n - 2 * m
        for i in range(count):
            diff = S[i + 2 * m] - 2.0 * S[i + m] + S[i]
            sq_sum += diff * diff
        return math.sqrt(sq_sum / (2.0 * m * m * count))

    def sdev(self, tau: int) -> float:
        """Structure-function deviation (variogram) at lag *tau*.

        S(τ) = sqrt( 0.5 * mean_i (x[i+τ] - x[i])² )

        Secondary diagnostic; classification uses :meth:`adev`.
        Returns NaN if fewer than τ+1 samples are available.
        """
        n = len(self._buf)
        if n < tau + 1 or tau < 1:
            return float("nan")
        data = list(self._buf)
        sq_sum = 0.0
        count = n - tau
        for i in range(count):
            diff = data[i + tau] - data[i]
            sq_sum += diff * diff
        return math.sqrt(0.5 * sq_sum / count)

    def noise_type(self) -> str:
        """Classify the fuzzing regime from the edge-discovery-rate signal.

        Uses the true overlapping Allan deviation (:meth:`adev`) and its
        log-log slope.  Returns one of: ``"active"``, ``"fatiguing"``,
        ``"stalled"``, ``"unknown"``.

        - ``active``: adev(2) > stall and slope < fatigue.
          White / stationary exploration (Allan slope ≲ 0).
        - ``fatiguing``: adev(2) > stall and slope ≥ fatigue.
          Random-walk or downward-trending discovery rate (slope ≳ +0.4).
        - ``stalled``: adev(2) ≤ stall.  Effectively constant signal.
        - ``unknown``: insufficient samples.
        """
        n = len(self._buf)
        if n < self._min_samples:
            return "unknown"

        dev2 = self.adev(2)
        if not math.isfinite(dev2):
            return "unknown"

        # Near-zero Allan deviation → stalled
        if dev2 <= _ADEV_STALL_THRESHOLD:
            return "stalled"

        # Log-log slope of adev(τ) over τ = 4..64 (weighted OLS).
        # For true Allan: weights ∝ (N - 2τ) / τ² reflect equivalent DOF.
        max_pow = min(int(math.log2(n // 2)), 6)
        if max_pow < 1:
            return "unknown"

        points: list[tuple[float, float, float]] = []  # (log_tau, log_dev, weight)
        for p in range(2, max_pow + 1):  # start from tau=4
            tau = 2**p
            dev = self.adev(tau)
            if math.isfinite(dev) and dev > 0:
                w = max(n - 2 * tau, 1) / float(tau * tau)
                points.append((math.log(tau), math.log(dev), w))

        if len(points) < 2:
            return "active" if dev2 > _ADEV_ACTIVE_THRESHOLD else "fatiguing"

        slope = self._weighted_slope(points)
        if slope is None:
            return "active" if dev2 > _ADEV_ACTIVE_THRESHOLD else "fatiguing"

        if slope >= _FATIGUE_SLOPE_THRESHOLD:
            return "fatiguing"
        return "active"

    def noise_slope(self) -> float | None:
        """Return the log-log Allan deviation slope, or None if unknown.

        Uses weighted OLS with weights ∝ (N − 2τ) / τ² so large-τ points,
        which have fewer overlapping pairs, do not dominate the fit.
        """
        n = len(self._buf)
        if n < self._min_samples:
            return None
        max_pow = min(int(math.log2(n // 2)), 6)
        if max_pow < 1:
            return None
        points: list[tuple[float, float, float]] = []
        for p in range(2, max_pow + 1):
            tau = 2**p
            dev = self.adev(tau)
            if math.isfinite(dev) and dev > 0:
                w = max(n - 2 * tau, 1) / float(tau * tau)
                points.append((math.log(tau), math.log(dev), w))
        if len(points) < 2:
            return None
        return self._weighted_slope(points)

    @staticmethod
    def _weighted_slope(
        points: list[tuple[float, float, float]],
    ) -> float | None:
        """Weighted least-squares slope of log-dev vs log-tau.

        points: list of (log_tau, log_dev, weight).
        """
        sw = sum(p[2] for p in points)
        if sw <= 0:
            return None
        sx = sum(p[0] * p[2] for p in points)
        sy = sum(p[1] * p[2] for p in points)
        sxx = sum(p[0] * p[0] * p[2] for p in points)
        sxy = sum(p[0] * p[1] * p[2] for p in points)
        denom = sw * sxx - sx * sx
        if denom == 0:
            return None
        return (sw * sxy - sx * sy) / denom

    @property
    def n_samples(self) -> int:
        """Number of samples currently in the buffer."""
        return len(self._buf)

    @property
    def buffer_full(self) -> bool:
        """Whether the buffer has reached its maximum capacity."""
        return len(self._buf) >= self._maxlen

    # ── Dispersion Index ─────────────────────────────────────────────

    def dispersion(self) -> float | None:
        """Index of Dispersion (Fano factor) of the current buffer.

        D = variance / mean

        This raw ratio is provided for diagnostics/logging. For a decision
        about whether D is *significantly* bursty or stalled (rather than
        just noisy at low sample counts), use :meth:`is_overdispersed` /
        :meth:`is_underdispersed`, which apply a chi-squared significance
        test instead of a fixed cutoff.

        Returns None if fewer than 2 observations or mean is effectively zero.
        """
        return self._disp.value

    def dispersion_pvalue(self) -> float | None:
        """One-sided upper-tail p-value of the current D under the Poisson
        dispersion test (chi-squared(n-1) survival function of
        ``(n-1) * D``).

        A small p-value means D is significantly *higher* than 1 (Poisson) —
        i.e. overdispersed/bursty. Use ``1 - dispersion_pvalue()`` reasoning
        via :meth:`is_underdispersed` for the opposite tail. Returns None if
        :meth:`dispersion` returns None.
        """
        return self._disp.dispersion_pvalue()

    def is_overdispersed(self, alpha: float = _DISPERSION_ALPHA) -> bool:
        """True if D is significantly greater than 1 (bursty) at level
        *alpha*, via the chi-squared dispersion test. False (not None) if
        there isn't enough data to tell, so this can be used directly in
        boolean stall-detection logic without an extra None-check.
        """
        return self._disp.is_overdispersed(alpha)

    def is_underdispersed(self, alpha: float = _DISPERSION_ALPHA) -> bool:
        """True if D is significantly less than 1 (near-constant / stalled)
        at level *alpha*, via the chi-squared dispersion test. False (not
        None) if there isn't enough data to tell.
        """
        return self._disp.is_underdispersed(alpha)

    def reset(self) -> None:
        """Clear all samples."""
        self._buf.clear()
        self._cum = collections.deque([0.0], maxlen=self._maxlen + 1)
        self._disp = DispersionIndex(window=self._maxlen)

    def save(self) -> dict:
        """Serialize state for persistence."""
        return {
            "max_buffer_pow": int(math.log2(self._maxlen)),
            "min_samples": self._min_samples,
            "samples": list(self._buf),
        }

    def load(self, data: dict) -> None:
        """Restore state from *save()* output."""
        self._maxlen = 2 ** data.get("max_buffer_pow", int(math.log2(self._maxlen)))
        self._min_samples = data.get("min_samples", self._min_samples)
        self._buf = collections.deque(data.get("samples", []), maxlen=self._maxlen)
        # Rebuild cumulative sum and dispersion index from the restored buffer.
        self._cum = collections.deque([0.0], maxlen=self._maxlen + 1)
        for v in self._buf:
            self._cum.append(self._cum[-1] + v)
        self._disp = DispersionIndex(window=self._maxlen)
        for v in self._buf:
            self._disp.update(v)


class DispersionIndex:
    """Sliding-window Index of Dispersion (Fano factor), D = σ²/μ.

    Uses :class:`RunningMoments` for O(1) per-update mean + variance and
    returns the ratio.  Tracks only the most recent *window* observations.

    D itself is not compared against fixed constants — a fixed cutoff is
    only well-calibrated at one particular sample count, and window fill
    level varies (especially early in a run). Use :meth:`is_overdispersed`
    / :meth:`is_underdispersed`, which apply the standard Poisson
    dispersion test (``(n-1)*D ~ chi-squared(n-1)`` under the null
    hypothesis of a Poisson/stationary process) so the effective threshold
    adapts to how many samples are actually in the window.

    Args:
        window: Max number of recent observations to retain.
    """

    def __init__(self, window: int = 200):
        self._moments = RunningMoments(window=window)

    def update(self, value: float) -> None:
        """Record a new observation."""
        self._moments.update(value)

    @property
    def value(self) -> float | None:
        """D = variance / mean, or None if mean ≈ 0 or insufficient data."""
        if self._moments.count < 2:
            return None
        mean = self._moments.mean
        if abs(mean) < 1e-12:
            return None
        return self._moments.variance / mean

    @property
    def count(self) -> int:
        """Number of observations incorporated."""
        return self._moments.count

    def dispersion_pvalue(self) -> float | None:
        """One-sided upper-tail p-value of the current D under the Poisson
        dispersion test. See :meth:`StructureFunctionDetector.dispersion_pvalue`
        for the underlying test. Returns None if :attr:`value` is None.
        """
        n = self._moments.count
        d = self.value
        if d is None or n < 2:
            return None
        t = (n - 1) * d
        return chi2_sf(t, n - 1)

    def is_overdispersed(self, alpha: float = _DISPERSION_ALPHA) -> bool:
        """True if D is significantly greater than 1 (bursty) at level
        *alpha*. False if there isn't enough data to tell."""
        p = self.dispersion_pvalue()
        return p is not None and p < alpha

    def is_underdispersed(self, alpha: float = _DISPERSION_ALPHA) -> bool:
        """True if D is significantly less than 1 (near-constant) at level
        *alpha*. False if there isn't enough data to tell."""
        p = self.dispersion_pvalue()
        return p is not None and (1.0 - p) < alpha

    def save(self) -> dict:
        """Serialize state for persistence."""
        return {"moments": self._moments.save()}

    def load(self, data: dict) -> None:
        """Restore state from *save()* output."""
        if "moments" in data:
            self._moments.load(data["moments"])
