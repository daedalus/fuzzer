"""Overlapping Allan variance for noise-type identification.

Allan variance measures how the variance of a time series changes with
averaging time τ. For stall detection, the first-difference Allan deviation
on the incremental edge-discovery rate reveals the fuzzing regime:

  σ²(τ) = ⟨(x[i+τ] - x[i])²⟩ / 2

Key insight from empirical testing of edge-discovery-rate signals:

  - **Active** (healthy random exploration):  adev(2) > 0.5, slope ≈ 0.0
  - **Fatiguing** (approaching saturation):   adev(2) > 0.01, slope > 0.1
  - **Stalled** (effectively zero discovery): adev(2) ≤ 0.01

The slope measures how the variance changes with averaging time:
  - slope ≈ 0: stationary variance (normal or stalled)
  - slope > 0.1: variance grows with τ (rate decreasing, pre-stall)

This module also provides :class:`DispersionIndex` — a sliding-window
Index of Dispersion (Fano factor, D = σ²/μ).  D complements the Allan
variance by resolving a key ambiguity the latter cannot: a buffer full
of zeros and a buffer with rare bursts both produce low Allan deviation,
but D discriminates them.

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

from fuzzer_tool.core.running_stats import RunningMoments

# Allan deviation thresholds (empirically derived from edge-discovery rate signals)
_ADEV_ACTIVE_THRESHOLD = 0.5  # adev(2) above this → signal has meaningful variance
_ADEV_STALL_THRESHOLD = 0.01  # adev(2) below this → signal is effectively constant
_FATIGUE_SLOPE_THRESHOLD = 0.1  # slope above this → variance grows with averaging

# Default significance level for the chi-squared dispersion test.
_DISPERSION_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Chi-squared distribution (pure Python — project deliberately has no scipy
# dependency; see core/running_stats.py and the scipy-removal fix in
# edge_tracker.py for the same rationale).
#
# Implementation follows the standard Numerical-Recipes-style split:
# series expansion for x < a+1, continued fraction for x >= a+1, both
# built on the regularized incomplete gamma function. Verified against
# reference chi-squared critical values in tests/test_allan_variance.py.
# ---------------------------------------------------------------------------


def _gammainc_lower_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) via series expansion.

    Valid (fast-converging) for x < a + 1.
    """
    if x <= 0.0:
        return 0.0
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(500):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gammainc_upper_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) via continued fraction.

    Valid (fast-converging) for x >= a + 1. Uses the modified Lentz
    algorithm for numerical stability.
    """
    gln = math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x: float, k: int) -> float:
    """Survival function (1 - CDF) of the chi-squared distribution.

    ``x``: test statistic. ``k``: degrees of freedom (must be > 0).
    Returns the probability of observing a value >= x under the
    chi-squared(k) distribution — i.e. the one-sided upper-tail p-value.
    """
    if k <= 0:
        raise ValueError("degrees of freedom must be positive")
    if x <= 0.0:
        return 1.0
    a = k / 2.0
    xh = x / 2.0
    if xh < a + 1.0:
        return 1.0 - _gammainc_lower_series(a, xh)
    return _gammainc_upper_cf(a, xh)


def chi2_cdf(x: float, k: int) -> float:
    """CDF of the chi-squared distribution. See :func:`chi2_sf`."""
    return 1.0 - chi2_sf(x, k)


class AllanVarianceDetector:
    """Overlapping Allan deviation detector for fuzzing stall analysis.

    Maintains a fixed-size buffer of incremental edge counts and computes the
    overlapping Allan deviation at power-of-two averaging times. Classification
    is tailored to edge-discovery-rate signals, not generic noise theory.

    Args:
        max_buffer_pow: Buffer capacity = 2**max_buffer_pow. Default 8 → 256.
        min_samples: Minimum samples before noise_type() returns a result.
    """

    def __init__(self, max_buffer_pow: int = 8, min_samples: int = 8):
        self._maxlen = 2**max_buffer_pow
        self._min_samples = min_samples
        self._buf: collections.deque[float] = collections.deque(maxlen=self._maxlen)

    # ── Public API ────────────────────────────────────────────────────

    def update(self, value: float) -> None:
        """Record a new observation (incremental edge count)."""
        self._buf.append(value)

    def adev(self, tau: int) -> float:
        """Overlapping Allan deviation at averaging time *tau*.

        Returns NaN if fewer than 2*tau+1 samples are available.
        """
        n = len(self._buf)
        if n < 2 * tau + 1:
            return float("nan")
        data = list(self._buf)
        sq_sum = 0.0
        count = n - 2 * tau
        for i in range(count):
            diff = data[i + tau] - data[i]
            sq_sum += diff * diff
        return math.sqrt(0.5 * sq_sum / count)

    def noise_type(self) -> str:
        """Classify the fuzzing regime from the edge-discovery-rate signal.

        Returns one of: ``"active"``, ``"fatiguing"``, ``"stalled"``,
        ``"unknown"``.

        - ``active``: adev(2) > threshold and slope < fatigue threshold.
          Normal random exploration — variance is stationary.
        - ``fatiguing``: adev(2) > stall threshold and slope >= fatigue
          threshold. Discovery rate is trending downward — approaching stall.
        - ``stalled``: adev(2) <= stall threshold. The signal is effectively
          constant — no new edges are being discovered.
        - ``unknown``: insufficient samples for a classification.
        """
        n = len(self._buf)
        if n < self._min_samples:
            return "unknown"

        dev2 = self.adev(2)
        if not math.isfinite(dev2):
            return "unknown"

        # Near-zero variance → stalled
        if dev2 <= _ADEV_STALL_THRESHOLD:
            return "stalled"

        # Compute log-log slope from larger tau values
        max_pow = min(int(math.log2(n // 2)), 6)
        if max_pow < 1:
            return "unknown"

        points: list[tuple[float, float]] = []
        for p in range(2, max_pow + 1):  # start from tau=4 to avoid tau-2 noise
            tau = 2**p
            dev = self.adev(tau)
            if math.isfinite(dev) and dev > 0:
                points.append((math.log(tau), math.log(dev)))

        if len(points) < 2:
            return "active" if dev2 > _ADEV_ACTIVE_THRESHOLD else "fatiguing"

        # Least-squares linear fit: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        n_pts = len(points)
        sx = sum(p[0] for p in points)
        sy = sum(p[1] for p in points)
        sxx = sum(p[0] * p[0] for p in points)
        sxy = sum(p[0] * p[1] for p in points)
        denom = n_pts * sxx - sx * sx
        if denom == 0:
            return "active" if dev2 > _ADEV_ACTIVE_THRESHOLD else "fatiguing"
        slope = (n_pts * sxy - sx * sy) / denom

        if slope >= _FATIGUE_SLOPE_THRESHOLD:
            return "fatiguing"
        return "active"

    def noise_slope(self) -> float | None:
        """Return the log-log Allan deviation slope, or None if unknown."""
        n = len(self._buf)
        if n < self._min_samples:
            return None
        max_pow = min(int(math.log2(n // 2)), 6)
        if max_pow < 1:
            return None
        points: list[tuple[float, float]] = []
        for p in range(2, max_pow + 1):
            tau = 2**p
            dev = self.adev(tau)
            if math.isfinite(dev) and dev > 0:
                points.append((math.log(tau), math.log(dev)))
        if len(points) < 2:
            return None
        n_pts = len(points)
        sx = sum(p[0] for p in points)
        sy = sum(p[1] for p in points)
        sxx = sum(p[0] * p[0] for p in points)
        sxy = sum(p[0] * p[1] for p in points)
        denom = n_pts * sxx - sx * sx
        if denom == 0:
            return None
        return (n_pts * sxy - sx * sy) / denom

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
        n = len(self._buf)
        if n < 2:
            return None
        data = list(self._buf)
        mean = sum(data) / n
        if abs(mean) < 1e-12:
            return None
        var = sum((x - mean) ** 2 for x in data) / (n - 1)  # sample variance
        return var / mean

    def dispersion_pvalue(self) -> float | None:
        """One-sided upper-tail p-value of the current D under the Poisson
        dispersion test (chi-squared(n-1) survival function of
        ``(n-1) * D``).

        A small p-value means D is significantly *higher* than 1 (Poisson) —
        i.e. overdispersed/bursty. Use ``1 - dispersion_pvalue()`` reasoning
        via :meth:`is_underdispersed` for the opposite tail. Returns None if
        :meth:`dispersion` returns None.
        """
        n = len(self._buf)
        d = self.dispersion()
        if d is None or n < 2:
            return None
        t = (n - 1) * d
        return chi2_sf(t, n - 1)

    def is_overdispersed(self, alpha: float = _DISPERSION_ALPHA) -> bool:
        """True if D is significantly greater than 1 (bursty) at level
        *alpha*, via the chi-squared dispersion test. False (not None) if
        there isn't enough data to tell, so this can be used directly in
        boolean stall-detection logic without an extra None-check.
        """
        p = self.dispersion_pvalue()
        return p is not None and p < alpha

    def is_underdispersed(self, alpha: float = _DISPERSION_ALPHA) -> bool:
        """True if D is significantly less than 1 (near-constant / stalled)
        at level *alpha*, via the chi-squared dispersion test. False (not
        None) if there isn't enough data to tell.
        """
        p = self.dispersion_pvalue()
        return p is not None and (1.0 - p) < alpha

    def reset(self) -> None:
        """Clear all samples."""
        self._buf.clear()

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
        dispersion test. See :meth:`AllanVarianceDetector.dispersion_pvalue`
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
