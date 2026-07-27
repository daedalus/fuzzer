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
"""

from __future__ import annotations

import collections
import math

# Allan deviation thresholds (empirically derived from edge-discovery rate signals)
_ADEV_ACTIVE_THRESHOLD = 0.5   # adev(2) above this → signal has meaningful variance
_ADEV_STALL_THRESHOLD = 0.01   # adev(2) below this → signal is effectively constant
_FATIGUE_SLOPE_THRESHOLD = 0.1  # slope above this → variance grows with averaging


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
        self._buf = collections.deque(
            data.get("samples", []), maxlen=self._maxlen
        )
