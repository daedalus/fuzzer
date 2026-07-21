"""Online running statistics — Welford/Pébay algorithms.

Single-pass, O(1) per update computation of mean, variance, skewness,
and excess kurtosis.  Two variants:

  RunningMoments          — unbounded, all history counts equally
  RunningMoments(window=N) — bounded deque, recent observations dominate

The windowed variant is better for series where recent behavior should
drive decisions (discovery rate, exec time).  The unbounded variant is
better for series where all observations matter equally (corpus seed
sizes across a session).

Reference
---------
Pébay, Terriberry, Kolla, Bennett (2016).
"Numerically stable, scalable formulas for parallel and online
computation of higher-order central moments with arbitrary weights."
"""

from __future__ import annotations

import collections
import math


class RunningMoments:
    """Welford/Pébay online moments — mean, variance, skewness, kurtosis.

    Args:
        window: If set, only the last *window* observations are kept
            (sliding-window mode).  ``None`` (default) keeps all history.
    """

    def __init__(self, window: int | None = None) -> None:
        self._window = window
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._m3: float = 0.0
        self._m4: float = 0.0
        if window is not None:
            self._buf: collections.deque[float] = collections.deque(maxlen=window)
        else:
            self._buf = collections.deque()  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, x: float) -> None:
        """Incorporate a new observation."""
        if self._window is not None and self._n >= self._window:
            self._buf.popleft()
            self._buf.append(x)
            self._recompute()
        else:
            self._n += 1
            self._buf.append(x)
            self._inc_update(x)

    def _inc_update(self, x: float) -> None:
        """Pébay's incremental recurrence for M2, M3, M4."""
        n = self._n
        delta = x - self._mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * (n - 1)

        self._mean += delta_n
        self._m4 += (
            term1 * delta_n2 * (n * n - 3 * n + 3)
            + 6 * delta_n2 * self._m2
            - 4 * delta_n * self._m3
        )
        self._m3 += term1 * delta_n * (n - 2) - 3 * delta_n * self._m2
        self._m2 += term1

    def _recompute(self) -> None:
        """Recompute all moments from scratch over the window buffer."""
        n = len(self._buf)
        self._n = n
        if n == 0:
            self._mean = self._m2 = self._m3 = self._m4 = 0.0
            return
        if n == 1:
            self._mean = self._buf[0]
            self._m2 = self._m3 = self._m4 = 0.0
            return
        mean = sum(self._buf) / n
        m2 = m3 = m4 = 0.0
        for v in self._buf:
            d = v - mean
            d2 = d * d
            m2 += d2
            m3 += d2 * d
            m4 += d2 * d2
        self._mean = mean
        self._m2 = m2
        self._m3 = m3
        self._m4 = m4

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mean(self) -> float:
        """Arithmetic mean of observations."""
        return self._mean if self._n >= 1 else 0.0

    @property
    def variance(self) -> float:
        """Sample variance (Bessel-corrected, /n-1). 0.0 if n < 2."""
        if self._n < 2:
            return 0.0
        return self._m2 / (self._n - 1)

    @property
    def stddev(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance) if self._n >= 2 else 0.0

    @property
    def skewness(self) -> float:
        """Sample skewness (adjusted Fisher-Pearson). 0.0 if n < 3."""
        if self._n < 3:
            return 0.0
        var = self._m2 / self._n
        if var <= 0.0:
            return 0.0
        stddev = math.sqrt(var)
        return (self._m3 / self._n) / (stddev**3)

    @property
    def kurtosis(self) -> float:
        """Excess kurtosis (Fisher). 0.0 if n < 4."""
        if self._n < 4:
            return 0.0
        var = self._m2 / self._n
        if var <= 0.0:
            return 0.0
        return (self._m4 / self._n) / (var * var) - 3.0

    @property
    def count(self) -> int:
        """Number of observations incorporated."""
        return self._n

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def z_score(self, x: float) -> float:
        """Z-score of *x* relative to the running distribution.

        Returns 0.0 if fewer than 2 observations (stddev is 0).
        """
        sd = self.stddev
        if sd <= 0.0:
            return 0.0
        return (x - self._mean) / sd

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self) -> dict:
        """Serialize for persistence."""
        return {
            "n": self._n,
            "mean": self._mean,
            "m2": self._m2,
            "m3": self._m3,
            "m4": self._m4,
            "window": self._window,
            "buf": list(self._buf),
        }

    def load(self, data: dict) -> None:
        """Restore from persistence."""
        self._n = data.get("n", 0)
        self._mean = data.get("mean", 0.0)
        self._m2 = data.get("m2", 0.0)
        self._m3 = data.get("m3", 0.0)
        self._m4 = data.get("m4", 0.0)
        self._window = data.get("window")
        if self._window is not None:
            self._buf = collections.deque(data.get("buf", []), maxlen=self._window)
        else:
            self._buf = collections.deque(data.get("buf", []))
