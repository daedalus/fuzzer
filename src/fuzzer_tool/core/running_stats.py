"""Online running statistics — Welford/Pébay algorithms.

Single-pass, O(1) per update computation of mean, variance, skewness,
and excess kurtosis.  Two variants:

  RunningMoments          — unbounded, all history counts equally
  RunningMoments(window=N) — bounded window, O(1) per update

The windowed variant runs a two-stack queue of Pébay aggregates
(n, mean, M2, M3, M4): amortised O(1) per update, and eviction never
subtracts, so no cancellation however far the stream drifts.

    oldest ──────────────────────────────▶ newest
    [ front: suffix aggregates ][ back: one running aggregate ]
      pop() evicts the oldest     push() folds in the newest
    window = merge(front[-1], back); empty front → refill from buffer

See https://orlp.net/blog/two-stack-sliding-window-aggregation/.

Reference
---------
Pébay, Terriberry, Kolla, Bennett (2016).
"Numerically stable, scalable formulas for parallel and online
computation of higher-order central moments with arbitrary weights."
"""

from __future__ import annotations

import collections
import math

#: Aggregate tuple layout: (n, mean, M2, M3, M4).
_Agg = tuple[int, float, float, float, float]
_EMPTY: _Agg = (0, 0.0, 0.0, 0.0, 0.0)


def _merge(a: _Agg, b: _Agg) -> _Agg:
    """Combine two disjoint aggregates (Pébay pairwise formulas, associative)."""
    na, ma, m2a, m3a, m4a = a
    nb, mb, m2b, m3b, m4b = b
    n = na + nb
    d = mb - ma
    d_n = d / n
    d2 = d * d_n
    nab = na * nb
    m2 = m2a + m2b + d2 * nab
    m3 = m3a + m3b + d2 * d_n * nab * (na - nb) + 3 * d_n * (na * m2b - nb * m2a)
    m4 = (
        m4a
        + m4b
        + d2 * d_n * d_n * nab * (na * na - nab + nb * nb)
        + 6 * d_n * d_n * (na * na * m2b + nb * nb * m2a)
        + 4 * d_n * (na * m3b - nb * m3a)
    )
    return (n, ma + d_n * nb, m2, m3, m4)


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
        # Cached stddev; invalidated on every update()/load(). Read-heavy
        # usage (e.g. GP-UCB select_op reading stddev per arm per select)
        # recomputed sqrt per access otherwise.
        self._stddev_cache: float | None = None
        # Window moments await a merge before the next read:
        # _stale → mean/M2 (_sync_lo), _stale_hi → M3/M4 (_sync).
        self._stale: bool = False
        self._stale_hi: bool = False
        if window is not None:
            self._buf: collections.deque[float] = collections.deque(maxlen=window)
            # Two-stack queue; an empty front refills from _buf on next slide.
            self._front: list[_Agg] = []
            self._back: _Agg = _EMPTY
        else:
            self._buf = collections.deque()  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, x: float) -> None:
        """Incorporate a new observation."""
        self._stddev_cache = None
        if self._window is not None and self._n >= self._window:
            self._slide_update(x)
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

    def _slide_update(self, x: float) -> None:
        """Evict the oldest, fold *x* into back: amortised O(1).

        Inlines ``_inc_update``'s step on a tuple: this runs once per exec.
        Moments merge lazily on read (``_sync_lo`` / ``_sync``).
        """
        front = self._front
        if not front:
            self._refill()
        front.pop()
        self._buf.append(x)  # maxlen evicts the oldest

        n0, mean, m2, m3, m4 = self._back
        n = n0 + 1
        delta = x - mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n0
        m4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3
        m3 += term1 * delta_n * (n - 2) - 3 * delta_n * m2
        self._back = (n, mean + delta_n, m2 + term1, m3, m4)
        self._stale = self._stale_hi = True

    def _sync_lo(self) -> None:
        """Merge mean and M2 only: what mean/variance/stddev read."""
        self._stale = False
        front = self._front
        nb, mb, m2b, _, _ = self._back
        if not front:
            self._mean, self._m2 = mb, m2b
            return

        na, ma, m2a, _, _ = front[-1]
        d = mb - ma
        d_n = d / (na + nb)
        self._mean = ma + d_n * nb
        self._m2 = m2a + m2b + (d * d_n) * (na * nb)  # same rounding as _merge

    def _sync(self) -> None:
        """Window moments = merge(front[-1], back), all four."""
        self._stale = self._stale_hi = False
        front = self._front
        agg = _merge(front[-1], self._back) if front else self._back
        _, self._mean, self._m2, self._m3, self._m4 = agg

    def _refill(self) -> None:
        """Move the back segment into front as suffix aggregates.

        Called only when front is empty, so the back segment is all of
        ``_buf``, e.g. [a, b, c] → front [agg(c), agg(b, c), agg(a, b, c)];
        front[-1] covers the whole window, pop() drops a first.
        """
        front = self._front
        append = front.append
        n0, mean, m2, m3, m4 = _EMPTY

        # _inc_update's step, inlined: a call per element costs ~2x.
        for x in reversed(self._buf):
            n = n0 + 1
            delta = x - mean
            delta_n = delta / n
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * n0
            m4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3
            m3 += term1 * delta_n * (n - 2) - 3 * delta_n * m2
            m2 += term1
            mean += delta_n
            n0 = n
            append((n, mean, m2, m3, m4))
        self._back = _EMPTY

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mean(self) -> float:
        """Arithmetic mean of observations."""
        if self._stale:
            self._sync_lo()
        return self._mean if self._n >= 1 else 0.0

    @property
    def variance(self) -> float:
        """Sample variance (Bessel-corrected, /n-1). 0.0 if n < 2.

        Clamped at zero: a guard for ``stddev``'s sqrt. The former
        power-sum window went a few ULP negative on zero-variance streams
        (-2.2e-19 at window=2); Pébay merges only add non-negative terms.
        """
        if self._n < 2:
            return 0.0
        if self._stale:
            self._sync_lo()
        return max(self._m2 / (self._n - 1), 0.0)

    @property
    def stddev(self) -> float:
        """Sample standard deviation."""
        if self._stddev_cache is None:
            self._stddev_cache = math.sqrt(self.variance) if self._n >= 2 else 0.0
        return self._stddev_cache

    @property
    def skewness(self) -> float:
        """Sample skewness (adjusted Fisher-Pearson). 0.0 if n < 3."""
        if self._n < 3:
            return 0.0
        if self._stale_hi:
            self._sync()
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
        if self._stale_hi:
            self._sync()
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
        return (x - self.mean) / sd

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self) -> dict:
        """Serialize for persistence."""
        if self._stale_hi:
            self._sync()
        data = {
            "n": self._n,
            "mean": self._mean,
            "m2": self._m2,
            "m3": self._m3,
            "m4": self._m4,
            "window": self._window,
            "buf": list(self._buf),
        }
        if self._window is not None:
            # Stacks fix merge order, so a resumed run is bit-identical.
            data["front"] = list(self._front)
            data["back"] = self._back
        return data

    def load(self, data: dict) -> None:
        """Restore from persistence."""
        self._stddev_cache = None
        self._stale = self._stale_hi = False
        self._n = data.get("n", 0)
        self._mean = data.get("mean", 0.0)
        self._m2 = data.get("m2", 0.0)
        self._m3 = data.get("m3", 0.0)
        self._m4 = data.get("m4", 0.0)
        self._window = data.get("window")
        if self._window is not None:
            self._buf = collections.deque(data.get("buf", []), maxlen=self._window)
            # Legacy saves (s1..s4, no stacks) rebuild from buf on next slide.
            self._front = [tuple(a) for a in data.get("front", [])]
            self._back = tuple(data.get("back", _EMPTY))
        else:
            self._buf = collections.deque(data.get("buf", []))


# ------------------------------------------------------------------
# Sharpe / Kelly helpers (module-level, no class dependency)
# ------------------------------------------------------------------


def sharpe_ratio(mean: float, stddev: float) -> float:
    """Instantaneous Sharpe ratio from running mean and stddev.

    Annualization factors are not meaningful in fuzzing; returns the raw
    ratio.  Returns 0.0 when stddev is zero and mean is also zero, or
    when stddev is negative (invalid input); returns +inf when stddev
    is zero and mean is positive (a perfect risk-free return); returns
    -inf when stddev is zero and mean is negative.
    """
    if stddev < 0.0:
        return 0.0
    if stddev == 0.0:
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return 0.0
    return mean / stddev


def kelly_fraction(mean: float, variance: float) -> float:
    """Kelly criterion optimal allocation fraction from running moments.

    Computes f* = mean / variance, which is the fraction of remaining
    budget that maximises log-growth under i.i.d. returns.  Clamped to
    [0.0, 1.0]: negative values mean the arm is a net loser and should
    receive zero allocation.
    """
    if variance <= 0.0:
        return 0.0
    return max(0.0, min(mean / variance, 1.0))
