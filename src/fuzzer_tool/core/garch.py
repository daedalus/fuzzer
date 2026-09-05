"""Online GARCH(1,1) conditional variance of the edge-discovery series.

    sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}

with ``omega > 0``, ``alpha, beta >= 0`` and ``alpha + beta < 1``.

Where this sits
---------------
``core/critical_slowing.py`` reacts to variance that is *already* rising.
``core/allan_variance.py`` classifies the noise type at several averaging
times.  Neither maintains an autoregressive model of the conditional
variance process itself, so neither can say what the variance will be on the
next tick.  This module supplies that one-step forecast plus a clustering
verdict, and nothing else: strategy stays in the callers, exactly as
``CoverageRegimeDetector`` keeps it out of the phase classifier.

What it is fed, and why it matters
----------------------------------
The input is the **non-overlapping per-tick edge delta** — the same value
``services/fuzzer.py`` already hands to ``AllanVarianceDetector.update()``.

It is deliberately *not* ``services/stats_reporter.discovery_rate()``.  That
function averages over a sliding window of the last five snapshots, so
consecutive samples share four of their five snapshots; it is an MA(4)
smoother.  Pushing a constant-rate Poisson discovery process (no clustering
whatsoever) through it yields a squared-residual autocorrelation of
+0.55 / +0.23 / +0.05 at lags 1-3 and a Ljung-Box(10) statistic of 1419
against a 5% critical value of 18.3, while the same process measured without
the window gives +0.03 and a statistic of 15.3 and correctly fails to
reject.  A GARCH(1,1) fitted to the windowed version of that pure noise
returns alpha = 0.54.  In other words the window alone manufactures the very
effect this model exists to detect.

``critical_slowing.py`` already learned this in the other direction: it keeps
``_raw_history`` apart from ``_history`` precisely because Kalman smoothing
inflates lag-1 autocorrelation.  The five-snapshot window is the smoother
nobody had applied that lesson to.

The artifact is also self-identifying, which is what :func:`arch_effect`
exploits: an MA(k) smoother can only correlate squares out to lag ``k``, so
window-induced structure dies at :data:`OVERLAP_ARTIFACT_LAGS`, whereas real
GARCH persistence decays geometrically well past it.

Cost
----
:meth:`OnlineGarch11.update` is O(1).  Parameter fitting is a bounded
coarse-to-fine grid search over the stationary simplex, run at most every
``refit_interval`` observations over a bounded ring buffer — observations
arrive one per stats tick (~10 s of work), so the refit is invisible.
"""

from __future__ import annotations

import collections
import math

from fuzzer_tool.core.chi_squared import chi_squared_pvalue

# Footprint of the discovery_rate() sliding window (5 snapshots -> MA(4)).
# Squared-residual autocorrelation at or below this lag can be produced by
# the estimator alone; only structure beyond it is evidence of ARCH.
OVERLAP_ARTIFACT_LAGS = 4

# Stationarity: alpha + beta must stay strictly below one.  The margin keeps
# the unconditional variance omega / (1 - alpha - beta) finite.
_MAX_PERSISTENCE = 0.995

# Ljung-Box lag count for the clustering verdict, and its significance level.
_LJUNG_BOX_LAGS = 10
_CLUSTER_ALPHA = 0.05

# Persistence below this is too weak to be worth acting on even if the
# Ljung-Box test rejects; it keeps a marginally significant fit from
# flipping the clustering flag on a long, nearly homoskedastic campaign.
_CLUSTER_MIN_PERSISTENCE = 0.30

# Grid search: levels of refinement and steps per axis per level.
_FIT_LEVELS = 3
_FIT_STEPS = 8

_DEFAULT_MIN_OBS = 64
_DEFAULT_REFIT_INTERVAL = 128
_DEFAULT_BUFFER = 512

_VARIANCE_FLOOR = 1e-12


def squared_acf(values, lags: int = _LJUNG_BOX_LAGS) -> list[float]:
    """Autocorrelation of the squared, mean-removed series.

    Returns ``[]`` when the series is too short for the requested lags or is
    exactly constant (zero variance in the squares, so no correlation is
    defined).  Index ``i`` holds lag ``i + 1``.
    """
    n = len(values)
    if n < lags + 2:
        return []

    mean = sum(values) / n
    sq = [(v - mean) ** 2 for v in values]
    sq_mean = sum(sq) / n
    dev = [s - sq_mean for s in sq]
    denom = sum(d * d for d in dev)
    if denom <= _VARIANCE_FLOOR:
        return []

    return [sum(dev[i] * dev[i + k] for i in range(n - k)) / denom for k in range(1, lags + 1)]


def ljung_box(acf: list[float], n: int) -> tuple[float, float]:
    """Ljung-Box portmanteau statistic and its p-value for *acf* on *n* points.

    ``(0.0, 1.0)`` — inconclusive — when there is nothing to test.
    """
    h = len(acf)
    if h == 0 or n <= h + 1:
        return 0.0, 1.0

    stat = n * (n + 2) * sum(r * r / (n - k - 1) for k, r in enumerate(acf))
    return stat, chi_squared_pvalue(stat, h)


class OnlineGarch11:
    """Recursive GARCH(1,1) over a scalar observation stream.

    Args:
        omega: Constant variance term.  ``None`` defers to the first fit,
            using the sample variance until then.
        alpha: Weight on the previous squared residual (ARCH term).
        beta: Weight on the previous conditional variance (GARCH term).
        min_obs: Observations required before forecasts and verdicts are
            offered.
        refit_interval: Observations between parameter refits.  ``0``
            disables fitting and pins the parameters given here.
        buffer: Ring-buffer capacity used for fitting and diagnostics.
    """

    def __init__(
        self,
        omega: float | None = None,
        alpha: float = 0.10,
        beta: float = 0.80,
        min_obs: int = _DEFAULT_MIN_OBS,
        refit_interval: int = _DEFAULT_REFIT_INTERVAL,
        buffer: int = _DEFAULT_BUFFER,
    ) -> None:
        _validate_params(omega, alpha, beta)

        self._omega = omega
        self._alpha = alpha
        self._beta = beta
        self._min_obs = max(2, min_obs)
        self._refit_interval = max(0, refit_interval)
        self._buffer_size = max(self._min_obs, buffer)

        self._buf: collections.deque[float] = collections.deque(maxlen=self._buffer_size)
        self._count = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._sigma2 = 0.0
        self._eps = 0.0
        self._last_fit_at = 0

    # ── observation ───────────────────────────────────────────────────

    def update(self, value: float) -> None:
        """Record one observation and advance the variance recursion."""
        v = float(value)
        self._count += 1
        self._sum += v
        self._sum_sq += v * v
        self._buf.append(v)

        # Advance sigma2 with the PREVIOUS residual before computing the
        # current one -- sigma2_t is known at t-1, which is what makes the
        # one-step forecast well defined.
        self._sigma2 = self._omega_eff() + self._alpha * self._eps**2 + self._beta * self._sigma2
        self._eps = v - self.mean

        self._maybe_refit()

    def reset(self) -> None:
        """Clear all state, keeping the configured parameters."""
        self._buf.clear()
        self._count = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._sigma2 = 0.0
        self._eps = 0.0
        self._last_fit_at = 0

    # ── readouts ──────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    @property
    def residual(self) -> float:
        """Most recent residual eps_t = x_t - mean."""
        return self._eps

    @property
    def variance(self) -> float:
        """Current conditional variance sigma2_t."""
        return self._sigma2

    @property
    def initial_variance(self) -> float:
        """Seed value of the recursion (zero: the first update supplies omega)."""
        return 0.0

    @property
    def omega(self) -> float:
        return self._omega_eff()

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def persistence(self) -> float:
        """alpha + beta: how long a variance shock survives."""
        return self._alpha + self._beta

    def forecast(self) -> float | None:
        """One-step-ahead conditional variance, or ``None`` before *min_obs*."""
        if self._count < self._min_obs:
            return None
        return self._omega_eff() + self._alpha * self._eps**2 + self._beta * self._sigma2

    def standardised_residual(self) -> float:
        """eps_t / sigma_t — unit variance under a correct fit."""
        if self._sigma2 <= _VARIANCE_FLOOR:
            return 0.0
        return self._eps / math.sqrt(self._sigma2)

    # ── clustering verdict ────────────────────────────────────────────

    def squared_residual_acf(self, lags: int = _LJUNG_BOX_LAGS) -> list[float]:
        """Squared-residual autocorrelation over the current buffer."""
        return squared_acf(list(self._buf), lags=lags)

    def ljung_box(self, lags: int = _LJUNG_BOX_LAGS) -> tuple[float, float]:
        """Ljung-Box statistic and p-value on the squared residuals."""
        if self._count < self._min_obs:
            return 0.0, 1.0
        return ljung_box(self.squared_residual_acf(lags), len(self._buf))

    def arch_effect(self) -> bool:
        """True when volatility clustering survives the estimator artifact.

        Requires the Ljung-Box test to reject *and* the autocorrelation to
        still be positive past :data:`OVERLAP_ARTIFACT_LAGS`, which a sliding
        window cannot produce on its own.
        """
        acf = self.squared_residual_acf()
        if len(acf) <= OVERLAP_ARTIFACT_LAGS:
            return False

        _stat, p = ljung_box(acf, len(self._buf))
        if p >= _CLUSTER_ALPHA:
            return False

        beyond = acf[OVERLAP_ARTIFACT_LAGS:]
        threshold = 2.0 / math.sqrt(len(self._buf))
        return max(beyond) > threshold

    @property
    def clustering(self) -> bool:
        """Actionable clustering: a real ARCH effect with usable persistence."""
        return self.persistence >= _CLUSTER_MIN_PERSISTENCE and self.arch_effect()

    # ── fitting ───────────────────────────────────────────────────────

    def _omega_eff(self) -> float:
        """omega, or a sample-variance stand-in while unfitted."""
        if self._omega is not None:
            return self._omega
        return max(self._sample_variance() * (1.0 - self.persistence), _VARIANCE_FLOOR)

    def _sample_variance(self) -> float:
        if self._count < 2:
            return 1.0
        mean = self.mean
        var = self._sum_sq / self._count - mean * mean
        return max(var, _VARIANCE_FLOOR)

    def _maybe_refit(self) -> None:
        if not self._refit_interval or self._count < self._min_obs:
            return
        if self._count - self._last_fit_at < self._refit_interval:
            return

        self._last_fit_at = self._count
        self.fit()

    def fit(self) -> None:
        """Re-estimate (omega, alpha, beta) on the buffer by quasi-MLE.

        Variance targeting pins ``omega = s2 * (1 - alpha - beta)`` from the
        sample variance, leaving a two-parameter search over the stationary
        simplex.  Coarse-to-fine grid rather than a general optimiser: the
        project carries no scipy dependency (same rationale as
        ``core/running_stats.py``), and the surface is two-dimensional and
        bounded.
        """
        data = list(self._buf)
        if len(data) < self._min_obs:
            return

        s2 = _series_variance(data)
        best = (self._alpha, self._beta)
        best_nll = _neg_log_lik(data, s2, *best)

        lo_a, hi_a, lo_b, hi_b = 0.0, 0.6, 0.0, _MAX_PERSISTENCE
        for _ in range(_FIT_LEVELS):
            best, best_nll = _grid_step(data, s2, (lo_a, hi_a, lo_b, hi_b), best, best_nll)
            span_a = (hi_a - lo_a) / _FIT_STEPS
            span_b = (hi_b - lo_b) / _FIT_STEPS
            lo_a, hi_a = max(0.0, best[0] - span_a), min(0.6, best[0] + span_a)
            lo_b, hi_b = max(0.0, best[1] - span_b), min(_MAX_PERSISTENCE, best[1] + span_b)

        self._alpha, self._beta = best
        self._omega = max(s2 * (1.0 - self.persistence), _VARIANCE_FLOOR)

    # ── persistence ───────────────────────────────────────────────────

    def save(self) -> dict:
        return {
            "omega": self._omega,
            "alpha": self._alpha,
            "beta": self._beta,
            "min_obs": self._min_obs,
            "refit_interval": self._refit_interval,
            "buffer_size": self._buffer_size,
            "buf": list(self._buf),
            "count": self._count,
            "sum": self._sum,
            "sum_sq": self._sum_sq,
            "sigma2": self._sigma2,
            "eps": self._eps,
            "last_fit_at": self._last_fit_at,
        }

    def load(self, data: dict) -> None:
        if not data:
            return

        self._omega = data.get("omega")
        self._alpha = data.get("alpha", self._alpha)
        self._beta = data.get("beta", self._beta)
        self._min_obs = data.get("min_obs", self._min_obs)
        self._refit_interval = data.get("refit_interval", self._refit_interval)
        self._buffer_size = data.get("buffer_size", self._buffer_size)
        self._buf = collections.deque(data.get("buf", []), maxlen=self._buffer_size)
        self._count = data.get("count", 0)
        self._sum = data.get("sum", 0.0)
        self._sum_sq = data.get("sum_sq", 0.0)
        self._sigma2 = data.get("sigma2", 0.0)
        self._eps = data.get("eps", 0.0)
        self._last_fit_at = data.get("last_fit_at", 0)


def _validate_params(omega: float | None, alpha: float, beta: float) -> None:
    if omega is not None and omega <= 0.0:
        raise ValueError("omega must be positive")
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha and beta must be non-negative")
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be < 1 for a stationary GARCH(1,1)")


def _series_variance(data: list[float]) -> float:
    mean = sum(data) / len(data)
    var = sum((v - mean) ** 2 for v in data) / len(data)
    return max(var, _VARIANCE_FLOOR)


def _neg_log_lik(data: list[float], s2: float, alpha: float, beta: float) -> float:
    """Gaussian quasi negative log-likelihood under variance targeting."""
    persistence = alpha + beta
    if persistence >= _MAX_PERSISTENCE:
        return math.inf

    omega = max(s2 * (1.0 - persistence), _VARIANCE_FLOOR)
    mean = sum(data) / len(data)
    sigma2 = s2
    eps = 0.0
    total = 0.0
    for v in data:
        sigma2 = omega + alpha * eps * eps + beta * sigma2
        sigma2 = max(sigma2, _VARIANCE_FLOOR)
        eps = v - mean
        total += math.log(sigma2) + eps * eps / sigma2

    return 0.5 * total


def _grid_step(
    data: list[float],
    s2: float,
    box: tuple[float, float, float, float],
    best: tuple[float, float],
    best_nll: float,
) -> tuple[tuple[float, float], float]:
    """One coarse-to-fine sweep of the (alpha, beta) box."""
    lo_a, hi_a, lo_b, hi_b = box
    step_a = (hi_a - lo_a) / _FIT_STEPS
    step_b = (hi_b - lo_b) / _FIT_STEPS

    for i in range(_FIT_STEPS + 1):
        alpha = lo_a + i * step_a
        for j in range(_FIT_STEPS + 1):
            beta = lo_b + j * step_b
            if alpha + beta >= _MAX_PERSISTENCE:
                continue

            nll = _neg_log_lik(data, s2, alpha, beta)
            if nll < best_nll:
                best, best_nll = (alpha, beta), nll

    return best, best_nll
