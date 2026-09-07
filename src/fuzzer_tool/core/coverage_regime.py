"""Coverage regime detector: percolation phase classification for fuzzing.

Combines CriticalSlowingDown (discovery-rate CSD), CoverageHomogeneityDetector
(spatial clustering), AllanVariance (stall detection), and edge-delta tracking
into a single phase classifier.  Emits actionable regime labels to the main
loop via a property; the loop does not embed strategy logic in the detector.

``CoverageRegime`` is defined in ``fuzzer_tool.core.percolation`` and imported
here for use by the detector.
"""

import logging
import math

from fuzzer_tool.core.critical_slowing import (
    CoverageHomogeneityDetector,
    CriticalSlowingDown,
)
from fuzzer_tool.core.percolation import CoverageRegime

log = logging.getLogger(__name__)

# A one-step-ahead conditional variance this many times the model's
# unconditional level counts as a volatility spike.  Same shape and default
# as ``csd_rise_threshold``: both ask "how far above baseline".
_GARCH_SPIKE_FACTOR = 1.5

# Ordinal rank of each percolation phase.  CoverageRegime is a plain
# enum.Enum (no IntEnum ordering), so the monotonic association between the
# discrete label and the continuous Reynolds number is defined here, not by
# enum comparison.  SUBCRITICAL < CRITICAL < SUPERCRITICAL.
_REGIME_RANK: dict[CoverageRegime, int] = {
    CoverageRegime.SUBCRITICAL: 0,
    CoverageRegime.CRITICAL: 1,
    CoverageRegime.SUPERCRITICAL: 2,
}


def _average_ranks(values: list[float]) -> list[float]:
    """Rank *values*, averaging across ties (competition ranking)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, df: float) -> float:
    """CDF of Student's t with *df* degrees of freedom at *t*."""
    if df <= 0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * _betai(0.5 * df, 0.5, x)
    return 1.0 - tail if t >= 0.0 else tail


def _spearman_rank_correlation(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Spearman's rho and two-sided p-value for two paired series.

    Pearson on average-tied ranks.  The p-value comes from the t-statistic
    t = rho * sqrt((n-2)/(1-rho^2)); a constant series carries no rank
    information and returns rho 0.0, p 1.0 rather than dividing by zero.
    """
    n = len(xs)
    if n < 3:
        return 0.0, 1.0
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    sy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if sx == 0.0 or sy == 0.0:
        return 0.0, 1.0
    rho = cov / (sx * sy)
    # Clamp to [-1, 1] and snap anything within floating-point noise of a
    # perfect correlation to exactly +/-1 with p=0.0. Without the snap the
    # t-statistic is computed from (1 - rho^2) and a perfect rank order
    # comes out as 0.9999999999999998, which yields a tiny-but-nonzero p.
    if rho >= 1.0 - 1e-12:
        return 1.0, 0.0
    if rho <= -1.0 + 1e-12:
        return -1.0, 0.0
    t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    p = 2.0 * (1.0 - _student_t_cdf(abs(t), n - 2))
    return rho, min(1.0, max(0.0, p))


class CoverageRegimeDetector:
    """Classify coverage exploration into percolation phases.

    Args:
        csd: CriticalSlowingDown instance (pre-existing).
        homogeneity: CoverageHomogeneityDetector instance (pre-existing).
        stall_threshold: Executions without a new edge before declaring
            subcritical.  Defaults to the fuzzer's _stall_threshold.
        csd_rise_threshold: Multiplier above baseline for CSD detection.
        regime_history_size: Observations kept in regime_history.
        garch: Optional :class:`~fuzzer_tool.core.garch.OnlineGarch11`.  When
            supplied it can only *raise* an otherwise-supercritical tick to
            CRITICAL; it never overrides a stall or a CSD detection.  The
            model is fed from the main loop, not from here -- this detector
            reads state, it does not drive detectors.
        continuum: Optional
            :class:`~fuzzer_tool.core.navier_stokes.ContinuumField`.
            Instrumentation only: it annotates ``reason`` and feeds
            :meth:`continuum_correlation`, and never changes the label.
            Handover §6 orders the work that way -- the Reynolds diagnostic
            is promoted to a classification input only once it has been
            shown to track the existing CRITICAL label, and correlation is
            what makes that check possible.  If it never does, the lift adds
            no information and the field should be dropped.
    """

    def __init__(
        self,
        csd: CriticalSlowingDown,
        homogeneity: CoverageHomogeneityDetector,
        stall_threshold: int = 5000,
        csd_rise_threshold: float = 1.5,
        regime_history_size: int = 100,
        garch=None,
        continuum=None,
    ) -> None:
        self._csd = csd
        self._homogeneity = homogeneity
        self._stall_threshold = stall_threshold
        self._csd_rise_threshold = csd_rise_threshold
        self._regime_history_size = regime_history_size
        self._garch = garch
        self._continuum = continuum
        self._continuum_history: list[tuple[CoverageRegime, float]] = []

        self._regime: CoverageRegime = CoverageRegime.SUPERCRITICAL
        self._last_regime: CoverageRegime | None = None
        self._reason: str = "establishing baseline"
        self._actionable: bool = False
        self._regime_history: list[tuple[int, CoverageRegime]] = []
        self._stall_triggered: bool = False
        self._last_discovery_rate: float = 0.0
        self._last_allan_delta: int = 0
        self._last_exec_count: int = 0

    def observe(
        self,
        discovery_rate: float,
        allan_delta: int,
        homogeneity_result: dict | None,
        execs_since_edge: int,
        exec_count: int,
    ) -> CoverageRegime:
        """Update regime classification from current observations.

        Args:
            discovery_rate: Edges per 1000 executions.  Used as a weak
                subcritical signal when every other detector is silent and
                the rate has collapsed near zero under a long stall window.
            allan_delta: Edge count delta since last observation.  Stored for
                diagnostics; classification reads CSD's own window instead.
            homogeneity_result: Output of CoverageHomogeneityDetector.detect(),
                or None when the detector is not configured.
            execs_since_edge: Executions since the last new edge was found.
            exec_count: Total executions so far (history axis only).

        Returns:
            The current :class:`CoverageRegime` after classification.
        """
        # Last-seen signals for diagnostics / continuum correlation.  These
        # used to be passed into _classify and then ignored (P2-5), which is
        # the hole any sixth argument would fall into.  Store them explicitly
        # and only feed _classify what it actually reads.
        self._last_discovery_rate = discovery_rate
        self._last_allan_delta = allan_delta
        self._last_exec_count = exec_count

        self._regime_history.append((exec_count, self._regime))
        if len(self._regime_history) > self._regime_history_size:
            self._regime_history = self._regime_history[-self._regime_history_size :]

        self._actionable = False
        self._last_regime = self._regime

        self._classify(
            discovery_rate=discovery_rate,
            homogeneity_result=homogeneity_result,
            execs_since_edge=execs_since_edge,
        )

        self._record_continuum()

        if execs_since_edge >= self._stall_threshold:
            self._stall_triggered = True
        return self._regime

    def _classify(
        self,
        discovery_rate: float,
        homogeneity_result: dict | None,
        execs_since_edge: int,
    ) -> None:
        """Recompute _regime and _reason from current signals.

        Only arguments that participate in a branch are parameters.  Signals
        kept for diagnostics live on the detector (``_last_*``) rather than
        as ignored formal args (P2-5).
        """
        is_stalled = execs_since_edge >= self._stall_threshold

        # Check CSD first — it's the most sensitive near-transition signal.
        if self._csd is not None:
            csd_detected, csd_reason = self._csd.is_approaching_transition()
        else:
            csd_detected, csd_reason = (False, "no csd detector")

        # Stall trumps everything: no new edges for a long time = stuck.
        if is_stalled:
            self._regime = CoverageRegime.SUBCRITICAL
            self._reason = (
                f"stall ({execs_since_edge} execs without new edge) — "
                f"subcritical: exponential decay"
            )
            self._actionable = self._last_regime != CoverageRegime.SUBCRITICAL
            return

        # CSD: rising variance + autocorrelation = near a transition.
        if csd_detected:
            if "productive" in csd_reason:
                self._regime = CoverageRegime.CRITICAL
                self._reason = f"approaching transition (productive): {csd_reason}"
            else:
                self._regime = CoverageRegime.CRITICAL
                self._reason = f"approaching transition: {csd_reason}"
            self._actionable = self._last_regime != CoverageRegime.CRITICAL
            return

        # GARCH: the conditional variance is forecast well above its own
        # unconditional level while the clustering is statistically real.
        # Rising variance is the same precursor CSD looks for; this leg only
        # sees it one tick earlier, so it sits directly after CSD.
        spike_reason = self._garch_spike()
        if spike_reason:
            self._regime = CoverageRegime.CRITICAL
            self._reason = spike_reason
            self._actionable = self._last_regime != CoverageRegime.CRITICAL
            return

        # Homogeneity: clustered coverage without a CSD signal = biased
        # exploration, i.e. subcritical in the percolation sense.
        if homogeneity_result is not None and not homogeneity_result.get("homogeneous", True):
            chi2 = homogeneity_result.get("chi2", 0.0)
            p = homogeneity_result.get("p_value", 1.0)
            self._regime = CoverageRegime.SUBCRITICAL
            self._reason = (
                f"clustered coverage (χ²={chi2:.2f}, p={p:.4f}) — subcritical: biased exploration"
            )
            self._actionable = self._last_regime != CoverageRegime.SUBCRITICAL
            return

        # Near-zero discovery rate with a long quiet stretch: subcritical even
        # when homogeneity has not yet fired.  discovery_rate was previously
        # accepted by _classify and never read (P2-5).
        if (
            discovery_rate is not None
            and discovery_rate <= 0.0
            and execs_since_edge >= max(1, self._stall_threshold // 4)
        ):
            self._regime = CoverageRegime.SUBCRITICAL
            self._reason = (
                f"discovery rate collapsed ({discovery_rate:.4g}) after "
                f"{execs_since_edge} execs without new edge — subcritical"
            )
            self._actionable = self._last_regime != CoverageRegime.SUBCRITICAL
            return

        # Otherwise the fuzzer is compounding coverage normally.
        self._regime = CoverageRegime.SUPERCRITICAL
        self._reason = "healthy compounding — supercritical"
        self._actionable = (
            self._last_regime is not None and self._last_regime != CoverageRegime.SUPERCRITICAL
        )

    def _record_continuum(self) -> None:
        """Annotate the reason and log (regime, Re) for the correlation check."""
        diag = self._continuum.diagnostics if self._continuum is not None else None
        if diag is None:
            return

        self._reason += f" | Re={diag.reynolds:.3g} |grad p|={diag.pressure_gradient:.3g}"
        self._continuum_history.append((self._regime, diag.reynolds))
        if len(self._continuum_history) > self._regime_history_size:
            self._continuum_history = self._continuum_history[-self._regime_history_size :]

    def continuum_correlation(self) -> dict:
        """Spearman correlation between regime rank and Reynolds number.

        The recorded window is a set of (regime, Re) pairs -- regime ordinal,
        Re continuous and possibly non-linear in it.  Pearson on either raw
        or rank-transformed values would be the wrong coefficient; Spearman
        asks exactly the question the continuum lift is gated on: does Re
        separate the discrete labels the detector already produces?

        Returns ``rho`` and a two-sided p-value alongside the bucket means,
        which are kept (under ``bucket_means``) so callers that only read the
        per-regime Reynolds figure keep working.
        """
        if not self._continuum_history:
            return {"rho": 0.0, "p_value": 1.0, "bucket_means": {}}

        regimes = [_REGIME_RANK[r] for r, _ in self._continuum_history]
        re_values = [re_value for _, re_value in self._continuum_history]
        rho, p_value = _spearman_rank_correlation(regimes, re_values)

        buckets: dict[CoverageRegime, list[float]] = {}
        for regime, re_value in self._continuum_history:
            buckets.setdefault(regime, []).append(re_value)
        bucket_means = {k: sum(v) / len(v) for k, v in buckets.items()}
        return {"rho": rho, "p_value": p_value, "bucket_means": bucket_means}

    @property
    def continuum(self):
        return self._continuum

    def _garch_spike(self) -> str:
        """Reason string for a forecast volatility spike, or "" for none."""
        if self._garch is None or not self._garch.clustering:
            return ""

        forecast = self._garch.forecast()
        baseline = self._garch.unconditional_variance
        if forecast is None or forecast < _GARCH_SPIKE_FACTOR * baseline:
            return ""

        return (
            f"volatility clustering (sigma2 {forecast:.3g} vs {baseline:.3g} baseline, "
            f"persistence {self._garch.persistence:.2f}) -- critical: variance spike ahead"
        )

    @property
    def garch(self):
        return self._garch

    @property
    def regime(self) -> CoverageRegime:
        return self._regime

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def actionable(self) -> bool:
        return self._actionable

    def acknowledge(self) -> None:
        """Consume the actionable signal after the loop has acted on it."""
        self._actionable = False

    @property
    def regime_history(self) -> list[tuple[int, CoverageRegime]]:
        return list(self._regime_history)

    def reset(self) -> None:
        self._regime = CoverageRegime.SUPERCRITICAL
        self._last_regime = None
        self._reason = "reset"
        self._actionable = False
        self._regime_history.clear()
        self._continuum_history.clear()
        self._stall_triggered = False
        if self._continuum is not None:
            self._continuum.reset()
        if self._garch is not None:
            self._garch.reset()
        if self._csd is not None:
            self._csd.reset()
        if self._homogeneity is not None:
            self._homogeneity = CoverageHomogeneityDetector(
                num_columns=self._homogeneity.num_columns,
                window_size=self._homogeneity.window_size,
                homogeneity_p_threshold=self._homogeneity.homogeneity_p_threshold,
            )

    def save(self) -> dict:
        data = {
            "regime": self._regime.value,
            "reason": self._reason,
            "actionable": self._actionable,
            "regime_history": self._regime_history,
            "stall_triggered": self._stall_triggered,
            "csd": self._csd.save(),
            # CoverageHomogeneityDetector has no save(); its
            # _column_histories are recomputed from the replay buffer.
        }
        if self._garch is not None:
            data["garch"] = self._garch.save()
        if self._continuum is not None:
            data["continuum"] = self._continuum.save()
        return data

    def load(self, data: dict) -> None:
        if not data:
            return
        self._regime = CoverageRegime(data.get("regime", "supercritical"))
        self._reason = data.get("reason", "")
        self._actionable = data.get("actionable", False)
        self._regime_history = data.get("regime_history", [])
        self._stall_triggered = data.get("stall_triggered", False)
        if "csd" in data:
            self._csd.load(data["csd"])
        if "garch" in data and self._garch is not None:
            self._garch.load(data["garch"])
        if "continuum" in data and self._continuum is not None:
            self._continuum.load(data["continuum"])
        # CoverageHomogeneityDetector is re-created fresh on load;
        # its column history is rebuilt by replaying the fuzzer's
        # record_coverage_snapshot() calls during resume.
