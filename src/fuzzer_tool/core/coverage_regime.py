"""Coverage regime detector: percolation phase classification for fuzzing.

Combines CriticalSlowingDown (discovery-rate CSD), CoverageHomogeneityDetector
(spatial clustering), AllanVariance (stall detection), and edge-delta tracking
into a single phase classifier.  Emits actionable regime labels to the main
loop via a property; the loop does not embed strategy logic in the detector.

``CoverageRegime`` is defined in ``fuzzer_tool.core.percolation`` and imported
here for use by the detector.
"""

import logging

from fuzzer_tool.core.critical_slowing import (
    CoverageHomogeneityDetector,
    CriticalSlowingDown,
)
from fuzzer_tool.core.percolation import CoverageRegime

log = logging.getLogger(__name__)


class CoverageRegimeDetector:
    """Classify coverage exploration into percolation phases.

    Args:
        csd: CriticalSlowingDown instance (pre-existing).
        homogeneity: CoverageHomogeneityDetector instance (pre-existing).
        stall_threshold: Executions without a new edge before declaring
            subcritical.  Defaults to the fuzzer's _stall_threshold.
        csd_rise_threshold: Multiplier above baseline for CSD detection.
        regime_history_size: Observations kept in regime_history.
    """

    def __init__(
        self,
        csd: CriticalSlowingDown,
        homogeneity: CoverageHomogeneityDetector,
        stall_threshold: int = 5000,
        csd_rise_threshold: float = 1.5,
        regime_history_size: int = 100,
    ) -> None:
        self._csd = csd
        self._homogeneity = homogeneity
        self._stall_threshold = stall_threshold
        self._csd_rise_threshold = csd_rise_threshold
        self._regime_history_size = regime_history_size

        self._regime: CoverageRegime = CoverageRegime.SUPERCRITICAL
        self._last_regime: CoverageRegime | None = None
        self._reason: str = "establishing baseline"
        self._actionable: bool = False
        self._regime_history: list[tuple[int, CoverageRegime]] = []
        self._stall_triggered: bool = False

    def observe(
        self,
        discovery_rate: float,
        allan_delta: int,
        homogeneity_result: dict | None,
        execs_since_edge: int,
        exec_count: int,
    ) -> None:
        """Record one observation tick and recompute the regime.

        Args:
            discovery_rate: Edges per 1000 executions.  The CriticalSlowingDown
                detector is already fed this value from the stats reporter
                (stats.py:583).  We only read its state here.
            allan_delta: Edge count delta since last observation (unused directly;
                _stall_triggered is based on execs_since_edge alone).
            homogeneity_result: Dict from CoverageHomogeneityDetector.is_homogeneous()
                or None.  The homogeneity detector is already fed from the main loop
                (fuzzer.py:5720).  We only read its state here.
            execs_since_edge: Executions since the last new edge.
            exec_count: Total executions so far.
        """
        # Store the observation for the history.
        self._regime_history.append((exec_count, self._regime))
        if len(self._regime_history) > self._regime_history_size:
            self._regime_history = self._regime_history[-self._regime_history_size :]

        # Reset actionable whenever we re-observe; it flips True only
        # when the regime actually changes (see classify).
        self._actionable = False
        self._last_regime = self._regime

        # Classify (reads the detectors' internal state; does NOT re-feed them).
        self._classify(
            discovery_rate=discovery_rate,
            allan_delta=allan_delta,
            homogeneity_result=homogeneity_result,
            execs_since_edge=execs_since_edge,
            exec_count=exec_count,
        )

        # Persist the stall-triggered flag.
        if execs_since_edge >= self._stall_threshold:
            self._stall_triggered = True
        return self._regime

    def _classify(
        self,
        discovery_rate: float,
        allan_delta: int,
        homogeneity_result: dict | None,
        execs_since_edge: int,
        exec_count: int,
    ) -> None:
        """Recompute _regime and _reason from current signals."""
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

        # Otherwise the fuzzer is compounding coverage normally.
        self._regime = CoverageRegime.SUPERCRITICAL
        self._reason = "healthy compounding — supercritical"
        self._actionable = (
            self._last_regime is not None and self._last_regime != CoverageRegime.SUPERCRITICAL
        )

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
        self._stall_triggered = False
        if self._csd is not None:
            self._csd.reset()
        if self._homogeneity is not None:
            self._homogeneity = CoverageHomogeneityDetector(
                num_columns=self._homogeneity.num_columns,
                window_size=self._homogeneity.window_size,
                homogeneity_p_threshold=self._homogeneity.homogeneity_p_threshold,
            )

    def save(self) -> dict:
        return {
            "regime": self._regime.value,
            "reason": self._reason,
            "actionable": self._actionable,
            "regime_history": self._regime_history,
            "stall_triggered": self._stall_triggered,
            "csd": self._csd.save(),
            # CoverageHomogeneityDetector has no save(); its
            # _column_histories are recomputed from the replay buffer.
        }

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
        # CoverageHomogeneityDetector is re-created fresh on load;
        # its column history is rebuilt by replaying the fuzzer's
        # record_coverage_snapshot() calls during resume.
