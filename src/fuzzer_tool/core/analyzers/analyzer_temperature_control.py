"""Closed-loop control of the seed-picker exploration temperature.

``SeedPicker.pick_seed`` sets the annealing temperature from a clock::

    f._temperature = max(0.1, 1.0 - f.exec_count / f._anneal_budget)

That is pure feed-forward. Every discovery-rate estimator in the tree
(``RobustKF``, ``StructureFunctionDetector``, ``DispersionIndex``,
``CriticalSlowingDown``, ``coverage_growth_model``) feeds a *detector*; none
feeds an *actuator*. This module closes that loop: the clock schedule stays
as the feed-forward term and a PI correction is added on top.

Keeping the clock as feed-forward is not caution for its own sake. A
feed-forward term is unaffected by the process feedback, so it cannot
contribute to oscillation, and it means the existing behaviour is exactly
what happens when the loop is switched off.

Three deliberate choices, each of which has a cheaper-looking alternative:

**Its own fixed exec-count clock.** The natural place to hook this is the
stats tick, which is where every other periodic analysis runs — but that
tick is ``max(1, int(10 * last_avg_eps))``, so its width tracks the plant's
own throughput. A PI whose sampling period varies with the plant has a
drifting integral gain, and a per-tick discovery *count* additionally
changes scale when the window widens. So the control period is a constant
number of executions, and the process variable is normalised to edges per
exec rather than edges per tick.

**Setpoint as a fraction of the running maximum.** An absolute edges-per-exec
target is tunable but target-specific and campaign-length-specific; a
fraction of the best filtered rate the campaign has itself achieved is
self-normalising. This resolves the open question the handover posed without
claiming the answer is obvious: the absolute form is still reachable by
setting the reference explicitly, and the choice is recorded here rather than
buried in a default.

**Disturbance compensation before the error.** The controller sees the
ESO-compensated rate rather than the raw one, which damps its reaction to a
sudden move it cannot yet have caused — useful given a sensing chain whose
dead time runs from ~20 s to ~6 min. Note this is weaker than the handover
claimed when it proposed the module: the rejection is transient, so a
*sustained* plateau is still chased once the observer has tracked it. See
``eso.ExtendedStateObserver.compensated`` for why the stronger property
needs a measured ``b0`` and therefore is not available here.

**Unvalidated.** The sign and magnitude of ``d(discovery rate)/d(temperature)``
have not been measured. The loop assumes it is positive: below setpoint means
explore more. If it turns out to be near zero over the operating range then
this loop cannot work at all and the right response is to delete this module,
not to retune it. That measurement is the kill criterion recorded in
docs/handover/handover_control_theory_loops_2026-09-12.md and it is the
reason this is opt-in and off by default.
"""

from __future__ import annotations

from fuzzer_tool.core.eso import DEFAULT_ESO_BANDWIDTH, ExtendedStateObserver
from fuzzer_tool.core.pi_controller import PIController

# Control period, in executions. Fixed by construction -- see the module
# docstring on why this is not the stats interval. 2000 execs is roughly the
# same order as SATURATION_REFRESH_EXECS, and well above the ~8-sample
# transport delay in the operator credit path, so a correction is not
# competing with reward attribution that has not landed yet.
TEMPERATURE_CONTROL_EXECS = 2000

# Default gains. Not derived from a plant model -- there isn't one -- and not
# from Ziegler-Nichols either, which is explicitly poor on time-delay
# processes and the sensing chain's dead time here is both large and badly
# conditioned (median 2-34 ticks, p10=1, p90~36). They are set low on
# purpose: the loop should be visibly sluggish until the relay-derived
# ultimate gain and period from _stall_relay_stats() are available to tune
# against. Kp dominating Ki follows normal practice.
DEFAULT_KP = 0.25
DEFAULT_KI = 0.02

# Setpoint as a fraction of the best filtered discovery rate the campaign has
# achieved. Not 1.0: asking a campaign to match its own peak forever is a
# setpoint it can never hold, which means a permanently saturated output and
# a controller that has stopped carrying information.
DEFAULT_SETPOINT_FRACTION = 0.5

# The correction is added to the feed-forward schedule and the sum is clamped
# to the knob's existing range, so the correction alone needs to be able to
# span it.
CORRECTION_MIN = -0.9
CORRECTION_MAX = 0.9

# Temperature clamp, matching seed_picker's existing max(0.1, ...) floor.
TEMPERATURE_MIN = 0.1
TEMPERATURE_MAX = 1.0

# Ticks before the reference is trusted. The running maximum of a Poisson
# rate over a handful of samples is mostly noise, and a setpoint derived from
# it would be too, so the loop outputs nothing until it has a few.
MIN_TICKS_BEFORE_ACTING = 4


class TemperatureController:
    """PI-over-ESO regulation of the exploration temperature.

    Call :meth:`observe` freely — it self-gates on its own exec-count clock
    and is cheap when the clock has not advanced. Read :meth:`correction`
    for the additive adjustment to the feed-forward schedule.
    """

    def __init__(
        self,
        setpoint_fraction: float = DEFAULT_SETPOINT_FRACTION,
        kp: float = DEFAULT_KP,
        ki: float = DEFAULT_KI,
        bandwidth: float = DEFAULT_ESO_BANDWIDTH,
        period_execs: int = TEMPERATURE_CONTROL_EXECS,
        reference_rate: float | None = None,
    ):
        if not 0.0 < setpoint_fraction <= 1.0:
            raise ValueError(f"setpoint_fraction must be in (0, 1], got {setpoint_fraction!r}")
        if period_execs < 1:
            raise ValueError(f"period_execs must be >= 1, got {period_execs!r}")
        self.setpoint_fraction = float(setpoint_fraction)
        self.period_execs = int(period_execs)
        # reference_rate set explicitly turns the setpoint absolute; left
        # None it tracks the campaign's own running maximum.
        self.reference_rate = reference_rate
        self._pi = PIController(
            kp=kp, ki=ki, out_min=CORRECTION_MIN, out_max=CORRECTION_MAX
        )
        self._eso = ExtendedStateObserver(bandwidth=bandwidth)
        self._last_exec = 0
        self._last_edges = 0
        self._running_max = 0.0
        self._ticks = 0
        self._last_rate = None
        self._last_compensated = None
        self._last_setpoint = None
        self._applied_correction = 0.0

    # ── driving ───────────────────────────────────────────────────────────

    def observe(self, exec_count: int, cumulative_edges: int) -> bool:
        """Advance the loop if the control period has elapsed.

        Returns True when a tick was taken. Safe and cheap to call every
        iteration.
        """
        if exec_count - self._last_exec < self.period_execs:
            return False
        d_execs = exec_count - self._last_exec
        d_edges = max(0, cumulative_edges - self._last_edges)
        self._last_exec = exec_count
        self._last_edges = cumulative_edges
        if d_execs <= 0:
            return False

        rate = d_edges / d_execs
        self._ticks += 1
        self._last_rate = rate

        # The control that was actually applied over the window just
        # elapsed, not the one about to be: the observer is explaining what
        # already happened.
        self._eso.update(rate, control=self._applied_correction)
        compensated = self._eso.compensated(rate)
        self._last_compensated = compensated

        # Running maximum tracks the compensated rate, not the raw one, so a
        # single lucky burst that the observer attributes to a disturbance
        # cannot set a reference the knob is then held to forever.
        self._running_max = max(self._running_max, compensated)

        reference = self.reference_rate if self.reference_rate is not None else self._running_max
        setpoint = self.setpoint_fraction * reference
        self._last_setpoint = setpoint

        if self._ticks < MIN_TICKS_BEFORE_ACTING or reference <= 0.0:
            return True

        # Direct-acting: below setpoint means explore more, so a positive
        # error raises the temperature. This is the unmeasured sign
        # assumption named in the module docstring.
        self._pi.update(setpoint - compensated)
        self._applied_correction = self._pi.output
        return True

    # ── reading ───────────────────────────────────────────────────────────

    def correction(self) -> float:
        """Additive adjustment to the feed-forward temperature schedule."""
        if self._ticks < MIN_TICKS_BEFORE_ACTING:
            return 0.0
        return self._pi.output

    def temperature(self, feed_forward: float) -> float:
        """Apply the correction to ``feed_forward`` and clamp to range."""
        return min(TEMPERATURE_MAX, max(TEMPERATURE_MIN, feed_forward + self.correction()))

    def stats(self) -> dict:
        return {
            "ticks": self._ticks,
            "rate": self._last_rate,
            "compensated": self._last_compensated,
            "disturbance": self._eso.disturbance if self._eso.is_initialized else None,
            "setpoint": self._last_setpoint,
            "reference": (
                self.reference_rate if self.reference_rate is not None else self._running_max
            ),
            "correction": self.correction(),
            "integral": self._pi.integral_term,
            "saturated": self._pi.saturated,
            "active": self._ticks >= MIN_TICKS_BEFORE_ACTING,
        }

    # ── persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "setpoint_fraction": self.setpoint_fraction,
            "period_execs": self.period_execs,
            "reference_rate": self.reference_rate,
            "pi": self._pi.to_dict(),
            "eso": self._eso.to_dict(),
            "last_exec": self._last_exec,
            "last_edges": self._last_edges,
            "running_max": self._running_max,
            "ticks": self._ticks,
            "applied_correction": self._applied_correction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TemperatureController:
        ctl = cls(
            setpoint_fraction=data.get("setpoint_fraction", DEFAULT_SETPOINT_FRACTION),
            period_execs=data.get("period_execs", TEMPERATURE_CONTROL_EXECS),
            reference_rate=data.get("reference_rate"),
        )
        if "pi" in data:
            ctl._pi = PIController.from_dict(data["pi"])
        if "eso" in data:
            ctl._eso = ExtendedStateObserver.from_dict(data["eso"])
        ctl._last_exec = data.get("last_exec", 0)
        ctl._last_edges = data.get("last_edges", 0)
        ctl._running_max = data.get("running_max", 0.0)
        ctl._ticks = data.get("ticks", 0)
        ctl._applied_correction = data.get("applied_correction", 0.0)
        return ctl
