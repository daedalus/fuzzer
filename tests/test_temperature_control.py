"""Tests for the closed temperature loop: ESO, PI controller, composition.

Tier 2 of docs/handover/handover_control_theory_loops_2026-09-12.md.

The properties worth pinning here are not "does it converge" -- that depends
on a plant gain nobody has measured -- but the structural ones the handover
argued for specifically, each of which has a plausible-looking wrong version:

* no derivative term anywhere (the PV is a Poisson count);
* anti-windup that actually unwinds, rather than freezing forever;
* a fixed control period, not one inherited from the EPS-tracking stats tick;
* the disturbance state absorbing a plateau instead of the knob chasing it;
* the feed-forward clock schedule surviving untouched when the loop is off.
"""

import math

import pytest

from fuzzer_tool.core.eso import ExtendedStateObserver
from fuzzer_tool.core.pi_controller import PIController
from fuzzer_tool.core.temperature_control import (
    CORRECTION_MAX,
    CORRECTION_MIN,
    MIN_TICKS_BEFORE_ACTING,
    TEMPERATURE_CONTROL_EXECS,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    TemperatureController,
)


class TestPIControllerHasNoDerivative:
    def test_no_derivative_gain_is_accepted(self):
        assert not hasattr(PIController(0.5, 0.1), "kd")
        with pytest.raises(TypeError):
            PIController(0.5, 0.1, kd=0.1)  # type: ignore[call-arg]

    def test_output_is_independent_of_error_history_shape(self):
        """A D term would make these two diverge; a PI must not.

        Same error sequence, same total, different ordering: with derivative
        action the final outputs differ because the last step's slope
        differs. Without it they agree, because P sees only the current
        error and I only the sum.
        """
        a = PIController(0.5, 0.1)
        b = PIController(0.5, 0.1)
        for e in (0.1, 0.2, 0.3, 0.4, 0.3):
            a.update(e)
        for e in (0.4, 0.3, 0.2, 0.1, 0.3):
            b.update(e)
        assert a.output == pytest.approx(b.output)


class TestPIProportionalAndIntegral:
    def test_proportional_alone_leaves_a_standing_offset(self):
        """The reason Ki exists: P needs an error to produce any output."""
        p_only = PIController(kp=0.5, ki=0.0)
        for _ in range(50):
            p_only.update(0.2)
        assert p_only.output == pytest.approx(0.1)
        assert p_only.integral_term == 0.0

    def test_integral_accumulates_a_persistent_error(self):
        pi = PIController(kp=0.0, ki=0.05)
        for _ in range(10):
            pi.update(0.2)
        assert pi.integral_term == pytest.approx(0.1)
        assert pi.output == pytest.approx(0.1)

    def test_negative_gains_are_rejected(self):
        """Reverse action belongs at the call site, not in a gain sign."""
        with pytest.raises(ValueError):
            PIController(kp=-0.5, ki=0.1)
        with pytest.raises(ValueError):
            PIController(kp=0.5, ki=-0.1)

    def test_inverted_bounds_are_rejected(self):
        with pytest.raises(ValueError):
            PIController(0.5, 0.1, out_min=1.0, out_max=-1.0)


class TestPIAntiWindup:
    def test_integral_does_not_grow_while_output_is_railed(self):
        pi = PIController(kp=0.1, ki=0.5, out_min=-1.0, out_max=1.0)
        for _ in range(200):
            pi.update(1.0)  # huge sustained error, output pinned high
        assert pi.output == pytest.approx(1.0)
        assert pi.saturated
        assert abs(pi.integral_term) <= pi.integral_limit + 1e-9

    def test_a_railed_controller_recovers_promptly_when_the_error_flips(self):
        """The failure this guards: windup that takes as long to unwind.

        Without conditional integration, 200 ticks of maximum error leave an
        accumulator far past the rail, and the output stays pinned for
        roughly as many ticks of opposite error before it moves at all.
        """
        pi = PIController(kp=0.1, ki=0.5, out_min=-1.0, out_max=1.0)
        for _ in range(200):
            pi.update(1.0)
        ticks_to_leave_rail = None
        for k in range(1, 40):
            out = pi.update(-1.0)
            if out < 1.0:
                ticks_to_leave_rail = k
                break
        assert ticks_to_leave_rail is not None, "never left the rail"
        assert ticks_to_leave_rail <= 5, f"took {ticks_to_leave_rail} ticks to respond"

    def test_conditional_integration_matters_when_the_limit_is_generous(self):
        """Separates the two anti-windup mechanisms, which are independent.

        At the default ``integral_limit`` (the wider output bound) the hard
        clamp alone already bounds the accumulator tightly enough that
        recovery is fast, so the conditional-integration branch is not
        exercised -- verified by making integration unconditional, which
        left every other case in this file passing. The branch earns its
        place only when the limit is loose relative to the output range,
        which is reachable since ``integral_limit`` is a parameter.
        """
        loose = PIController(
            kp=0.1, ki=0.5, out_min=-1.0, out_max=1.0, integral_limit=50.0
        )
        for _ in range(200):
            loose.update(1.0)
        # Conditional integration stops the accumulator near the rail rather
        # than letting it run to the 50.0 clamp.
        assert loose.integral_term < 5.0, f"wound up to {loose.integral_term}"
        ticks = None
        for k in range(1, 200):
            if loose.update(-1.0) < 1.0:
                ticks = k
                break
        assert ticks is not None and ticks <= 5, f"took {ticks} ticks"

    def test_integration_resumes_when_the_error_moves_off_the_rail(self):
        """Freezing unconditionally while saturated never recovers."""
        pi = PIController(kp=0.1, ki=0.2, out_min=-1.0, out_max=1.0)
        for _ in range(50):
            pi.update(1.0)
        railed_integral = pi.integral_term
        for _ in range(5):
            pi.update(-1.0)
        assert pi.integral_term < railed_integral

    def test_reset_clears_the_accumulator(self):
        pi = PIController(kp=0.1, ki=0.5)
        for _ in range(20):
            pi.update(1.0)
        pi.reset()
        assert pi.integral_term == 0.0
        assert pi.output == 0.0
        assert not pi.saturated

    def test_accumulator_stores_ki_times_integral(self):
        """Changing Ki mid-run must not step the output."""
        pi = PIController(kp=0.0, ki=0.1)
        for _ in range(10):
            pi.update(0.5)
        before = pi.output
        pi.ki = 0.9  # would multiply a raw-integral store by 9
        after = pi.update(0.0)
        assert after == pytest.approx(before)


class TestExtendedStateObserver:
    def test_snap_initialises_on_the_first_observation(self):
        eso = ExtendedStateObserver()
        assert not eso.is_initialized
        eso.update(0.05)
        assert eso.is_initialized
        assert eso.value == pytest.approx(0.05)
        assert eso.disturbance == 0.0

    def test_tracks_a_constant_signal_with_no_disturbance(self):
        eso = ExtendedStateObserver(bandwidth=0.3)
        for _ in range(80):
            eso.update(0.02)
        assert eso.value == pytest.approx(0.02, abs=2e-3)
        assert abs(eso.disturbance) < 5e-3

    def test_a_step_is_attributed_to_disturbance_transiently(self):
        """The rejection is transient, and this pins that it is.

        Nothing touched the knob, so the step is by construction outside
        the model -- but the disturbance state holds unexplained
        *acceleration*, not an unexplained level. It spikes during the step
        and decays once the value state has tracked the new level. This
        test exists because the handover proposed the module claiming
        permanent absorption; that property needs a measured b0 and is not
        what this delivers.
        """
        eso = ExtendedStateObserver(bandwidth=0.3)
        for _ in range(60):
            eso.update(0.05, control=0.0)
        assert abs(eso.disturbance) < 5e-3
        eso.update(0.01, control=0.0)  # region exhausted, knob unchanged
        assert eso.disturbance < -1e-4, "the drop was not attributed anywhere"
        for _ in range(59):
            eso.update(0.01, control=0.0)
        assert abs(eso.disturbance) < 1e-5, "disturbance did not decay"
        assert eso.value == pytest.approx(0.01, abs=1e-4)

    def test_compensated_damps_the_transient_only(self):
        eso = ExtendedStateObserver(bandwidth=0.3)
        for _ in range(60):
            eso.update(0.05)
        # During the transient the controller is handed less than the full
        # drop...
        eso.update(0.01)
        assert eso.compensated(0.01) > 0.01
        # ...and afterwards it is handed all of it, which is the documented
        # limitation rather than a bug.
        for _ in range(59):
            eso.update(0.01)
        assert eso.compensated(0.01) == pytest.approx(0.01, abs=1e-5)

    def test_transient_window_is_short_and_bounded(self):
        """Quantifies the claim in compensated()'s docstring."""
        eso = ExtendedStateObserver(bandwidth=0.3)
        for _ in range(60):
            eso.update(0.05)
        deltas = []
        for _ in range(40):
            eso.update(0.01)
            deltas.append(abs(eso.compensated(0.01) - 0.01))
        assert sum(1 for d in deltas if d > 0.0005) <= 6
        assert max(deltas) < 0.01  # under the new level itself

    def test_disturbance_is_clamped_against_a_long_plateau(self):
        eso = ExtendedStateObserver(bandwidth=0.5, clamp_factor=2.0)
        eso.update(0.1)
        for _ in range(5000):
            eso.update(0.0)
        assert math.isfinite(eso.disturbance)
        assert abs(eso.disturbance) <= 2.0 * 0.1 + 1e-9

    def test_bad_parameters_are_rejected(self):
        with pytest.raises(ValueError):
            ExtendedStateObserver(bandwidth=0.0)
        with pytest.raises(ValueError):
            ExtendedStateObserver(clamp_factor=-1.0)

    def test_round_trips_through_dict(self):
        eso = ExtendedStateObserver(bandwidth=0.4, b0=2.0)
        for i in range(30):
            eso.update(0.01 * (i % 5), control=0.1)
        clone = ExtendedStateObserver.from_dict(eso.to_dict())
        assert clone.to_dict() == eso.to_dict()
        assert clone.update(0.03, 0.1) == pytest.approx(eso.update(0.03, 0.1))


class TestTemperatureControllerClock:
    def test_period_is_a_fixed_exec_count(self):
        ctl = TemperatureController()
        assert ctl.period_execs == TEMPERATURE_CONTROL_EXECS
        assert not ctl.observe(TEMPERATURE_CONTROL_EXECS - 1, 10)
        assert ctl.observe(TEMPERATURE_CONTROL_EXECS, 10)

    def test_observe_is_cheap_and_idempotent_between_ticks(self):
        ctl = TemperatureController()
        ctl.observe(TEMPERATURE_CONTROL_EXECS, 5)
        before = ctl.stats()["ticks"]
        for e in range(TEMPERATURE_CONTROL_EXECS + 1, TEMPERATURE_CONTROL_EXECS + 500):
            assert not ctl.observe(e, 5)
        assert ctl.stats()["ticks"] == before

    def test_rate_is_normalised_per_exec_not_per_tick(self):
        """A per-tick count would change scale with the window width."""
        a = TemperatureController(period_execs=1000)
        a.observe(1000, 10)
        b = TemperatureController(period_execs=2000)
        b.observe(2000, 20)
        assert a.stats()["rate"] == pytest.approx(b.stats()["rate"])

    def test_no_correction_before_the_warmup_ticks(self):
        ctl = TemperatureController(period_execs=100)
        edges = 0
        for k in range(1, MIN_TICKS_BEFORE_ACTING):
            edges += 5
            ctl.observe(100 * k, edges)
            assert ctl.correction() == 0.0

    def test_bad_setpoint_fraction_is_rejected(self):
        with pytest.raises(ValueError):
            TemperatureController(setpoint_fraction=0.0)
        with pytest.raises(ValueError):
            TemperatureController(setpoint_fraction=1.5)
        with pytest.raises(ValueError):
            TemperatureController(period_execs=0)


class TestTemperatureControllerLoop:
    def _drive(self, ctl, rates, period=None):
        """Feed a sequence of edges-per-exec rates; return corrections."""
        period = period or ctl.period_execs
        edges = 0.0
        out = []
        for k, r in enumerate(rates, start=1):
            edges += r * period
            ctl.observe(period * k, int(edges))
            out.append(ctl.correction())
        return out

    def test_falling_discovery_rate_raises_the_correction(self):
        """The direct-acting sign assumption, made explicit.

        Below setpoint means explore more. This is the unmeasured premise of
        the whole loop, so it is asserted rather than left implied.
        """
        ctl = TemperatureController(period_execs=1000)
        corrections = self._drive(ctl, [0.05] * 6 + [0.001] * 12)
        assert corrections[-1] > corrections[5]
        assert corrections[-1] > 0.0

    def test_correction_stays_within_its_bounds(self):
        ctl = TemperatureController(period_execs=1000)
        corrections = self._drive(ctl, [0.2] * 8 + [0.0] * 200)
        assert all(CORRECTION_MIN <= c <= CORRECTION_MAX for c in corrections)

    def test_temperature_stays_in_the_pickers_range(self):
        ctl = TemperatureController(period_execs=1000)
        self._drive(ctl, [0.2] * 8 + [0.0] * 100)
        for ff in (0.1, 0.5, 1.0):
            t = ctl.temperature(ff)
            assert TEMPERATURE_MIN <= t <= TEMPERATURE_MAX

    def test_setpoint_tracks_the_running_maximum_by_default(self):
        ctl = TemperatureController(period_execs=1000, setpoint_fraction=0.5)
        self._drive(ctl, [0.01, 0.02, 0.08, 0.02, 0.02, 0.02])
        st = ctl.stats()
        assert st["reference"] > 0.0
        assert st["setpoint"] == pytest.approx(0.5 * st["reference"])

    def test_an_explicit_reference_makes_the_setpoint_absolute(self):
        ctl = TemperatureController(
            period_execs=1000, setpoint_fraction=0.5, reference_rate=0.04
        )
        self._drive(ctl, [0.5] * 10)  # far above the reference
        assert ctl.stats()["setpoint"] == pytest.approx(0.02)
        assert ctl.stats()["reference"] == pytest.approx(0.04)

    def test_a_sustained_plateau_does_not_drive_the_knob_to_its_rail(self):
        """The most useful property this composition has, and not by design.

        Open loop the disturbance estimate decays once the value state
        tracks a step (see TestExtendedStateObserver). Closed loop it does
        not, because a nonzero control input has to be explained: if the
        knob moves and the rate does not follow, the observer attributes the
        difference to disturbance. So an assumed ``b0`` that is too large --
        which is the situation here, since the true plant gain is
        unmeasured and may be near zero -- makes the loop throttle itself
        rather than wind to its rail.

        That is graceful degradation toward "do nothing" in exactly the case
        where the controller has no authority, which is the failure mode
        worth having. Asserted here so a later refactor that "fixes" the
        observer's steady-state bias does not silently remove it.
        """
        ctl = TemperatureController(period_execs=1000)
        self._drive(ctl, [0.05] * 8)
        early = self._drive(ctl, [0.0] * 3)
        late = self._drive(ctl, [0.0] * 60)
        assert late[-1] > early[-1], "loop is not responding to the plateau at all"
        # Nowhere near the rail after 60 ticks of maximum sustained error.
        assert late[-1] < 0.1 * CORRECTION_MAX
        st = ctl.stats()
        # The plateau is held in the disturbance state, not in the error.
        assert st["rate"] == 0.0
        assert st["disturbance"] < 0.0
        assert st["compensated"] > 0.5 * st["setpoint"]
        assert not st["saturated"]

    def test_round_trips_through_dict(self):
        ctl = TemperatureController(period_execs=1000)
        self._drive(ctl, [0.05] * 10 + [0.01] * 10)
        clone = TemperatureController.from_dict(ctl.to_dict())
        assert clone.correction() == pytest.approx(ctl.correction())
        assert clone.stats()["ticks"] == ctl.stats()["ticks"]
        assert clone.to_dict() == ctl.to_dict()


class TestSeedPickerIntegration:
    """The feed-forward schedule must survive the loop being off."""

    def _picker(self, controller=None, anneal_budget=10000, exec_count=0, edges=0):
        import types

        from fuzzer_tool.services.seed_picker import SeedPicker

        f = types.SimpleNamespace(
            _anneal_budget=anneal_budget,
            exec_count=exec_count,
            _temperature=1.0,
            _temp_controller=controller,
            _edge_tracker=types.SimpleNamespace(get_cumulative_edge_count=lambda: edges),
            corpus=[],
        )
        return f, SeedPicker(f)

    def _apply(self, f, p):
        """Drive the REAL temperature computation, not a copy of it."""
        return p._update_temperature()

    def test_loop_off_reproduces_the_clock_schedule_exactly(self):
        for exec_count in (0, 2500, 5000, 9000, 20000):
            f, p = self._picker(controller=None, exec_count=exec_count)
            expected = max(0.1, 1.0 - exec_count / 10000)
            assert self._apply(f, p) == pytest.approx(expected)

    def test_loop_off_with_no_annealing_is_still_one(self):
        f, p = self._picker(controller=None, anneal_budget=0, exec_count=50000)
        assert self._apply(f, p) == 1.0

    def test_a_warming_up_controller_does_not_move_the_temperature(self):
        ctl = TemperatureController(period_execs=1000)
        f, p = self._picker(controller=ctl, exec_count=2500, edges=40)
        expected = max(0.1, 1.0 - 2500 / 10000)
        assert self._apply(f, p) == pytest.approx(expected)

    def test_pick_seed_delegates_to_the_extracted_method(self):
        """Pin the call, so the block cannot drift back inline."""
        import inspect

        from fuzzer_tool.services import seed_picker

        src = inspect.getsource(seed_picker.SeedPicker.pick_seed)
        assert "_update_temperature()" in src
        assert "_anneal_budget" not in src, "temperature logic drifted back inline"

    def test_the_controller_actually_moves_the_temperature(self):
        ctl = TemperatureController(period_execs=1000)
        edges = 0.0
        for k in range(1, 40):
            edges += 0.05 * 1000 if k <= 8 else 0.0
            f, p = self._picker(controller=ctl, exec_count=1000 * k, edges=int(edges))
            t = self._apply(f, p)
        assert ctl.correction() != 0.0
        assert t == pytest.approx(
            min(1.0, max(0.1, max(0.1, 1.0 - 39000 / 10000) + ctl.correction()))
        )
