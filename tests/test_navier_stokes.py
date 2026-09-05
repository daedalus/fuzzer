"""Tests for the steady continuum diagnostics.

The module is deliberately time-integration-free, so the invariants under
test are boundedness and monotonicity, not conservation.  The adversarial
class at the bottom is the one that matters: it pins the reason no advective
step exists.
"""

import math

import pytest

from fuzzer_tool.core.navier_stokes import (
    MAX_FLUX,
    ContinuumField,
    flux,
    gradient_magnitude,
    pressure_field,
    reynolds,
    viscosity,
)


class TestPressureField:
    def test_empty_input(self):
        assert pressure_field({}) == {}

    def test_scarce_coverage_has_high_pressure(self):
        p = pressure_field({"rare": 1.0, "common": 100.0})
        assert p["rare"] > p["common"]

    def test_bounded_to_unit_interval(self):
        p = pressure_field({k: float(k) for k in range(50)})
        assert all(0.0 <= v <= 1.0 for v in p.values())

    def test_monotone_decreasing_in_occupancy(self):
        p = pressure_field({"a": 1.0, "b": 4.0, "c": 16.0, "d": 64.0})
        assert p["a"] > p["b"] > p["c"] > p["d"]

    def test_uniform_occupancy_is_flat(self):
        p = pressure_field(dict.fromkeys("abcd", 7.0))
        assert len(set(round(v, 12) for v in p.values())) == 1

    def test_all_zero_occupancy_is_max_pressure(self):
        p = pressure_field(dict.fromkeys("abc", 0.0))
        assert all(v == 1.0 for v in p.values())

    def test_negative_occupancy_is_clamped(self):
        """ADVERSARIAL: a bad counter must not produce pressure > 1."""
        p = pressure_field({"a": -5.0, "b": 10.0})
        assert 0.0 <= p["a"] <= 1.0
        assert 0.0 <= p["b"] <= 1.0


class TestGradient:
    def test_no_adjacency_is_zero(self):
        p = {"a": 0.1, "b": 0.9}
        assert gradient_magnitude(p, {}) == {"a": 0.0, "b": 0.0}

    def test_gradient_is_max_neighbour_drop(self):
        p = {"a": 0.1, "b": 0.9, "c": 0.5}
        g = gradient_magnitude(p, {"a": {"b", "c"}, "b": {"a"}, "c": {"a"}})
        assert g["a"] == pytest.approx(0.8)
        assert g["b"] == pytest.approx(0.8)

    def test_unknown_neighbours_ignored(self):
        g = gradient_magnitude({"a": 0.5}, {"a": {"missing"}})
        assert g["a"] == 0.0

    def test_bounded(self):
        p = {"a": 0.0, "b": 1.0}
        g = gradient_magnitude(p, {"a": {"b"}, "b": {"a"}})
        assert all(0.0 <= v <= 1.0 for v in g.values())


class TestViscosityAndReynolds:
    def test_viscosity_is_strictly_positive(self):
        assert viscosity(0.0) > 0.0

    def test_viscosity_rises_with_failure_rate(self):
        assert viscosity(0.9) > viscosity(0.1)

    def test_viscosity_clamps_out_of_range_input(self):
        assert viscosity(-3.0) == viscosity(0.0)
        assert viscosity(50.0) == viscosity(1.0)

    def test_reynolds_zero_velocity(self):
        assert reynolds(0.0, 4.0, 0.5) == 0.0

    def test_reynolds_scales_with_velocity_and_length(self):
        assert reynolds(2.0, 4.0, 0.5) == pytest.approx(16.0)

    def test_reynolds_never_divides_by_zero(self):
        assert math.isfinite(reynolds(1.0, 1.0, 0.0))


class TestFlux:
    def test_unobserved_operator_gets_optimistic_flux(self):
        """Matches invasion's zero-resistance rule for unseen arms."""
        assert flux(0.0, 0.0, 0.0, viscosity(0.0)) == MAX_FLUX

    def test_hopeless_operator_gets_zero_flux(self):
        assert flux(0.0, 50.0, 0.5, viscosity(1.0)) == 0.0

    def test_pressure_gradient_increases_flux(self):
        lo = flux(5.0, 5.0, 0.0, viscosity(0.5))
        hi = flux(5.0, 5.0, 1.0, viscosity(0.5))
        assert hi > lo

    def test_viscosity_decreases_flux(self):
        lo_visc = flux(5.0, 5.0, 0.5, viscosity(0.0))
        hi_visc = flux(5.0, 5.0, 0.5, viscosity(1.0))
        assert lo_visc > hi_visc

    def test_flux_is_bounded(self):
        assert flux(1e9, 0.0, 1.0, viscosity(0.0)) <= MAX_FLUX


class TestContinuumField:
    def _field(self):
        f = ContinuumField()
        f.observe(
            occupancy={"a": 1.0, "b": 20.0, "c": 3.0},
            adjacency={"a": {"b"}, "b": {"a", "c"}, "c": {"b"}},
            velocity=2.0,
            failure_rate=0.25,
        )
        return f

    def test_diagnostics_none_before_observe(self):
        assert ContinuumField().diagnostics is None

    def test_diagnostics_after_observe(self):
        d = self._field().diagnostics
        assert 0.0 <= d.pressure_gradient <= 1.0
        assert d.viscosity > 0.0
        assert d.reynolds >= 0.0

    def test_flux_map_covers_every_operator(self):
        stats = {"havoc": (10.0, 5.0), "splice": (1.0, 40.0)}
        fm = self._field().flux_map(stats)
        assert set(fm) == set(stats)
        assert fm["havoc"] > fm["splice"]

    def test_flux_map_before_observe_is_empty(self):
        assert ContinuumField().flux_map({"havoc": (1.0, 1.0)}) == {}

    def test_empty_frontier_gives_zero_gradient(self):
        f = ContinuumField()
        f.observe(occupancy={}, adjacency={}, velocity=1.0, failure_rate=0.5)
        assert f.diagnostics.pressure_gradient == 0.0

    def test_save_load_round_trip(self):
        src = self._field()
        dst = ContinuumField()
        dst.load(src.save())
        assert dst.diagnostics == src.diagnostics

    def test_load_empty_is_noop(self):
        f = ContinuumField()
        f.load({})
        assert f.diagnostics is None

    def test_reset(self):
        f = self._field()
        f.reset()
        assert f.diagnostics is None


class TestNoTimeIntegration:
    """ADVERSARIAL: the module must expose no advective evolution.

    Tao (2014) builds an *averaged* Navier-Stokes whose nonlinearity keeps
    the cancellation law -- and therefore the energy identity -- plus
    essentially every function-space upper bound the true nonlinearity
    obeys, and still admits solutions that blow up in finite time.  Any
    surrogate obtained by replacing the nonlinear term with "something
    analogous" sits in that class, so qualitative behaviour is not inherited
    from real Navier-Stokes and cannot be assumed benign.

    This module is worse placed than Tao's construction, not better: there
    is no Leray projection on the horizon graph, so incompressibility is not
    enforced and the energy identity does not even hold.  The response is to
    integrate nothing.  Every quantity here is a bounded function of the
    current state, so there is no state to diverge.
    """

    def test_module_exposes_no_step_function(self):
        import fuzzer_tool.core.navier_stokes as ns

        forbidden = {"step", "advect", "integrate", "solve", "evolve", "lbm", "collide"}
        exported = {n.lower() for n in dir(ns) if not n.startswith("_")}
        assert not (forbidden & exported)

    def test_repeated_observation_does_not_accumulate(self):
        """No hidden integrator: the same input yields the same state."""
        args = dict(
            occupancy={"a": 1.0, "b": 8.0},
            adjacency={"a": {"b"}, "b": {"a"}},
            velocity=3.0,
            failure_rate=0.4,
        )
        f = ContinuumField()
        f.observe(**args)
        first = f.diagnostics
        for _ in range(500):
            f.observe(**args)
        assert f.diagnostics == first

    def test_all_outputs_stay_bounded_under_extreme_input(self):
        f = ContinuumField()
        f.observe(
            occupancy={"a": 0.0, "b": 1e12},
            adjacency={"a": {"b"}, "b": {"a"}},
            velocity=1e9,
            failure_rate=1.0,
        )
        d = f.diagnostics
        assert 0.0 <= d.pressure_gradient <= 1.0
        assert math.isfinite(d.reynolds)
        assert all(0.0 <= v <= MAX_FLUX for v in f.flux_map({"op": (1.0, 1.0)}).values())
