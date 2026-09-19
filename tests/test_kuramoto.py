"""Tests for core/kuramoto.py.

Expectations are derived from the Kuramoto model's definitions (order
parameter, the ODE itself, Restrepo-Ott-Hunt's eigenvalue approximation
for K_c) rather than asserted against the implementation's own output --
same discipline as test_centrality.py.
"""

import math

import numpy as np
import pytest

from fuzzer_tool.core.kuramoto import (
    critical_coupling,
    frequency_density_at_zero,
    kuramoto_step,
    order_parameter,
    simulate,
    spectral_radius,
)


class TestOrderParameter:
    def test_empty(self):
        assert order_parameter(np.array([])) == (0.0, 0.0)

    def test_identical_phases_fully_synchronized(self):
        phases = np.full(5, 1.2345)
        r, psi = order_parameter(phases)
        assert abs(r - 1.0) < 1e-9
        assert abs(psi - 1.2345) < 1e-9

    def test_uniformly_spread_phases_incoherent(self):
        # N=4 evenly spaced points on the circle sum to exactly zero.
        phases = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
        r, _ = order_parameter(phases)
        assert abs(r - 0.0) < 1e-9

    def test_two_antipodal_phases_cancel(self):
        phases = np.array([0.0, math.pi])
        r, _ = order_parameter(phases)
        assert abs(r) < 1e-9

    def test_r_in_unit_interval(self):
        rng = np.random.default_rng(0)
        phases = rng.uniform(-math.pi, math.pi, size=37)
        r, _ = order_parameter(phases)
        assert 0.0 <= r <= 1.0 + 1e-12


class TestKuramotoStep:
    def test_zero_coupling_reduces_to_free_rotation(self):
        phases = np.array([0.0, 1.0])
        omega = np.array([2.0, -3.0])
        coupling = np.zeros((2, 2))
        out = kuramoto_step(phases, omega, coupling, k=5.0, dt=0.1)
        expected = phases + 0.1 * omega
        assert np.allclose(out, expected)

    def test_zero_k_ignores_coupling_matrix_entirely(self):
        phases = np.array([0.0, 3.0])
        omega = np.array([1.0, 1.0])
        coupling = np.array([[0.0, 1.0], [1.0, 0.0]])
        out = kuramoto_step(phases, omega, coupling, k=0.0, dt=0.1)
        expected = phases + 0.1 * omega
        assert np.allclose(out, expected)

    def test_empty_input(self):
        out = kuramoto_step(np.array([]), np.array([]), np.zeros((0, 0)), k=1.0)
        assert out.shape == (0,)

    def test_symmetric_coupling_pulls_phases_together(self):
        # Two oscillators, same natural frequency, strong symmetric
        # coupling, starting out of phase: the gap should shrink each step.
        phases = np.array([0.0, math.pi / 2])
        omega = np.array([0.0, 0.0])
        coupling = np.array([[0.0, 1.0], [1.0, 0.0]])
        gap0 = abs(phases[1] - phases[0])
        phases = kuramoto_step(phases, omega, coupling, k=10.0, dt=0.01)
        gap1 = abs(phases[1] - phases[0])
        assert gap1 < gap0


class TestSimulate:
    def test_strongly_coupled_identical_oscillators_synchronize(self):
        rng = np.random.default_rng(1)
        n = 8
        phases0 = rng.uniform(-math.pi, math.pi, size=n)
        omega = np.zeros(n)  # identical frequencies: nothing to fight
        coupling = np.ones((n, n)) - np.eye(n)  # all-to-all
        r_trace = simulate(phases0, omega, coupling, k=20.0, steps=300, dt=0.02)
        assert r_trace[0] < 0.9  # started roughly incoherent
        assert r_trace[-1] > 0.99  # ends locked

    def test_zero_coupling_never_synchronizes_heterogeneous_oscillators(self):
        rng = np.random.default_rng(2)
        n = 6
        phases0 = rng.uniform(-math.pi, math.pi, size=n)
        omega = rng.normal(0, 1, size=n)
        coupling = np.zeros((n, n))
        r_trace = simulate(phases0, omega, coupling, k=0.0, steps=200, dt=0.02)
        # Free-running oscillators at distinct frequencies drift apart and
        # r wanders with no restoring force -- it can pass through a
        # higher value by chance at any single tick, but with zero
        # coupling it can never lock into sustained near-full coherence.
        assert r_trace.max() < 0.9

    def test_trace_length(self):
        r_trace = simulate(np.zeros(3), np.zeros(3), np.zeros((3, 3)), k=1.0, steps=10)
        assert r_trace.shape == (11,)


class TestSpectralRadius:
    def test_empty(self):
        assert spectral_radius(np.zeros((0, 0))) == 0.0

    def test_diagonal_matrix_is_its_own_max_entry(self):
        a = np.diag([1.0, 5.0, 2.0])
        assert abs(spectral_radius(a) - 5.0) < 1e-9

    def test_scales_linearly(self):
        a = np.array([[0.0, 1.0], [1.0, 0.0]])
        rho1 = spectral_radius(a)
        rho2 = spectral_radius(2.0 * a)
        assert abs(rho2 - 2.0 * rho1) < 1e-9


class TestFrequencyDensityAtZero:
    def test_too_few_points(self):
        assert frequency_density_at_zero(np.array([1.0])) == 0.0

    def test_zero_variance(self):
        assert frequency_density_at_zero(np.full(10, 3.0)) == 0.0

    def test_gaussian_matches_analytic_density_at_zero(self):
        rng = np.random.default_rng(3)
        sigma = 2.0
        omega = rng.normal(0.0, sigma, size=20000)
        g0_hat = frequency_density_at_zero(omega)
        g0_true = 1.0 / (sigma * math.sqrt(2 * math.pi))
        # KDE with a data-driven bandwidth over a finite sample -- loose
        # tolerance, this is checking "right ballpark," not exactness.
        assert abs(g0_hat - g0_true) / g0_true < 0.15


class TestCriticalCoupling:
    def test_no_coupling_is_infinite(self):
        omega = np.array([-1.0, 0.0, 1.0])
        assert critical_coupling(np.zeros((3, 3)), omega) == float("inf")

    def test_degenerate_frequencies_is_infinite(self):
        coupling = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert critical_coupling(coupling, np.zeros(2)) == float("inf")

    def test_stronger_coupling_lowers_threshold(self):
        omega = np.array([-1.0, -0.5, 0.5, 1.0])
        weak = np.array(
            [
                [0.0, 0.1, 0.0, 0.0],
                [0.1, 0.0, 0.1, 0.0],
                [0.0, 0.1, 0.0, 0.1],
                [0.0, 0.0, 0.1, 0.0],
            ]
        )
        strong = weak * 10.0
        kc_weak = critical_coupling(weak, omega)
        kc_strong = critical_coupling(strong, omega)
        assert kc_strong < kc_weak

    def test_matches_katz_style_spectral_radius_on_a_real_transition_matrix(self):
        # Reuse op_katz's own transition-matrix builder so this test
        # exercises the exact object this module is meant to consume.
        from fuzzer_tool.core.schedulers.op_katz import build_transition_matrix

        transition_counts = {
            "havoc": {"havoc": 3, "splice": 7},
            "splice": {"havoc": 6, "splice": 4},
        }
        ops = ["havoc", "splice"]
        a = build_transition_matrix(transition_counts, ops)
        omega = np.array([-1.0, 1.0])
        rho = spectral_radius(a)
        assert rho > 0.0
        kc = critical_coupling(a, omega)
        assert kc > 0.0 and math.isfinite(kc)
        # K_c must equal K_0/rho exactly for this closed form.
        g0 = frequency_density_at_zero(omega)
        expected = (2.0 / (math.pi * g0)) / rho
        assert abs(kc - expected) < 1e-9

    @pytest.mark.parametrize("n", [0, 1])
    def test_degenerate_sizes_do_not_raise(self, n):
        coupling = np.zeros((n, n))
        omega = np.zeros(n)
        assert critical_coupling(coupling, omega) == float("inf")
