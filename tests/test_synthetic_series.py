"""Synthetic null-series generators (core/synthetic_series.py).

Each generator is checked against the closed-form property it exists to
produce, with a control run of the same estimator on a reference series
(Hard Rule 46) and the degenerate parameter that must remove the property.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.synthetic_series import ar1, cox, garch, pink

_N = 20_000
_SDS = 6.0  # tolerance in standard errors


def _lag1(x: np.ndarray) -> float:
    """Sample lag-1 autocorrelation."""
    d = x - x.mean()
    return float(np.dot(d[:-1], d[1:]) / np.dot(d, d))


def _slope(x: np.ndarray) -> float:
    """Log-log periodogram slope over the two-sided bins (no DC, no Nyquist)."""
    n = x.size
    power = np.abs(np.fft.rfft(x)) ** 2
    k = np.arange(1, (n + 1) // 2)
    f = np.fft.rfftfreq(n)[k]
    return float(np.polyfit(np.log(f), np.log(power[k]), 1)[0])


def _dispersion(x: np.ndarray) -> float:
    """Index of dispersion var/mean: 1 for Poisson."""
    return float(x.var() / x.mean())


class TestAr1:
    @pytest.mark.parametrize("phi", [0.7, -0.6, 0.9])
    def test_lag1_autocorrelation_is_phi(self, phi):
        r = _lag1(ar1(RandPool(seed=1), _N, phi))
        se = math.sqrt((1.0 - phi * phi) / _N)
        assert abs(r - phi) < _SDS * se

    def test_control_two_runs_agree(self):
        """Hard Rule 46: the estimator agrees with itself across seeds."""
        a = _lag1(ar1(RandPool(seed=2), _N, 0.7))
        b = _lag1(ar1(RandPool(seed=3), _N, 0.7))
        assert abs(a - b) < _SDS * math.sqrt(2 * 0.51 / _N)

    def test_falsification_phi_zero_is_white(self):
        assert abs(_lag1(ar1(RandPool(seed=4), _N, 0.0))) < _SDS / math.sqrt(_N)

    def test_starts_stationary(self):
        """x[0] has the stationary variance 1/(1-phi^2), not 1."""
        phi = 0.9
        x0 = np.array([ar1(RandPool(seed=s), 1, phi)[0] for s in range(4000)])
        assert abs(x0.var() * (1 - phi * phi) - 1.0) < 0.1

    def test_edge_lengths(self):
        assert ar1(RandPool(seed=5), 0, 0.5).size == 0
        assert ar1(RandPool(seed=5), 1, 0.5).size == 1

    @pytest.mark.parametrize("phi", [1.0, -1.0, 1.5])
    def test_adversarial_nonstationary_phi_raises(self, phi):
        with pytest.raises(ValueError):
            ar1(RandPool(seed=6), 10, phi)


class TestPink:
    @pytest.mark.parametrize(("n", "beta"), [(512, 1.0), (512, 2.0), (511, 1.5)])
    def test_spectrum_slope_is_minus_beta(self, n, beta):
        """Amplitudes are set exactly, so the slope holds to float precision."""
        assert _slope(pink(RandPool(seed=7), n, beta)) == pytest.approx(-beta, abs=1e-6)

    def test_control_white_has_flat_spectrum(self):
        """Hard Rule 46: beta=0 runs the same estimator to slope 0."""
        assert _slope(pink(RandPool(seed=8), 512, 0.0)) == pytest.approx(0.0, abs=1e-6)

    def test_falsification_seed_moves_phases_not_spectrum(self):
        a, b = pink(RandPool(seed=9), 256, 1.0), pink(RandPool(seed=10), 256, 1.0)
        assert not np.allclose(a, b)
        np.testing.assert_allclose(np.abs(np.fft.rfft(a))[1:-1], np.abs(np.fft.rfft(b))[1:-1])

    def test_zero_mean(self):
        assert abs(pink(RandPool(seed=11), 512, 1.0).mean()) < 1e-12

    def test_edge_lengths(self):
        assert pink(RandPool(seed=12), 0, 1.0).size == 0
        np.testing.assert_array_equal(pink(RandPool(seed=12), 1, 1.0), [0.0])


class TestCox:
    def test_counts_are_overdispersed(self):
        x = cox(RandPool(seed=13), _N)
        assert x.min() >= 0 and x.dtype.kind == "i"
        assert _dispersion(x) > 1.5

    def test_control_poisson_of_same_mean_is_not(self):
        """Hard Rule 46: the dispersion estimator reads ~1 on true Poisson."""
        rng = RandPool(seed=14)
        mean = float(cox(rng, _N).mean())
        assert abs(_dispersion(rng.poisson_array(np.full(_N, mean))) - 1.0) < 0.1

    def test_falsification_frozen_rate_is_poisson(self):
        x = cox(RandPool(seed=15), _N, sigma=0.0)
        assert abs(_dispersion(x) - 1.0) < 0.1
        assert abs(x.mean() - 3.0) < _SDS * math.sqrt(3.0 / _N)

    def test_rate_drifts_slowly(self):
        """The OU rate is persistent: counts are positively autocorrelated."""
        assert _lag1(cox(RandPool(seed=16), _N).astype(float)) > 0.2

    def test_edge_zero_length(self):
        assert cox(RandPool(seed=17), 0).size == 0

    @pytest.mark.parametrize("kw", [{"tau": 0.0}, {"tau": -1.0}, {"base": -1.0}])
    def test_adversarial_bad_parameters_raise(self, kw):
        with pytest.raises(ValueError):
            cox(RandPool(seed=18), 10, **kw)


class TestGarch:
    def test_unconditional_variance(self):
        omega, alpha, beta = 0.05, 0.1, 0.8
        x = garch(RandPool(seed=19), 100_000, omega, alpha, beta)
        assert abs(x.var() / (omega / (1 - alpha - beta)) - 1.0) < 0.05

    def test_volatility_clusters(self):
        x = garch(RandPool(seed=20), _N, 0.05, 0.25, 0.70)
        assert _lag1(x * x) > 0.1

    def test_control_no_arch_term_is_iid(self):
        """Hard Rule 46: alpha=0 runs the same estimator to ~0 clustering."""
        x = garch(RandPool(seed=21), _N, 0.05, 0.0, 0.70)
        assert abs(_lag1(x * x)) < _SDS / math.sqrt(_N)
        assert abs(x.var() / (0.05 / 0.30) - 1.0) < 0.05

    def test_falsification_white_noise_has_variance_omega(self):
        x = garch(RandPool(seed=22), _N, 2.0, 0.0, 0.0)
        assert abs(x.var() / 2.0 - 1.0) < 0.05

    def test_mean_offset(self):
        x = garch(RandPool(seed=23), _N, 0.05, 0.1, 0.8, mean=20.0)
        assert abs(x.mean() - 20.0) < 0.05

    def test_edge_zero_length(self):
        assert garch(RandPool(seed=24), 0, 0.05, 0.1, 0.8).size == 0

    @pytest.mark.parametrize(
        ("omega", "alpha", "beta"),
        [(0.05, 0.5, 0.5), (0.0, 0.1, 0.8), (0.05, -0.1, 0.8), (0.05, 0.1, -0.2)],
    )
    def test_adversarial_nonstationary_or_negative_raises(self, omega, alpha, beta):
        with pytest.raises(ValueError):
            garch(RandPool(seed=25), 10, omega, alpha, beta)
