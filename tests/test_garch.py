"""Tests for the online GARCH(1,1) conditional-variance model.

Falsification and adversarial cases (Hard Rule 23) live at the bottom.
"""

import math
import random

import pytest

from fuzzer_tool.core.garch import (
    OVERLAP_ARTIFACT_LAGS,
    OnlineGarch11,
    squared_acf,
)


def _feed(g, values):
    for v in values:
        g.update(v)
    return g


def _garch_series(n, omega, alpha, beta, seed=0):
    """Generate a series that genuinely follows GARCH(1,1)."""
    rng = random.Random(seed)
    s2 = omega / max(1.0 - alpha - beta, 1e-6)
    eps = 0.0
    out = []
    for _ in range(n):
        s2 = omega + alpha * eps * eps + beta * s2
        eps = rng.gauss(0.0, math.sqrt(s2))
        out.append(eps)
    return out


class TestRecursion:
    def test_starts_unfitted(self):
        g = OnlineGarch11()
        assert g.count == 0
        assert g.forecast() is None

    def test_forecast_available_after_min_obs(self):
        g = _feed(OnlineGarch11(min_obs=8), [1.0, 5.0, 2.0, 9.0, 1.0, 7.0, 3.0, 8.0])
        f = g.forecast()
        assert f is not None
        assert f > 0.0

    def test_recursion_matches_closed_form(self):
        """sigma2_t = omega + alpha*eps_{t-1}^2 + beta*sigma2_{t-1}, exactly.

        Residuals are taken against the *running* mean, not the final one:
        the model is online, so using the whole-series mean would be
        lookahead bias.  The replay below mirrors that.
        """
        g = OnlineGarch11(omega=0.5, alpha=0.2, beta=0.7, min_obs=1, refit_interval=0)
        values = [4.0, 6.0, 2.0, 8.0, 5.0]
        _feed(g, values)

        s2 = g.initial_variance
        eps = 0.0
        running = 0.0
        for i, v in enumerate(values, start=1):
            s2 = 0.5 + 0.2 * eps * eps + 0.7 * s2
            running += v
            eps = v - running / i

        assert g.mean == pytest.approx(sum(values) / len(values))
        assert g.variance == pytest.approx(s2, rel=1e-9)

    def test_forecast_is_one_step_ahead(self):
        g = OnlineGarch11(omega=0.5, alpha=0.2, beta=0.7, min_obs=1, refit_interval=0)
        _feed(g, [4.0, 6.0, 2.0, 8.0, 5.0])
        expected = 0.5 + 0.2 * g.residual**2 + 0.7 * g.variance
        assert g.forecast() == pytest.approx(expected, rel=1e-9)

    def test_variance_stays_positive_on_constant_input(self):
        g = _feed(OnlineGarch11(min_obs=4), [3.0] * 200)
        assert g.variance > 0.0
        assert g.forecast() > 0.0


class TestStationarity:
    def test_rejects_nonstationary_parameters(self):
        with pytest.raises(ValueError):
            OnlineGarch11(alpha=0.6, beta=0.6)

    def test_rejects_negative_parameters(self):
        with pytest.raises(ValueError):
            OnlineGarch11(alpha=-0.1, beta=0.5)
        with pytest.raises(ValueError):
            OnlineGarch11(omega=0.0)

    def test_fit_never_leaves_the_stationary_region(self):
        g = OnlineGarch11(min_obs=64, refit_interval=64)
        _feed(g, _garch_series(400, 0.2, 0.30, 0.65, seed=3))
        assert g.persistence < 1.0
        assert g.alpha >= 0.0
        assert g.beta >= 0.0
        assert g.omega > 0.0


class TestFit:
    def test_recovers_persistence_on_a_real_garch_series(self):
        """A genuine GARCH(1,1) series should fit with high persistence."""
        g = OnlineGarch11(min_obs=64, refit_interval=128)
        _feed(g, _garch_series(1200, 0.05, 0.15, 0.80, seed=11))
        assert g.persistence > 0.6
        assert g.arch_effect() is True

    def test_no_arch_effect_on_iid_noise(self):
        """FALSIFICATION: i.i.d. noise has no volatility clustering."""
        rng = random.Random(5)
        g = OnlineGarch11(min_obs=64, refit_interval=128)
        _feed(g, [rng.gauss(20.0, 4.0) for _ in range(1200)])
        assert g.arch_effect() is False
        assert g.clustering is False


class TestDiagnostics:
    def test_squared_acf_of_iid_is_near_zero(self):
        rng = random.Random(7)
        vals = [rng.gauss(0.0, 1.0) for _ in range(4000)]
        acf = squared_acf(vals, lags=4)
        assert max(abs(v) for v in acf) < 0.1

    def test_squared_acf_detects_clustering(self):
        acf = squared_acf(_garch_series(4000, 0.05, 0.25, 0.70, seed=13), lags=4)
        assert acf[0] > 0.15

    def test_squared_acf_short_series_returns_empty(self):
        assert squared_acf([1.0, 2.0], lags=4) == []

    def test_ljung_box_rejects_on_clustered_series(self):
        g = OnlineGarch11(min_obs=64, refit_interval=128)
        _feed(g, _garch_series(1200, 0.05, 0.25, 0.70, seed=17))
        stat, p = g.ljung_box()
        assert stat > 0.0
        assert p < 0.05

    def test_ljung_box_accepts_on_iid(self):
        rng = random.Random(23)
        g = OnlineGarch11(min_obs=64, refit_interval=128)
        _feed(g, [rng.gauss(10.0, 2.0) for _ in range(1200)])
        _stat, p = g.ljung_box()
        assert p > 0.01


class TestOverlapArtifact:
    """ADVERSARIAL: the estimator that must NOT be used as input.

    ``services/stats_reporter.discovery_rate()`` is a 5-snapshot sliding
    window, so consecutive samples share four of five snapshots.  That is an
    MA(4) smoother: it manufactures a decaying positive autocorrelation in the
    squared residuals of a process that has none, and the artifact dies at
    lag 4 by construction.  Feeding it to this model reports clustering that
    is not there.  The production feed is the non-overlapping per-tick edge
    delta instead; this test pins the reason.
    """

    @staticmethod
    def _windowed(counts, window=5):
        cum, execs, edges = 0, [], []
        for i, c in enumerate(counts, start=1):
            cum += c
            execs.append(i * 1000)
            edges.append(cum)
        out = []
        for i in range(2, len(execs) + 1):
            lo = max(0, i - window)
            d_e = execs[i - 1] - execs[lo]
            d_g = edges[i - 1] - edges[lo]
            out.append(d_g / d_e * 1000 if d_e > 0 else 0.0)
        return out

    def test_sliding_window_fabricates_clustering(self):
        rng = random.Random(1234)
        counts = [rng.gauss(20.0, 4.5) for _ in range(3000)]

        raw = squared_acf(counts, lags=6)
        windowed = squared_acf(self._windowed(counts), lags=6)

        # Control first (Hard Rule 46): the unwindowed source has no ARCH.
        assert max(abs(v) for v in raw) < 0.1
        # The window alone produces a large lag-1 effect ...
        assert windowed[0] > 0.3
        # ... that collapses right after the window's own footprint.
        assert abs(windowed[OVERLAP_ARTIFACT_LAGS]) < 0.1

    def test_arch_effect_requires_persistence_beyond_the_artifact(self):
        """A signal whose ACF dies at lag 4 must not be called clustering."""
        rng = random.Random(99)
        counts = [rng.gauss(20.0, 4.5) for _ in range(3000)]
        g = OnlineGarch11(min_obs=64, refit_interval=256)
        _feed(g, self._windowed(counts))
        assert g.arch_effect() is False


class TestPersistence:
    def test_save_load_round_trip(self):
        g = OnlineGarch11(min_obs=16, refit_interval=32)
        _feed(g, _garch_series(300, 0.1, 0.2, 0.7, seed=31))
        restored = OnlineGarch11()
        restored.load(g.save())
        assert restored.variance == pytest.approx(g.variance)
        assert restored.forecast() == pytest.approx(g.forecast())
        assert restored.persistence == pytest.approx(g.persistence)
        assert restored.count == g.count

    def test_load_of_empty_dict_is_a_noop(self):
        g = OnlineGarch11()
        g.load({})
        assert g.count == 0

    def test_reset_clears_state(self):
        g = _feed(OnlineGarch11(min_obs=4), [1.0, 9.0, 2.0, 8.0])
        g.reset()
        assert g.count == 0
        assert g.forecast() is None


class TestEdgeCases:
    def test_all_zero_series_has_no_arch_effect(self):
        g = _feed(OnlineGarch11(min_obs=8, refit_interval=32), [0.0] * 500)
        assert g.arch_effect() is False
        assert g.variance > 0.0

    def test_single_spike_does_not_claim_clustering(self):
        g = OnlineGarch11(min_obs=8, refit_interval=64)
        _feed(g, [1.0] * 400 + [900.0] + [1.0] * 400)
        assert g.clustering is False

    def test_negative_values_are_accepted(self):
        g = _feed(OnlineGarch11(min_obs=8), [-3.0, 5.0, -7.0, 2.0, -1.0, 6.0, -4.0, 8.0])
        assert g.forecast() > 0.0

    def test_ljung_box_before_min_obs_is_inconclusive(self):
        g = _feed(OnlineGarch11(min_obs=64), [1.0, 2.0, 3.0])
        stat, p = g.ljung_box()
        assert stat == 0.0
        assert p == 1.0
