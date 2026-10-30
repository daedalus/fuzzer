"""Tests for the Poisson index-of-dispersion discovery detector.

The load-bearing tests here mirror ``test_randomness.py``'s split: a
calibration test (stationary Poisson counts must not trip the detector at
much more than the nominal alpha) and discrimination tests (a genuine rate
change, and a burstiness pattern with the *same* mean rate, must trip it).
A detector whose false-positive rate on its own null isn't controlled is
worse than no detector -- see the module docstring for the mid-p/KS design
this replaced after that measurement came back badly miscalibrated.
"""

import numpy as np

from fuzzer_tool.core.analyzers.analyzer_discovery_uniformity import (
    DiscoveryUniformityDetector,
    dispersion_pvalue,
    fit_negative_binomial,
)
from fuzzer_tool.core.rand_pool import RandPool


def _feed(d, counts):
    for c in counts:
        d.update(c)
    return d


def _poisson_series(n, lam, seed=0):
    return RandPool(seed=seed).poisson_array(np.full(n, lam)).tolist()


class TestDispersionPvalue:
    def test_too_few_counts_is_inconclusive(self):
        assert dispersion_pvalue([5]) == 1.0
        assert dispersion_pvalue([]) == 1.0

    def test_zero_mean_is_inconclusive(self):
        assert dispersion_pvalue([0, 0, 0, 0]) == 1.0

    def test_a_genuine_poisson_draw_is_not_a_rejection(self):
        """A real Poisson(lambda) sample lands well inside the null."""
        counts = _poisson_series(200, lam=6.0, seed=42)
        assert dispersion_pvalue(counts) > 0.05

    def test_wildly_overdispersed_is_rejected(self):
        counts = [0] * 40 + [200] * 5
        assert dispersion_pvalue(counts) < 0.01

    def test_perfectly_constant_is_rejected_as_underdispersed(self):
        """Zero variance at a positive mean is as informative as too much."""
        assert dispersion_pvalue([7] * 50) < 0.01


class TestCalibration:
    def test_stationary_poisson_reads_homogeneous(self):
        d = DiscoveryUniformityDetector(window=300, min_obs=32)
        _feed(d, _poisson_series(300, lam=8.0, seed=1))
        v = d.verdict()
        assert v["homogeneous"]
        assert v["p"] > 0.01

    def test_false_positive_rate_near_alpha(self):
        """Over many independent stationary runs, rejections stay bounded."""
        rejections = 0
        trials = 200
        for seed in range(trials):
            d = DiscoveryUniformityDetector(window=300, min_obs=32, alpha=0.01)
            _feed(d, _poisson_series(300, lam=6.0, seed=seed + 1000))
            if not d.verdict()["homogeneous"]:
                rejections += 1
        assert rejections / trials < 0.05

    def test_below_min_obs_is_inconclusive(self):
        d = DiscoveryUniformityDetector(min_obs=32)
        _feed(d, _poisson_series(10, lam=5.0, seed=2))
        v = d.verdict()
        assert v["homogeneous"]
        assert v["p"] == 1.0

    def test_empty_detector_is_inconclusive(self):
        d = DiscoveryUniformityDetector()
        v = d.verdict()
        assert v["homogeneous"]
        assert v["n"] == 0


class TestDiscrimination:
    def test_catches_a_rate_step_change(self):
        """First half at lam=5, second half at lam=25 -- same detector, one run."""
        d = DiscoveryUniformityDetector(window=400, min_obs=32)
        _feed(d, _poisson_series(200, lam=5.0, seed=3))
        _feed(d, _poisson_series(200, lam=25.0, seed=4))
        v = d.verdict()
        assert not v["homogeneous"]

    def test_catches_burstiness_at_a_constant_mean_rate(self):
        """Alternating feast/famine ticks with the *same* long-run mean as a
        steady process -- a plain rate/mean check would miss this; the
        dispersion statistic is exactly what exposes the per-tick shape."""
        rng = RandPool(seed=5)
        bursty = []
        for _ in range(150):
            bursty.append(0)
            bursty.append(0)
            bursty.append(30 + rng.randint(-2, 2))
        d = DiscoveryUniformityDetector(window=450, min_obs=32)
        _feed(d, bursty)
        v = d.verdict()
        assert not v["homogeneous"]

    def test_accepts_stdlib_and_plain_ints(self):
        """Non-negative ints/floats both work; a stray negative delta from a
        counter reset must not raise."""
        d = DiscoveryUniformityDetector(min_obs=8, window=32)
        for c in [1, 2.0, -1, 0, 3]:
            d.update(c)
        d.verdict()  # must not raise


class TestFitNegativeBinomial:
    def test_too_few_counts_is_none(self):
        assert fit_negative_binomial([5]) is None
        assert fit_negative_binomial([]) is None

    def test_zero_mean_is_none(self):
        assert fit_negative_binomial([0, 0, 0, 0]) is None

    def test_poisson_like_var_not_exceeding_mean_is_none(self):
        # Perfectly constant: var == 0 <= mean, no overdispersion to fit.
        assert fit_negative_binomial([7] * 20) is None

    def test_bursty_counts_fit_a_small_r(self):
        counts = [0] * 40 + [200] * 5
        nb = fit_negative_binomial(counts)
        assert nb is not None
        assert 0 < nb["r"] < 1
        assert 0 < nb["p"] < 1

    def test_recovers_known_parameters_from_a_negative_binomial_sample(self):
        """Sample from NB(r=3, p=0.4) via r iid geometric draws (the sum
        representation the wiki article gives) and check the moment
        estimator lands near the generating parameters."""
        rng = RandPool(seed=7)
        r_true, p_true = 3, 0.4
        counts = [sum(rng.geometric(p_true) for _ in range(r_true)) for _ in range(4000)]
        nb = fit_negative_binomial(counts)
        assert nb is not None
        assert abs(nb["r"] - r_true) / r_true < 0.15
        assert abs(nb["p"] - p_true) / p_true < 0.15

    def test_verdict_surfaces_nb_fit_only_when_overdispersed(self):
        d = DiscoveryUniformityDetector(window=300, min_obs=32)
        _feed(d, _poisson_series(300, lam=8.0, seed=1))
        v = d.verdict()
        assert v["homogeneous"]
        assert "nb_r" not in v

        d2 = DiscoveryUniformityDetector(window=450, min_obs=32)
        rng = RandPool(seed=5)
        bursty = []
        for _ in range(150):
            bursty.append(0)
            bursty.append(0)
            bursty.append(30 + rng.randint(-2, 2))
        _feed(d2, bursty)
        v2 = d2.verdict()
        assert not v2["homogeneous"]
        assert "nb_r" in v2 and "nb_p" in v2


class TestReset:
    def test_reset_clears_state(self):
        d = DiscoveryUniformityDetector(min_obs=8)
        _feed(d, _poisson_series(50, lam=5.0, seed=6))
        d.reset()
        assert d.verdict() == {"homogeneous": True, "p": 1.0, "n": 0}
