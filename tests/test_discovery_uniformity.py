"""Tests for the Poisson index-of-dispersion discovery detector.

The load-bearing tests here mirror ``test_randomness.py``'s split: a
calibration test (stationary Poisson counts must not trip the detector at
much more than the nominal alpha) and discrimination tests (a genuine rate
change, and a burstiness pattern with the *same* mean rate, must trip it).
A detector whose false-positive rate on its own null isn't controlled is
worse than no detector -- see the module docstring for the mid-p/KS design
this replaced after that measurement came back badly miscalibrated.
"""

import random

from fuzzer_tool.core.discovery_uniformity import (
    DiscoveryUniformityDetector,
    dispersion_pvalue,
)


def _feed(d, counts):
    for c in counts:
        d.update(c)
    return d


def _poisson_sample(rng, lam):
    # Knuth's algorithm -- fine for the small lambdas a stats-tick edge
    # delta actually takes, avoids a numpy/scipy test dependency.
    if lam <= 0:
        return 0
    lim = pow(2.718281828459045, -lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= lim:
            return k - 1


def _poisson_series(n, lam, seed=0):
    rng = random.Random(seed)
    return [_poisson_sample(rng, lam) for _ in range(n)]


class TestDispersionPvalue:
    def test_too_few_counts_is_inconclusive(self):
        assert dispersion_pvalue([5]) == 1.0
        assert dispersion_pvalue([]) == 1.0

    def test_zero_mean_is_inconclusive(self):
        assert dispersion_pvalue([0, 0, 0, 0]) == 1.0

    def test_a_genuine_poisson_draw_is_not_a_rejection(self):
        """A real Poisson(lambda) sample lands well inside the null."""
        rng = random.Random(42)
        counts = [_poisson_sample(rng, 6.0) for _ in range(200)]
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
        rng = random.Random(5)
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


class TestReset:
    def test_reset_clears_state(self):
        d = DiscoveryUniformityDetector(min_obs=8)
        _feed(d, _poisson_series(50, lam=5.0, seed=6))
        d.reset()
        assert d.verdict() == {"homogeneous": True, "p": 1.0, "n": 0}
