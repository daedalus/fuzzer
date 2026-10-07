"""Hierarchical pooling is an empirical-Bayes (BLUP) prior, not a convex blend.

The old blend ``(1 - h) * alpha_i + h * (1 + pooled_successes)`` handed every
unit a pseudocount of ``h`` times the whole population's evidence: at 20
seeds x 100 observations and h = 1, a seed's Beta had ~2000 pseudocounts, so
Thompson draws collapsed onto the population mean and pooling turned
exploration off. The shift also ignored the unit's own evidence: a seed with
1000 observations moved as far as one with 10.

Now the prior is Beta(h m mu, h m (1 - mu)) with (mu, m) fitted from the
units' dispersion (core/blup.py), added to each unit's own posterior.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.core.blup import MIN_STRENGTH, POOL_MAX_STRENGTH
from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler
from fuzzer_tool.core.seed_quality import BayesianSeedQuality
from tests.support.blup_ref import ref_fit, ref_strength


def _expected(alpha, beta, succ, n, h):
    """(alpha, beta) of a unit after the reference BLUP prior is added."""
    mu, rho = ref_fit(succ, n)
    m = ref_strength(rho, sum(n), POOL_MAX_STRENGTH, MIN_STRENGTH)
    return alpha + h * m * mu, beta + h * m * (1.0 - mu)


def _feed_seeds(bsq, rates):
    """Seed j gets rates[j][0] hits out of rates[j][1] observations."""
    for j, (hits, total) in enumerate(rates):
        sid = f"s{j}"
        bsq.init_seed(sid)
        for k in range(total):
            bsq.record_outcome(sid, discovered=k < hits)


def test_regression_pooling_concentration_is_bounded():
    bsq = BayesianSeedQuality(hierarchical_pooling=1.0)
    rates = [(j, 100) for j in range(20)]
    _feed_seeds(bsq, rates)

    a, b = bsq._get_pooled_params("s0")

    succ = [float(h) for h, _ in rates]
    n = [float(t) for _, t in rates]
    assert (a, b) == pytest.approx(_expected(1.0, 101.0, succ, n, 1.0))
    assert a + b < 102.0 + POOL_MAX_STRENGTH


def test_regression_well_observed_seed_keeps_its_rate():
    """Falsification of the old blend: 1000 observations at 10% beside
    seeds at 50% must stay near 10%; the blend moved it to ~23%."""
    bsq = BayesianSeedQuality(hierarchical_pooling=0.5)
    _feed_seeds(bsq, [(100, 1000)] + [(50, 100)] * 10)
    assert bsq.posterior_mean("s0") == pytest.approx(0.1, abs=0.02)


def test_poorly_observed_seed_shrinks_more():
    bsq = BayesianSeedQuality(hierarchical_pooling=1.0)
    _feed_seeds(bsq, [(100, 1000), (1, 10)] + [(50, 100), (30, 100), (70, 100)])
    big = abs(bsq.posterior_mean("s0") - 101 / 1002)
    small = abs(bsq.posterior_mean("s1") - 2 / 12)
    assert big < small


def test_falsify_identical_seeds_pool_at_ceiling():
    """Seeds with exactly equal rates: a fresh seed is predicted at the
    population rate with all the population's evidence (500 observations)."""
    bsq = BayesianSeedQuality(hierarchical_pooling=1.0)
    _feed_seeds(bsq, [(20, 100)] * 5)
    bsq.init_seed("fresh")
    a, b = bsq._get_pooled_params("fresh")
    assert a + b == pytest.approx(2.0 + 500.0)
    assert a / (a + b) == pytest.approx(0.2, abs=0.01)


@pytest.mark.parametrize(
    "rates",
    [
        [(0, 100), (0, 50)],  # mu = 0: dispersion undefined
        [(10, 100)],  # one seed: nothing to pool across
    ],
)
def test_adversarial_undefined_fit_disables_pooling(rates):
    bsq = BayesianSeedQuality(hierarchical_pooling=1.0)
    _feed_seeds(bsq, rates)
    hits, total = rates[0]
    assert bsq._get_pooled_params("s0") == (1.0 + hits, 1.0 + total - hits)


def test_regression_mc_pooling_concentration_is_bounded():
    mc = MonteCarloScheduler(arm_decay=1.0, hierarchical_pooling=1.0)
    rates = [(j, 50) for j in range(10)]
    for j, (hits, total) in enumerate(rates):
        mc.init_arm(f"op{j}")
        for k in range(total):
            mc.record(f"op{j}", success=k < hits)

    a, b = mc._get_effective_params("op0")

    succ = [float(h) for h, _ in rates]
    n = [float(t) for _, t in rates]
    assert (a, b) == pytest.approx(_expected(1.0, 51.0, succ, n, 1.0))


class _RecordingRng:
    """Captures betavariate_array's parameters; returns draws that pick *win*."""

    def __init__(self, win):
        self.win = win
        self.params = None

    def betavariate_array(self, alphas, betas):
        self.params = list(zip(alphas.tolist(), betas.tolist(), strict=True))
        draws = np.zeros(len(alphas))
        draws[self.win] = 1.0
        return draws


def test_pooled_select_draws_each_seeds_pooled_params():
    """The vectorised path must draw from exactly _get_pooled_params per
    seed; an unregistered candidate keeps the bare default prior."""
    bsq = BayesianSeedQuality(hierarchical_pooling=0.5)
    _feed_seeds(bsq, [(5, 50), (20, 50), (40, 50)])
    rng = _RecordingRng(win=2)
    bsq._rng = rng
    ids = ["s0", "s1", "ghost", "s2"]

    assert bsq.select_index(ids) == 2

    expect = [bsq._get_pooled_params(s) for s in ids]
    assert expect[2] == (1.0, 1.0)
    assert rng.params == pytest.approx(expect)
