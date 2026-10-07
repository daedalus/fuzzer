"""core/blup.py: empirical-Bayes prior strength from between-unit dispersion."""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.core.blup import MIN_STRENGTH, PoolCache, fit_groups, fit_pool
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.blup_ref import ref_fit, ref_strength

M_MAX = 500.0


def _beta_binomial(rng: RandPool, units: int, n: int, mu: float, m: float):
    """Synthetic units: p_i ~ Beta(m mu, m (1 - mu)), s_i ~ Bin(n, p_i)."""
    succ = []
    for _ in range(units):
        p = rng.betavariate(m * mu, m * (1.0 - mu))
        succ.append(float(sum(rng.random() < p for _ in range(n))))
    return np.array(succ), np.full(units, float(n))


def test_matches_reference_derivation():
    succ = np.array([3.0, 0.5, 7.0, 12.0, 0.0])
    n = np.array([10.0, 4.0, 20.0, 30.0, 6.0])
    mu, rho = ref_fit(list(succ), list(n))
    expect_m = ref_strength(rho, n.sum(), M_MAX, MIN_STRENGTH)

    got = fit_pool(succ, n, M_MAX)

    assert got is not None
    assert got[0] == pytest.approx(mu)
    assert got[1] == pytest.approx(expect_m)


def test_groups_match_per_group_fit():
    """Vectorised grouping equals fitting each group alone; empty and
    single-unit groups report not ok."""
    succ = np.array([3.0, 0.5, 7.0, 12.0, 0.0, 1.0, 9.0, 2.0])
    n = np.array([10.0, 4.0, 20.0, 30.0, 6.0, 5.0, 10.0, 8.0])
    groups = np.array([0, 0, 0, 1, 1, 1, 1, 3])

    fit = fit_groups(succ, n, groups, 4, M_MAX)

    for g in range(4):
        mask = groups == g
        alone = fit_pool(succ[mask], n[mask], M_MAX)
        assert bool(fit.ok[g]) == (alone is not None)
        if alone is None:
            continue
        assert fit.mu[g] == pytest.approx(alone[0])
        assert fit.m[g] == pytest.approx(alone[1])


def test_recovers_known_strength():
    """400 beta-binomial units with m = 10: the fit lands near 10."""
    succ, n = _beta_binomial(RandPool(7), 400, 50, 0.3, 10.0)
    mu, m = fit_pool(succ, n, M_MAX)
    assert mu == pytest.approx(0.3, abs=0.03)
    assert 7.0 < m < 14.0


def test_falsify_identical_units_pool_fully():
    """Falsification: units with exactly the same rate carry no
    between-unit variance, so the strength hits its ceiling."""
    n = np.array([100.0, 200.0, 50.0, 400.0])
    _, m = fit_pool(0.2 * n, n, M_MAX)
    assert m == M_MAX


def test_adversarial_strength_never_exceeds_evidence():
    """Decayed, tiny evidence shows no dispersion; the prior is still
    capped at the population's own total, not at m_max."""
    n = np.array([4.7, 5.3])
    _, m = fit_pool(np.array([0.0, 0.66]), n, M_MAX)
    assert m == pytest.approx(n.sum())


def test_all_or_nothing_units_pool_least():
    """Every unit is 0% or 100%: rho = 1, no shrinkage beyond the floor."""
    n = np.array([10.0, 10.0, 10.0, 10.0])
    _, m = fit_pool(np.array([0.0, 10.0, 10.0, 0.0]), n, M_MAX)
    assert m == MIN_STRENGTH


@pytest.mark.parametrize(
    ("succ", "n"),
    [
        ([], []),
        ([3.0], [10.0]),  # one unit: no between-unit variance
        ([0.0, 0.0], [10.0, 5.0]),  # mu = 0: dispersion undefined
        ([4.0, 9.0], [4.0, 9.0]),  # mu = 1
        ([1.0, 0.0, 1.0], [1.0, 1.0, 1.0]),  # n = 1 units: E[S] flat in rho
        ([0.0, 0.0, 2.0], [0.0, 0.0, 5.0]),  # zero-n units are ignored
    ],
)
def test_adversarial_undefined_fits(succ, n):
    assert fit_pool(np.array(succ, dtype=float), np.array(n, dtype=float), M_MAX) is None


def test_adversarial_outlier_lowers_strength():
    """One strong unit among weak ones must weaken pooling, not be crushed."""
    n = np.full(6, 200.0)
    flat = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    outlier = flat.copy()
    outlier[-1] = 100.0
    _, m_flat = fit_pool(flat, n, M_MAX)
    _, m_out = fit_pool(outlier, n, M_MAX)
    shrunk = (outlier[-1] + m_out * outlier.sum() / n.sum()) / (n[-1] + m_out)
    assert m_out < m_flat
    assert shrunk == pytest.approx(0.5, abs=0.05)


def test_cache_refits_on_cadence():
    calls = []

    def evidence():
        calls.append(1)
        return np.array([2.0, 8.0]), np.array([10.0, 10.0])

    cache = PoolCache(M_MAX, refit_every=10)
    first = cache.prior(0, evidence)
    cache.prior(9, evidence)
    assert len(calls) == 1
    assert cache.prior(10, evidence) == first
    assert len(calls) == 2
