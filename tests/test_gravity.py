"""Gravity-model splice donor weighting and its online PPML fit (core/gravity.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.gravity import (
    MIN_POSITIVES,
    SOFTENING,
    WINDOW,
    GravityModel,
    fit_ppml,
    pair_terms,
    pick_index,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

# Ground truth for the synthetic PPML recovery tests: small enough that the
# largest Poisson mean stays ~15 and Knuth sampling stays exact and fast.
_TRUE = (-4.0, 0.2, 0.5, 1.0)
_FIT_TOL = 0.15
_N_OBS = 3000


def _poisson(rng: RandPool, mu: float) -> int:
    """Knuth sampler: exact for the small means these tests use."""
    limit = math.exp(-mu)
    k, p = 0, rng.random()
    while p > limit:
        k += 1
        p *= rng.random()
    return k


def _synthetic(theta, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows (1, log1p m_i, log1p m_j, -½log(d²+ε²)) with Poisson(exp(x·θ)) yields."""
    rng = RandPool(seed=seed)
    rows = np.empty((_N_OBS, 4))
    ys = np.empty(_N_OBS)
    for n in range(_N_OBS):
        m_i, m_j, d = rng.randint(0, 200), rng.randint(0, 200), rng.random()
        x = GravityModel.features(m_i, m_j, d)
        rows[n] = x
        ys[n] = _poisson(rng, math.exp(float(np.dot(x, theta))))
    return rows, ys


class TestPairTerms:
    def test_identical_sets_have_no_mass_and_no_distance(self):
        assert pair_terms(40, 40, 1.0) == (0.0, 0.0, 0.0)

    def test_disjoint_sets_keep_full_mass(self):
        assert pair_terms(30, 50, 0.0) == (30.0, 50.0, 1.0)

    def test_complement_from_jaccard(self):
        # |A|=|B|=60, |A∩B|=20 → |A∪B|=100, J=0.2, each side owns 40 the other lacks.
        m_i, m_j, d = pair_terms(60, 60, 0.2)
        assert (m_i, m_j) == pytest.approx((40.0, 40.0))
        assert d == pytest.approx(0.8)

    def test_adversarial_inconsistent_estimate_clamps_to_zero(self):
        # MinHash can report J=1 for sets of different size; mass never goes negative.
        m_i, m_j, _ = pair_terms(10, 90, 1.0)
        assert m_i == 0.0 and m_j >= 0.0


class TestWeight:
    def test_falsification_zero_exponents_is_uniform(self):
        g = GravityModel(alpha=0.0, beta=0.0, gamma=0.0)
        ws = {g.weight(m_i, m_j, d) for m_i, m_j, d in [(1, 5, 0.1), (90, 1, 0.9), (0, 300, 0.5)]}
        assert ws == {1.0}

    def test_adversarial_identical_donor_is_inert(self):
        # A donor with nothing the base lacks gets weight 0, even at d=0 where
        # an unsoftened 1/d^γ would make it the heaviest body in the corpus.
        g = GravityModel(alpha=1.0, beta=1.0, gamma=4.0)
        assert g.weight(*pair_terms(50, 50, 1.0)) == 0.0

    def test_softening_keeps_zero_distance_finite(self):
        g = GravityModel(alpha=0.0, beta=0.0, gamma=2.0)
        assert g.weight(0, 3, 0.0) == pytest.approx(1.0 / SOFTENING**2)

    def test_closer_and_heavier_donors_attract_more(self):
        g = GravityModel(alpha=0.0, beta=1.0, gamma=1.0)
        assert g.weight(0, 10, 0.2) > g.weight(0, 10, 0.8)
        assert g.weight(0, 50, 0.5) > g.weight(0, 5, 0.5)


class TestPickIndex:
    def test_scripted_draw_selects_cdf_bucket(self):
        # weights 1,3 → CDF 0.25, 1.0; r=0.5 lands in bucket 1.
        assert pick_index([1.0, 3.0], ScriptedRng(randoms=(0.5,))) == 1
        assert pick_index([1.0, 3.0], ScriptedRng(randoms=(0.1,))) == 0

    def test_zero_mass_returns_sentinel_without_drawing(self):
        # Empty ScriptedRng: any draw would raise StopIteration.
        assert pick_index([0.0, 0.0], ScriptedRng()) == -1
        assert pick_index([], ScriptedRng()) == -1


class TestFitPpml:
    def test_recovers_known_exponents(self):
        rows, ys = _synthetic(_TRUE, seed=1)
        theta = fit_ppml(rows, ys, np.zeros(4))
        assert theta is not None
        assert theta == pytest.approx(np.array(_TRUE), abs=_FIT_TOL)

    def test_control_two_samples_of_same_process_agree(self):
        # Hard Rule 46: the oracle must pass against itself first.
        a = fit_ppml(*_synthetic(_TRUE, seed=2), np.zeros(4))
        b = fit_ppml(*_synthetic(_TRUE, seed=3), np.zeros(4))
        assert a is not None and b is not None
        assert a == pytest.approx(b, abs=2 * _FIT_TOL)

    def test_falsification_no_distance_effect_fits_gamma_zero(self):
        truth = (_TRUE[0], _TRUE[1], _TRUE[2], 0.0)
        theta = fit_ppml(*_synthetic(truth, seed=4), np.zeros(4))
        assert theta is not None
        assert abs(theta[3]) < _FIT_TOL

    def test_adversarial_sparse_positives_decline(self):
        # A handful of hits is quasi-separation, not signal: on fuzzgoat 6 hits
        # in 512 rows drove γ to the clamp. Below the gate the fit declines.
        rows, _ = _synthetic(_TRUE, seed=7)
        ys = np.zeros(len(rows))
        ys[: MIN_POSITIVES - 1] = 2.0
        assert fit_ppml(rows, ys, np.zeros(4)) is None
        ys[MIN_POSITIVES - 1] = 2.0
        assert fit_ppml(rows, ys, np.zeros(4)) is not None

    def test_adversarial_all_zero_yield_declines(self):
        rows, _ = _synthetic(_TRUE, seed=5)
        assert fit_ppml(rows, np.zeros(len(rows)), np.zeros(4)) is None


class TestOnline:
    def test_observe_splits_yield_across_staged_pairs(self):
        g = GravityModel()
        g.stage(1, 2, 0.5)
        g.stage(3, 4, 0.25)
        g.observe(6)
        assert g.summary()["observations"] == 2
        assert g.pending == 0

    def test_summary_counts_positive_rows(self):
        g = GravityModel()
        for y in (0, 3, 0, 1):
            g.stage(1, 2, 0.5)
            g.observe(y)
        assert g.summary()["positives"] == 2

    def test_observe_without_stage_is_noop(self):
        g = GravityModel()
        g.observe(9)
        assert g.summary()["observations"] == 0

    def test_discard_drops_unobserved_pairs(self):
        g = GravityModel()
        g.stage(1, 2, 0.5)
        g.discard()
        g.observe(3)
        assert g.summary()["observations"] == 0

    def test_adversarial_memory_bounded(self):
        g = GravityModel()
        for n in range(10 * WINDOW):
            g.stage(n % 7, n % 11, (n % 13) / 13)
            g.observe(n % 3)
        assert g.summary()["observations"] == WINDOW

    def test_adversarial_stage_flood_bounded(self):
        g = GravityModel()
        for _ in range(10 * WINDOW):
            g.stage(1, 1, 0.5)
        assert g.pending <= WINDOW

    def test_online_refit_moves_exponents_toward_truth(self):
        rows, ys = _synthetic(_TRUE, seed=6)
        g = GravityModel(alpha=1.0, beta=1.0, gamma=1.0)
        for x, y in zip(rows, ys, strict=True):
            g.stage_features(x)
            g.observe(int(y))
        alpha, beta, _ = g.exponents
        # Starting exponents are 1.0; truth is α=0.2, β=0.5.
        assert abs(alpha - _TRUE[1]) < 0.3
        assert abs(beta - _TRUE[2]) < 0.3

    def test_state_round_trip(self):
        g = GravityModel(alpha=0.3, beta=0.7, gamma=1.9)
        g.stage(1, 2, 0.5)
        g.observe(4)
        h = GravityModel()
        h.load_state_dict(g.state_dict())
        assert h.exponents == g.exponents
        assert h.summary() == g.summary()

    def test_adversarial_state_missing_keys_keeps_defaults(self):
        h = GravityModel()
        h.load_state_dict({})
        assert h.exponents == GravityModel().exponents

    def test_adversarial_wrapped_ring_round_trips_in_order(self):
        g = GravityModel()
        extra = 5
        for n in range(WINDOW + extra):
            g.stage_features(np.array([1.0, n, 0.0, 0.0]))
            g.observe(n)
        state = g.state_dict()
        # Oldest surviving row is n=extra, newest n=WINDOW+extra-1.
        assert state["ys"][0] == extra
        assert state["ys"][-1] == WINDOW + extra - 1
        h = GravityModel()
        h.load_state_dict(state)
        assert h.state_dict()["ys"] == state["ys"]
