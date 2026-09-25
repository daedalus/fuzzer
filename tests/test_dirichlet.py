"""Tests for core/dirichlet.py and RandPool.dirichlet."""

import math

import numpy as np
import pytest

from fuzzer_tool.core.dirichlet import (
    DirichletPicker,
    _digamma_arr,
    _histograms,
    digamma,
    dm_alpha,
)
from fuzzer_tool.core.rand_pool import RandPool

EULER_GAMMA = 0.5772156649015329
BYTE_VALUES = 256


def _dm_rows(alpha: float, n_ctx: int, n_per_ctx: int, seed: int) -> list[list[int]]:
    """Counts rows drawn from a symmetric Dirichlet-Multinomial with known alpha."""
    g = np.random.default_rng(seed)
    rows = []
    for _ in range(n_ctx):
        p = g.dirichlet(np.full(BYTE_VALUES, alpha))
        counts = g.multinomial(n_per_ctx, p)
        rows.append([int(c) for c in counts if c])
    return rows


# ── digamma ─────────────────────────────────────────────────────────────


class TestDigamma:
    def test_known_values(self):
        assert digamma(1.0) == pytest.approx(-EULER_GAMMA, abs=1e-12)
        assert digamma(0.5) == pytest.approx(-EULER_GAMMA - 2 * math.log(2), abs=1e-12)

    def test_recurrence(self):
        # psi(x+1) = psi(x) + 1/x, checked over small, mid and large x
        for x in (1e-4, 0.3, 2.5, 7.0, 1e5):
            assert digamma(x + 1) == pytest.approx(digamma(x) + 1 / x, rel=1e-10)

    def test_matches_lgamma_derivative(self):
        # Oracle independent of the code under test: central difference of lgamma
        h = 1e-5
        for x in (0.05, 0.9, 3.3, 40.0):
            deriv = (math.lgamma(x + h) - math.lgamma(x - h)) / (2 * h)
            assert digamma(x) == pytest.approx(deriv, rel=1e-6)

    def test_adversarial_non_positive(self):
        with pytest.raises(ValueError):
            digamma(0.0)
        with pytest.raises(ValueError):
            digamma(-1.0)

    def test_vectorised_matches_scalar(self):
        xs = np.array([1e-6, 0.01, 0.7, 5.9, 6.0, 12.5, 3e4])
        expected = np.array([digamma(float(x)) for x in xs])
        assert np.allclose(_digamma_arr(xs), expected, rtol=1e-13, atol=0)


class TestHistograms:
    def test_example(self):
        cv, cm, tv, tm = _histograms([[3, 1], [3]])
        assert dict(zip(cv.tolist(), cm.tolist(), strict=True)) == {1: 1, 3: 2}
        assert dict(zip(tv.tolist(), tm.tolist(), strict=True)) == {3: 1, 4: 1}

    def test_adversarial_empty_and_zero_rows(self):
        """Empty, all-zero and single-observation rows add nothing."""
        cv, cm, tv, tm = _histograms([[], [3, 0, 1], [], [0, 0], [1], [3], []])
        assert dict(zip(cv.tolist(), cm.tolist(), strict=True)) == {1: 1, 3: 2}
        assert dict(zip(tv.tolist(), tm.tolist(), strict=True)) == {3: 1, 4: 1}


# ── dm_alpha ────────────────────────────────────────────────────────────


class TestDmAlpha:
    def test_control_two_samples_agree(self):
        """Control (Hard Rule 46): two samples of the same alpha agree first."""
        a = dm_alpha(_dm_rows(0.1, 300, 40, seed=1), BYTE_VALUES)
        b = dm_alpha(_dm_rows(0.1, 300, 40, seed=2), BYTE_VALUES)
        assert a == pytest.approx(b, rel=0.2)

    @pytest.mark.parametrize("true_alpha", [0.02, 0.1, 1.0])
    def test_recovers_known_alpha(self, true_alpha):
        est = dm_alpha(_dm_rows(true_alpha, 300, 40, seed=7), BYTE_VALUES)
        assert est == pytest.approx(true_alpha, rel=0.2)

    def test_falsification_ordering(self):
        """Peaked data must yield a smaller alpha than flat data."""
        peaked = dm_alpha(_dm_rows(0.01, 200, 40, seed=3), BYTE_VALUES)
        flat = dm_alpha(_dm_rows(5.0, 200, 40, seed=3), BYTE_VALUES)
        assert peaked < flat

    def test_initial_alpha_irrelevant(self):
        rows = _dm_rows(0.3, 200, 40, seed=4)
        assert dm_alpha(rows, BYTE_VALUES, alpha=1e-3) == pytest.approx(
            dm_alpha(rows, BYTE_VALUES, alpha=50.0), rel=1e-3
        )

    def test_adversarial_empty_returns_initial(self):
        assert dm_alpha([], BYTE_VALUES, alpha=0.7) == 0.7
        assert dm_alpha([[], [0, 0]], BYTE_VALUES, alpha=0.7) == 0.7

    def test_adversarial_deterministic_contexts_clamp_low(self):
        """One byte per context: MLE alpha -> 0, clamped to a finite floor."""
        est = dm_alpha([[50]] * 100, BYTE_VALUES)
        assert 0.0 < est < 1e-3
        assert math.isfinite(est)

    def test_adversarial_all_singletons_clamp_high(self):
        """Every observation distinct: MLE alpha -> inf, clamped to a finite cap."""
        est = dm_alpha([[1] * 40] * 50, BYTE_VALUES)
        assert est > 1.0
        assert math.isfinite(est)

    def test_adversarial_bad_initial_alpha(self):
        rows = _dm_rows(0.3, 50, 40, seed=5)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            assert math.isfinite(dm_alpha(rows, BYTE_VALUES, alpha=bad))

    def test_adversarial_single_category(self):
        """k=1 has no free parameter; return the initial alpha unchanged."""
        assert dm_alpha([[10], [3]], 1, alpha=0.4) == 0.4


# ── RandPool.dirichlet ──────────────────────────────────────────────────


class TestRandPoolDirichlet:
    def test_on_simplex(self):
        p = RandPool(seed=1).dirichlet([0.5, 1.0, 2.0, 3.0])
        assert p.shape == (4,)
        assert (p >= 0).all()
        assert p.sum() == pytest.approx(1.0)

    def test_seed_reproducible(self):
        a = RandPool(seed=9).dirichlet([1.0] * 8)
        b = RandPool(seed=9).dirichlet([1.0] * 8)
        assert np.array_equal(a, b)

    def test_mean_matches_normalized_alpha(self):
        alphas = np.array([1.0, 2.0, 7.0])
        pool = RandPool(seed=3)
        draws = np.array([pool.dirichlet(alphas) for _ in range(4000)])
        assert draws.mean(axis=0) == pytest.approx(alphas / alphas.sum(), abs=0.02)

    def test_adversarial_tiny_alpha_no_nan(self):
        p = RandPool(seed=2).dirichlet([1e-4] * BYTE_VALUES)
        assert not np.isnan(p).any()
        assert p.sum() == pytest.approx(1.0)

    def test_adversarial_invalid_alpha_raises(self):
        pool = RandPool(seed=2)
        for bad in ([], [1.0, 0.0], [1.0, -2.0], [1.0, float("nan")], [float("inf")]):
            with pytest.raises(ValueError):
                pool.dirichlet(bad)


# ── DirichletPicker ─────────────────────────────────────────────────────


class _CountingPool(RandPool):
    """RandPool that counts dirichlet() calls (one per round expected)."""

    def __init__(self, seed):
        super().__init__(seed=seed)
        self.dirichlet_calls = 0

    def dirichlet(self, alphas):
        self.dirichlet_calls += 1
        return super().dirichlet(alphas)


class TestDirichletPicker:
    TOKENS = [b"GET", b"POST", b"HEAD", b"PUT"]

    def test_one_simplex_draw_per_round(self):
        """Thompson: p is sampled once per round, not once per token draw."""
        pool = _CountingPool(seed=1)
        picker = DirichletPicker(pool)
        idx = picker.draw(self.TOKENS, 64)
        assert len(idx) == 64
        assert all(0 <= i < len(self.TOKENS) for i in idx)
        assert pool.dirichlet_calls == 1

    def test_rewarded_token_dominates(self):
        """Posterior mean of GET after 200 wins is 201/204; draws follow it."""
        picker = DirichletPicker(RandPool(seed=2))
        for _ in range(200):
            picker.draw(self.TOKENS, 1)
            picker._pending = [0]  # the round used GET
            picker.reward(1)
        idx = picker.draw(self.TOKENS, 1000)
        assert idx.count(0) > 900

    def test_falsification_no_reward_stays_spread(self):
        """Control: with no wins every token keeps real mass."""
        picker = DirichletPicker(RandPool(seed=3))
        counts = [0] * len(self.TOKENS)
        for _ in range(200):
            for i in picker.draw(self.TOKENS, 4):
                counts[i] += 1
            picker.clear()
        assert min(counts) > 100

    def test_reward_credits_only_used_prefix(self):
        picker = DirichletPicker(RandPool(seed=4))
        picker.draw(self.TOKENS, 8)
        picker._pending = [1, 1, 3, 0, 0, 0, 0, 0]
        picker.reward(3)
        assert picker.wins == {b"POST": 1, b"PUT": 1}

    def test_adversarial_dictionary_truncated_between_draw_and_reward(self):
        """fuzzer.py truncates with dictionary[-keep:]; credit follows the bytes."""
        picker = DirichletPicker(RandPool(seed=5))
        tokens = list(self.TOKENS)
        picker.draw(tokens, 4)
        picker._pending = [3, 3, 3, 3]
        tokens = tokens[-2:]  # indices shift: PUT is now index 1
        picker.reward(4)
        assert picker.wins == {b"PUT": 1}
        idx = picker.draw(tokens, 500)
        assert idx.count(1) > idx.count(0)

    def test_adversarial_reward_without_pending_is_noop(self):
        picker = DirichletPicker(RandPool(seed=6))
        picker.reward(5)
        picker.draw(self.TOKENS, 4)
        picker.clear()
        picker.reward(5)
        assert picker.wins == {}

    def test_adversarial_used_beyond_drawn_is_clamped(self):
        picker = DirichletPicker(RandPool(seed=7))
        picker.draw(self.TOKENS, 2)
        picker._pending = [2, 2]
        picker.reward(99)
        assert picker.wins == {b"HEAD": 1}

    def test_adversarial_empty_dictionary(self):
        picker = DirichletPicker(RandPool(seed=8))
        assert picker.draw([], 16) == []
        picker.reward(16)
        assert picker.wins == {}


class TestCategorical:
    def test_zero_weight_never_drawn(self):
        idx = RandPool(seed=1).categorical(np.array([0.0, 1.0, 0.0, 3.0]), 2000)
        assert set(idx) <= {1, 3}
        assert idx.count(3) > idx.count(1)

    def test_frequencies_match_weights(self):
        w = np.array([1.0, 2.0, 7.0])
        idx = RandPool(seed=2).categorical(w, 20000)
        freq = np.bincount(idx, minlength=3) / 20000
        assert freq == pytest.approx(w / w.sum(), abs=0.015)

    def test_adversarial_count_zero_and_single(self):
        pool = RandPool(seed=3)
        assert pool.categorical(np.array([1.0]), 0) == []
        assert pool.categorical(np.array([5.0]), 4) == [0, 0, 0, 0]
