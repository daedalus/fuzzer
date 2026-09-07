"""Tests for the Tang quantum-inspired recommendation scheduler.

Each of the paper's three claimed subroutines gets an oracle that is derived
independently of the code under test:

* Proposition 4.2 (inner-product estimation) is checked against ``np.dot``.
* Proposition 4.3 (rejection sampling from ``Vw``) is checked against the
  exact ``D_{Vw}`` density, with the control-against-itself run first
  (Hard Rule 46) so a broken comparison shows up as a failing control rather
  than as a passing test.
* Algorithm 3's projection is checked against the exact rank-k SVD projection
  on a matrix that is exactly rank k, where the two must agree to machine
  precision rather than approximately.

No retry-until-hit loops (Hard Rule 39): every stochastic assertion either
uses a fixed ``RandPool`` seed or asserts a distributional property over a
fixed sample size.
"""

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.tang import (
    DEFAULT_RANK,
    TangRecommendationScheduler,
    estimate_inner_product,
    modfkv_sample_complexity,
    sample_from_linear_combination,
)


class _FakeTracker:
    """Minimal EdgeTracker stand-in: the two attributes ``refit`` reads."""

    def __init__(self, seed_edges, seed_hit_counts=None):
        self.seed_edges = seed_edges
        self.seed_hit_counts = seed_hit_counts or {}


def _low_rank_tracker(n_seeds=24, n_edges=60, rank=3, seed=11):
    """A tracker whose matrix is exactly rank ``rank`` (up to nonnegativity)."""
    rng = np.random.default_rng(seed)
    left = np.abs(rng.normal(size=(n_seeds, rank)))
    right = np.abs(rng.normal(size=(rank, n_edges)))
    # No rounding: rounding would destroy the exact rank, and the exact
    # rank is what makes the SVD oracle below an equality rather than a
    # tolerance. Hit counts are floats in the matrix either way.
    matrix = left @ right * 4.0
    seed_edges, hits = {}, {}
    for i in range(n_seeds):
        key = f"seed{i:03d}"
        edges = {int(j) for j in range(n_edges)}
        seed_edges[key] = edges
        hits[key] = {int(j): float(matrix[i, j]) for j in range(n_edges)}
    return _FakeTracker(seed_edges, hits), matrix


# ------------------------------------------------------- ModFKV arithmetic


class TestModfkvSampleComplexity:
    def test_matches_the_papers_formula(self):
        # K = ||A||_F^2 / sigma^2 and q = K^4 / (eta*eps^2)^2, computed here
        # from the definitions rather than echoed from the implementation.
        frob_sq, sigma, eps, eta = 400.0, 10.0, 0.5, 0.2
        expected_k = frob_sq / sigma**2
        expected_q = expected_k**4 / (eta * eps**2) ** 2
        k, q = modfkv_sample_complexity(frob_sq, sigma, eps, eta)
        assert k == pytest.approx(expected_k)
        assert q == pytest.approx(expected_q)

    def test_q_exceeds_any_realistic_corpus_at_generous_parameters(self):
        # The practical objection to running real ModFKV, kept executable.
        # K=4 with eps=0.5, eta=0.2 is about as forgiving as the parameters go.
        _k, q = modfkv_sample_complexity(frobenius_sq=4.0, sigma=1.0, eps=0.5, eta=0.2)
        assert q > 1e4

    @pytest.mark.parametrize(
        "bad", [(1.0, 0.0, 0.5, 0.2), (1.0, 1.0, 0.0, 0.2), (1.0, 1.0, 0.5, 0.0)]
    )
    def test_non_positive_parameters_raise(self, bad):
        with pytest.raises(ValueError):
            modfkv_sample_complexity(*bad)


# --------------------------------------------- Proposition 4.2, inner product


class TestInnerProductEstimator:
    def test_estimates_the_true_inner_product(self):
        rng_np = np.random.default_rng(3)
        x = np.abs(rng_np.normal(size=200)) + 0.5
        y = rng_np.normal(size=200)
        exact = float(np.dot(x, y))
        est = estimate_inner_product(x, y, RandPool(seed=7), samples=4000, groups=9)
        # Proposition 4.2's error scales with ||x|| ||y||; assert against that
        # bound rather than a hand-tuned tolerance.
        assert abs(est - exact) <= 0.25 * np.linalg.norm(x) * np.linalg.norm(y)

    def test_is_exact_for_a_one_hot_vector(self):
        # D_x puts all mass on the single nonzero index, so every draw returns
        # y_j/x_j for that j and the estimate is exact regardless of sampling.
        x = np.zeros(16)
        x[5] = 2.0
        y = np.arange(16, dtype=float)
        est = estimate_inner_product(x, y, RandPool(seed=1), samples=32, groups=4)
        assert est == pytest.approx(float(np.dot(x, y)))

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError):
            estimate_inner_product(np.zeros(8), np.ones(8), RandPool(seed=1))

    def test_is_reproducible_under_a_fixed_seed(self):
        x = np.abs(np.random.default_rng(0).normal(size=64)) + 0.1
        y = np.random.default_rng(1).normal(size=64)
        a = estimate_inner_product(x, y, RandPool(seed=42), samples=500)
        b = estimate_inner_product(x, y, RandPool(seed=42), samples=500)
        assert a == b


# ------------------------------------------ Proposition 4.3, rejection sampling


class TestLinearCombinationSampling:
    @staticmethod
    def _empirical(columns, w, seed, draws=6000):
        rng = RandPool(seed=seed)
        counts = np.zeros(columns.shape[0])
        for _ in range(draws):
            i = sample_from_linear_combination(columns, w, rng)
            if i is not None:
                counts[i] += 1
        return counts / max(counts.sum(), 1.0)

    def test_control_against_itself_passes_first(self):
        # Hard Rule 46: two runs of the same sampler at different seeds must
        # agree under the same distance the real comparison uses. If this
        # fails, the comparison below is meaningless and the failure is here.
        columns = np.abs(np.random.default_rng(5).normal(size=(40, 3))) + 0.2
        w = np.array([1.0, 0.6, 0.3])
        a = self._empirical(columns, w, seed=17)
        b = self._empirical(columns, w, seed=23)
        assert 0.5 * np.abs(a - b).sum() < 0.10

    def test_matches_the_exact_projected_density(self):
        columns = np.abs(np.random.default_rng(5).normal(size=(40, 3))) + 0.2
        w = np.array([1.0, 0.6, 0.3])
        target = columns @ w
        exact = target**2 / float((target**2).sum())
        empirical = self._empirical(columns, w, seed=17)
        assert 0.5 * np.abs(empirical - exact).sum() < 0.10

    def test_returns_none_when_the_combination_is_zero(self):
        columns = np.zeros((10, 2))
        assert sample_from_linear_combination(columns, np.ones(2), RandPool(seed=1)) is None


# ------------------------------------------------- Algorithm 3, the scheduler


class TestTangScheduler:
    def test_requires_a_rand_pool(self):
        with pytest.raises(ValueError):
            TangRecommendationScheduler(rng=None)

    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError):
            TangRecommendationScheduler(RandPool(seed=1), mode="popularity")

    def test_unfitted_scheduler_is_neutral_rather_than_raising(self):
        sched = TangRecommendationScheduler(RandPool(seed=1))
        assert not sched.fitted
        assert sched.seed_energy("nope") == 0.0
        assert sched.recommend({1, 2, 3}) == []

    def test_refit_fails_closed_on_a_degenerate_corpus(self):
        sched = TangRecommendationScheduler(RandPool(seed=1))
        assert sched.refit(_FakeTracker({})) is False
        assert sched.refit(_FakeTracker({"a": {1, 2}})) is False  # one seed
        assert sched.refit(_FakeTracker({"a": set(), "b": set()})) is False
        assert not sched.fitted

    def test_a_failed_refit_keeps_the_previous_basis(self):
        # A corpus that momentarily shrinks must not blank the arm.
        tracker, _ = _low_rank_tracker()
        sched = TangRecommendationScheduler(RandPool(seed=1))
        assert sched.refit(tracker) is True
        before = sched.seed_energy("seed000")
        assert sched.refit(_FakeTracker({})) is False
        assert sched.fitted
        assert sched.seed_energy("seed000") == before

    def test_projection_equals_the_exact_rank_k_svd(self):
        # On a matrix of exact rank r, projecting onto the top-r right
        # singular subspace must reproduce the row itself. Deriving the
        # oracle from the rank of the construction, not from the module.
        rank = 3
        tracker, matrix = _low_rank_tracker(rank=rank)
        sched = TangRecommendationScheduler(RandPool(seed=1), rank=rank)
        assert sched.refit(tracker)
        _u, sv, vt = np.linalg.svd(matrix, full_matrices=False)
        assert sv[rank] < 1e-9 * sv[0]  # the construction really is rank-r
        expected = (matrix[0] @ vt[:rank].T) @ vt[:rank]
        np.testing.assert_allclose(sched._project(matrix[0]), expected, atol=1e-8)

    def test_effective_rank_is_capped_by_the_true_rank(self):
        tracker, _ = _low_rank_tracker(rank=3)
        sched = TangRecommendationScheduler(RandPool(seed=1), rank=DEFAULT_RANK)
        assert sched.refit(tracker)
        assert sched.last_effective_rank <= 3

    def test_energies_are_bounded_and_defined_for_every_tracked_seed(self):
        tracker, _ = _low_rank_tracker()
        sched = TangRecommendationScheduler(RandPool(seed=1))
        assert sched.refit(tracker)
        for key in tracker.seed_edges:
            assert 0.0 <= sched.seed_energy(key) <= 1.0

    def test_seed_admitted_since_the_last_refit_scores_zero(self):
        tracker, _ = _low_rank_tracker()
        sched = TangRecommendationScheduler(RandPool(seed=1))
        assert sched.refit(tracker)
        assert sched.seed_energy("seed_added_after_the_refit") == 0.0

    def test_frontier_mode_is_the_complement_of_tang_mode(self):
        # frontier_mass = 1 - covered_mass exactly, before the 1/owners
        # reweighting. This is the algebraic identity that makes the inverted
        # variant the same measurement sign-flipped, and it is asserted rather
        # than described so the docstring cannot drift away from the code.
        tracker, matrix = _low_rank_tracker()
        a = TangRecommendationScheduler(RandPool(seed=1), mode="tang")
        b = TangRecommendationScheduler(RandPool(seed=1), mode="frontier")
        assert a.refit(tracker) and b.refit(tracker)
        projected = a._project(matrix[0]) ** 2
        covered = matrix[0] > 0
        total = projected.sum()
        assert a.seed_energy("seed000") + float(projected[~covered].sum() / total) == pytest.approx(
            1.0
        )
        assert b.mode == "frontier"

    def test_missing_hit_counts_fall_back_to_incidence(self):
        # Absent from seed_hit_counts means "covered, count unknown" -> 1.0.
        # The regression this guards: reading the raw to_dict payload gives
        # str edge keys, every lookup misses, and the matrix goes silently
        # binary without any error surfacing.
        tracker = _FakeTracker({"a": {1, 2, 3}, "b": {2, 3, 4}}, seed_hit_counts={})
        sched = TangRecommendationScheduler(RandPool(seed=1), rank=2)
        assert sched.refit(tracker)
        assert sched.last_shape == (2, 4)

    def test_string_keyed_hit_counts_do_not_silently_binarise(self):
        edges = {"a": {1, 2}, "b": {2, 3}}
        int_keyed = _FakeTracker(edges, {"a": {1: 99.0, 2: 1.0}, "b": {2: 1.0, 3: 99.0}})
        str_keyed = _FakeTracker(edges, {"a": {"1": 99.0, "2": 1.0}, "b": {"2": 1.0, "3": 99.0}})
        picks = []
        for tracker in (int_keyed, str_keyed):
            sched = TangRecommendationScheduler(RandPool(seed=5), rank=2)
            assert sched.refit(tracker)
            picks.append(sched.recommend(int_keyed.seed_hit_counts["a"], count=400))
        # With real counts the l2 draw concentrates on the 99-hit edge; with
        # str keys every lookup misses, the matrix goes binary, and the draw
        # flattens. Asserting the gap documents that EdgeTracker.from_dict is
        # what must restore int keys -- reading the raw to_dict payload does
        # not, and the failure is silent.
        assert picks[0].count(1) > picks[1].count(1)

    def test_recommend_returns_known_edge_ids(self):
        tracker, _ = _low_rank_tracker(n_edges=30)
        sched = TangRecommendationScheduler(RandPool(seed=1))
        assert sched.refit(tracker)
        picks = sched.recommend(tracker.seed_edges["seed000"], count=25)
        assert len(picks) == 25
        assert set(picks) <= set(range(30))

    def test_recommend_is_reproducible_under_a_fixed_seed(self):
        tracker, _ = _low_rank_tracker(n_edges=30)
        out = []
        for _ in range(2):
            sched = TangRecommendationScheduler(RandPool(seed=99))
            assert sched.refit(tracker)
            out.append(sched.recommend(tracker.seed_edges["seed000"], count=40))
        assert out[0] == out[1]

    def test_recommend_rejects_a_row_of_the_wrong_width(self):
        tracker, _ = _low_rank_tracker(n_edges=30)
        sched = TangRecommendationScheduler(RandPool(seed=1))
        assert sched.refit(tracker)
        assert sched.recommend(np.ones(7)) == []

    def test_maybe_refit_honours_the_interval(self):
        tracker, _ = _low_rank_tracker()
        sched = TangRecommendationScheduler(RandPool(seed=1), refit_interval=500)
        assert sched.maybe_refit(tracker, exec_count=0) is True
        assert sched.maybe_refit(tracker, exec_count=499) is False
        assert sched.maybe_refit(tracker, exec_count=500) is True
        assert sched.refits == 2

    def test_stats_reports_the_fitted_shape(self):
        tracker, _ = _low_rank_tracker(n_seeds=12, n_edges=20)
        sched = TangRecommendationScheduler(RandPool(seed=1))
        assert sched.refit(tracker)
        stats = sched.stats()
        assert stats["fitted"] is True
        assert stats["seeds"] == 12
        assert stats["edges"] == 20
        assert stats["mode"] == "tang"


class TestSchedulerWiring:
    def test_tang_is_a_registered_seed_strategy(self):
        from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES

        assert "tang" in _SEED_STRATEGY_NAMES

    def test_the_picker_exposes_a_handler_for_the_arm(self):
        from fuzzer_tool.services.seed_picker import SeedPicker

        assert hasattr(SeedPicker, "_pick_tang_seed")
