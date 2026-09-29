"""Tests for core/group_testing.py -- non-adaptive pooled which-items-matter inference."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core import group_testing as gt


def _oracle(defective: set[int]):
    """Positive iff the pool contains at least one defective item."""
    calls: list[frozenset[int]] = []

    def run(pool: frozenset[int]) -> bool:
        calls.append(pool)
        return bool(pool & defective)

    run.calls = calls  # type: ignore[attr-defined]
    return run


class TestDesign:
    def test_shape(self):
        pools = gt.design(100, 7, d=3, rng=random.Random(1))
        assert len(pools) == 7
        assert all(p <= set(range(100)) for p in pools)

    def test_density_near_one_over_d_plus_one(self):
        n, d = 2000, 4
        pools = gt.design(n, 40, d=d, rng=random.Random(2))
        mean = sum(len(p) for p in pools) / (40 * n)
        assert abs(mean - 1 / (d + 1)) < 0.02

    def test_reproducible_under_seed(self):
        a = gt.design(50, 10, d=2, rng=random.Random(9))
        b = gt.design(50, 10, d=2, rng=random.Random(9))
        assert a == b

    def test_tests_needed_grows_with_d_and_n(self):
        assert gt.tests_needed(1000, 4) > gt.tests_needed(1000, 1)
        assert gt.tests_needed(100_000, 2) > gt.tests_needed(100, 2)

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            gt.design(0, 5, d=1, rng=random.Random(0))
        with pytest.raises(ValueError):
            gt.design(10, 0, d=1, rng=random.Random(0))
        with pytest.raises(ValueError):
            gt.design(10, 5, d=0, rng=random.Random(0))


class TestComp:
    def test_keeps_all_defectives(self):
        n = 64
        pools = gt.design(n, 40, d=2, rng=random.Random(3))
        defective = {5, 40}
        out = [bool(p & defective) for p in pools]
        assert defective <= gt.comp(n, pools, out)

    def test_removes_items_in_negative_pools(self):
        assert gt.comp(4, [frozenset({0, 1}), frozenset({1, 2})], [False, True]) == {2, 3}

    def test_all_positive_keeps_everything(self):
        assert gt.comp(3, [frozenset({0}), frozenset({1})], [True, True]) == {0, 1, 2}


class TestIdentify:
    @pytest.mark.parametrize("seed", range(20))
    def test_exact_for_random_defective_sets(self, seed):
        rng = random.Random(seed)
        n = 200
        d = rng.randint(1, 5)
        defective = set(rng.sample(range(n), d))
        res = gt.identify(n, _oracle(defective), d=d, rng=random.Random(seed + 100))
        assert res.defective == defective

    def test_no_defectives(self):
        res = gt.identify(50, _oracle(set()), d=2, rng=random.Random(0))
        assert res.defective == set()

    def test_all_defective(self):
        n = 12
        res = gt.identify(n, _oracle(set(range(n))), d=n, rng=random.Random(0))
        assert res.defective == set(range(n))

    def test_single_item(self):
        assert gt.identify(1, _oracle({0}), d=1, rng=random.Random(0)).defective == {0}
        assert gt.identify(1, _oracle(set()), d=1, rng=random.Random(0)).defective == set()

    def test_underestimated_d_stays_exact_but_costs_more(self):
        # Adversarial: true d=8, told d=1. Correctness must not depend on d.
        rng = random.Random(4)
        n = 300
        defective = set(rng.sample(range(n), 8))
        good = gt.identify(n, _oracle(defective), d=8, rng=random.Random(5))
        bad = gt.identify(n, _oracle(defective), d=1, rng=random.Random(5))
        assert good.defective == bad.defective == defective
        assert bad.tests > good.tests

    def test_counts_match_oracle_calls(self):
        defective = {3, 77}
        orc = _oracle(defective)
        res = gt.identify(100, orc, d=2, rng=random.Random(6))
        assert res.tests == len(orc.calls)

    def test_first_round_is_non_adaptive(self):
        # Pools of round 1 must not depend on any answer: same design
        # whatever the defective set is.
        a, b = _oracle({1}), _oracle({90})
        gt.identify(100, a, d=1, rng=random.Random(7))
        gt.identify(100, b, d=1, rng=random.Random(7))
        t = gt.tests_needed(100, 1)
        assert a.calls[:t] == b.calls[:t]

    def test_far_fewer_tests_than_individual_for_sparse(self):
        n = 2000
        res = gt.identify(n, _oracle({17, 1234}), d=2, rng=random.Random(8))
        assert res.tests < n // 4

    def test_budget_exhaustion_flags_truncated(self):
        defective = {1, 2, 3, 4}
        res = gt.identify(100, _oracle(defective), d=1, rng=random.Random(9), max_tests=5)
        assert res.truncated

    def test_control_split_baseline_is_exact_too(self):
        # Hard Rule 46: the reference must pass on its own first.
        for seed in range(10):
            rng = random.Random(seed)
            defective = set(rng.sample(range(150), rng.randint(0, 6)))
            base = gt.split_search(150, _oracle(defective))
            assert base.defective == defective


class TestSplitBaseline:
    def test_rounds_are_tree_depth(self):
        res = gt.split_search(64, _oracle({10}))
        assert res.rounds <= 7  # 1 + log2(64)

    def test_single_defective_cost_is_logarithmic(self):
        res = gt.split_search(1024, _oracle({500}))
        assert res.tests <= 2 * 11
