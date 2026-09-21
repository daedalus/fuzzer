"""Tests for core/covering_array.py -- generic t-way covering array construction."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core import covering_array as ca


class TestRequiredTupleCount:
    def test_two_binary_params_pairwise(self):
        # C(2,2)=1 subset, 2*2=4 combos.
        assert ca.required_tuple_count([[0, 1], [0, 1]], t=2) == 4

    def test_three_params_pairwise(self):
        # C(3,2)=3 subsets: (0,1)->2*3=6, (0,2)->2*2=4, (1,2)->3*2=6 = 16
        assert ca.required_tuple_count([[0, 1], [0, 1, 2], [0, 1]], t=2) == 16

    def test_t_clamped_to_param_count(self):
        # Only 1 param: t=2 clamps to t=1, one subset of size 1.
        assert ca.required_tuple_count([[0, 1, 2]], t=2) == 3

    def test_empty_value_sets(self):
        assert ca.required_tuple_count([], t=2) == 0


class TestGenerate:
    def test_empty_value_sets_returns_empty(self):
        assert ca.generate([], t=2) == []

    def test_rejects_empty_domain(self):
        with pytest.raises(ValueError):
            ca.generate([[0, 1], []], t=2)

    def test_single_param_covers_every_value_once_each(self):
        rows = ca.generate([[0, 1, 2, 3]], t=2, rng=random.Random(1))
        assert ca.verify_coverage(rows, [[0, 1, 2, 3]], t=2)
        # t=1-equivalent: each value must appear in some row's only field.
        assert {r[0] for r in rows} == {0, 1, 2, 3}

    def test_two_param_full_coverage(self):
        value_sets = [[0, 1, 2], [0, 1]]
        rows = ca.generate(value_sets, t=2, rng=random.Random(2))
        assert ca.verify_coverage(rows, value_sets, t=2)
        # Pairwise over exactly 2 params is the full cross product; no
        # row can be skipped, but no row should repeat a covered pair.
        assert len(rows) <= 3 * 2

    def test_rows_drawn_from_declared_domains(self):
        value_sets = [[10, 20, 30], ["a", "b"], [True, False]]
        rows = ca.generate(value_sets, t=2, rng=random.Random(3))
        for row in rows:
            for value, domain in zip(row, value_sets, strict=True):
                assert value in domain

    def test_png_ihdr_shaped_domains_fully_covered(self):
        # The actual domains covering_array_mutate.py uses -- exercised
        # here too so a change to core/covering_array.py that silently
        # breaks coverage on a 7-parameter, mixed-cardinality case is
        # caught independently of the PNG operator's own tests.
        value_sets = [
            [0, 1, 2, 0x7FFFFFFF, 0xFFFFFFFF],
            [0, 1, 2, 0x7FFFFFFF, 0xFFFFFFFF],
            [0, 1, 2, 4, 8, 16, 255],
            [0, 1, 2, 3, 4, 6, 255],
            [0, 1, 255],
            [0, 1, 255],
            [0, 1, 42, 255],
        ]
        rows = ca.generate(value_sets, t=2, rng=random.Random(1234))
        assert ca.verify_coverage(rows, value_sets, t=2)
        # Far fewer rows than the exhaustive cross product (44,100), and
        # a loose ceiling so a future domain tweak doesn't silently
        # regress the greedy construction into something pathological
        # without a test noticing.
        assert 0 < len(rows) < 500

    def test_deterministic_given_same_rng_seed(self):
        value_sets = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        rows_a = ca.generate(value_sets, t=2, rng=random.Random(99))
        rows_b = ca.generate(value_sets, t=2, rng=random.Random(99))
        assert rows_a == rows_b

    def test_t_greater_than_param_count_clamps(self):
        # t=5 over 2 params must not raise or hang -- clamps to t=2.
        value_sets = [[0, 1], [0, 1]]
        rows = ca.generate(value_sets, t=5, rng=random.Random(4))
        assert ca.verify_coverage(rows, value_sets, t=2)

    def test_max_rows_stops_early(self):
        value_sets = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        rows = ca.generate(value_sets, t=2, rng=random.Random(5), max_rows=1)
        assert len(rows) == 1

    def test_full_coverage_across_many_seeds_png_ihdr_domains(self):
        # Regression for a real bug: the greedy loop used to give up
        # (break) the first time a round of `candidate_pool` random rows
        # failed to improve coverage at all, which happens by chance
        # for the last few tuples remaining (e.g. one specific pair out
        # of ~500, each row only a few percent likely to hit it) --
        # silently leaving the array incomplete instead of retrying.
        # Swept 200 seeds while fixing it; keep a slice of that sweep
        # here so it can't regress quietly again.
        value_sets = [
            [0, 1, 2, 0x7FFFFFFF, 0xFFFFFFFF],
            [0, 1, 2, 0x7FFFFFFF, 0xFFFFFFFF],
            [0, 1, 2, 4, 8, 16, 255],
            [0, 1, 2, 3, 4, 6, 255],
            [0, 1, 255],
            [0, 1, 255],
            [0, 1, 42, 255],
        ]
        for seed in range(25):
            rows = ca.generate(value_sets, t=2, rng=random.Random(seed))
            assert ca.verify_coverage(rows, value_sets, t=2), seed

    def test_accepts_randpool_style_rng(self):
        # RandPool has .choice()/.randint() but not the rest of
        # random.Random's surface -- generate() must not reach for
        # anything else.
        from fuzzer_tool.core.rand_pool import RandPool

        value_sets = [[0, 1, 2], [0, 1], [5, 6, 7]]
        rng = RandPool(seed=7)
        rows = ca.generate(value_sets, t=2, rng=rng)
        assert ca.verify_coverage(rows, value_sets, t=2)


class TestVerifyCoverageAndMissingTuples:
    def test_missing_tuples_nonempty_for_incomplete_rows(self):
        value_sets = [[0, 1], [0, 1]]
        rows = [(0, 0)]
        missing = ca.missing_tuples(rows, value_sets, t=2)
        assert missing == {
            ((0, 1), (0, 1)),
            ((0, 1), (1, 0)),
            ((0, 1), (1, 1)),
        }
        assert not ca.verify_coverage(rows, value_sets, t=2)

    def test_verify_true_for_full_cross_product(self):
        value_sets = [[0, 1], [0, 1]]
        rows = [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert ca.verify_coverage(rows, value_sets, t=2)
        assert ca.missing_tuples(rows, value_sets, t=2) == set()
