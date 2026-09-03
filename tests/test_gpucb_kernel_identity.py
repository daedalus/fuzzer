"""The GP-UCB RBF kernel over one-hot features takes exactly two values.

Feature vectors are exact category one-hots, so the squared distance
between any two is 0 (same category) or 2 (different). Substituting into
exp(-d / (2*ls^2)) leaves

    K(i, j) = 1.0            if cat_i == cat_j
              exp(-1/ls^2)   otherwise

``_compute_kernel_row`` was rewritten onto that closed form. ``_rbf`` is
kept as the general vector-space definition, so it doubles as this
module's oracle: the fast path must agree with it on every pair, at every
length scale, bit for bit.
"""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.schedulers.gp_ucb import GPUCBScheduler

ALL_OPS = sorted({op for ops in OPERATOR_CATEGORIES.values() for op in ops})


def rbf_row(sched: GPUCBScheduler, op: str, candidates: list[str]) -> dict[str, float]:
    """Oracle: the row as the general RBF definition produces it."""
    f_i = sched._features.get(op)
    if f_i is None:
        return dict.fromkeys(candidates, 0.0)
    return {
        c: (sched._rbf(f_i, sched._features[c]) if c in sched._features else 0.0)
        for c in candidates
    }


def armed(length_scale: float = 1.0, ops: list[str] | None = None) -> GPUCBScheduler:
    sched = GPUCBScheduler(length_scale=length_scale)
    for op in ops if ops is not None else ALL_OPS:
        sched.init_arm(op)
    return sched


class TestKernelIsTwoValued:
    def test_only_two_distinct_values_appear(self):
        sched = armed()
        values = set()
        for op in ALL_OPS:
            values |= {round(v, 12) for v in sched._compute_kernel_row(op, ALL_OPS).values()}
        assert values == {1.0, round(math.exp(-1.0), 12)}

    def test_same_category_is_exactly_one(self):
        sched = armed()
        for op in ALL_OPS:
            cat = sched._op_to_cat[op]
            row = sched._compute_kernel_row(op, ALL_OPS)
            same = [c for c in ALL_OPS if sched._op_to_cat[c] == cat]
            assert same, "every operator shares a category with itself"
            for c in same:
                assert row[c] == 1.0

    @pytest.mark.parametrize("length_scale", [0.25, 0.5, 1.0, 2.0, 5.0])
    def test_cross_category_value_tracks_the_length_scale(self, length_scale):
        sched = armed(length_scale)
        expected = math.exp(-1.0 / length_scale**2)
        assert sched._cross_category_similarity == pytest.approx(expected)


class TestAgreesWithTheGeneralRbf:
    @pytest.mark.parametrize("length_scale", [0.25, 0.5, 1.0, 2.0, 5.0])
    def test_every_pair_matches(self, length_scale):
        """156 x 156 pairs, exact equality — this is an identity, not an approximation."""
        sched = armed(length_scale)
        for op in ALL_OPS:
            fast = sched._compute_kernel_row(op, ALL_OPS)
            slow = rbf_row(sched, op, ALL_OPS)
            assert fast == slow

    def test_unknown_operator_yields_a_zero_row(self):
        sched = armed()
        assert sched._compute_kernel_row("not-an-operator", ALL_OPS) == dict.fromkeys(ALL_OPS, 0.0)

    def test_unknown_candidate_scores_zero(self):
        sched = armed()
        row = sched._compute_kernel_row(ALL_OPS[0], [*ALL_OPS, "not-an-operator"])
        assert row["not-an-operator"] == 0.0
        assert row == rbf_row(sched, ALL_OPS[0], [*ALL_OPS, "not-an-operator"])

    def test_empty_candidate_list(self):
        assert armed()._compute_kernel_row(ALL_OPS[0], []) == {}


class TestRuntimeRegisteredOperators:
    """An operator added after import gets a category, so it gets a real row."""

    def test_new_category_does_not_disturb_existing_rows(self):
        sched = armed(ops=ALL_OPS[:20])
        before = {op: sched._compute_kernel_row(op, ALL_OPS[:20]) for op in ALL_OPS[:20]}
        sched.init_arm("some-operator-registered-at-runtime")
        after = {op: sched._compute_kernel_row(op, ALL_OPS[:20]) for op in ALL_OPS[:20]}
        assert before == after

    def test_runtime_operator_row_matches_the_rbf(self):
        sched = armed(ops=ALL_OPS[:20])
        sched.init_arm("some-operator-registered-at-runtime")
        ops = [*ALL_OPS[:20], "some-operator-registered-at-runtime"]
        for op in ops:
            assert sched._compute_kernel_row(op, ops) == rbf_row(sched, op, ops)
