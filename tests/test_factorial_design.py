"""Tests for tools/lib/factorial_design.py -- two-level orthogonal screening designs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from factorial_design import (  # noqa: E402
    design_matrix,
    fold_over,
    main_effects,
    screen,
)


def _col(rows, j):
    return [r[j] for r in rows]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestDesignMatrix:
    @pytest.mark.parametrize(
        ("k", "runs"), [(1, 4), (3, 4), (4, 8), (7, 8), (8, 12), (11, 12), (12, 16), (15, 16)]
    )
    def test_smallest_run_count(self, k, runs):
        assert len(design_matrix(k)) == runs

    @pytest.mark.parametrize("k", [3, 7, 11, 15, 19, 31])
    def test_columns_are_orthogonal_and_balanced(self, k):
        rows = design_matrix(k)
        for j in range(k):
            assert sum(_col(rows, j)) == 0
            for i in range(j):
                assert _dot(_col(rows, i), _col(rows, j)) == 0

    def test_entries_are_plus_minus_one(self):
        assert {v for r in design_matrix(11) for v in r} == {-1, 1}

    def test_rows_are_distinct(self):
        rows = design_matrix(11)
        assert len(set(rows)) == len(rows)

    def test_width_equals_requested_factors(self):
        assert all(len(r) == 5 for r in design_matrix(5))

    def test_zero_or_negative_factors_rejected(self):
        with pytest.raises(ValueError):
            design_matrix(0)
        with pytest.raises(ValueError):
            design_matrix(-2)

    def test_too_many_factors_rejected(self):
        with pytest.raises(ValueError):
            design_matrix(10_000)

    def test_far_fewer_runs_than_full_factorial(self):
        # Falsification of "this is just the grid": 11 factors, 12 vs 2048 runs.
        assert len(design_matrix(11)) * 100 < 2**11


class TestFoldOver:
    def test_doubles_runs_and_negates(self):
        rows = design_matrix(5)
        folded = fold_over(rows)
        assert len(folded) == 2 * len(rows)
        assert folded[len(rows) :] == [tuple(-v for v in r) for r in rows]

    def test_folded_columns_stay_orthogonal(self):
        rows = fold_over(design_matrix(7))
        for j in range(7):
            for i in range(j):
                assert _dot(_col(rows, i), _col(rows, j)) == 0

    def test_folded_main_effect_unbiased_by_two_factor_interaction(self):
        # y = 3*A + 5*(B*C): a 12-run PB aliases B*C into other main effects;
        # folding cancels it. A's estimate must be exactly 6 (2 * coefficient).
        rows = fold_over(design_matrix(11))
        y = [3 * r[0] + 5 * r[1] * r[2] for r in rows]
        eff = main_effects(rows, y)
        assert eff[0] == pytest.approx(6.0)
        assert all(abs(e) < 1e-9 for e in eff[1:])

    def test_unfolded_design_is_aliased(self):
        # Control: same response without folding must show contamination,
        # otherwise the test above proves nothing.
        rows = design_matrix(11)
        y = [3 * r[0] + 5 * r[1] * r[2] for r in rows]
        eff = main_effects(rows, y)
        assert any(abs(e) > 1e-9 for e in eff[1:])


class TestMainEffects:
    def test_recovers_linear_coefficients(self):
        rows = design_matrix(7)
        coef = [4.0, -2.0, 0.0, 1.0, 0.0, 0.0, 3.0]
        y = [sum(c * v for c, v in zip(coef, r, strict=True)) + 10 for r in rows]
        eff = main_effects(rows, y)
        for e, c in zip(eff, coef, strict=True):
            assert e == pytest.approx(2 * c)

    def test_constant_response_has_no_effects(self):
        rows = design_matrix(7)
        assert main_effects(rows, [5.0] * len(rows)) == [0.0] * 7

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            main_effects(design_matrix(3), [1.0])

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            main_effects([], [])


class TestScreen:
    FACTORS = {"kp": (0.1, 0.9), "ki": (0.01, 0.5), "floor": (0.0, 0.2)}

    def test_ranks_the_active_factor_first(self):
        def run(cfg):
            return 10 * cfg["kp"] + 0.0 * cfg["ki"]

        ranked = screen(self.FACTORS, run)
        assert ranked[0].name == "kp"
        assert {r.name for r in ranked} == set(self.FACTORS)

    def test_effect_is_in_response_units(self):
        ranked = screen(self.FACTORS, lambda c: 2.0 * c["kp"])
        top = ranked[0]
        # high - low = 0.8, slope 2 -> effect 1.6
        assert top.effect == pytest.approx(1.6)

    def test_run_count_is_design_size_not_grid(self):
        calls = []

        def run(cfg):
            calls.append(cfg)
            return 0.0

        screen(self.FACTORS, run)
        assert len(calls) == 4

    def test_configs_only_use_declared_levels(self):
        seen = []
        screen(self.FACTORS, lambda c: seen.append(c) or 0.0)
        for cfg in seen:
            for name, (lo, hi) in self.FACTORS.items():
                assert cfg[name] in (lo, hi)

    def test_fold_flag_doubles_runs(self):
        calls = []
        screen(self.FACTORS, lambda c: calls.append(c) or 0.0, fold=True)
        assert len(calls) == 8

    def test_replicates_average_noise(self):
        state = {"n": 0}

        def run(cfg):
            state["n"] += 1
            return cfg["kp"] + (1.0 if state["n"] % 2 else -1.0)

        ranked = screen(self.FACTORS, run, replicates=2)
        assert ranked[0].effect == pytest.approx(0.8)

    def test_empty_factors_rejected(self):
        with pytest.raises(ValueError):
            screen({}, lambda c: 0.0)

    def test_degenerate_level_range_rejected(self):
        with pytest.raises(ValueError):
            screen({"a": (1.0, 1.0)}, lambda c: 0.0)
