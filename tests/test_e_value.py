"""Mixture e-value diagnostic: hand values, null calibration, Ville bound."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

import pytest
from bench_paired import compare

from fuzzer_tool.core.exact_tests import mcnemar_exact, mixture_e_value


class TestMixtureEValue:
    def test_no_discordant_pairs_is_neutral(self):
        assert mixture_e_value(0, 0) == 1.0

    def test_matches_beta_integral_by_hand(self):
        # b=1, c=1: 2^2 * 1!1!/3! = 4/6
        assert mixture_e_value(1, 1) == pytest.approx(4 / 6)
        # b=9, c=1: 2^10 * 9!1!/11! = 1024/110
        assert mixture_e_value(9, 1) == pytest.approx(1024 / 110)

    def test_symmetric_in_direction(self):
        assert mixture_e_value(2, 11) == pytest.approx(mixture_e_value(11, 2))

    def test_strong_split_exceeds_the_alpha_line(self):
        assert mixture_e_value(10, 0) >= 20  # McNemar p = 2/1024

    def test_balanced_split_is_below_one(self):
        assert mixture_e_value(10, 10) < 1

    def test_finite_for_huge_counts(self):
        assert math.isfinite(mixture_e_value(10**6, 0))

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            mixture_e_value(-1, 3)

    def test_less_powerful_than_mcnemar_at_fixed_n(self):
        for b, c in [(9, 1), (12, 3), (20, 5)]:
            assert 1 / mixture_e_value(b, c) >= mcnemar_exact(b, c) / 2


class TestUnderNull:
    def test_mean_is_one_exactly(self):
        # E[E] = sum_k C(n,k) 2^-n E(k, n-k) = 1 for every n.
        for n in (1, 5, 17):
            tot = sum(math.comb(n, k) * mixture_e_value(k, n - k) for k in range(n + 1)) / 2**n
            assert tot == pytest.approx(1.0)

    def test_ville_bound_holds_with_peeking(self):
        rng = random.Random(7)
        trials, hits = 2000, 0
        for _ in range(trials):
            w = loss = 0
            for _ in range(60):
                if rng.random() < 0.5:
                    w += 1
                else:
                    loss += 1
                if mixture_e_value(w, loss) >= 20:
                    hits += 1
                    break
        assert hits / trials <= 0.05 + 0.02  # alpha plus sampling slack


def test_compare_reports_e_value():
    def rows(arm, vals):
        return [
            {"arm": arm, "target": "t", "seed": i, "edges": v, "coverage_attached": True}
            for i, v in enumerate(vals)
        ]

    r = compare(rows("a", [1] * 10), rows("b", [5] * 10))
    assert r["wins"] == 10 and r["e_value"] == pytest.approx(mixture_e_value(10, 0))
    assert compare(rows("a", [3] * 4), rows("b", [3] * 4))["e_value"] == 1.0
