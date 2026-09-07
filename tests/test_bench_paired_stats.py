"""Statistics for the paired benchmark harness.

A benchmark harness that reports a wrong p-value is worse than no harness:
it launders noise into a claim. These check the two tests against values
that can be computed by hand, and check that pairing is preserved when
rows are matched up.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from bench_paired import (  # noqa: E402
    _fisher_exact,
    _mcnemar_exact,
    _wilcoxon_signed_rank,
    compare,
)


class TestMcNemarExact:
    def test_matches_binomial_by_hand(self):
        # b=1, c=9: two-sided exact = 2 * P(X <= 1) for X ~ Binom(10, 0.5)
        assert _mcnemar_exact(1, 9) == 2 * (1 + 10) / 1024
        assert _mcnemar_exact(0, 6) == 2 * (1 / 64)

    def test_no_discordant_pairs_is_not_evidence(self):
        """All ties must give p=1, not a division by zero or a false win."""
        assert _mcnemar_exact(0, 0) == 1.0

    def test_symmetric_split_is_p_one(self):
        assert _mcnemar_exact(5, 5) == 1.0

    def test_symmetry_in_arguments(self):
        assert _mcnemar_exact(3, 11) == _mcnemar_exact(11, 3)

    def test_bounded(self):
        for b in range(8):
            for c in range(8):
                assert 0.0 <= _mcnemar_exact(b, c) <= 1.0


class TestFisherExact:
    def test_tea_tasting(self):
        # Fisher's original 2x2, two-sided p = 0.4857
        assert round(_fisher_exact(3, 1, 1, 3), 4) == 0.4857

    def test_complete_separation(self):
        # 2 / C(20,10)
        assert abs(_fisher_exact(10, 0, 0, 10) - 2 / 184756) < 1e-12

    def test_empty_table(self):
        assert _fisher_exact(0, 0, 0, 0) == 1.0


class TestWilcoxonSignedRank:
    def test_all_positive_gives_zero_statistic(self):
        # Ranks 1+2+3 = 6, the smaller of the two rank sums is 0. Two-sided
        # exact: only the all-negative assignment gives W+ <= 0 and only the
        # all-positive gives W+ >= 6, so p = 2 * 1/8 = 0.25.
        assert _wilcoxon_signed_rank([1, 2, 3]) == (0.0, 0.25)

    def test_all_negative_gives_zero_statistic(self):
        assert _wilcoxon_signed_rank([-1, -2, -3]) == (0.0, 0.25)

    def test_balanced_signs_is_not_evidence(self):
        # +1,-1,+2,-2: every sign assignment with equal magnitude pairs is
        # consistent with the null, so the two-sided p must be 1.0.
        stat, p = _wilcoxon_signed_rank([1, -1, 2, -2])
        assert stat == 5.0
        assert p == 1.0

    def test_zeros_are_dropped_before_ranking(self):
        # Three non-zero deltas of equal magnitude, all the same sign: the
        # zero carries no sign and must not inflate n or the rank sum.
        assert _wilcoxon_signed_rank([0, 1, 1, 1]) == (0.0, 0.25)

    def test_empty_input_is_not_evidence(self):
        assert _wilcoxon_signed_rank([]) == (0.0, 1.0)

    def test_single_delta(self):
        # One non-zero delta: W- = 0, W+ = 1, and under the null both
        # assignments are equally likely, so the two-sided p is 1.0.
        assert _wilcoxon_signed_rank([5]) == (0.0, 1.0)

    def test_symmetry_in_sign(self):
        # Flipping every sign swaps the two rank sums but keeps the smaller one
        # and the p-value unchanged.
        assert _wilcoxon_signed_rank([1, 2, 3, 4, 5]) == _wilcoxon_signed_rank([-1, -2, -3, -4, -5])

    def test_bounded(self):
        for deltas in ([1, 2, 3], [-1, 2, -3], [1, -1, 2, -2, 3]):
            stat, p = _wilcoxon_signed_rank(deltas)
            assert stat >= 0.0
            assert 0.0 <= p <= 1.0


class TestPairing:
    def _rows(self, arm, edges_by_cell):
        return [
            {
                "arm": arm,
                "target": t,
                "seed": s,
                "edges": e,
                "corpus": 0,
                "crashes": 0,
                "coverage_attached": True,
            }
            for (t, s), e in edges_by_cell.items()
        ]

    def test_pairs_by_target_and_seed_not_by_order(self):
        """Rows must be matched by cell identity, not by list position.

        If the two arms' rows are in different orders -- which they will be
        after a resumed or partial run -- positional matching silently
        compares unrelated campaigns.
        """
        base = self._rows("baseline", {("t1", 0): 100, ("t1", 1): 200})
        test = self._rows("arm", {("t1", 1): 210, ("t1", 0): 110})
        r = compare(base, test)
        assert r["wins"] == 2
        assert r["losses"] == 0
        assert r["median_delta"] == 10

    def test_unmatched_cells_are_ignored(self):
        base = self._rows("baseline", {("t1", 0): 100, ("t2", 0): 50})
        test = self._rows("arm", {("t1", 0): 90})
        r = compare(base, test)
        assert r["cells"] == 1
        assert r["losses"] == 1

    def test_coverage_failures_are_dropped_not_scored(self):
        """A run where coverage never attached is a hole, not a zero.

        Scoring it would let an instrumentation failure look like a
        catastrophic arm regression.
        """
        base = self._rows("baseline", {("t1", 0): 100, ("t1", 1): 100})
        test = self._rows("arm", {("t1", 0): 110, ("t1", 1): 0})
        test[1]["coverage_attached"] = False
        r = compare(base, test)
        assert r["cells"] == 1
        assert r["dropped_no_coverage"] == 1
        assert r["wins"] == 1

    def test_ties_are_not_wins(self):
        base = self._rows("baseline", {("t1", i): 100 for i in range(5)})
        test = self._rows("arm", {("t1", i): 100 for i in range(5)})
        r = compare(base, test)
        assert (r["wins"], r["losses"], r["ties"]) == (0, 0, 5)
        assert r["mcnemar_p"] == 1.0

    def _rows_with_coverage(self, arm, edges_by_cell, coverage_by_cell=None):
        """Like _rows but lets each cell declare whether coverage attached."""
        coverage_by_cell = coverage_by_cell or {}
        return [
            {
                "arm": arm,
                "target": t,
                "seed": s,
                "edges": e,
                "corpus": 0,
                "crashes": 0,
                "coverage_attached": coverage_by_cell.get((t, s), True),
            }
            for (t, s), e in edges_by_cell.items()
        ]

    def test_drop_counts_are_per_arm(self):
        """Unequal drop rates between arms are the MNAR red flag.

        If the base arm drops 3 cells and the test arm drops 9, the surviving-cell
        win rate is optimistic for the test arm no matter what the deltas say.
        The diagnostic has to say which arm dropped, not just that something did.
        """
        base = self._rows("baseline", {("t1", i): 100 for i in range(12)})
        for i in (0, 1, 2):
            base[i]["coverage_attached"] = False
        test = self._rows("arm", {("t1", i): 110 for i in range(12)})
        for i in range(9):
            test[i]["coverage_attached"] = False
        r = compare(base, test)
        assert r["dropped_base"] == 3
        assert r["dropped_test"] == 9
        assert r["dropped_no_coverage"] == 12
        # A cell is scorable only when BOTH arms attached coverage, so the 3
        # survivors are cells 9, 10, 11 -- and they are all wins. The drop
        # asymmetry (3 vs 9) is the point being diagnosed.
        assert r["cells"] == 3
        assert r["wins"] == 3
