"""Regression: operator credit must go to operators that did something.

Three stacked bugs meant Elo and every bandit scheduler credited every
operator in a round identically:

1. The call site built ``winners = set(self._last_ops_used)`` while
   ``operators`` was ``list(dict.fromkeys(self._last_ops_used))`` -- the
   same set. ``losers`` in ``record_round()`` was therefore always empty,
   so the winners-beat-losers branch, including the proportional
   ``edge_counts`` path, was unreachable from the fuzzer. Measured:
   1000/1000 rounds fell through to the cross-iteration fallback.

2. An operator that was selected but left the buffer unchanged was scored
   as a full winner. On a single-seed corpus ``splice``, ``crossover``,
   ``byte_shuffle`` and ``redqueen_xform`` change nothing 100% of the time
   (measured over 2500 execs) yet were credited on every successful round.

3. The cross-iteration fallback ignored the previous round's outcome,
   scoring the earlier round at 0.7 over the later one on every failure.
   With ~95% of rounds failing, the dominant rating update was "whichever
   operators ran first win" -- adjacency, not merit. Cross-seed Spearman
   correlation of the final ranking was +0.294 before the fix and +0.978
   after, i.e. the original ratings were mostly a random walk.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.elo import BayesianEloTracker, EloTracker

TRACKERS = [EloTracker, BayesianEloTracker]


def _rating(tracker, op: str) -> float:
    return tracker.get_rating(op)


class TestWinnersMustBeAProperSubset:
    @pytest.mark.parametrize("cls", TRACKERS)
    def test_all_operators_as_winners_records_no_within_round_match(self, cls):
        """The shape of the original bug: winners == operators."""
        t = cls()
        for op in ("a", "b"):
            t.init_arm(op)
        t.record_round(["a", "b"], {"a", "b"})
        assert sum(t._match_count.values()) == 0

    @pytest.mark.parametrize("cls", TRACKERS)
    def test_effective_subset_produces_a_real_match(self, cls):
        """With a proper subset the winners-beat-losers branch is live."""
        t = cls()
        for op in ("did_work", "no_op"):
            t.init_arm(op)
        t.record_round(["did_work", "no_op"], {"did_work"})
        assert _rating(t, "did_work") > _rating(t, "no_op")
        assert sum(t._match_count.values()) > 0


class TestProportionalEdgeCountsPathIsReachable:
    def test_edge_counts_scale_scores_among_multiple_winners(self):
        """The cost-aware proportional branch needs >1 winner AND a loser.

        It was dead code from the fuzzer because losers was always empty.
        """
        t = EloTracker()
        for op in ("big", "small", "loser"):
            t.init_arm(op)
        t.record_round(
            ["big", "small", "loser"],
            {"big", "small"},
            edge_counts={"big": 10.0, "small": 1.0},
        )
        assert t.ratings["big"] > t.ratings["small"] > 1500
        assert t.ratings["loser"] < 1500

    def test_edge_counts_accepts_floats(self):
        """_cost_adjusted_weight returns floats, not ints."""
        t = EloTracker()
        for op in ("w", "l"):
            t.init_arm(op)
        t.record_round(["w", "l"], {"w"}, edge_counts={"w": 0.375})
        assert t.ratings["w"] > 1500


class TestCrossRoundComparisonIsOutcomeAware:
    def test_two_failed_rounds_record_nothing(self):
        """Failure-vs-failure carries no relative information.

        Recording it -- in either direction, or as a draw -- swamps the
        real signal: 171742 of 171972 matches in a 2500-exec run were
        failure-vs-failure, against 24 genuine wins.
        """
        t = EloTracker()
        for op in ("a", "b", "c", "d"):
            t.init_arm(op)
        t.record_round(["a", "b"], set())
        before = sum(t._match_count.values())
        t.record_round(["c", "d"], set())
        assert sum(t._match_count.values()) == before

    def test_two_successful_rounds_record_nothing(self):
        t = EloTracker()
        for op in ("a", "b", "c", "d"):
            t.init_arm(op)
        t.record_round(["a", "b"], {"a", "b"})
        before = sum(t._match_count.values())
        t.record_round(["c", "d"], {"c", "d"})
        assert sum(t._match_count.values()) == before

    def test_success_after_failure_favours_the_successful_round(self):
        t = EloTracker()
        for op in ("failed", "won"):
            t.init_arm(op)
        t.record_round(["failed", "failed2"], set())
        t.record_round(["won", "won2"], {"won", "won2"})
        assert t.ratings["won"] > 1500
        assert t.ratings["failed"] < 1500

    def test_failure_after_success_penalises_the_failed_round(self):
        t = EloTracker()
        t.record_round(["won", "won2"], {"won", "won2"})
        t.record_round(["failed", "failed2"], set())
        assert t.ratings["failed"] < 1500
        assert t.ratings["won"] > 1500

    def test_shared_operators_are_excluded_from_the_comparison(self):
        """An operator in both rounds sits on both sides of the match.

        The old code paired it with itself via record_match(a, a), which
        updates one rating twice in opposite directions, leaving only the
        adaptive-k residue -- noise.
        """
        t = EloTracker()
        for op in ("shared", "only_prev", "only_cur"):
            t.init_arm(op)
        t.record_round(["shared", "only_prev"], {"shared", "only_prev"})
        t.record_round(["shared", "only_cur"], set())
        assert t.ratings["shared"] == 1500.0

    def test_prev_success_is_tracked_across_rounds(self):
        t = EloTracker()
        t.record_round(["a"], {"a"})
        assert t._prev_success is True
        t.record_round(["b"], set())
        assert t._prev_success is False


class TestSharedImplementation:
    def test_both_trackers_share_one_record_round(self):
        """It existed as two byte-identical copies; every fix had to be
        applied twice, and the failed-round fix duly was."""
        from fuzzer_tool.core.elo import RoundRecorderMixin

        assert EloTracker.record_round is RoundRecorderMixin.record_round
        assert BayesianEloTracker.record_round is RoundRecorderMixin.record_round
