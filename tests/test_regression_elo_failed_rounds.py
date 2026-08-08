"""Regression: Elo must learn from failed rounds, not only successes.

Two stacked bugs meant operator-level Elo ratings were driven exclusively
by iterations that found new coverage:

1. ``record_round()`` computed ``losers`` and branched on ``if losers:``.
   When nothing found coverage every operator is a loser, so that branch
   was taken, its ``for w in winners`` loops iterated an empty set, and
   the function returned having changed nothing -- leaving the
   all-losers cross-iteration branch under the ``elif`` unreachable.
   That branch is the only path by which a failed round can move a
   rating, so it was dead code.

2. The call site in ``Fuzzer.fuzz_one()`` additionally guarded with
   ``if winners:``, so ``record_round()`` was not even called on a failed
   round.

Combined effect: a systematic positive bias (ratings only ever reflect
what worked, never what didn't) and zero learning during a stall.
Measured before the fix: a 4000-exec run sitting in ``random_stall``
recorded zero matches across all 106 registered arms, every rating
sitting at exactly the initial 1500.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.elo import BayesianEloTracker, EloTracker


def _seeded(tracker_cls):
    """Tracker whose ``_prev_operators`` is a *disjoint* successful round.

    The cross-iteration branch needs a previous round to compare against,
    and only operators unique to one side are compared -- pairing a round
    against itself is not evidence -- so the seed round uses different
    arms than the rounds under test.
    """
    t = tracker_cls()
    for op in ("a", "b", "x", "y"):
        t.init_arm(op)
    t.record_round(["x", "y"], {"x", "y"})
    return t


class TestFailedRoundIsRecorded:
    @pytest.mark.parametrize("cls", [EloTracker, BayesianEloTracker])
    def test_failed_round_changes_ratings(self, cls):
        """The core regression: a round where nobody found coverage must
        move ratings. Previously this was a silent no-op."""
        t = _seeded(cls)
        before = dict(t.ratings) if hasattr(t, "ratings") else dict(t.mu)

        t.record_round(["a", "b"], set())  # nobody found coverage

        after = dict(t.ratings) if hasattr(t, "ratings") else dict(t.mu)
        assert after != before, "failed round had no effect on ratings"

    @pytest.mark.parametrize("cls", [EloTracker, BayesianEloTracker])
    def test_failed_round_accrues_matches(self, cls):
        t = _seeded(cls)
        before = sum(t._match_count.values())

        t.record_round(["a", "b"], set())

        assert sum(t._match_count.values()) > before

    def test_failed_round_lowers_current_operators(self):
        """Directionality: the operators that just failed should not come
        out ahead of the previous round's operators."""
        t = EloTracker()
        for op in ("a", "b", "c", "d"):
            t.init_arm(op)
        t.record_round(["a", "b"], {"a", "b"})  # seed prev = [a, b]
        t.record_round(["c", "d"], set())  # c, d failed against a, b

        assert t.ratings["c"] < t.ratings["a"]
        assert t.ratings["d"] < t.ratings["b"]


class TestSuccessPathUnchanged:
    """The winner-beats-loser path must behave exactly as before."""

    def test_partial_winners_still_beat_losers(self):
        t = EloTracker()
        for op in ("w", "l"):
            t.init_arm(op)
        t.record_round(["w", "l"], {"w"})
        assert t.ratings["w"] > 1500
        assert t.ratings["l"] < 1500

    def test_all_winners_against_identical_prev_round_is_a_noop(self):
        """Comparing a round to a previous round with the same operators is
        not evidence. This previously changed ratings, but only because
        record_match(a, a) and the symmetric (a, b) / (b, a) pair failed to
        cancel exactly under the adaptive k-factor -- an artifact, not signal.
        """
        t = _seeded(EloTracker)
        t.record_round(["a", "b"], {"a", "b"})
        before = dict(t.ratings)
        t.record_round(["a", "b"], {"a", "b"})
        assert t.ratings == before

    def test_all_winners_beat_a_disjoint_failed_prev_round(self):
        """Cross-iteration comparison fires when the rounds differ."""
        t = EloTracker()
        for op in ("a", "b", "c", "d"):
            t.init_arm(op)
        t.record_round(["c", "d"], set())  # seed prev: failed round
        t.record_round(["a", "b"], {"a", "b"})  # all winners
        assert t.ratings["a"] > 1500
        assert t.ratings["c"] < 1500

    def test_first_round_is_a_noop_without_prev_operators(self):
        """No previous round to compare against: nothing to record yet."""
        t = EloTracker()
        for op in ("a", "b"):
            t.init_arm(op)
        t.record_round(["a", "b"], set())
        assert sum(t._match_count.values()) == 0

    def test_prev_operators_tracked_across_failed_rounds(self):
        """A failed round must still update _prev_operators, otherwise the
        comparison baseline goes stale during a stall."""
        t = _seeded(EloTracker)
        t.record_round(["c", "d"], set())
        assert t._prev_operators == ["c", "d"]
