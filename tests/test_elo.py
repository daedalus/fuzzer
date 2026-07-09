"""Tests for Elo rating tracker."""

import json
import tempfile
from pathlib import Path

from fuzzer_tool.core.elo import EloTracker


class TestEloTracker:
    def test_init_arm(self):
        elo = EloTracker()
        elo.init_arm("bit_flip")
        assert "bit_flip" in elo.ratings
        assert elo.ratings["bit_flip"] == 1500.0

    def test_init_arm_idempotent(self):
        elo = EloTracker()
        elo.init_arm("bit_flip")
        elo.init_arm("bit_flip")
        assert elo.ratings["bit_flip"] == 1500.0

    def test_expected_score(self):
        elo = EloTracker()
        # Equal ratings → 0.5
        assert elo._expected_score(1500, 1500) == 0.5
        # Higher rating → > 0.5
        assert elo._expected_score(1600, 1500) > 0.5
        # Lower rating → < 0.5
        assert elo._expected_score(1400, 1500) < 0.5

    def test_record_match_win(self):
        elo = EloTracker(k_factor=32)
        elo.init_arm("A")
        elo.init_arm("B")
        elo.record_match("A", "B", score_a=1.0)
        assert elo.ratings["A"] > 1500.0
        assert elo.ratings["B"] < 1500.0

    def test_record_match_loss(self):
        elo = EloTracker(k_factor=32)
        elo.init_arm("A")
        elo.init_arm("B")
        elo.record_match("A", "B", score_a=0.0)
        assert elo.ratings["A"] < 1500.0
        assert elo.ratings["B"] > 1500.0

    def test_record_match_draw(self):
        elo = EloTracker(k_factor=32)
        elo.init_arm("A")
        elo.init_arm("B")
        elo.record_match("A", "B", score_a=0.5)
        # Ratings should stay near 1500
        assert abs(elo.ratings["A"] - 1500.0) < 1.0
        assert abs(elo.ratings["B"] - 1500.0) < 1.0

    def test_record_match_count(self):
        elo = EloTracker()
        elo.init_arm("A")
        elo.init_arm("B")
        elo.record_match("A", "B", score_a=1.0)
        assert elo._match_count["A"] == 1
        assert elo._match_count["B"] == 1

    def test_record_round(self):
        elo = EloTracker()
        for op in ["A", "B", "C"]:
            elo.init_arm(op)
        elo.record_round(["A", "B", "C"], winners={"A"})
        # A should beat B and C
        assert elo.ratings["A"] > 1500.0
        assert elo.ratings["B"] < 1500.0
        assert elo.ratings["C"] < 1500.0
        # A should have 2 matches
        assert elo._match_count["A"] == 2

    def test_record_round_all_winners(self):
        elo = EloTracker()
        for op in ["A", "B"]:
            elo.init_arm(op)
        # All winners → no matches recorded (no losers to beat)
        elo.record_round(["A", "B"], winners={"A", "B"})
        assert elo.ratings["A"] == 1500.0
        assert elo.ratings["B"] == 1500.0

    def test_crash_track(self):
        elo = EloTracker(crash_track=True)
        elo.init_arm("A")
        elo.init_arm("B")
        elo.record_match("A", "B", score_a=1.0, crash=True)
        assert elo.crash_ratings["A"] > 1500.0
        assert elo.crash_ratings["B"] < 1500.0

    def test_crash_track_disabled(self):
        elo = EloTracker(crash_track=False)
        elo.init_arm("A")
        elo.init_arm("B")
        elo.record_match("A", "B", score_a=1.0, crash=True)
        assert "A" not in elo.crash_ratings

    def test_select_op_weighted(self):
        elo = EloTracker(k_factor=100)
        elo.init_arm("good")
        elo.init_arm("bad")
        # Make "good" much better
        for _ in range(20):
            elo.record_match("good", "bad", score_a=1.0)
        # Select many times — "good" should be selected more often
        counts = {"good": 0, "bad": 0}
        for _ in range(1000):
            op = elo.select_op(["good", "bad"])
            counts[op] += 1
        assert counts["good"] > counts["bad"]

    def test_select_op_single(self):
        elo = EloTracker()
        assert elo.select_op(["A"]) == "A"

    def test_select_op_empty(self):
        elo = EloTracker()
        assert elo.select_op([]) == ""

    def test_get_ranking(self):
        elo = EloTracker(k_factor=100, min_matches=0)
        elo.init_arm("A")
        elo.init_arm("B")
        elo.init_arm("C")
        elo.record_match("A", "B", score_a=1.0)
        elo.record_match("A", "C", score_a=1.0)
        ranking = elo.get_ranking()
        assert ranking[0][0] == "A"
        assert ranking[-1][0] in ("B", "C")

    def test_apply_decay(self):
        elo = EloTracker(k_factor=100, decay=0.9)
        elo.init_arm("A")
        elo.init_arm("B")
        elo.record_match("A", "B", score_a=1.0)
        old_a = elo.ratings["A"]
        elo.apply_decay()
        # Rating should move toward default (1500)
        assert elo.ratings["A"] != old_a
        assert abs(elo.ratings["A"] - 1500.0) < abs(old_a - 1500.0)

    def test_save_load(self):
        elo = EloTracker(k_factor=42)
        elo.init_arm("A")
        elo.init_arm("B")
        elo.record_match("A", "B", score_a=1.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        assert elo.save(path)

        elo2 = EloTracker()
        assert elo2.load(path)
        assert elo2.k_factor == 42
        assert abs(elo2.ratings["A"] - elo.ratings["A"]) < 0.01
        assert abs(elo2.ratings["B"] - elo.ratings["B"]) < 0.01

        Path(path).unlink()

    def test_load_nonexistent(self):
        elo = EloTracker()
        assert not elo.load("/nonexistent/path.json")

    def test_multiple_matches_converge(self):
        elo = EloTracker(k_factor=32)
        elo.init_arm("strong")
        elo.init_arm("weak")
        # Strong always wins
        for _ in range(100):
            elo.record_match("strong", "weak", score_a=1.0)
        # Strong should be much higher rated
        assert elo.ratings["strong"] > elo.ratings["weak"] + 200

    def test_k_factor_half_produces_smaller_swings(self):
        elo_32 = EloTracker(k_factor=32, min_matches=0)
        elo_16 = EloTracker(k_factor=16, min_matches=0)
        for elo in (elo_32, elo_16):
            elo.init_arm("A")
            elo.init_arm("B")
            elo.record_match("A", "B", score_a=1.0)
        # K=16 should produce smaller rating change
        assert abs(elo_16.ratings["A"] - 1500.0) < abs(elo_32.ratings["A"] - 1500.0)

    def test_min_matches_excludes_from_ranking(self):
        elo = EloTracker(min_matches=10)
        elo.init_arm("rated")
        elo.init_arm("unrated")
        # Give "rated" 10 matches but "unrated" only 3
        for _ in range(3):
            elo.record_match("rated", "unrated", score_a=1.0)
        elo.record_match("rated", "third", score_a=1.0)
        elo.record_match("rated", "third", score_a=1.0)
        elo.record_match("rated", "third", score_a=1.0)
        elo.record_match("rated", "third", score_a=1.0)
        elo.record_match("rated", "third", score_a=1.0)
        elo.record_match("rated", "third", score_a=1.0)
        elo.record_match("rated", "third", score_a=1.0)
        # rated=10, unrated=3, third=7
        ranking = elo.get_ranking()
        assert len(ranking) == 1  # only "rated" has >= 10 matches
        assert ranking[0][0] == "rated"

    def test_min_matches_excludes_from_select(self):
        elo = EloTracker(k_factor=100, min_matches=5)
        elo.init_arm("rated")
        elo.init_arm("unrated")
        for _ in range(10):
            elo.record_match("rated", "unrated", score_a=1.0)
        # select_op should skip "unrated" (only 5 matches < 5 threshold... wait)
        # "unrated" has 5 matches, threshold is 5, so it IS rated
        # Let's use min_matches=10
        elo2 = EloTracker(k_factor=100, min_matches=10)
        elo2.init_arm("rated")
        elo2.init_arm("unrated")
        for _ in range(5):
            elo2.record_match("rated", "unrated", score_a=1.0)
        # "unrated" has 5 matches < 10 threshold
        # select_op should skip it and return "rated"
        counts = {"rated": 0, "unrated": 0}
        for _ in range(100):
            op = elo2.select_op(["rated", "unrated"])
            counts[op] += 1
        assert counts["rated"] == 100
        assert counts["unrated"] == 0

    def test_get_unrated(self):
        elo = EloTracker(min_matches=5)
        elo.init_arm("A")
        elo.init_arm("B")
        elo.init_arm("C")
        elo.record_match("A", "B", score_a=1.0)  # A: 1, B: 1
        elo.record_match("A", "C", score_a=1.0)  # A: 2, C: 1
        elo.record_match("A", "B", score_a=1.0)  # A: 3, B: 2
        elo.record_match("A", "C", score_a=1.0)  # A: 4, C: 2
        elo.record_match("A", "B", score_a=1.0)  # A: 5, B: 3
        elo.record_match("A", "C", score_a=1.0)  # A: 6, C: 3
        elo.record_match("A", "B", score_a=1.0)  # A: 7, B: 4
        elo.record_match("A", "C", score_a=1.0)  # A: 8, C: 4
        elo.record_match("A", "B", score_a=1.0)  # A: 9, B: 5
        elo.record_match("A", "C", score_a=1.0)  # A: 10, C: 5
        elo.record_match("B", "C", score_a=1.0)  # B: 6, C: 6
        elo.record_match("B", "C", score_a=1.0)  # B: 7, C: 7
        elo.record_match("B", "C", score_a=1.0)  # B: 8, C: 8
        elo.record_match("B", "C", score_a=1.0)  # B: 9, C: 9
        elo.record_match("B", "C", score_a=1.0)  # B: 10, C: 10
        # All have >= 5 matches
        assert elo.get_unrated() == []

    def test_min_matches_save_load(self):
        elo = EloTracker(min_matches=20)
        elo.init_arm("A")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        elo.save(path)
        elo2 = EloTracker()
        elo2.load(path)
        assert elo2.min_matches == 20
        Path(path).unlink()

    def test_record_round_edge_counts(self):
        elo = EloTracker(k_factor=32, min_matches=0)
        for op in ["A", "B", "C"]:
            elo.init_arm(op)
        # A found 10 edges, B found 5, C found 0
        elo.record_round(
            ["A", "B", "C"], winners={"A", "B"},
            edge_counts={"A": 10, "B": 5, "C": 0}
        )
        # A should have higher rating than B (proportional scoring)
        assert elo.ratings["A"] > elo.ratings["B"]
        # Both should beat C
        assert elo.ratings["B"] > elo.ratings["C"]
