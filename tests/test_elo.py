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
        elo = EloTracker(k_factor=100)
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
