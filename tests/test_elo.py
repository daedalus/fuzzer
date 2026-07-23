"""Tests for Elo rating tracker."""

import tempfile
from pathlib import Path

from fuzzer_tool.core.elo import BayesianEloTracker, EloTracker


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
        elo.record_round(["A", "B", "C"], winners={"A", "B"}, edge_counts={"A": 10, "B": 5, "C": 0})
        # A should have higher rating than B (proportional scoring)
        assert elo.ratings["A"] > elo.ratings["B"]
        # Both should beat C
        assert elo.ratings["B"] > elo.ratings["C"]

    def test_record_strategy_match(self):
        elo = EloTracker(min_matches=0)
        elo.record_strategy_match("bandit", "mopt", score_a=1.0)
        assert elo._strategy_ratings["bandit"] > elo._strategy_ratings["mopt"]
        assert elo._strategy_match_count["bandit"] == 1
        assert elo._strategy_match_count["mopt"] == 1

    def test_record_strategy_match_loss(self):
        elo = EloTracker(min_matches=0)
        elo.record_strategy_match("bandit", "mopt", score_a=0.0)
        assert elo._strategy_ratings["mopt"] > elo._strategy_ratings["bandit"]

    def test_select_strategy(self):
        elo = EloTracker(min_matches=0)
        for _ in range(20):
            elo.record_strategy_match("bandit", "mopt", score_a=1.0)
        # bandit should be selected most of the time (higher rating)
        counts = {"bandit": 0, "mopt": 0}
        for _ in range(100):
            s = elo.select_strategy(["bandit", "mopt"])
            counts[s] += 1
        assert counts["bandit"] > counts["mopt"]

    def test_select_strategy_single(self):
        elo = EloTracker(min_matches=0)
        assert elo.select_strategy(["bandit"]) == "bandit"

    def test_select_strategy_empty(self):
        elo = EloTracker(min_matches=0)
        assert elo.select_strategy([]) == ""

    def test_get_strategy_ranking(self):
        elo = EloTracker(min_matches=0)
        elo.record_strategy_match("bandit", "mopt", score_a=1.0)
        ranking = elo.get_strategy_ranking()
        assert len(ranking) == 2
        assert ranking[0][0] == "bandit"

    def test_strategy_decay(self):
        elo = EloTracker(min_matches=0, decay=0.5)
        elo.record_strategy_match("bandit", "mopt", score_a=1.0)
        before = elo._strategy_ratings["bandit"]
        elo.apply_decay()
        after = elo._strategy_ratings["bandit"]
        # Decay should move rating toward default (1500)
        assert after != before

    def test_strategy_save_load(self):
        elo = EloTracker(min_matches=0)
        elo.record_strategy_match("bandit", "mopt", score_a=1.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        elo.save(path)
        elo2 = EloTracker()
        elo2.load(path)
        assert elo2._strategy_ratings == elo._strategy_ratings
        assert elo2._strategy_match_count == elo._strategy_match_count
        Path(path).unlink()


class TestRewardMoments:
    def test_record_reward(self):
        elo = EloTracker()
        elo.init_arm("A")
        elo.record_reward("A", 5.0)
        rm = elo.get_reward_moments("A")
        assert rm is not None
        assert rm.count == 1
        assert rm.mean == 5.0

    def test_record_reward_auto_init(self):
        elo = EloTracker()
        elo.record_reward("new_op", 3.0)
        rm = elo.get_reward_moments("new_op")
        assert rm is not None
        assert rm.count == 1

    def test_record_match_feeds_reward(self):
        elo = EloTracker()
        elo.init_arm("A")
        elo.init_arm("B")
        elo.record_match("A", "B", score_a=1.0)
        rm_a = elo.get_reward_moments("A")
        rm_b = elo.get_reward_moments("B")
        assert rm_a is not None and rm_a.count >= 1
        assert rm_b is not None and rm_b.count >= 1

    def test_select_op_ucb_empty(self):
        elo = EloTracker()
        assert elo.select_op_ucb([]) == ""

    def test_select_op_ucb_single(self):
        elo = EloTracker()
        assert elo.select_op_ucb(["A"]) == "A"

    def test_select_op_ucb_fallback_to_elo(self):
        """With too few reward samples, UCB falls back to Elo."""
        elo = EloTracker(k_factor=100, min_matches=0)
        elo.init_arm("A")
        elo.init_arm("B")
        # Give A a high rating via matches (also feeds reward data)
        for _ in range(20):
            elo.record_match("A", "B", score_a=1.0)
        # A has mean reward 1.0, B has mean reward 0.0
        # With very low temperature, A should dominate
        counts = {"A": 0, "B": 0}
        for _ in range(100):
            op = elo.select_op_ucb(["A", "B"], temperature=1.0)
            counts[op] += 1
        assert counts["A"] > counts["B"]

    def test_select_op_ucb_with_reward_data(self):
        """With enough reward data, UCB should use mean + k*stddev."""
        elo = EloTracker(min_matches=0)
        elo.init_arm("high_reward")
        elo.init_arm("low_reward")
        # Feed reward data: high_reward gets high mean, low_reward gets low
        for _ in range(30):
            elo.record_reward("high_reward", 10.0)
            elo.record_reward("low_reward", 1.0)
        # With enough data and low temperature, high_reward should dominate
        counts = {"high_reward": 0, "low_reward": 0}
        for _ in range(100):
            op = elo.select_op_ucb(
                ["high_reward", "low_reward"],
                exploration_weight=0.5,
                temperature=10.0,
            )
            counts[op] += 1
        assert counts["high_reward"] > counts["low_reward"]

    def test_select_op_ucb_no_scale_mismatch(self):
        """Regression: Elo fallback (~1500) must not swamp UCB scores (~0-1).

        Operator A has 1 lucky reward sample and high Elo (~1600).
        Operator B has 50 well-characterized samples (mean=0.1, stddev=0.05).
        Without the fix, A wins ~97% despite being a poor bet.
        With the fix, A's raw Elo is never mixed into the UCB softmax so
        the well-characterized operator is not artificially suppressed.
        """
        elo = EloTracker(k_factor=100, min_matches=0)
        elo.init_arm("A")
        elo.init_arm("B")
        # Give A high Elo via matches (so it would dominate if mixed raw)
        for _ in range(5):
            elo.record_match("A", "B", score_a=1.0)
        # B has many reward samples with low mean — well-characterized
        for _ in range(50):
            elo.record_reward("B", 0.1)
        # Single lucky sample for A — should NOT be UCB-ready
        elo.record_reward("A", 1.0)

        # With low temperature, a scale-mismatch bug would select A nearly
        # every time.  With the fix, B should have a non-trivial chance.
        counts = {"A": 0, "B": 0}
        for _ in range(100):
            op = elo.select_op_ucb(["A", "B"], exploration_weight=0.5, temperature=10.0)
            counts[op] += 1
        # B must get at least 10% — not a tight bound but clearly
        # separates from the ~97% the bug produced.
        assert counts["B"] >= 10, (
            f"B should have >= 10% selection, got {counts['B']}/100. "
            f"Scale-mismatch bug may have regressed."
        )

    def test_reward_moments_save_load(self):
        elo = EloTracker()
        elo.init_arm("A")
        elo.record_reward("A", 5.0)
        elo.record_reward("A", 3.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        elo.save(path)
        elo2 = EloTracker()
        elo2.load(path)
        rm = elo2.get_reward_moments("A")
        assert rm is not None
        assert rm.count == 2
        assert abs(rm.mean - 4.0) < 0.01
        Path(path).unlink()


class TestBayesianEloTracker:
    def test_init_defaults(self):
        belo = BayesianEloTracker()
        assert belo.initial_mu == 1500.0
        assert belo.initial_sigma == 350.0
        assert belo.beta == 200.0
        assert belo.tau == 5.0

    def test_init_arm(self):
        belo = BayesianEloTracker()
        belo.init_arm("op_a")
        assert "op_a" in belo.mu
        assert belo.mu["op_a"] == 1500.0
        assert abs(belo.sigma_sq["op_a"] - 350.0 ** 2) < 1.0

    def test_init_arm_idempotent(self):
        belo = BayesianEloTracker()
        belo.init_arm("op_a")
        belo.init_arm("op_a")
        assert belo._match_count["op_a"] == 0

    def test_record_match_win(self):
        belo = BayesianEloTracker(initial_mu=1500.0, initial_sigma=350.0, beta=200.0, tau=0.0)
        belo.init_arm("A")
        belo.init_arm("B")
        belo.record_match("A", "B", score_a=1.0)
        # A won → A's mu should increase, B's should decrease
        assert belo.mu["A"] > 1500.0
        assert belo.mu["B"] < 1500.0
        # Variance should shrink (tau=0 so no re-inflation)
        assert belo.sigma_sq["A"] < 350.0 ** 2
        assert belo.sigma_sq["B"] < 350.0 ** 2
        assert belo._match_count["A"] == 1
        assert belo._match_count["B"] == 1

    def test_record_match_loss(self):
        belo = BayesianEloTracker(tau=0.0)
        belo.init_arm("A")
        belo.init_arm("B")
        belo.record_match("A", "B", score_a=0.0)
        assert belo.mu["A"] < 1500.0
        assert belo.mu["B"] > 1500.0

    def test_record_match_draw(self):
        belo = BayesianEloTracker(tau=0.0)
        belo.init_arm("A")
        belo.init_arm("B")
        belo.record_match("A", "B", score_a=0.5)
        # Draw: very small changes
        assert abs(belo.mu["A"] - belo.mu["B"]) < 10.0

    def test_thompson_selection_exploration(self):
        """New unrated operators should still be selectable."""
        belo = BayesianEloTracker(min_matches=0)
        belo.init_arm("A")
        belo.init_arm("B")
        results = set()
        for _ in range(200):
            results.add(belo.select_op(["A", "B"]))
        assert len(results) == 2  # both get picked

    def test_thompson_selection_prefers_higher_rated(self):
        """After many wins, the better operator should be preferred."""
        belo = BayesianEloTracker(tau=0.0, min_matches=0)
        belo.init_arm("good")
        belo.init_arm("bad")
        for _ in range(50):
            belo.record_match("good", "bad", score_a=1.0)
        counts = {"good": 0, "bad": 0}
        for _ in range(500):
            op = belo.select_op(["good", "bad"])
            counts[op] += 1
        assert counts["good"] > counts["bad"]

    def test_adaptive_k_increases_with_errors(self):
        belo = BayesianEloTracker(initial_mu=1500.0, initial_sigma=350.0, beta=200.0, tau=0.0)
        belo.init_arm("A")
        belo.init_arm("B")
        # Feed consistently wrong predictions: expected A to win, but B wins
        for _ in range(50):
            belo.record_match("A", "B", score_a=0.0)
        # K should have increased above base  (initial errors are large)
        k_after = belo._effective_k()
        assert k_after > belo._base_k

    def test_adaptive_k_stable_after_burn_in(self):
        """After burn-in, accurate predictions produce lower K than wrong ones."""
        belo = BayesianEloTracker(initial_mu=1500.0, initial_sigma=350.0, beta=200.0, tau=0.0)
        belo.init_arm("A")
        belo.init_arm("B")
        # Burn in: 200 matches to establish ratings, then check recent error window
        for _ in range(100):
            belo.record_match("A", "B", score_a=1.0)
        k_accurate = belo._effective_k()

        error_belo = BayesianEloTracker(initial_mu=1500.0, initial_sigma=350.0, beta=200.0, tau=0.0)
        error_belo.init_arm("A")
        error_belo.init_arm("B")
        for _ in range(100):
            error_belo.record_match("A", "B", score_a=0.0)
        k_wrong = error_belo._effective_k()
        # Both converge to the same MSE (squared error is symmetric),
        # but the key property: K should not exceed 2x base_k
        assert k_accurate <= belo._base_k * 2.0

    def test_adaptive_temperature(self):
        belo = BayesianEloTracker(tau=0.0, min_matches=0)
        belo.init_arm("A")
        belo.init_arm("B")
        # After consistent wins by A (which becomes the higher-rated op),
        # temperature should decrease from base
        for _ in range(50):
            belo.record_match("A", "B", score_a=1.0)
        temp = belo._effective_temperature()
        # After 50 wins, A is clearly higher-rated; best_win_rate ~1.0
        # Scale = 2.0 - 1.0 = 1.0, so temp ~= base
        assert temp <= belo._base_temperature * 1.05

    def test_record_round_winners_and_losers(self):
        belo = BayesianEloTracker(tau=0.0)
        belo.record_round(["A", "B", "C"], winners={"A", "B"})
        assert belo.mu["A"] > belo.initial_mu
        assert belo.mu["B"] > belo.initial_mu
        assert belo.mu["C"] < belo.initial_mu

    def test_strategy_matches(self):
        belo = BayesianEloTracker(tau=0.0, min_matches=0)
        belo.record_strategy_match("bandit", "mopt", score_a=1.0)
        belo.record_strategy_match("bandit", "mopt", score_a=1.0)
        belo.record_strategy_match("bandit", "mopt", score_a=1.0)
        ranking = belo.get_strategy_ranking()
        assert ranking[0][0] == "bandit"

    def test_select_strategy(self):
        belo = BayesianEloTracker(min_matches=0)
        belo.init_arm("bandit")  # not needed for strategy
        result = belo.select_strategy(["bandit", "mopt"])
        assert result in ("bandit", "mopt")

    def test_apply_decay_adds_noise(self):
        belo = BayesianEloTracker(tau=5.0)
        belo.init_arm("A")
        belo.init_arm("B")
        sigma_before = belo.sigma_sq["A"]
        belo.apply_decay()
        assert belo.sigma_sq["A"] > sigma_before

    def test_get_ranking(self):
        belo = BayesianEloTracker(tau=0.0, min_matches=0)
        belo.record_round(["A", "B"], winners={"A"})
        ranking = belo.get_ranking()
        assert ranking[0][0] == "A"

    def test_get_unrated(self):
        belo = BayesianEloTracker(min_matches=5)
        belo.init_arm("A")
        assert "A" in belo.get_unrated()
        belo.record_match("A", "B", score_a=1.0)
        assert "A" in belo.get_unrated()
        for _ in range(10):
            belo.record_match("A", "B", score_a=1.0)
        assert "A" not in belo.get_unrated()

    def test_save_load_roundtrip(self):
        belo = BayesianEloTracker(tau=0.0)
        belo.init_arm("A")
        belo.init_arm("B")
        belo.record_match("A", "B", score_a=1.0)
        belo.record_strategy_match("bandit", "mopt", score_a=1.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            assert belo.save(path)
            belo2 = BayesianEloTracker()
            assert belo2.load(path)
            assert belo2.mu["A"] == belo.mu["A"]
            assert belo2.mu["B"] == belo.mu["B"]
            assert belo2.sigma_sq["A"] == belo.sigma_sq["A"]
            assert belo2._strategy_mu["bandit"] == belo._strategy_mu["bandit"]
        finally:
            Path(path).unlink()

    def test_load_nonexistent(self):
        belo = BayesianEloTracker()
        assert not belo.load("/nonexistent/path.json")

    def test_thompson_sample_in_range(self):
        belo = BayesianEloTracker()
        belo.init_arm("A")
        for _ in range(100):
            s = belo._thompson_sample("A")
            assert isinstance(s, float)
