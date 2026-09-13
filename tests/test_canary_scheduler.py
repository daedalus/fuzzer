"""Falsification and adversarial coverage for CanaryScheduler.

Hard Rule 23: every new capability ships one falsification test and one
adversarial test. CanaryScheduler's whole point is to be the worst
scheduler in the Elo pool given the same signal everyone else gets, so
the adversarial test here is a small bandit simulation proving that claim
rather than just exercising the interface.
"""

from __future__ import annotations

import json
import random

import pytest

from fuzzer_tool.core.elo import BayesianEloTracker
from fuzzer_tool.core.schedulers.canary import CanaryScheduler
from fuzzer_tool.core.schedulers.consolidated import ConsolidatedScheduler

ARMS = ["bit_flip", "byte_flip", "havoc"]


class TestSharedContract:
    """Mirrors TestSharedContract in the other scheduler test files."""

    def test_declares_supports_priors_false(self):
        # A "good" prior would work against the one property that
        # matters: seeding it toward a known-worse arm would help, not
        # hurt, its purpose -- but that's not what init_arm's prior args
        # are for on any other scheduler, and pretending to honor them
        # while ignoring them would be a silent lie. Declaring False is
        # honest about not implementing prior-shrinkage at all.
        assert CanaryScheduler.supports_priors is False

    def test_empty_candidate_list(self):
        s = CanaryScheduler()
        assert s.select_op([]) == ""

    def test_single_candidate(self):
        s = CanaryScheduler()
        assert s.select_op(["only"]) == "only"

    def test_record_for_unregistered_arm_does_not_raise(self):
        s = CanaryScheduler()
        s.record("never_initialized", True, weight=1.0)  # must not raise
        assert s.bandit_stats()["never_initialized"] == (2.0, 1.0)

    def test_bandit_stats_are_json_safe(self):
        s = CanaryScheduler()
        s.init_arm("a")
        s.record("a", True)
        json.dumps(s.bandit_stats())  # must not raise

    def test_reregistering_an_arm_does_not_reset_it(self):
        s = CanaryScheduler()
        s.init_arm("a")
        s.record("a", True, weight=1.0)
        s.init_arm("a")
        assert s.bandit_stats()["a"] == (2.0, 1.0)


class TestArgminSelection:
    """The core behavior: select_op must return the WORST-looking candidate."""

    def test_selects_lowest_posterior_mean(self):
        s = CanaryScheduler()
        # "good" gets fed successes, "bad" gets fed failures.
        for _ in range(20):
            s.record("good", True, weight=1.0)
            s.record("bad", False)
        assert s.select_op(["good", "bad"]) == "bad"

    def test_never_selects_the_best_arm_of_three(self):
        s = CanaryScheduler()
        for _ in range(30):
            s.record("best", True, weight=1.0)
            s.record("mid", True, weight=0.5)
            s.record("mid", False)
            s.record("worst", False)
        picks = {s.select_op(["best", "mid", "worst"]) for _ in range(10)}
        assert picks == {"worst"}

    def test_ties_broken_by_candidate_order(self):
        """Untouched arms all sit at the same Beta(1, 1) prior."""
        s = CanaryScheduler()
        assert s.select_op(["x", "y", "z"]) == "x"
        assert s.select_op(["z", "y", "x"]) == "z"

    def test_reachability_within_the_currently_worst_arm(self):
        """Not a fixed single string forever -- if the worst arm changes,
        canary must follow it (this is what distinguishes it from a
        signal-blind scheduler like round-robin)."""
        s = CanaryScheduler()
        for _ in range(20):
            s.record("a", False)
        assert s.select_op(["a", "b"]) == "a"
        # Now b becomes worse than a.
        for _ in range(40):
            s.record("b", False)
        assert s.select_op(["a", "b"]) == "b"

    def test_unregistered_candidates_are_registered_on_the_fly(self):
        s = CanaryScheduler()
        op = s.select_op(["fresh_one", "fresh_two"])
        assert op in ("fresh_one", "fresh_two")
        assert set(s.bandit_stats()) == {"fresh_one", "fresh_two"}


class TestAdversarialFloor:
    """CanaryScheduler must actually be worse than a real scheduler on the
    same bandit environment, not just structurally different from one."""

    def test_scores_below_a_real_scheduler_on_the_same_environment(self):
        """Same arm-quality environment, same number of pulls, independent
        state: canary's cumulative reward must land below a scheduler that
        is actually trying to win (Consolidated's flat Thompson sampling).
        """
        true_p = {"a": 0.9, "b": 0.5, "c": 0.1}
        rng = random.Random(1234)

        def run(select, record, rounds=4000):
            total = 0.0
            for _ in range(rounds):
                op = select(list(true_p))
                ok = rng.random() < true_p[op]
                record(op, ok)
                total += 1.0 if ok else 0.0
            return total

        canary = CanaryScheduler()
        canary_reward = run(canary.select_op, canary.record)

        rng = random.Random(1234)  # same draw stream for a fair comparison
        consolidated = ConsolidatedScheduler()
        consolidated_reward = run(consolidated.select_op, consolidated.record)

        assert canary_reward < consolidated_reward
        # Canary should track close to the worst arm's true rate (0.1),
        # not merely "below Consolidated" by luck.
        assert canary_reward / 4000 < 0.2

    def test_elo_rates_canary_below_a_normal_strategy(self):
        """Integration-level falsification: after enough rounds of
        record_strategy_match, BayesianEloTracker.strategies_below_canary
        must be empty when a real strategy is actually winning most of its
        matches against canary -- i.e. canary should NOT need inspection
        under ordinary conditions.
        """
        elo = BayesianEloTracker(initial_mu=1500, initial_sigma=350, beta=200, tau=5.0, min_matches=10)
        rng = random.Random(7)
        for _ in range(200):
            # "real" wins ~80% of its matches against canary.
            score = 1.0 if rng.random() < 0.8 else 0.0
            elo.record_strategy_match("real", "canary", score)
        assert elo.strategies_below_canary() == []
        ranking = dict(elo.get_strategy_ranking())
        assert ranking["real"] > ranking["canary"]

    def test_flags_a_real_strategy_that_actually_regressed(self):
        """When a 'real' strategy is losing to canary as often as it wins,
        strategies_below_canary must surface it."""
        elo = BayesianEloTracker(initial_mu=1500, initial_sigma=350, beta=200, tau=5.0, min_matches=10)
        rng = random.Random(99)
        for _ in range(200):
            # "broken" loses more often than it wins against canary --
            # canary should never be beating a real strategy this badly.
            score = 1.0 if rng.random() < 0.3 else 0.0
            elo.record_strategy_match("broken", "canary", score)
        flagged = elo.strategies_below_canary()
        assert any(name == "broken" for name, _, _ in flagged)
