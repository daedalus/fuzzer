"""Regression: Elo strategy arenas never left their first arm.

e7363be made ``record_strategy_matches`` rate only the arm that played, and
skip opponents never played. Selection only considered arms with
``min_matches`` matches, else returned the first. So the first arm played
forever, its opponents were never played, and nothing was ever rated.
Measured: fuzzgoat, ``--elo --position-arena --pos-centroid``, 5k execs:
19,134 position picks, every one ``uniform``; status line ``top_seed=?(0)``.

Fix: while any arm is below ``min_matches``, play the least-matched one
(fewest plays breaks ties, then list order).
"""

import pytest

from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker, EloTracker

ARMS = ["pos_uniform", "pos_a", "pos_b"]
WINNER = "pos_b"
MAX_ROUNDS = 1000  # bound, not a retry: the cold phase is deterministic


def _play(tracker, arms=ARMS):
    """One round: pick, then record the played arm against the others."""
    s = tracker.select_strategy(arms)
    score = 1.0 if s == WINNER else 0.0
    tracker.record_strategy_matches(s, [o for o in arms if o != s], score)
    return s


def _cold(tracker, arms=ARMS) -> bool:
    return any(tracker._strategy_match_count.get(a, 0) < tracker.min_matches for a in arms)


def _warm(tracker):
    """Play until every arm is rated; fails instead of spinning."""
    for _ in range(MAX_ROUNDS):
        if not _cold(tracker):
            return
        _play(tracker)
    raise AssertionError("cold start never ended: an arm is never played")


@pytest.mark.parametrize("cls", [BayesianEloTracker, EloTracker])
def test_regression_every_arm_is_played_first(cls):
    t = cls()
    assert [_play(t) for _ in ARMS] == ARMS


@pytest.mark.parametrize("cls", [BayesianEloTracker, EloTracker])
def test_every_arm_gets_rated(cls):
    t = cls()
    _warm(t)
    assert all(t._strategy_match_count[a] >= t.min_matches for a in ARMS)


def test_sampler_explores_cold_arm():
    # The position arena reads strategy_sampler, not select_strategy.
    t = BayesianEloTracker()
    _play(t)
    assert t.strategy_sampler(ARMS)() == "pos_a"


def test_warm_arms_use_the_posterior():
    # Control: once every arm is rated, the winner leads the posterior.
    t = BayesianEloTracker()
    _warm(t)
    mu = {a: t._strategy_mu[a] for a in ARMS}
    assert max(mu, key=mu.get) == WINNER


def test_late_arm_joins_cold():
    # Adversarial: a gated arm entering a warm pool is explored, not ignored.
    t = BayesianEloTracker()
    _warm(t)
    assert t.strategy_sampler(ARMS + ["pos_late"])() == "pos_late"
