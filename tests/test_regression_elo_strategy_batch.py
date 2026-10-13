"""Elo strategy matches: only the arm that played is rated.

Every round the served strategy is scored against the other pool members
(position arena, seed and operator schedulers). Rating the idle ones with
``1 - score`` credited each of them on every miss, so ratings tracked pick
frequency instead of hit rate. Now the played arm is judged against each
idle arm's historical mean score; idle arms are not rated, counted or
credited.
"""

import pytest

from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker, EloTracker
from fuzzer_tool.core.rand_pool import RandPool

_MAKERS = [lambda: BayesianEloTracker(rng=RandPool(seed=0)), EloTracker]
_IDS = ["bayesian", "point"]
_POOL = ("a", "b", "c", "d")
_PICK_CYCLE = ("a",) * 7 + ("b", "c", "d")
_HITS_IN_TEN = 3
_TOL = 25.0


def _ratings(t):
    """Raw ratings; the ranking hides arms below min_matches."""
    return dict(t._strategy_mu if hasattr(t, "_strategy_mu") else t._strategy_ratings)


def _play(t, arm, score, pool=_POOL):
    t.record_strategy_matches(arm, [o for o in pool if o != arm], score)


def _rate(make, rounds):
    """Equal hit rate (3 in 10 per arm), arm 'a' picked 7x as often as the rest."""
    t = make()
    plays = dict.fromkeys(_POOL, 0)
    for i in range(rounds):
        arm = _PICK_CYCLE[i % len(_PICK_CYCLE)]
        _play(t, arm, 1.0 if plays[arm] % 10 < _HITS_IN_TEN else 0.0)
        plays[arm] += 1
    return _ratings(t)


@pytest.mark.parametrize("make", _MAKERS, ids=_IDS)
def test_rating_tracks_hit_rate_not_pick_frequency(make):
    """Falsification: equal hit rates -> the frequent arm is not rated lower."""
    r = _rate(make, 4000)
    rest = [r["b"], r["c"], r["d"]]

    assert abs(r["a"] - sum(rest) / len(rest)) < _TOL


@pytest.mark.parametrize("make", _MAKERS, ids=_IDS)
def test_better_hit_rate_rates_higher(make):
    """Adversarial: frequency is irrelevant, hit rate is not."""
    t = make()
    plays = dict.fromkeys(_POOL, 0)
    for i in range(4000):
        arm = _PICK_CYCLE[i % len(_PICK_CYCLE)]
        good = arm == "d"  # rarest arm hits 6 in 10, the rest 3 in 10
        score = 1.0 if plays[arm] % 10 < (6 if good else 3) else 0.0
        plays[arm] += 1
        _play(t, arm, score)

    r = _ratings(t)
    assert r["d"] > max(r["a"], r["b"], r["c"]) + _TOL


def test_rating_control_is_stable_against_itself():
    """Control (Hard Rule 46): the same script twice gives the same ratings."""
    assert _rate(_MAKERS[0], 600) == _rate(_MAKERS[0], 600)


@pytest.mark.parametrize("make", _MAKERS, ids=_IDS)
def test_idle_arm_untouched_by_others_play(make):
    """A miss must not move, count or credit an arm that did not play."""
    t = make()
    _play(t, "a", 1.0)  # first round has no reference: nothing rated yet
    _play(t, "b", 1.0)  # rated against a's history
    _play(t, "a", 1.0)
    before = _ratings(t)["b"], t._strategy_match_count["b"], dict(t._strategy_win_count)
    for _ in range(50):
        _play(t, "a", 0.0)

    after = _ratings(t)["b"], t._strategy_match_count["b"], dict(t._strategy_win_count)
    assert after[:2] == before[:2]
    assert after[2].get("b") == before[2].get("b")


@pytest.mark.parametrize("make", _MAKERS, ids=_IDS)
def test_unplayed_opponent_is_no_reference(make):
    """Adversarial: an opponent that never played gives A no match and is not created."""
    t = make()
    t.record_strategy_matches("a", ["ghost"], 0.0)

    assert _ratings(t) == {}
    assert "ghost" not in t._strategy_match_count


@pytest.mark.parametrize("make", _MAKERS, ids=_IDS)
def test_self_and_empty_opponents_are_skipped(make):
    """Adversarial: self in the opponent list, and an empty list."""
    t = make()
    t.record_strategy_matches("a", [], 1.0)
    t.record_strategy_matches("a", ["a", "a"], 1.0)

    assert _ratings(t) == {}


def test_point_tracker_matches_reference():
    """One miss against one idle arm, checked against the formula by hand."""
    t = EloTracker()
    t.record_strategy_matches("b", ["a"], 0.5)  # b played once, score 0.5, no reference
    t.record_strategy_matches("a", ["b"], 1.0)

    expected_s = 0.5 + (1.0 - 0.5) / 2.0
    ea = t._expected_score(t.default_rating, t.default_rating)
    assert _ratings(t)["a"] == pytest.approx(t.default_rating + t.k_factor * (expected_s - ea))


@pytest.mark.parametrize("make", _MAKERS, ids=_IDS)
def test_baseline_survives_roundtrip(make):
    """Falsification: the baseline is part of the persisted state."""
    t = make()
    for _ in range(5):
        _play(t, "b", 1.0)
    fresh = make()
    fresh.from_dict(t.to_dict())
    _play(fresh, "a", 1.0)
    _play(t, "a", 1.0)

    assert _ratings(fresh) == _ratings(t)
