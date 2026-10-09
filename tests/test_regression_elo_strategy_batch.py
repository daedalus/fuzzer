"""Elo strategy matches: one batched call per (strategy, opponents, score).

Every round plays the served strategy against each other pool member
(position arena, seed and operator schedulers): 268k record_strategy_match
calls per 2000 ``--hail-mary`` execs, each re-doing three setdefaults, the
attribute reads and the K lookup. ``record_strategy_matches`` hoists those
and runs the same arithmetic in the same order; the oracle is the per-match
loop on a copy, compared exactly, dict order included.
"""

import copy
import random

import pytest

from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker, EloTracker
from fuzzer_tool.core.rand_pool import RandPool

NAMES = [f"s{i}" for i in range(12)]


def _state(t):
    return [
        list(d.items())
        for d in (
            t._strategy_mu,
            t._strategy_sigma_sq,
            t._strategy_match_count,
            t._strategy_win_count,
        )
    ]


def _loop(t, a, opponents, score):
    for b in opponents:
        t.record_strategy_match(a, b, score)


def _script(seed):
    rnd = random.Random(seed)
    for _ in range(60):
        a = rnd.choice(NAMES)
        opponents = rnd.sample(NAMES, rnd.randrange(0, len(NAMES)))
        score = rnd.choice((0.0, 0.5, 1.0, rnd.random()))
        yield a, opponents, score


@pytest.mark.parametrize("seed", range(4))
def test_regression_elo_strategy_batch(seed):
    """Bayesian tracker: batched equals the per-match loop, bit for bit."""
    batched = BayesianEloTracker(rng=RandPool(seed=0))
    looped = copy.deepcopy(batched)
    control = copy.deepcopy(batched)
    for a, opponents, score in _script(seed):
        batched.record_strategy_matches(a, opponents, score)
        _loop(looped, a, opponents, score)
        _loop(control, a, opponents, score)
    # Control (Hard Rule 46): the oracle against a second run of itself.
    assert _state(control) == _state(looped)
    assert _state(batched) == _state(looped)


def test_self_match_and_repeats_match_loop():
    """Adversarial: the strategy in its own opponent list, and repeated opponents."""
    batched = BayesianEloTracker(rng=RandPool(seed=0))
    looped = copy.deepcopy(batched)
    opponents = ["x", "a", "x", "a", "y"]
    for score in (0.9, 0.1, 0.5):
        batched.record_strategy_matches("a", opponents, score)
        _loop(looped, "a", opponents, score)
    assert _state(batched) == _state(looped)


def test_empty_opponents_creates_nothing():
    """Adversarial: no opponents leaves the tables untouched."""
    t = BayesianEloTracker(rng=RandPool(seed=0))
    t.record_strategy_matches("a", [], 1.0)
    assert _state(t) == [[], [], [], []]


def test_point_elo_tracker_has_the_batch_too():
    """Falsification: the point-estimate tracker batches through the same loop."""
    batched = EloTracker()
    looped = copy.deepcopy(batched)
    for a, opponents, score in _script(9):
        batched.record_strategy_matches(a, opponents, score)
        _loop(looped, a, opponents, score)
    assert batched._strategy_ratings == looped._strategy_ratings
    assert batched._strategy_match_count == looped._strategy_match_count
