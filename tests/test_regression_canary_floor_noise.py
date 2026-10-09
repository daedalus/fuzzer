"""Regression: canary-floor checks flagged noise.

Each round the played arm "loses" to every arm that sat out when it
misses. At a ~1% gain rate the Elo ratings then track pick frequency,
not quality, and the floor checks flagged most of the pool -- even with
identical arms, and even with a canary 5x worse than everyone else.
The floor check now compares per-arm hit rates (Beta posteriors).
"""

import logging
import math
import random

import numpy as np

from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker, _prob_below
from fuzzer_tool.core.schedulers.pos_canary import PositionCanaryScheduler
from fuzzer_tool.services.fuzzer import Fuzzer

_ARMS = ["canary"] + [f"a{i}" for i in range(19)]
_HIT_RATE = 0.01
_ROUNDS = 4000
_GRID = 4000


def _simulate(seed: int, canary_rate: float) -> BayesianEloTracker:
    """Thompson-selected arena; the canary hits at ``canary_rate``."""
    elo = BayesianEloTracker(min_matches=10)
    rng = random.Random(seed)
    sampler = None
    for t in range(_ROUNDS):
        if t % 50 == 0:
            sampler = elo.strategy_sampler(_ARMS)
        arm = sampler()
        rate = canary_rate if arm == "canary" else _HIT_RATE
        score = 1.0 if rng.random() < rate else 0.0
        elo.record_strategy_matches(arm, [o for o in _ARMS if o != arm], score)
        if t % 100 == 99:
            elo.apply_decay()
    return elo


def _rated(rates: dict[str, tuple[int, int]]) -> BayesianEloTracker:
    elo = BayesianEloTracker(min_matches=10)
    for key, (hits, trials) in rates.items():
        elo._strategy_hits[key] = hits
        elo._strategy_trials[key] = trials
    return elo


def _grid_oracle(h_a: int, n_a: int, h_b: int, n_b: int) -> float:
    """P(p_a < p_b) by midpoint integration of the two Beta densities."""
    x = (np.arange(_GRID) + 0.5) / _GRID

    def pdf(h, n):
        a, b = h + 1, n - h + 1
        log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        return np.exp(log_norm + (a - 1) * np.log(x) + (b - 1) * np.log1p(-x)) / _GRID

    # Half the diagonal cell: an inclusive CDF biases ties toward a < b.
    mass_a = pdf(h_a, n_a)
    cdf_a = np.cumsum(mass_a) - mass_a / 2
    return float(np.sum(pdf(h_b, n_b) * cdf_a))


def test_regression_identical_arms_not_flagged():
    """Falsification: identical arms are never below the floor."""
    elo = _simulate(seed=0, canary_rate=_HIT_RATE)
    assert elo.strategies_below_canary() == []


def test_better_arms_not_flagged_against_worse_canary():
    """Adversarial: a canary 5x worse must not flag the arms beating it."""
    elo = _simulate(seed=3, canary_rate=_HIT_RATE / 5)
    assert elo.strategies_below_canary() == []


def test_credibly_worse_arm_flagged():
    elo = _rated({"canary": (100, 2000), "broken": (0, 2000), "fine": (110, 2000)})
    flagged = elo.strategies_below_canary()
    assert [s for s, *_ in flagged] == ["broken"]
    _, rate, floor_rate = flagged[0]
    assert rate < floor_rate


def test_unrated_floor_flags_nothing():
    """Adversarial: the floor needs min_matches trials of its own."""
    elo = _rated({"canary": (0, 5), "broken": (0, 2000)})
    assert elo.strategies_below_canary() == []


def test_counts_only_the_arm_that_played():
    elo = BayesianEloTracker()
    elo.record_strategy_matches("a", ["b", "c"], 1.0)
    elo.record_strategy_matches("b", ["a", "c"], 0.0)
    assert elo._strategy_trials == {"a": 1, "b": 1}
    assert elo._strategy_hits == {"a": 1, "b": 0}


def test_counts_survive_round_trip():
    elo = _rated({"canary": (3, 40)})
    restored = BayesianEloTracker()
    restored.from_dict(elo.to_dict())
    assert restored._strategy_hits == {"canary": 3}
    assert restored._strategy_trials == {"canary": 40}


def test_prob_below_control_is_half():
    """Control (Hard Rule 46): oracle and code on identical arms give 0.5."""
    assert abs(_grid_oracle(7, 300, 7, 300) - 0.5) < 1e-3
    assert abs(_prob_below(7, 300, 7, 300) - 0.5) < 1e-9


def test_prob_below_matches_grid_oracle():
    cases = [(0, 50, 5, 50), (12, 400, 3, 100), (30, 60, 25, 70), (2, 1000, 9, 1000)]
    for case in cases:
        assert abs(_prob_below(*case) - _grid_oracle(*case)) < 2e-3, case


def test_prob_below_large_counts_use_bounded_path():
    """Adversarial: huge hit counts stay finite and ordered."""
    lo = _prob_below(400_000, 1_000_000, 410_000, 1_000_000)
    assert 0.99 < lo <= 1.0
    assert _prob_below(410_000, 1_000_000, 400_000, 1_000_000) < 0.01


def _checker(elo: BayesianEloTracker) -> Fuzzer:
    f = Fuzzer.__new__(Fuzzer)
    f._elo = elo
    f._position_arena = object()
    f._pos_canary = PositionCanaryScheduler()
    f._use_seed_canary = False
    f._seed_canary = None
    return f


def test_regression_pos_arm_logged_once(caplog):
    """An arm below both position floors gets one warning, not two."""
    elo = _rated({"pos_uniform": (100, 2000), "pos_canary": (100, 2000), "pos_mi": (0, 2000)})
    with caplog.at_level(logging.WARNING):
        _checker(elo)._check_canary_inspection()
    hits = [r.getMessage() for r in caplog.records if "'pos_mi'" in r.getMessage()]
    assert len(hits) == 1
