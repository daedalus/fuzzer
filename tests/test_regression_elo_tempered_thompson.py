"""BayesianEloTracker's adaptive temperature was computed but never used.

``_effective_temperature`` now widens the Thompson draws (sigma * T_eff /
T_base, scale in [1, 2]) when the favourite stops winning, and strategy
matches feed a favourite-won history per arena.
"""

import numpy as np

from fuzzer_tool.core.analyzers.analyzer_elo import Arena, BayesianEloTracker
from fuzzer_tool.core.rand_pool import RandPool

FAV, DOG = "fav", "dog"  # operator arena (unprefixed)
OP = Arena.OPERATOR
DOG_MU, FAV_MU, SD = 1500.0, 1600.0, 100.0
Z_DOG = 0.8  # dog draws 1580 at scale 1, 1620 at scale 1.5


class _FixedNormal:
    """Scripted RNG: normal_array/gauss return fixed standard-normal draws."""

    def __init__(self, zs: list[float]):
        self._zs = zs

    def normal_array(self, n):
        return np.array(self._zs[:n], dtype=float)

    def gauss(self, mu, sigma):
        z, self._zs = self._zs[0], self._zs[1:]
        return mu + sigma * z


def _tracker(fav_hits: list[float]) -> BayesianEloTracker:
    t = BayesianEloTracker(rng=_FixedNormal([Z_DOG, 0.0]))
    for name, mu in ((DOG, DOG_MU), (FAV, FAV_MU)):
        t._strategy_mu[name] = mu
        t._strategy_sigma_sq[name] = SD**2
        t._strategy_match_count[name] = t.min_matches
        t.mu[name] = mu
        t.sigma_sq[name] = SD**2
        t._match_count[name] = t.min_matches
    t._track_strategy_wins(OP, fav_hits)
    t._best_win_rate.extend(fav_hits)
    return t


def _noisy(n: int) -> list[float]:
    """Favourite wins half the time: scale 1.5."""
    return [float(i % 2) for i in range(n)]


def test_regression_noisy_history_widens_strategy_draws():
    """Falsification: at scale 1 the favourite wins; 1.5 flips it."""
    assert _tracker([1.0] * 50).select_strategy([DOG, FAV]) == FAV
    assert _tracker(_noisy(50)).select_strategy([DOG, FAV]) == DOG


def test_short_history_keeps_scale_one():
    """Below the minimum history the draw is untempered."""
    assert _tracker(_noisy(10)).select_strategy([DOG, FAV]) == FAV


def test_noisy_history_widens_op_draws():
    assert _tracker([1.0] * 50).select_op([DOG, FAV]) == FAV
    assert _tracker(_noisy(50)).select_op([DOG, FAV]) == DOG


def test_strategy_history_separate_from_op_history():
    """Operator noise must not temper strategy draws."""
    t = _tracker([])
    t._best_win_rate.extend(_noisy(50))
    assert t.select_strategy([DOG, FAV]) == FAV


def test_strategy_match_records_favourite_outcome():
    t = BayesianEloTracker(rng=RandPool(1))
    t._strategy_mu.update({FAV: FAV_MU, DOG: DOG_MU})
    t.record_strategy_match(DOG, FAV, 1.0)  # upset
    t.record_strategy_match(FAV, DOG, 1.0)  # favourite wins
    assert list(t._strategy_best_win_rate[OP]) == [0.0, 1.0]


def test_strategy_matches_records_per_rated_opponent():
    t = BayesianEloTracker(rng=RandPool(1))
    t._strategy_mu.update({FAV: FAV_MU, DOG: DOG_MU})
    t._strategy_trials[FAV] = 4
    t._strategy_score_sum[FAV] = 0.0  # FAV's bar is 0: any score beats it
    t.record_strategy_matches(DOG, [FAV, DOG, "never_played"], 1.0)
    assert list(t._strategy_best_win_rate[OP]) == [0.0]


def test_adversarial_favourite_always_loses_caps_scale():
    """All upsets: scale 2, never beyond."""
    t = _tracker([0.0] * 50)
    assert t._strategy_draw_scale(OP) == 2.0


def test_adversarial_history_stays_bounded():
    t = BayesianEloTracker(rng=RandPool(1))
    t._strategy_mu.update({FAV: FAV_MU, DOG: DOG_MU})
    for _ in range(5000):
        t.record_strategy_match(FAV, DOG, 1.0)
    assert len(t._strategy_best_win_rate[OP]) <= 200


def test_strategy_history_persists():
    t = _tracker(_noisy(30))
    r = BayesianEloTracker(rng=RandPool(1))
    r.from_dict(t.to_dict())
    assert list(r._strategy_best_win_rate[OP]) == _noisy(30)


def test_arenas_temper_independently():
    """Position-arena noise must not temper operator-strategy draws."""
    t = _tracker([])
    t._track_strategy_wins(Arena.POSITION, _noisy(50))
    assert t.select_strategy([DOG, FAV]) == FAV


def test_scale_cache_resets_on_new_history():
    """Cached scale must follow the history it was computed from."""
    t = _tracker([1.0] * 50)
    assert t._strategy_draw_scale(OP) == 1.0
    t._track_strategy_wins(OP, [0.0] * 50)
    assert t._strategy_draw_scale(OP) == 2.0
