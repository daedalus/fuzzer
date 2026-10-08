"""BayesianEloTracker.select_strategy drew one Python gauss per rated arm.

The position arena calls it once per op application (~50 per exec, ~28
arms): 4.2M scalar draws per 3k --hail-mary execs. One standard-normal
array scaled per arm is the same numpy stream, so picks are identical.
"""

import math

from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker
from fuzzer_tool.core.rand_pool import RandPool

ARMS = [f"pos_{i}" for i in range(28)]
CALLS = 2000


def _old_select(t: BayesianEloTracker, strategies: list[str]) -> str:
    """The pre-change select_strategy body, verbatim."""
    rated = [s for s in strategies if t._strategy_match_count.get(s, 0) >= t.min_matches]
    if not rated:
        return strategies[0]
    samples = [
        (
            s,
            t._rng.gauss(
                t._strategy_mu.get(s, t.initial_mu),
                math.sqrt(t._strategy_sigma_sq.get(s, t.initial_sigma**2)),
            ),
        )
        for s in rated
    ]
    return max(samples, key=lambda x: x[1])[0]


def _tracker(seed: int) -> BayesianEloTracker:
    t = BayesianEloTracker(rng=RandPool(seed))
    for i, s in enumerate(ARMS):
        t._strategy_mu[s] = 1500.0 + (i % 7) * 4.0
        t._strategy_sigma_sq[s] = (60.0 + i) ** 2
        t._strategy_match_count[s] = t.min_matches if i % 5 else 0  # some unrated
    return t


def _picks(select, t) -> list[str]:
    out = [select(t, ARMS) for _ in range(CALLS)]
    out.append(str(t._rng.random()))  # the stream after must line up too
    return out


def test_old_matches_itself():
    """Control (Hard Rule 46)."""
    assert _picks(_old_select, _tracker(3)) == _picks(_old_select, _tracker(3))


def test_regression_elo_select_vectorized():
    new = _picks(lambda t, s: t.select_strategy(s), _tracker(3))
    assert new == _picks(_old_select, _tracker(3))


def test_no_rated_arm_returns_first():
    """Adversarial: nothing rated, nothing drawn."""
    t = BayesianEloTracker(rng=RandPool(1))
    before = RandPool(1).random()
    assert t.select_strategy(["a", "b"]) == "a"
    assert t._rng.random() == before


def test_sampler_matches_repeated_select():
    """A sampler built once draws exactly what per-call select_strategy draws."""
    a, b = _tracker(7), _tracker(7)
    sample = a.strategy_sampler(ARMS)
    assert [sample() for _ in range(CALLS)] == [b.select_strategy(ARMS) for _ in range(CALLS)]
    assert a._rng.random() == b._rng.random()


def test_sampler_edge_cases_draw_nothing():
    """Adversarial: empty, single, and all-unrated lists consume no randomness."""
    t = BayesianEloTracker(rng=RandPool(2))
    assert t.strategy_sampler([])() == ""
    assert t.strategy_sampler(["only"])() == "only"
    assert t.strategy_sampler(["a", "b"])() == "a"
    assert t._rng.random() == RandPool(2).random()
