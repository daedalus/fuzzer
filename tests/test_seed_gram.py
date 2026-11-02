"""SeedGramScheduler: eigenvector-centrality seed sampler."""

import numpy as np
import pytest

from fuzzer_tool.core.gram_matrixfree import Incidence, top_eigenpair
from fuzzer_tool.core.schedulers.seed_gram import FLOOR, SeedGramScheduler


class _Tracker:
    def __init__(self, seed_edges):
        self.seed_edges = seed_edges


class _Fixed:
    """Scripted RNG: replays the given draws."""

    def __init__(self, *draws):
        self._d = list(draws)

    def random(self):
        return self._d.pop(0)


# Seeds a,b,c share a tight cluster; d is disjoint. Core edge 100 is in all.
EDGES = {
    "a": {100, 1, 2, 3},
    "b": {100, 1, 2, 3},
    "c": {100, 1, 2},
    "d": {100, 50, 51},
    "e": {100, 60},
    "f": {100, 61},
    "g": {100, 62},
    "h": {100, 63},
    "i": {100, 64},
    "j": {100, 65},
}


def _expected_loading():
    keys = sorted(EDGES)
    inc = Incidence([np.array(sorted(EDGES[k])) for k in keys], weighting="core_stripped")
    _, v, _ = top_eigenpair(inc)
    return dict(zip(keys, (v / v.max()).tolist(), strict=True))


def test_refit_matches_independent_eigenvector():
    s = SeedGramScheduler(rng=_Fixed())
    assert s.refit(_Tracker(EDGES))
    exp = _expected_loading()
    assert s._loading == pytest.approx(exp)
    assert max(s._loading.values()) == pytest.approx(1.0)
    # The tight cluster is the dominant mode.
    assert s._loading["a"] > s._loading["d"] > s._loading["e"] - 1e-9


def test_peripheral_prefers_off_mode_seed():
    keys = sorted(EDGES)
    s = SeedGramScheduler(rng=_Fixed(0.0))
    s.refit(_Tracker(EDGES))
    # u = 0 lands on the first seed with a nonzero ticket in key order.
    tickets = [FLOOR + (1.0 - s._loading[k]) for k in keys]
    assert s.select_seed(keys) == keys[0]
    assert tickets[0] < max(tickets)


def test_central_mode_tickets_follow_loading():
    s = SeedGramScheduler(rng=_Fixed(), mode="central")
    s.refit(_Tracker(EDGES))
    assert s._ticket("a") == pytest.approx(FLOOR + s._loading["a"])
    p = SeedGramScheduler(rng=_Fixed(), mode="peripheral")
    p.refit(_Tracker(EDGES))
    assert p._ticket("a") == pytest.approx(FLOOR + 1.0 - p._loading["a"])


def test_unrefit_seed_takes_max_ticket_and_pick_is_inverse_cdf():
    keys = ["a", "d", "zz"]
    s = SeedGramScheduler(rng=_Fixed(0.999999))
    s.refit(_Tracker(EDGES))
    t_a, t_d = s._ticket("a"), s._ticket("d")
    tz = max(t_a, t_d)
    total = t_a + t_d + tz
    # 0.999999 of the mass falls in the last slot, the unrefit seed.
    assert s.select_seed(keys) == "zz" and (t_a + t_d) / total < 0.999999


def test_no_refit_is_uniform_and_empty_edges_decline():
    s = SeedGramScheduler(rng=_Fixed(0.5))
    assert s.select_seed(["x", "y"]) == "y"
    assert s.select_seed([]) == ""
    assert not s.refit(_Tracker({}))
    assert not s.refit(_Tracker({"a": set()}))


def test_maybe_refit_interval_gate():
    s = SeedGramScheduler(rng=_Fixed(), refit_interval=100)
    assert s.maybe_refit(_Tracker(EDGES), 0)
    assert not s.maybe_refit(_Tracker(EDGES), 1)


def test_bad_args():
    with pytest.raises(ValueError):
        SeedGramScheduler(rng=None)
    with pytest.raises(ValueError):
        SeedGramScheduler(rng=_Fixed(), mode="x")
