"""core/set_cover.quick_cover and the live min-cover stat (Fuzzer._refresh_cover)."""

from itertools import combinations

from fuzzer_tool.core.set_cover import min_cover, quick_cover
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.test_set_cover import TRAP


def _union(seed_edges, keys):
    out = set()
    for k in keys:
        out |= seed_edges[k]
    return out


def _brute_min(seed_edges):
    """Smallest cover size by exhaustive search (independent oracle)."""
    keys = [k for k, e in seed_edges.items() if e]
    goal = _union(seed_edges, keys)
    for r in range(len(keys) + 1):
        if any(_union(seed_edges, c) == goal for c in combinations(keys, r)):
            return r
    return len(keys)


def test_quick_cover_covers_every_edge():
    """Falsification: result covers the union and drops redundant picks."""
    cover = quick_cover(TRAP)
    assert _union(TRAP, cover) == _union(TRAP, TRAP)
    assert len(cover) == _brute_min(TRAP)


def test_quick_cover_control_matches_min_cover():
    """Control first: brute force agrees with min_cover, then quick_cover."""
    assert len(min_cover(TRAP, dict.fromkeys(TRAP, 1))) == _brute_min(TRAP)
    assert len(quick_cover(TRAP)) == len(min_cover(TRAP, dict.fromkeys(TRAP, 1)))


def test_quick_cover_adversarial_inputs():
    """Empty corpus, empty edge sets, identical seeds."""
    assert quick_cover({}) == []
    assert quick_cover({"a": set(), "b": set()}) == []
    assert quick_cover({"a": {1, 2}, "b": {1, 2}, "c": set()}) == ["a"]


class _Tracker:
    def __init__(self, seed_edges):
        self.seed_edges = seed_edges
        self.owner_version = 0


def _fuzzer(seed_edges):
    f = Fuzzer.__new__(Fuzzer)
    f._edge_tracker = _Tracker(seed_edges)
    f._cover_version = -1
    f._cover_cost = 0.0
    f._cover_at = 0.0
    f._min_cover_size = -1
    return f


def test_refresh_cover_sets_size():
    f = _fuzzer(dict(TRAP))
    f._refresh_cover(now=100.0)
    assert f._min_cover_size == _brute_min(TRAP)


def test_refresh_cover_skips_unchanged_version():
    """Adversarial: no coverage change -> no recompute, even if edges mutate."""
    f = _fuzzer(dict(TRAP))
    f._refresh_cover(now=100.0)
    f._edge_tracker.seed_edges["D"] = {99}
    f._refresh_cover(now=10_000.0)
    assert f._min_cover_size == _brute_min(TRAP)

    f._edge_tracker.owner_version += 1
    f._refresh_cover(now=20_000.0)
    assert f._min_cover_size == _brute_min(TRAP) + 1


def test_refresh_cover_respects_budget():
    """A solve costing c seconds is not repeated within COVER_BUDGET * c."""
    f = _fuzzer(dict(TRAP))
    f._refresh_cover(now=100.0)
    f._cover_cost = 1.0
    f._edge_tracker.seed_edges["D"] = {99}
    f._edge_tracker.owner_version += 1
    f._refresh_cover(now=101.0)
    assert f._min_cover_size == _brute_min(TRAP)


def test_stats_line_reports_min_cover():
    from fuzzer_tool.services.stats import StatsReporter
    from tests.test_stats_reporter import _mock_fuzzer

    f = _mock_fuzzer()
    f.corpus = [b"ab"]
    f._min_cover_size = 3
    parts: list[str] = []
    StatsReporter(f)._supp_corpus(f, parts)
    assert "min-cover:3" in parts

    f._min_cover_size = -1
    parts = []
    StatsReporter(f)._supp_corpus(f, parts)
    assert not any(p.startswith("min-cover:") for p in parts)
