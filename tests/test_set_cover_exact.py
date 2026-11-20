"""exact_min_cover / min_cover(exact_nodes=...): branch-and-bound on top of greedy."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from fuzzer_tool.core.set_cover import exact_min_cover, min_cover


def _instance(rng: random.Random, n_seeds: int, n_edges: int, p: float):
    edges = {f"s{i}": {e for e in range(n_edges) if rng.random() < p} for i in range(n_seeds)}
    edges = {k: v for k, v in edges.items() if v}
    # make every edge coverable so the optimum is defined
    for e in range(n_edges):
        if not any(e in v for v in edges.values()):
            edges.setdefault("fix", set()).add(e)
    sizes = {k: rng.randrange(1, 50) for k in edges}
    return edges, sizes


def _optimum(edges) -> int:
    universe = set().union(*edges.values())
    keys = list(edges)
    for r in range(1, len(keys) + 1):
        for combo in combinations(keys, r):
            if set().union(*(edges[k] for k in combo)) == universe:
                return r
    raise AssertionError


def _covers(edges, picked) -> bool:
    return set().union(*(edges[k] for k in picked)) == set().union(*edges.values())


@pytest.mark.parametrize("seed", range(60))
def test_exact_matches_brute_force(seed):
    rng = random.Random(seed)
    edges, sizes = _instance(rng, rng.randrange(4, 11), rng.randrange(5, 16), 0.35)
    picked, proven = exact_min_cover(edges, sizes)
    assert proven
    assert _covers(edges, picked)
    assert len(picked) == _optimum(edges)
    assert len(picked) <= len(min_cover(edges, sizes))


def test_exact_beats_greedy_somewhere():
    gains = 0
    for seed in range(300):
        rng = random.Random(1000 + seed)
        edges, sizes = _instance(rng, 9, 14, 0.3)
        greedy = min_cover(edges, sizes)
        exact, proven = exact_min_cover(edges, sizes)
        assert proven and _covers(edges, exact)
        assert len(exact) <= len(greedy)
        gains += len(exact) < len(greedy)
    assert gains > 0


def test_exact_nodes_wires_through_min_cover():
    rng = random.Random(7)
    edges, sizes = _instance(rng, 10, 16, 0.3)
    assert len(min_cover(edges, sizes, exact_nodes=50_000)) == _optimum(edges)
    assert min_cover(edges, sizes) == min_cover(edges, sizes, exact_nodes=0)


def test_tiny_budget_never_worse_than_greedy_and_unproven():
    rng = random.Random(11)
    edges, sizes = _instance(rng, 40, 60, 0.12)
    greedy = min_cover(edges, sizes)
    picked, proven = exact_min_cover(edges, sizes, node_budget=1)
    assert not proven
    assert _covers(edges, picked)
    assert (len(picked), sum(sizes[k] for k in picked)) <= (
        len(greedy),
        sum(sizes[k] for k in greedy),
    )


def test_empty_and_single():
    assert exact_min_cover({}, {}) == ([], False)
    assert exact_min_cover({"a": {1, 2}}, {"a": 3})[0] == ["a"]


def test_ties_prefer_fewer_bytes():
    edges = {"big": {1, 2}, "small": {1, 2}, "x": {3}}
    sizes = {"big": 100, "small": 1, "x": 5}
    picked, _ = exact_min_cover(edges, sizes)
    assert sorted(picked) == ["small", "x"]
