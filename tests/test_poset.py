"""``core/poset.py``: chain height and Dilworth width against brute-force enumeration.

The oracle enumerates every vertex subset and keeps the antichains (width) and
chains (height), so it shares nothing with the matching. Rule 46: it is first
run against itself on a relabelled copy of the same poset.
"""

from __future__ import annotations

import itertools

import pytest

from fuzzer_tool.core.poset import chain_cover, chain_height, poset_width, topo_order
from fuzzer_tool.core.rand_pool import RandPool

# ── helpers ───────────────────────────────────────────────────────────


def _supersets(vals: list[int]) -> list[list[int]]:
    """succ[i] = strict supersets of vals[i]: a transitively closed order."""
    return [[j for j, w in enumerate(vals) if w != v and v & w == v] for v in vals]


def _random_sets(pool: RandPool, n: int, bits: int) -> list[int]:
    """n distinct nonzero bitsets over *bits* positions."""
    vals: set[int] = set()
    while len(vals) < n:
        vals.add(pool.randint(1, (1 << bits) - 1))
    return sorted(vals)


def _brute(succ: list[list[int]]) -> tuple[int, int]:
    """(width, height) by enumerating every subset of vertices."""
    n = len(succ)
    less = {(i, j) for i in range(n) for j in succ[i]}
    comp = less | {(j, i) for i, j in less}
    width = height = 0
    for k in range(1, n + 1):
        for sub in itertools.combinations(range(n), k):
            pairs = list(itertools.combinations(sub, 2))
            if all(p not in comp for p in pairs):
                width = max(width, k)
            if all(p in comp for p in pairs):
                height = max(height, k)
    return width, height


def _relabel(succ: list[list[int]], perm: list[int]) -> list[list[int]]:
    """Same poset with vertex i renamed perm[i]."""
    out: list[list[int]] = [[] for _ in succ]
    for i, js in enumerate(succ):
        out[perm[i]] = [perm[j] for j in js]
    return out


def _tower(n: int, chains: int) -> list[int]:
    """*chains* interleaved chains: value i is a prefix minus one of its low bits."""
    return [((1 << (i // chains + chains)) - 1) ^ (1 << (i % chains)) for i in range(n)]


# ── oracle control ────────────────────────────────────────────────────


def test_oracle_control_is_label_invariant():
    pool = RandPool(3)
    for n in range(1, 9):
        succ = _supersets(_random_sets(pool, n, 4))
        perm = list(range(n))
        pool.shuffle(perm)
        assert _brute(succ) == _brute(_relabel(succ, perm))


# ── falsification ─────────────────────────────────────────────────────


def test_width_and_height_match_enumeration():
    pool = RandPool(17)
    for n in range(1, 11):
        for _ in range(4):
            succ = _supersets(_random_sets(pool, n, 5))
            assert (poset_width(succ), chain_height(succ)) == _brute(succ)


def test_chain_cover_is_a_minimum_partition_into_chains():
    pool = RandPool(23)
    for n in range(1, 12):
        succ = _supersets(_random_sets(pool, n, 5))
        chains = chain_cover(succ)
        assert sorted(v for c in chains for v in c) == list(range(n))
        for c in chains:
            assert all(b in succ[a] for a, b in itertools.pairwise(c))
        assert len(chains) == poset_width(succ) == _brute(succ)[0]


def test_given_order_matches_derived_order():
    pool = RandPool(29)
    vals = _random_sets(pool, 60, 8)
    succ = _supersets(vals)
    by_size = sorted(range(len(vals)), key=lambda k: vals[k].bit_count())
    assert chain_height(succ, by_size) == chain_height(succ)


def test_empty_total_order_and_antichain():
    assert (poset_width([]), chain_height([]), chain_cover([])) == (0, 0, [])
    n = 7
    total = [list(range(i + 1, n)) for i in range(n)]
    assert (poset_width(total), chain_height(total)) == (1, n)
    assert chain_cover(total) == [list(range(n))]
    anti: list[list[int]] = [[] for _ in range(n)]
    assert (poset_width(anti), chain_height(anti)) == (n, 1)


def test_topo_order_puts_every_edge_forward():
    pool = RandPool(31)
    succ = _supersets(_random_sets(pool, 40, 7))
    pos = {v: k for k, v in enumerate(topo_order(succ))}
    assert all(pos[i] < pos[j] for i, js in enumerate(succ) for j in js)


# ── adversarial ───────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [[[1], [0]], [[0]], [[1], [2], [0]]])
def test_adversarial_cycles_rejected(bad):
    with pytest.raises(ValueError):
        topo_order(bad)
    with pytest.raises(ValueError):
        chain_height(bad)
    with pytest.raises(ValueError):
        chain_cover(bad)


def test_adversarial_reversed_adjacency_on_a_tower():
    # Reversed successor lists make the greedy seed pick the farthest chain
    # member, so the result rests on augmenting paths.
    chains, n = 4, 120
    succ = _supersets(_tower(n, chains))
    rev = [js[::-1] for js in succ]
    assert poset_width(succ) == poset_width(rev) == chains
    assert chain_height(succ) == chain_height(rev) == n // chains
    assert len(chain_cover(rev)) == chains


def test_adversarial_shuffled_adjacency_is_order_independent():
    pool = RandPool(37)
    succ = _supersets(_random_sets(pool, 200, 10))
    want = poset_width(succ)
    for _ in range(3):
        shuf = [list(js) for js in succ]
        for js in shuf:
            pool.shuffle(js)
        assert poset_width(shuf) == want
