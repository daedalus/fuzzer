"""Regression: min_cover on bitsets returns exactly what the set version did.

ffmpeg campaign, ~2,900 seeds x ~2,400 edges: each --minimize-every-execs
pass spent ~12 s in min_cover (four Tie x Reduce runs over frozensets, the
holder index rebuilt per reduction round). The rewrite keeps the algorithm
and swaps the representation, so its picks -- members and order -- must
match the frozen reference in tests/support/set_cover_ref.py.
"""

from __future__ import annotations

import time
from itertools import product

from fuzzer_tool.core import set_cover
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.set_cover import DOMINATOR_SCAN_CAP, _to_masks, min_cover
from tests.support import set_cover_ref as ref


def _instance(rp: RandPool, n_seeds: int, core: int, rare: int, width: int):
    """ffmpeg-shaped: every seed hits part of a shared core plus a few rare edges."""
    seed_edges = {}
    for i in range(n_seeds):
        edges = set(rp.randrange_list(core, rp.randrange(width + 1)))
        edges |= {core + e for e in rp.randrange_list(rare, rp.randrange(4))}
        seed_edges[f"s{i}"] = edges
    sizes = {k: rp.randint(1, 8) for k in seed_edges}  # few sizes: many ties
    return seed_edges, sizes


def _runs(module, seed_edges, sizes, masks):
    """All four Tie x Reduce runs of *module*'s _Cover, with *module*'s own enums."""
    order = {k: i for i, k in enumerate(seed_edges)}
    full = {k: frozenset(e) for k, e in seed_edges.items() if e}
    sz = {k: sizes[k] for k in full}
    return [
        module._Cover(order, masks(full), sz, tie, red).solve()
        for tie, red in product(module.Tie, module.Reduce)
    ]


def _assert_same(seed_edges, sizes):
    want = ref.min_cover(seed_edges, sizes)
    got = min_cover(seed_edges, sizes)
    assert got == want
    assert _runs(ref, seed_edges, sizes, dict) == _runs(set_cover, seed_edges, sizes, _to_masks)


def test_regression_set_cover_bitset_control():
    """Hard Rule 46: the reference agrees with itself on a fresh copy."""
    seed_edges, sizes = _instance(RandPool(seed=1), 40, 30, 20, 12)
    copy = {k: set(v) for k, v in seed_edges.items()}
    assert ref.min_cover(seed_edges, sizes) == ref.min_cover(copy, dict(sizes))


def test_regression_set_cover_bitset_matches_reference():
    rp = RandPool(seed=7)
    for _ in range(60):
        n = rp.randint(1, 60)
        _assert_same(*_instance(rp, n, rp.randint(1, 40), rp.randint(1, 30), rp.randint(0, 25)))


def test_regression_set_cover_bitset_adversarial():
    """Empty sets, all empty, twins, 63-bit ids, one edge over the scan cap."""
    big = (1 << 62) + 3
    cases = [
        ({"a": set(), "b": set()}, {"a": 1, "b": 1}),
        ({"a": {big}, "b": set(), "c": {big, 0}}, {"a": 1, "b": 1, "c": 2}),
        ({f"t{i}": {1, 2, 3} for i in range(5)}, {f"t{i}": 1 for i in range(5)}),
        ({"solo": {0}}, {"solo": 9}),
    ]
    crowd = {f"c{i}": {7, i % 3} for i in range(DOMINATOR_SCAN_CAP + 40)}
    crowd["keystone"] = {7, 0, 1, 2, 99}
    cases.append((crowd, {k: 1 + len(k) % 3 for k in crowd}))
    for seed_edges, sizes in cases:
        _assert_same(seed_edges, sizes)


def test_regression_set_cover_bitset_falsify_coverage():
    """Falsification: drop any pick from a minimal result and an edge goes uncovered."""
    seed_edges, sizes = _instance(RandPool(seed=11), 200, 60, 80, 20)
    picks = min_cover(seed_edges, sizes)
    every = set().union(*seed_edges.values())
    assert set().union(*(seed_edges[k] for k in picks)) == every
    for k in picks:
        rest = set().union(*(seed_edges[j] for j in picks if j != k))
        assert rest != every


def test_regression_set_cover_bitset_speed():
    """800 seeds x ~1,500 edges: at least 2x faster than the set version."""
    seed_edges, sizes = _instance(RandPool(seed=5), 800, 6000, 4000, 3000)

    t0 = time.perf_counter()
    want = ref.min_cover(seed_edges, sizes)
    reference = time.perf_counter() - t0

    t0 = time.perf_counter()
    got = min_cover(seed_edges, sizes)
    elapsed = time.perf_counter() - t0

    assert got == want
    assert elapsed * 2 < reference, f"bitset {elapsed:.2f}s vs sets {reference:.2f}s"
