"""Regression: EdgeTracker._maybe_prune at ffmpeg scale.

ffmpeg campaign (1,100 tracked seeds x ~2,400 edges): 20 prunes cost 39 s of
a 6,000-round run -- a Python owner-count pass plus a per-seed unique-loss
sum, both O(seeds x edges). The vectorized pass must evict the same seeds and
leave the same owner counts as the rule it replaces (reference below).
"""

from __future__ import annotations

import copy
import heapq
import time

from fuzzer_tool.core.edge_tracker import PRUNE_BATCH_FRAC, EdgeTracker
from fuzzer_tool.core.rand_pool import RandPool


def _reference_prune(seed_edges: dict[str, set[int]], ceiling: int):
    """The pre-vectorization eviction rule, in plain Python."""
    batch = int(ceiling * PRUNE_BATCH_FRAC)
    excess = len(seed_edges) - max(1, ceiling - batch)
    owners: dict[int, int] = {}
    for edges in seed_edges.values():
        for e in edges:
            owners[e] = owners.get(e, 0) + 1

    def loss(k):
        return sum(1 for e in seed_edges[k] if owners.get(e, 0) <= 1)

    heap = [(loss(k), i, k) for i, k in enumerate(seed_edges)]
    heapq.heapify(heap)
    pruned: list[str] = []
    while len(pruned) < excess and heap:
        cur, order, key = heapq.heappop(heap)
        now = loss(key)
        if now != cur:
            heapq.heappush(heap, (now, order, key))
            continue
        pruned.append(key)
        for e in seed_edges[key]:
            owners[e] -= 1
    return pruned, {e: n for e, n in owners.items() if n > 0}


def _fill(et: EdgeTracker, n: int, rng: RandPool, universe: int, width: int):
    """Record *n* seeds into *et* without triggering its own prune."""
    ceiling = et.max_tracked_seeds
    et.max_tracked_seeds = n + 1
    for i in range(n):
        edges = set(rng.randrange_list(universe, rng.randrange(width + 1)))
        et.record_edges(f"s{i}", edges)
    et.max_tracked_seeds = ceiling


def _check_against_reference(n: int, ceiling: int, universe: int, width: int, seed: int):
    et = EdgeTracker(max_tracked_seeds=n + 1)
    _fill(et, n, RandPool(seed=seed), universe, width)
    before = {k: set(v) for k, v in et.seed_edges.items()}
    want_pruned, want_owners = _reference_prune(before, ceiling)

    et.max_tracked_seeds = ceiling
    et._maybe_prune()

    assert set(et.seed_edges) == set(before) - set(want_pruned)
    assert dict(et._edge_owner_count) == want_owners


def test_regression_prune_vectorized_control():
    """Hard Rule 46: the reference against itself on one input agrees."""
    rng_state = {f"s{i}": {i % 7, i % 11, 100 + i} for i in range(40)}
    assert _reference_prune(rng_state, 30) == _reference_prune(rng_state, 30)


def test_regression_prune_vectorized_matches_reference():
    for seed in range(12):
        _check_against_reference(n=160, ceiling=100, universe=300, width=40, seed=seed)


def test_regression_prune_vectorized_cascading_losses():
    """Tiny universe: evictions make survivors' edges unique, losses go stale."""
    for seed in range(12):
        _check_against_reference(n=60, ceiling=20, universe=12, width=4, seed=seed)


def test_regression_prune_vectorized_adversarial_empty_and_huge_ids():
    """Empty edge sets, id 0 and 63-bit ids, all-identical seeds."""
    et = EdgeTracker(max_tracked_seeds=1000)
    big = (1 << 62) + 5
    sets = [set(), {0, big}, {big}, set(), {0}, {7}, {7}, {7}, set(), {big, 7, 0}]
    for i, s in enumerate(sets):
        et.record_edges(f"k{i}", s)
    before = {k: set(v) for k, v in et.seed_edges.items()}
    want_pruned, want_owners = _reference_prune(before, 5)

    et.max_tracked_seeds = 5
    et._maybe_prune()

    assert set(et.seed_edges) == set(before) - set(want_pruned)
    assert dict(et._edge_owner_count) == want_owners


def test_regression_prune_vectorized_falsify_wrong_victim():
    """Falsification: the seed owning the only copy of an edge must survive."""
    et = EdgeTracker(max_tracked_seeds=1000)
    for i in range(20):
        et.record_edges(f"dup{i}", {1, 2, 3})
    et.record_edges("keystone", {1, 2, 3, 4242})
    et.max_tracked_seeds = 10
    et._maybe_prune()
    assert "keystone" in et.seed_edges
    assert 4242 in et._edge_owner_count


def test_regression_prune_vectorized_speed():
    """700 seeds x 1,500 edges: at least 1.4x faster than the Python rule.

    Best of 3 on both sides: single runs swing ~2x on a loaded box.
    """
    base = EdgeTracker(max_tracked_seeds=701)
    _fill(base, 700, RandPool(seed=3), universe=20_000, width=3000)
    snapshot = {k: set(v) for k, v in base.seed_edges.items()}

    reference = elapsed = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        _reference_prune(snapshot, 600)
        reference = min(reference, time.perf_counter() - t0)

        et = copy.deepcopy(base)
        et.max_tracked_seeds = 600
        t0 = time.perf_counter()
        et._maybe_prune()
        elapsed = min(elapsed, time.perf_counter() - t0)
        assert len(et.seed_edges) <= 600

    assert elapsed * 1.4 < reference, f"prune {elapsed:.2f}s vs reference {reference:.2f}s"
