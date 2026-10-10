"""_record_sparse folds a hit set with invariants hoisted out of the edge loop.

It re-tested ``hit_counts`` truthiness and ``_f0_enabled`` per edge, set
``_spectrum_dirty`` and compared ``max_hit_count`` per edge, and added
edges to ``new_edges`` one by one: ~4% of FFmpeg fuzz-loop wall time.
State must match the per-edge loop exactly.
"""

import random

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker


def _ref_record_sparse(et, hc, hit_edges, hit_counts, new_edges):
    """Oracle: the pre-change loop, verbatim modulo self -> et."""
    for edge_id in hit_edges:
        val = hit_counts.get(edge_id, 1) if hit_counts else 1
        new_edges.add(edge_id)
        hc[edge_id] = val
        if et._f0_enabled:
            et._f0.update(edge_id)
        et._aggregate_totals[edge_id] = et._aggregate_totals.get(edge_id, 0) + val
        et._aggregate_total_count += val
        old_gh = et._global_edge_hits.get(edge_id, 0)
        et._global_edge_hits[edge_id] = old_gh + val
        et._spectrum_dirty = True
        if et._global_edge_hits[edge_id] > et.max_hit_count:
            et.max_hit_count = et._global_edge_hits[edge_id]


def _state(et, hc, new_edges):
    f0 = et._f0.estimate() if et._f0_enabled else None
    return (
        dict(hc),
        set(new_edges),
        dict(et._aggregate_totals),
        et._aggregate_total_count,
        dict(et._global_edge_hits),
        et.max_hit_count,
        et._spectrum_dirty,
        f0,
    )


def _drive(record, f0, seed, rounds=30):
    """Feed identical rounds of (hit_edges, hit_counts) through *record*."""
    et = EdgeTracker(map_size=4096, enable_f0=f0)
    et._spectrum_dirty = False
    rnd = random.Random(seed)
    hc, new_edges = {}, set()
    for r in range(rounds):
        edges = set(rnd.sample(range(1, 5000), rnd.randint(0, 300)))
        kind = r % 3
        if kind == 0:
            counts = None
        elif kind == 1:
            counts = {e: rnd.randint(1, 255) for e in edges}
        else:  # partial dict: missing ids default to 1
            counts = {e: rnd.randint(1, 255) for e in list(edges)[: len(edges) // 2]}
        record(et, hc, edges, counts, new_edges)
    return _state(et, hc, new_edges)


def _new(et, hc, edges, counts, new_edges):
    et._record_sparse(hc, edges, counts, new_edges)


@pytest.mark.parametrize("f0", [False, True])
def test_control_oracle_matches_itself(f0):
    """Rule 46: the oracle is deterministic per seed (F0 uses the RNG)."""
    assert _drive(_ref_record_sparse, f0, 1) == _drive(_ref_record_sparse, f0, 1)


@pytest.mark.parametrize("f0", [False, True])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_per_edge_loop(f0, seed):
    """Falsification: every piece of tracker state equals the old loop's,
    across None / full / partial hit-count dicts."""
    assert _drive(_new, f0, seed) == _drive(_ref_record_sparse, f0, seed)


def test_empty_hit_set_leaves_spectrum_clean():
    """Adversarial: no edges -> nothing marked dirty, nothing counted."""
    et = EdgeTracker(map_size=4096)
    et._spectrum_dirty = False
    hc, new = {}, set()
    et._record_sparse(hc, set(), None, new)
    assert (hc, new, et._spectrum_dirty, et._aggregate_total_count) == ({}, set(), False, 0)


def test_zero_counts_never_raise_max():
    """Adversarial: explicit zero hit counts add nothing and keep max."""
    et = EdgeTracker(map_size=4096)
    hc, new = {}, set()
    et._record_sparse(hc, {7, 9}, {7: 0, 9: 0}, new)
    assert et.max_hit_count == 0 and et._aggregate_total_count == 0
