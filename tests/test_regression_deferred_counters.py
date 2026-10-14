"""defer_counters(): buffered record_edges() must equal the per-edge path."""

import random

from fuzzer_tool.core.edge_tracker import EdgeTracker


def _seeds(n=200, universe=3000, rng_seed=3):
    rng = random.Random(rng_seed)
    ids = [rng.getrandbits(32) for _ in range(universe)]
    out = []
    for i in range(n):
        edges = set(rng.sample(ids, rng.choice([20, 200, 600])))
        out.append((f"s{i}", edges, {e: rng.choice([1, 1, 2, 5]) for e in edges}))
    return out


def _run(seeds, defer):
    tracker = EdgeTracker()
    tracker.defer_counters(defer)
    for key, edges, counts in seeds:
        tracker.record_edges(key, edges, hit_counts=counts)
    tracker.defer_counters(False)
    return tracker


def test_deferred_counters_match_immediate():
    seeds = _seeds()
    a, b = _run(seeds, False), _run(seeds, True)
    assert a._aggregate_totals == b._aggregate_totals
    assert a._global_edge_hits == b._global_edge_hits
    assert a._aggregate_total_count == b._aggregate_total_count
    assert a.max_hit_count == b.max_hit_count
    assert a.seed_hit_counts == b.seed_hit_counts
    assert a.cumulative_edges == b.cumulative_edges


def test_mismatched_keys_fall_back_to_immediate_path():
    tracker = EdgeTracker()
    tracker.defer_counters(True)
    tracker.record_edges("a", {1, 2, 3}, hit_counts={1: 4})  # keys differ
    assert tracker._global_edge_hits == {1: 4, 2: 1, 3: 1}
    tracker.defer_counters(False)
    assert tracker._global_edge_hits == {1: 4, 2: 1, 3: 1}
