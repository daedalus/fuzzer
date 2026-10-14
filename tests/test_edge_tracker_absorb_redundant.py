"""EdgeTracker.absorb_redundant folds edge-free seeds into the global counters."""

from __future__ import annotations

import pytest

from fuzzer_tool.core import edge_tracker as et
from fuzzer_tool.core.edge_tracker import EdgeTracker


def _globals(t: EdgeTracker):
    return (
        dict(t._aggregate_totals),
        dict(t._global_edge_hits),
        t._aggregate_total_count,
        t.max_hit_count,
    )


def test_absorb_matches_record_edges_global_statistics() -> None:
    """Absorbing is record_edges for the global counters, minus per-seed state."""
    full, light = EdgeTracker(), EdgeTracker()
    first = ({1, 2, 3, 4}, {1: 2, 2: 1, 3: 5, 4: 1})
    redundant = [({1, 3}, {1: 4, 3: 1}), ({2, 3, 4}, None), ({3}, {3: 9})]
    for t in (full, light):
        t.record_edges("s0", first[0], target_name="", hit_counts=first[1])
    for i, (edges, counts) in enumerate(redundant, 1):
        full.record_edges(f"s{i}", edges, target_name="", hit_counts=counts)
        assert light.is_redundant(edges)
        light.absorb_redundant(edges, counts)
    light.flush_absorbed()

    assert _globals(light) == _globals(full)
    assert light.cumulative_edges == full.cumulative_edges
    assert list(light.seed_edges) == ["s0"]  # no per-seed state for absorbed seeds
    assert len(light.seed_hit_counts) == 1


def test_is_redundant_requires_a_subset_and_prior_coverage() -> None:
    t = EdgeTracker()
    assert not t.is_redundant({1})  # empty tracker: the first seed is never redundant
    t.record_edges("s0", {1, 2}, target_name="")
    assert t.is_redundant({1})
    assert t.is_redundant({1, 2})
    assert not t.is_redundant({1, 3})


def test_absorb_flushes_when_batch_is_large(monkeypatch) -> None:
    monkeypatch.setattr(et, "ABSORB_FLUSH_ENTRIES", 5)
    t = EdgeTracker()
    t.record_edges("s0", {1, 2, 3}, target_name="")
    t.absorb_redundant({1, 2, 3})
    assert t._absorb_pending  # 3 < 5, still buffered
    t.absorb_redundant({1, 2, 3})
    assert not t._absorb_pending  # 6 >= 5, flushed
    assert t._global_edge_hits[1] == 3


@pytest.mark.parametrize("enabled", [False, True])
def test_absorb_feeds_f0_only_when_enabled(enabled: bool) -> None:
    t = EdgeTracker(enable_f0=enabled)
    t.record_edges("s0", {1, 2, 3}, target_name="")
    t.absorb_redundant({1, 2})
    t.flush_absorbed()
    if enabled:
        assert t._f0.size > 0
    assert t._aggregate_total_count == 5


def test_absorb_arrays_matches_dict_path() -> None:
    """The numpy-column entry point gives the same global counters as the dict one."""
    import numpy as np

    dict_t, arr_t = EdgeTracker(), EdgeTracker()
    for t in (dict_t, arr_t):
        t.record_edges("s0", {1, 2, 3, 4}, target_name="", hit_counts={1: 2, 2: 1, 3: 5, 4: 1})
    runs = [({1, 3}, {1: 4, 3: 1}), ({2, 3, 4}, {2: 1, 3: 2, 4: 7})]
    for edges, counts in runs:
        dict_t.absorb_redundant(edges, counts)
        ids = np.array(sorted(edges), dtype=np.uint32)
        arr_t.absorb_redundant_arrays(ids, np.array([counts[e] for e in sorted(edges)], np.int64))
    dict_t.flush_absorbed()
    arr_t.flush_absorbed()

    assert _globals(arr_t) == _globals(dict_t)
