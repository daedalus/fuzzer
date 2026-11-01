"""EdgeTracker stores surviving per-seed edge sets as compact EdgeSets.

New keys start as builtin sets (no per-exec sort); once a prune batch of
them accumulates, the survivors are packed into EdgeSet. Oracle: a dict of
builtin sets kept beside the tracker from the same inputs.
"""

from __future__ import annotations

import random

from fuzzer_tool.core.edge_set import EdgeSet
from fuzzer_tool.core.edge_tracker import PRUNE_BATCH_FRAC, EdgeTracker

_CAP = 50
_BATCH = int(_CAP * PRUNE_BATCH_FRAC)


def _fill(et: EdgeTracker, n: int, seed: int = 7) -> dict[str, set[int]]:
    """Record *n* seeds, each owning one unique edge; return the oracle."""
    rng = random.Random(seed)
    oracle = {}
    for i in range(n):
        edges = {rng.randrange(400) for _ in range(30)} | {100_000 + i}
        et.record_edges(f"s{i}", edges)
        oracle[f"s{i}"] = edges
    return oracle


def test_survivors_compacted_with_same_edges():
    et = EdgeTracker(max_tracked_seeds=_CAP)
    oracle = _fill(et, 3 * _CAP)

    packed = [k for k, v in et.seed_edges.items() if isinstance(v, EdgeSet)]
    assert len(packed) >= len(et.seed_edges) - _BATCH
    for k, v in et.seed_edges.items():
        assert v == oracle[k]


def test_below_batch_stays_builtin():
    et = EdgeTracker(max_tracked_seeds=_CAP)
    _fill(et, _BATCH - 1)
    assert all(type(v) is set for v in et.seed_edges.values())


def test_compacts_with_prune_disabled():
    et = EdgeTracker(max_tracked_seeds=_CAP)
    et._prune_enabled = False
    oracle = _fill(et, 2 * _CAP)

    assert len(et.seed_edges) == 2 * _CAP
    loose = [k for k, v in et.seed_edges.items() if type(v) is set]
    assert len(loose) < _BATCH
    assert et.seed_edges == oracle


def test_rerecord_compacted_key_merges():
    et = EdgeTracker(max_tracked_seeds=10_000)
    oracle = _fill(et, 10_000 // 10)
    key = "s0"
    assert isinstance(et.seed_edges[key], EdgeSet)
    owners_before = et._edge_owner_count[100_000]

    extra = {555_555, 100_000}
    new = et.record_edges(key, extra)

    assert et.seed_edges[key] == oracle[key] | extra
    assert isinstance(et.seed_edges[key], EdgeSet)
    assert 555_555 in new
    # Re-recording an owned edge must not inflate its owner count.
    assert et._edge_owner_count[100_000] == owners_before
    assert et._edge_owner_count[555_555] == 1


def test_roundtrip_restores_edgesets():
    et = EdgeTracker(max_tracked_seeds=_CAP)
    oracle = _fill(et, 2 * _CAP)

    back = EdgeTracker(max_tracked_seeds=_CAP)
    back.from_dict(et.to_dict())

    assert all(isinstance(v, EdgeSet) for v in back.seed_edges.values())
    assert back.seed_edges == {k: oracle[k] for k in et.seed_edges}


def test_roundtrip_accepts_legacy_unsorted_lists():
    et = EdgeTracker()
    et.from_dict({"seed_edges": {"a": [9, 1, 1, 5]}})
    assert list(et.seed_edges["a"]) == [1, 5, 9]


def test_prune_matches_builtin_sets():
    """Compaction must not change which seeds a prune evicts."""
    packed = EdgeTracker(max_tracked_seeds=_CAP)
    _fill(packed, 4 * _CAP, seed=3)

    plain = EdgeTracker(max_tracked_seeds=_CAP)
    plain._compact = lambda: None
    _fill(plain, 4 * _CAP, seed=3)

    assert all(type(v) is set for v in plain.seed_edges.values())
    assert list(packed.seed_edges) == list(plain.seed_edges)
    assert dict(packed._edge_owner_count) == dict(plain._edge_owner_count)
