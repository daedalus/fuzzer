"""``--packed-edges``: EdgeTracker stores surviving per-seed edge sets as EdgeSets.

Gated by EdgeStore (default BUILTIN: builtin sets, no packing). Under PACKED,
new keys start as builtin sets (no per-exec sort); once a prune batch of them
accumulates, the survivors are packed into EdgeSet. Oracle: a dict of builtin
sets kept beside the tracker from the same inputs.
"""

from __future__ import annotations

import copy
import random
import subprocess
import sys

import pytest

from fuzzer_tool.core.edge_set import EdgeSet
from fuzzer_tool.core.edge_tracker import PRUNE_BATCH_FRAC, EdgeStore, EdgeTracker

_CAP = 50
_BATCH = int(_CAP * PRUNE_BATCH_FRAC)


def _packed(cap: int = _CAP) -> EdgeTracker:
    et = EdgeTracker(max_tracked_seeds=cap)
    et.set_edge_store(EdgeStore.PACKED)
    return et


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
    et = _packed()
    oracle = _fill(et, 3 * _CAP)

    packed = [k for k, v in et.seed_edges.items() if isinstance(v, EdgeSet)]
    assert len(packed) >= len(et.seed_edges) - _BATCH
    for k, v in et.seed_edges.items():
        assert v == oracle[k]


def test_below_batch_stays_builtin():
    et = _packed()
    _fill(et, _BATCH - 1)
    assert all(type(v) is set for v in et.seed_edges.values())


def test_compacts_with_prune_disabled():
    et = _packed()
    et._prune_enabled = False
    oracle = _fill(et, 2 * _CAP)

    assert len(et.seed_edges) == 2 * _CAP
    loose = [k for k, v in et.seed_edges.items() if type(v) is set]
    assert len(loose) < _BATCH
    assert et.seed_edges == oracle


def test_rerecord_compacted_key_merges():
    et = _packed(10_000)
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
    et = _packed()
    oracle = _fill(et, 2 * _CAP)

    back = _packed()
    back.from_dict(et.to_dict())

    assert all(isinstance(v, EdgeSet) for v in back.seed_edges.values())
    assert back.seed_edges == {k: oracle[k] for k in et.seed_edges}


def test_roundtrip_accepts_legacy_unsorted_lists():
    et = _packed()
    et.from_dict({"seed_edges": {"a": [9, 1, 1, 5]}})
    assert list(et.seed_edges["a"]) == [1, 5, 9]


def test_prune_matches_builtin_sets():
    """Compaction must not change which seeds a prune evicts."""
    packed = _packed()
    _fill(packed, 4 * _CAP, seed=3)

    plain = EdgeTracker(max_tracked_seeds=_CAP)
    _fill(plain, 4 * _CAP, seed=3)

    assert all(type(v) is set for v in plain.seed_edges.values())
    assert list(packed.seed_edges) == list(plain.seed_edges)
    assert dict(packed._edge_owner_count) == dict(plain._edge_owner_count)


def test_default_is_builtin():
    et = EdgeTracker(max_tracked_seeds=_CAP)
    oracle = _fill(et, 3 * _CAP)

    assert et._edge_store is EdgeStore.BUILTIN
    assert all(type(v) is set for v in et.seed_edges.values())
    assert not et._loose  # nothing queued: the hooks are no-ops
    assert all(et.seed_edges[k] == oracle[k] for k in et.seed_edges)


def test_builtin_roundtrip_stays_builtin():
    et = EdgeTracker()
    et.from_dict({"seed_edges": {"a": [9, 1]}})
    assert type(et.seed_edges["a"]) is set


def test_deepcopy_keeps_own_loose_list():
    et = _packed()
    _fill(et, _BATCH - 2)  # one more record stays below the packing batch
    dup = copy.deepcopy(et)
    dup.record_edges("extra", {1, 2})

    assert "extra" in dup._loose
    assert "extra" not in et._loose


def test_cli_flag():
    out = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", "--help"], capture_output=True, text=True
    ).stdout
    assert "--packed-edges" in out


@pytest.mark.parametrize("store", list(EdgeStore))
def test_fuzzer_wires_store(store, tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "s").write_bytes(b"x")
    f = Fuzzer(
        target="/bin/true",
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "k"),
        use_coverage=False,
        packed_edges=store is EdgeStore.PACKED,
    )
    assert f._edge_tracker._edge_store is store
