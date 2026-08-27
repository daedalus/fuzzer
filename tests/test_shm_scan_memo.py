"""Regression tests for the sparse, memoized live-entry scan in ShmCoverage.

``_active_columns`` and ``_active_edge_ids`` used to mask the whole table on
every call, and both are called about once per execution -- two full-width
passes over a table that is typically under 1% occupied and unchanged between
them. The scan now selects occupied slots first and caches its result against
``_scan_key``.

The risk the memo introduces is staleness: a table that changed without moving
its key would be read from cache. These tests pin each invalidation path.
"""

import numpy as np

from fuzzer_tool.adapters.shm import _ENTRY_DTYPE, ShmCoverage


def _reference_scan(shm):
    """Full-width mask, i.e. the pre-change implementation, as the oracle."""
    arr = np.frombuffer(shm._map, dtype=_ENTRY_DTYPE, count=shm.num_entries)
    eid = arr["edge_id"]
    cnt = arr["count"]
    mask = (eid != 0) & (((cnt >> 24) & 0xFF) == shm.read_generation())
    return sorted(eid[mask].tolist()), sorted(cnt[mask].tolist())


def _assert_matches_reference(shm):
    ids, counts = shm._active_columns()
    ref_ids, ref_counts = _reference_scan(shm)
    assert sorted(ids.tolist()) == ref_ids
    assert sorted(counts.tolist()) == ref_counts
    assert sorted(shm._active_edge_ids().tolist()) == ref_ids


def test_sparse_scan_matches_full_mask():
    shm = ShmCoverage(2048)
    for edge in (3, 3, 17, 4001, 65537):
        shm.record_edge(edge)
    _assert_matches_reference(shm)


def test_empty_table_scans_clean():
    shm = ShmCoverage(256)
    ids, counts = shm._active_columns()
    assert ids.size == 0
    assert counts.size == 0
    assert shm.get_edge_ids() == set()


def test_second_call_is_served_from_memo():
    """Back-to-back reads of an unchanged table do not rescan.

    This is the whole point of the memo: _check_new_coverage and
    get_edge_ids each scan once per execution, of the same table.
    """
    shm = ShmCoverage(2048)
    shm.record_edge(11)
    first, _ = shm._active_columns()
    key = shm._scan_memo[0]
    second = shm._active_edge_ids()
    assert shm._scan_memo[0] == key
    assert second is first


def test_record_edge_invalidates_memo():
    """A count bump with no new slot still moves path_hash, so it is seen."""
    shm = ShmCoverage(2048)
    shm.record_edge(7)
    assert shm.get_edge_counts() == {7: 1}
    shm.record_edge(7)  # same slot: edge_count header does not move
    assert shm.get_edge_counts() == {7: 2}
    _assert_matches_reference(shm)


def test_reset_edge_map_invalidates_memo():
    shm = ShmCoverage(2048)
    shm.record_edge(21)
    assert shm.get_edge_ids() == {21}
    shm.reset_edge_map()
    assert shm.get_edge_ids() == set()
    shm.record_edge(22)
    assert shm.get_edge_ids() == {22}


def test_resize_invalidates_memo():
    """num_entries is part of the key: resize replaces the mapping."""
    shm = ShmCoverage(512)
    shm.record_edge(33)
    ids_before = shm.get_edge_ids()
    shm.resize(4096)
    assert shm.num_entries == 4096
    # Whatever resize leaves behind, the scan must reflect the new mapping
    # rather than the cached view of the old one.
    _assert_matches_reference(shm)
    assert shm._scan_memo is None or shm._scan_memo[0][3] == 4096
    assert isinstance(ids_before, set)


def test_zero_path_hash_tables_are_never_memoized():
    """Hand-built tables bypass the memo.

    edge_count counts new-slot insertions only and is blind to multiplicity,
    so a test that writes counts directly can change the table without moving
    the key. The fast path in _check_new_coverage documents the same carve-out.
    """
    shm = ShmCoverage(256)
    arr = np.frombuffer(shm._map, dtype=_ENTRY_DTYPE, count=shm.num_entries)
    gen = shm.read_generation()
    arr["edge_id"][0] = 99
    arr["count"][0] = (gen << 24) | 1
    assert shm.read_path_hash() == 0
    assert shm.get_edge_counts() == {99: 1}
    assert shm._scan_memo is None
    arr["count"][0] = (gen << 24) | 5
    assert shm.get_edge_counts() == {99: 5}
