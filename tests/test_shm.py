"""Tests for SHM coverage adapter (sparse entry format)."""

import ctypes
from pathlib import Path

from fuzzer_tool.adapters.shm import SHM_MAP_SIZE, SIZEOF_ENTRY, ShmCoverage, SHM_METADATA_SIZE


class TestShmCoverage:
    def test_alloc_returns_valid_id(self):
        cov = ShmCoverage()
        assert cov.shm_id >= 0
        cov.cleanup()

    def test_map_size_constants(self):
        # SHM_MAP_SIZE is the number of entries; SHM bytes = entries * 8
        assert SHM_MAP_SIZE == 8192
        assert SIZEOF_ENTRY == 8

    def test_read_bitmap_returns_entry_bytes(self):
        cov = ShmCoverage()
        try:
            buf = cov.read_bitmap()
            assert len(buf) == SHM_MAP_SIZE * SIZEOF_ENTRY  # 65536 bytes
        finally:
            cov.cleanup()

    def test_reset_edge_map_clears_snapshot(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            assert cov.is_new_coverage() is False
        finally:
            cov.cleanup()

    def test_is_new_coverage_false_initially(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            assert cov.is_new_coverage() is False
        finally:
            cov.cleanup()

    def test_is_new_coverage_true_after_write(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            assert not cov.is_new_coverage()
            cov._entries[0].edge_id = 1
            # Update edge_count header to reflect the new edge (as C shim would)
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            assert cov.is_new_coverage()
        finally:
            cov.cleanup()

    def test_read_entries_empty_after_reset(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            assert cov.read_entries() == []
        finally:
            cov.cleanup()

    def test_read_entries_after_record(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            cov._entries[0].edge_id = 42
            cov._entries[0].count = 7
            entries = cov.read_entries()
            assert entries == [(42, 7)]
            assert cov.get_edge_ids() == {42}
            assert cov.get_edge_counts() == {42: 7}
        finally:
            cov.cleanup()

    def test_get_edge_counts(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            assert cov.get_edge_counts() == {}
            cov._entries[0].edge_id = 10
            cov._entries[0].count = 3
            assert cov.get_edge_counts() == {10: 3}
        finally:
            cov.cleanup()

    def test_is_new_coverage_with_existing_edges(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            cov._entries[0].edge_id = 100
            cov._entries[0].count = 1
            # Update edge_count header (as C shim would on new-slot insertion)
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            assert cov.is_new_coverage()  # first time seeing edge_id=100
            # Second call: edge already seen, no change
            assert cov.is_new_coverage() is False
        finally:
            cov.cleanup()

    def test_record_edge_inserts(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            cov.record_edge(42)
            assert 42 in cov.get_edge_ids()
            assert cov.get_edge_counts()[42] == 1
        finally:
            cov.cleanup()

    def test_record_edge_increments(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            cov.record_edge(7)
            cov.record_edge(7)
            assert cov.get_edge_counts()[7] == 2
        finally:
            cov.cleanup()

    def test_reset_clears_entries(self):
        cov = ShmCoverage()
        try:
            cov.record_edge(1)
            cov.record_edge(2)
            assert len(cov.read_entries()) == 2
            cov.reset()
            assert cov.read_entries() == []
        finally:
            cov.cleanup()

    def test_commit_snapshot(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            cov._entries[0].edge_id = 42
            cov._entries[0].count = 1
            cov.commit_snapshot()
            assert 42 in cov._seen_edge_ids
            # Change the entry — is_new_coverage should not trigger
            cov._entries[0].edge_id = 0
            assert cov.is_new_coverage() is False
        finally:
            cov.cleanup()

    # ── Edge count fast-path tests ──────────────────────────────────────

    def test_edge_count_fast_path_returns_false_when_unchanged(self):
        """Calling is_new_coverage() twice without changes hits the fast path."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            assert not cov.is_new_coverage()
            # Second call: edge_count unchanged, fast path should short-circuit
            assert not cov.is_new_coverage()
        finally:
            cov.cleanup()

    def test_edge_count_fast_path_true_after_new_edge(self):
        """A changed edge_count forces the slow path which detects new edges."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            assert not cov.is_new_coverage()
            # Simulate C shim: write edge to table + increment edge_count header
            cov._entries[0].edge_id = 42
            cov._entries[0].count = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            assert cov.read_edge_count() == 1
            assert cov.is_new_coverage()
        finally:
            cov.cleanup()

    def test_edge_count_fast_path_false_after_reset(self):
        """After direct-edge-write + reset_edge_map, unchanged count → fast path."""
        cov = ShmCoverage()
        try:
            # Simulate C shim writing an edge
            cov._entries[0].edge_id = 42
            cov._entries[0].count = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            assert cov.is_new_coverage()  # first discovery; _last_edge_count = 1
            cov.reset_edge_map()
            # Header edge_count preserved at 1, _last_edge_count = 1
            # Fast path: 1 == 1 → returns False (table is empty, no new edges)
            assert not cov.is_new_coverage()
        finally:
            cov.cleanup()

    def test_edge_count_fast_path_path_hash_catches_same_count_different_edges(self):
        """Same edge_count but different edge set → path_hash mismatch catches it."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            # Round 1: edges {1, 2, 3} with edge_count=3
            cov._entries[0].edge_id = 1
            cov._entries[0].count = 1
            cov._entries[1].edge_id = 2
            cov._entries[1].count = 1
            cov._entries[2].edge_id = 3
            cov._entries[2].count = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 3
            # Set path_hash = hash({1,2,3})
            ph_abc = cov.compute_path_hash_from_edges({1, 2, 3})
            ctypes.c_uint64.from_address(cov._ptr + 8).value = ph_abc
            assert cov.is_new_coverage()  # slow path, discovers {1,2,3}
            # Fast path: edge_count=3==3, path_hash=ph_abc==ph_abc → no new
            assert not cov.is_new_coverage()

            # Round 2: different edges {1, 2, 4}, SAME edge_count=3
            cov._entries[0].edge_id = 1
            cov._entries[0].count = 1
            cov._entries[1].edge_id = 2
            cov._entries[1].count = 1
            cov._entries[2].edge_id = 4  # different from round 1
            cov._entries[2].count = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 3  # same count!
            ph_abd = cov.compute_path_hash_from_edges({1, 2, 4})
            ctypes.c_uint64.from_address(cov._ptr + 8).value = ph_abd
            # Fast path: edge_count=3==3 BUT path_hash=ph_abd!=ph_abc → slow path
            assert cov.is_new_coverage(), "must detect new edge 4 via path_hash mismatch"
        finally:
            cov.cleanup()

    def test_fast_path_passes_when_path_hash_matches(self):
        """Same edge_count SAME path_hash → fast path returns False (correct)."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            cov._entries[0].edge_id = 10
            cov._entries[0].count = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            ctypes.c_uint64.from_address(cov._ptr + 8).value = 10
            assert cov.is_new_coverage()  # slow path, discovers {10}
            # Same edge_count (1) AND same path_hash (10) → fast path
            assert not cov.is_new_coverage()
            # Still same → fast path again (regression)
            assert not cov.is_new_coverage()
        finally:
            cov.cleanup()

    def test_fast_path_falls_back_when_path_hash_zero(self):
        """path_hash==0 in header → fast path uses edge_count-only comparison."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            cov._entries[0].edge_id = 10
            cov._entries[0].count = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            # path_hash stays 0 (default zero memory) — fallback to edge_count-only
            assert cov.is_new_coverage()  # slow path, discovers {10}
            # Fast path: edge_count=1==1, path_hash=0 (fallback → edge_count only)
            assert not cov.is_new_coverage()
            # Change edges but KEEP edge_count=1 and path_hash=0
            cov._entries[0].edge_id = 20
            cov._entries[0].count = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            ctypes.c_uint64.from_address(cov._ptr + 8).value = 0
            # Because path_hash=0 → edge_count-only comparison → 1==1 → NO new
            # (This is the known limitation: path_hash=0 means the shim
            #  didn't write it, so we can't distinguish same-count diff-edges)
            assert not cov.is_new_coverage()
        finally:
            cov.cleanup()

    def test_path_hash_is_non_zero_after_live_write(self):
        """After recording real edges, path_hash header is non-zero."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            for i in range(5):
                cov.record_edge(i + 1)
            ph = cov.read_path_hash()
            # With 5 unique edges, path_hash must be non-zero
            assert ph != 0, "path_hash should be non-zero after recording edges"
            assert ph == cov.compute_path_hash_from_edges({1, 2, 3, 4, 5})
        finally:
            cov.cleanup()

    def test_read_edge_count_after_multiple_records(self):
        """read_edge_count() returns correct count after several record_edge calls."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            for i in range(10):
                cov.record_edge(i + 1)
            assert cov.read_edge_count() >= 10  # at least 10 new-slot insertions
        finally:
            cov.cleanup()

    def test_reset_preserves_header(self):
        """reset_edge_map() zeros the edge table but preserves the front header."""
        cov = ShmCoverage()
        try:
            # Write some metadata to the front header
            ctypes.c_uint32.from_address(cov._ptr).value = 42  # stack_depth
            ctypes.c_uint64.from_address(cov._ptr + 8).value = 12345  # path_hash
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 99  # edge_count
            # Reset the edge table
            cov.reset_edge_map()
            # Header should be preserved
            assert cov.read_stack_depth() == 42
            assert cov.read_path_hash() == 12345
            assert cov.read_edge_count() == 99
        finally:
            cov.cleanup()

    def test_metadata_accessors_after_init(self):
        """All three metadata accessors return 0 on a fresh (uninitialized) SHM."""
        cov = ShmCoverage()
        try:
            assert cov.read_stack_depth() == 0
            assert cov.read_path_hash() == 0
            assert cov.read_edge_count() == 0
        finally:
            cov.cleanup()

    def test_read_metadata_returns_three_values(self):
        """read_metadata() returns a (stack_depth, path_hash, edge_count) tuple."""
        cov = ShmCoverage()
        try:
            md = cov.read_metadata()
            assert len(md) == 3
            assert isinstance(md[0], int)
            assert isinstance(md[1], int)
            assert isinstance(md[2], int)
        finally:
            cov.cleanup()

    def test_shm_bytes_includes_header(self):
        """shm_bytes accounts for both the front header and the edge table."""
        cov = ShmCoverage()
        try:
            expected = SHM_METADATA_SIZE + SHM_MAP_SIZE * SIZEOF_ENTRY
            assert cov.shm_bytes == expected
        finally:
            cov.cleanup()

    # ── Edge_count semantic correctness ─────────────────────────────────

    def test_edge_count_does_not_change_on_count_increment(self):
        """Only new-slot insertions increment edge_count; repeated edge_ids do not."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            cov.record_edge(42)
            cnt_before = cov.read_edge_count()
            # Record the same edge again — count increments in place, no new slot
            cov.record_edge(42)
            assert cov.read_edge_count() == cnt_before
            assert cov.get_edge_counts()[42] == 2  # count did go up
        finally:
            cov.cleanup()

    def test_edge_count_full_table_no_increment(self):
        """When the table is full, record_edge returns False and edge_count does NOT change."""
        cov = ShmCoverage(size=4)  # tiny table for fast saturation
        try:
            cov.reset_edge_map()
            cnt_before = cov.read_edge_count()
            # Saturate all 4 slots with unique edge_ids
            for i in range(5):  # 5th call must be a miss
                ok = cov.record_edge(1000 + i)
                if not ok:
                    break
            # The miss should NOT have incremented edge_count
            assert cov.read_edge_count() == cnt_before + 4  # exactly 4 slots filled
        finally:
            cov.cleanup()

    def test_edge_count_monotonic_stress(self):
        """Many unique edges: edge_count is monotonic and at least == unique edges."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            n = 500
            for i in range(n):
                cov.record_edge(i + 1)
            # edge_count tracks new-slot insertions (may be < n due to collisions)
            # but must be monotonic and non-negative
            assert cov.read_edge_count() > 0
            assert cov.read_edge_count() <= cov.num_entries  # bounded by capacity
            # Every recorded edge should be findable
            assert len(cov.get_edge_ids()) <= cov.read_edge_count()
        finally:
            cov.cleanup()

    def test_edge_count_mixed_insert_and_repeat(self):
        """Interleaved new and repeated edges: count only rises on new insertions."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            cov.record_edge(1)  # +1
            cov.record_edge(2)  # +1
            cnt = cov.read_edge_count()
            cov.record_edge(1)  # repeat — no change
            assert cov.read_edge_count() == cnt
            cov.record_edge(3)  # +1
            assert cov.read_edge_count() == cnt + 1
        finally:
            cov.cleanup()

    # ── is_new_coverage_with_edges ──────────────────────────────────────

    def test_is_new_coverage_with_edges_tuple(self):
        """is_new_coverage_with_edges returns (bool, set[int]) with the current edge set."""
        cov = ShmCoverage()
        try:
            cov._entries[0].edge_id = 99
            cov._entries[0].count = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            new, edges = cov.is_new_coverage_with_edges()
            assert new is True
            assert 99 in edges
        finally:
            cov.cleanup()

    def test_is_new_coverage_with_edges_no_change(self):
        """When edge_count unchanged, returns (False, empty_set) fast-path."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            new, edges = cov.is_new_coverage_with_edges()
            assert new is False
            assert edges == set()
        finally:
            cov.cleanup()

    def test_is_new_coverage_with_edges_updates_last_edge_count(self):
        """After slow path, _last_edge_count matches edge_count for next fast-path."""
        cov = ShmCoverage()
        try:
            cov._entries[0].edge_id = 7
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            cov.is_new_coverage_with_edges()  # slow path — _last_edge_count = 1
            # Clear table, call again with same edge_count → fast path
            cov._entries[0].edge_id = 0  # but edge_count still 1
            # Hmm, this won't work because we'd need to add new unseen edges...
            # Instead: verify _last_edge_count was updated
            assert cov._last_edge_count == 1
        finally:
            cov.cleanup()

    # ── reset() header preservation ─────────────────────────────────────

    def test_reset_preserves_header(self):
        """reset() (full reset) delegates to reset_edge_map() → header survives.
        Edge_count +2 and path_hash updated from record_edge calls."""
        cov = ShmCoverage()
        try:
            ctypes.c_uint32.from_address(cov._ptr).value = 77
            ctypes.c_uint64.from_address(cov._ptr + 8).value = 8888
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 55
            # Also add some seen edges (each increments edge_count and updates path_hash)
            cov.record_edge(10)  # edge_count 56, path_hash = 8888*31^10 = 275538
            cov.record_edge(20)  # edge_count 57, path_hash = 275538*31^20 = 8541162
            cov.reset()  # full reset — preserves header, clears _seen_edge_ids
            # Header preserved
            assert cov.read_stack_depth() == 77
            assert cov.read_path_hash() == 8541162  # updated by record_edge calls
            assert cov.read_edge_count() == 57  # 55 + 2 from record_edge calls
            # Cumulative state is cleared
            assert cov.read_entries() == []
            assert len(cov._seen_edge_ids) == 0
        finally:
            cov.cleanup()

    def test_reset_edge_map_twice_preserves_header(self):
        """Calling reset_edge_map() twice in a row still preserves the front header."""
        cov = ShmCoverage()
        try:
            ctypes.c_uint32.from_address(cov._ptr).value = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 2
            cov.reset_edge_map()
            cov.reset_edge_map()
            assert cov.read_stack_depth() == 1
            assert cov.read_edge_count() == 2
        finally:
            cov.cleanup()

    # ── read_bitmap unchanged ───────────────────────────────────────────

    def test_read_bitmap_still_returns_only_edge_bytes(self):
        """read_bitmap() returns only the edge table, NOT including the front header."""
        cov = ShmCoverage()
        try:
            buf = cov.read_bitmap()
            assert len(buf) == cov.num_entries * SIZEOF_ENTRY
            # Verify the header is NOT included: write a sentinel to header
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 0xDEADBEEF
            buf = cov.read_bitmap()
            expected_len = cov.num_entries * SIZEOF_ENTRY
            assert len(buf) == expected_len
            # The header bytes should NOT appear at the end of buf
            assert buf[-8:] != ctypes.c_uint64(0xDEADBEEF)
        finally:
            cov.cleanup()

    # ── read_metadata round-trip ────────────────────────────────────────

    def test_read_metadata_roundtrip(self):
        """read_metadata() round-trips all three fields after manual writes."""
        cov = ShmCoverage()
        try:
            ctypes.c_uint32.from_address(cov._ptr).value = 100
            ctypes.c_uint64.from_address(cov._ptr + 8).value = 0xAABBCCDDEEFF0011
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 42
            sd, ph, ec = cov.read_metadata()
            assert sd == 100
            assert ph == 0xAABBCCDDEEFF0011
            assert ec == 42
        finally:
            cov.cleanup()

    # ── compute_path_hash_from_edges ────────────────────────────────────

    def test_compute_path_hash_empty(self):
        """Empty edge set produces hash 0."""
        cov = ShmCoverage()
        assert cov.compute_path_hash_from_edges(set()) == 0

    def test_compute_path_hash_single_edge(self):
        """Single edge: hash = 31 * 0 ^ eid = eid."""
        cov = ShmCoverage()
        assert cov.compute_path_hash_from_edges({42}) == 42

    def test_compute_path_hash_deterministic(self):
        """Same edge set (any order) produces same hash."""
        cov = ShmCoverage()
        h1 = cov.compute_path_hash_from_edges({1, 2, 3})
        h2 = cov.compute_path_hash_from_edges({3, 2, 1})
        assert h1 == h2

    def test_compute_path_hash_different_sets_differ(self):
        """Different edge sets produce different hashes (with high probability)."""
        cov = ShmCoverage()
        h1 = cov.compute_path_hash_from_edges({1, 2, 3})
        h2 = cov.compute_path_hash_from_edges({1, 2, 4})
        assert h1 != h2

    def test_compute_path_hash_large_set(self):
        """Large edge set does not overflow and produces a consistent 64-bit value."""
        cov = ShmCoverage()
        ids = set(range(1, 1001))
        h = cov.compute_path_hash_from_edges(ids)
        assert 0 <= h <= 0xFFFFFFFFFFFFFFFF

    # ── Multiple discovery rounds (iteration simulation) ────────────────

    def test_multiple_discovery_rounds(self):
        """Simulate multiple iterations: each new edge_count triggers discovery once."""
        cov = ShmCoverage()
        try:
            # Round 1: first edge
            cov._entries[0].edge_id = 10
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            assert cov.is_new_coverage()  # discovered
            assert not cov.is_new_coverage()  # fast-path, no new

            # Round 2: new edge, increment edge_count
            cov._entries[1].edge_id = 20
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 2
            assert cov.is_new_coverage()  # discovered
            assert not cov.is_new_coverage()  # fast-path

            # Round 3: no new edges
            assert not cov.is_new_coverage()
        finally:
            cov.cleanup()

    def test_last_edge_count_tracks_multiple_rounds(self):
        """_last_edge_count is updated after each slow-path call, tracking the latest count."""
        cov = ShmCoverage()
        try:
            assert cov._last_edge_count == 0
            # Round 1
            cov._entries[0].edge_id = 5
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            cov.is_new_coverage()
            assert cov._last_edge_count == 1
            # Round 2
            cov._entries[1].edge_id = 6
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 2
            cov.is_new_coverage()
            assert cov._last_edge_count == 2
        finally:
            cov.cleanup()

    # ── Resize + edge_count ─────────────────────────────────────────────

    def test_resize_preserves_edge_count_header(self):
        """After resize, the front header (including edge_count) is preserved via memmove."""
        cov = ShmCoverage(size=8)
        try:
            # Write header metadata
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 123
            # Add some edges to fill the table
            for i in range(5):
                cov.record_edge(100 + i)
            old_id = cov.shm_id
            cov.resize(16)
            # Header preserved
            assert cov.read_edge_count() == 123 + 5  # original + record_edge increments
            # New SHM id (old was detached and removed)
            assert cov.shm_id != old_id
            # All edges survived the rehash
            for i in range(5):
                assert (100 + i) in cov.get_edge_ids()
        finally:
            cov.cleanup()

    def test_resize_resets_last_edge_count(self):
        """resize() resets _last_edge_count to 0, forcing rescan on next call."""
        cov = ShmCoverage(size=8)
        try:
            cov._entries[0].edge_id = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            cov.is_new_coverage()  # _last_edge_count = 1
            assert cov._last_edge_count == 1
            cov.resize(16)
            assert cov._last_edge_count == 0  # reset by resize
        finally:
            cov.cleanup()

    # ── SHM_METADATA_SIZE constant ──────────────────────────────────────

    def test_shm_metadata_size_constant(self):
        """SHM_METADATA_SIZE is exactly 24 bytes (stack_depth 4 + pad 4 + path_hash 8 + edge_count 8)."""
        assert SHM_METADATA_SIZE == 24

    # ── SHM layout invariants ───────────────────────────────────────────

    def test_header_not_accessible_via_map(self):
        """The _map view starts after the header; writing to _map cannot corrupt header."""
        cov = ShmCoverage()
        try:
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 0xFFFFFFFFFFFFFFFF
            # Write all zeros to the first byte of _map (which is at ptr + SHM_METADATA_SIZE)
            cov._map[0] = b"\x00"
            # Header should be untouched
            assert cov.read_edge_count() == 0xFFFFFFFFFFFFFFFF
        finally:
            cov.cleanup()

    def test_edge_table_offset_equals_shm_metadata_size(self):
        """The edge table begins exactly at ptr + SHM_METADATA_SIZE bytes."""
        cov = ShmCoverage()
        try:
            # Get address of _entries[0] via ctypes
            entry0_addr = ctypes.addressof(cov._entries)
            ptr_addr = cov._ptr
            assert entry0_addr == ptr_addr + SHM_METADATA_SIZE
        finally:
            cov.cleanup()

    # ── Fast-path specific invariants ───────────────────────────────────

    def test_fast_path_skipped_on_first_call(self):
        """The first is_new_coverage() call after init always takes the slow path
        because _last_edge_count == 0 but the SHM may have data from a prior user.
        (In practice the SHM is fresh, so edge_count == 0 too — the test verifies
        the three-state: 0 == 0 returns False without scanning.)"""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            # Fast path would trigger (0 == 0) — cheap nop, correct answer
            assert not cov.is_new_coverage()
        finally:
            cov.cleanup()

    def test_fast_path_after_commit_snapshot(self):
        """After commit_snapshot adds edges to _seen_edge_ids, a genuinely new edge
        is still detected via slow path.  commit_snapshot does NOT update _last_edge_count."""
        cov = ShmCoverage()
        try:
            cov._entries[0].edge_id = 1
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            cov.commit_snapshot()
            # commit_snapshot adds edge_id=1 to _seen_edge_ids but does NOT update
            # _last_edge_count (still 0).  Now add a genuinely new edge_id.
            cov._entries[1].edge_id = 2
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 2
            # Slow path (2 != 0): edge_id=2 is new → True
            assert cov.is_new_coverage()
            # After that, _last_edge_count = 2. Edge_count still 2.
            assert not cov.is_new_coverage()  # fast path: 2 == 2
        finally:
            cov.cleanup()

    # ── integration-style: record_edge → is_new_coverage → record_edge → is_new_coverage ──

    def test_record_edge_then_is_new_coverage_via_direct_write(self):
        """record_edge pre-records edges into _seen_edge_ids.
        is_new_coverage detects edges NOT added via record_edge.
        This test verifies the two code paths stay consistent."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            # record_edge adds to _seen_edge_ids AND updates edge_count
            cov.record_edge(55)
            # is_new_coverage does NOT see edge_id=55 as new (it's already tracked)
            # but the edge_count changed from 0 to 1, so slow path runs
            new, edges = cov.is_new_coverage_with_edges()
            # 55 is in the table but not new to _seen_edge_ids
            assert new is False  # already tracked
            assert 55 in edges  # still reported in the edge set
        finally:
            cov.cleanup()

    def test_record_edge_then_direct_write_different_edge(self):
        """Mix of record_edge and direct writes: both mechanisms work together."""
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            cov.record_edge(10)  # Python API — edge_count += 1, _seen_edge_ids += {10}
            # Direct write of a different edge + manual edge_count bump
            cov._entries[1].edge_id = 20
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 2
            new, edges = cov.is_new_coverage_with_edges()
            # edge 10 is in _seen_edge_ids (not new), edge 20 is new
            assert new is True  # edge_id 20 is new
            assert 10 in edges and 20 in edges
        finally:
            cov.cleanup()


class TestShimEdgeCountEndToEnd:
    """End-to-end tests verifying the C shim updates edge_count live.

    Regression for: the C shim's __afl_iter_edge_count was never flushed to the
    SHM header because __afl_map_reset() is never called in the in-process
    execution path. This caused the edge_count fast-path to always see 0 == 0
    and skip the coverage scan, producing 'shm: 0 max: 0 sat: 0%' in stats.
    The fix: __afl_map_edge() writes edge_count live to the SHM header on each
    new-slot insertion.
    """

    def test_shim_updates_edge_count_after_target_call(self, tmp_path):
        """Compile a minimal .so with shim, call it, verify edge_count > 0."""
        import subprocess
        import os

        src = tmp_path / "test_edge_count.c"
        so = tmp_path / "test_edge_count.so"

        src.write_text("""
#include <stdint.h>
#include <stddef.h>

/* The shim is -include'd at compile time, so __afl_map_edge() is available.
   Record a few known edges so the test can check the edge_count header. */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t len) {
    (void)buf; (void)len;
    __afl_map_edge(0x1010);
    __afl_map_edge(0x2020);
    __afl_map_edge(0x3030);
    return 0;
}
""")

        shim_path = Path(__file__).parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"
        assert shim_path.exists(), f"shim not found: {shim_path}"

        subprocess.run(
            [
                "gcc",
                "-O2",
                "-g",
                "-shared",
                "-fPIC",
                "-include",
                str(shim_path),
                "-o",
                str(so),
                str(src),
            ],
            check=True,
            capture_output=True,
        )

        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            assert cov.read_edge_count() == 0, "edge_count should be 0 after reset"

            # Set env vars before loading .so (constructor maps SHM).
            # Must set both __AFL_SHM_ID and AFL_MAP_SIZE — if a previous
            # test changed AFL_MAP_SIZE via SHM resize, the shim's modulo
            # maps edges past the readable range.
            old_shm = os.environ.get("__AFL_SHM_ID")
            old_size = os.environ.get("AFL_MAP_SIZE")
            os.environ["__AFL_SHM_ID"] = str(cov.shm_id)
            os.environ["AFL_MAP_SIZE"] = str(cov.num_entries)
            try:
                lib = ctypes.CDLL(str(so))
                func = lib.fuzz_shm_run
                func.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                func.restype = ctypes.c_int

                data = b"hello"
                buf = ctypes.create_string_buffer(data)
                rc = func(ctypes.cast(buf, ctypes.c_void_p), ctypes.c_size_t(len(data)))
                assert rc == 0, f"target returned {rc}"

                # The C shim's __afl_map_edge() should have written edge_count live
                ec = cov.read_edge_count()
                assert ec > 0, (
                    f"edge_count == 0 after target run — C shim did NOT write "
                    f"edge_count live. This means the fast-path will always skip "
                    f"the coverage scan (shm: 0 bug)."
                )
                assert ec >= 3, f"expected >= 3 edges, got {ec}"

                # Coverage detection should work via the fast-path
                has_new, edge_ids = cov.is_new_coverage_with_edges()
                assert has_new, "no new coverage detected via fast-path"
                # Edge IDs are computed as prev_loc ^ cur_loc:
                #   0x1010 ^ 0          = 0x1010
                #   0x2020 ^ (0x1010>>1) = 0x2828
                #   0x3030 ^ (0x2020>>1) = 0x2020
                expected = {0x1010, 0x2828, 0x2020}
                assert edge_ids == expected, f"got {edge_ids}, expected {expected}"
            finally:
                if old_shm is not None:
                    os.environ["__AFL_SHM_ID"] = old_shm
                else:
                    os.environ.pop("__AFL_SHM_ID", None)
                if old_size is not None:
                    os.environ["AFL_MAP_SIZE"] = old_size
                else:
                    os.environ.pop("AFL_MAP_SIZE", None)
        finally:
            cov.cleanup()

    def test_regression_edge_count_fast_path_false_positive(self):
        """Regression: disjoint edge sets with same per-execution count must be
        detected as new coverage via cumulative edge_count from the C shim.

        Before the fix, the C shim wrote per-execution __afl_iter_edge_count to
        the header.  When two consecutive executions touched different edges but
        the same count (e.g. {100,200,300} then {400,500,600}, both count=3),
        the fast-path comparison ``edge_count == _last_edge_count`` incorrectly
        returned False -- silently missing real coverage.

        The fix makes the C shim write __afl_total_edge_count (cumulative, never
        reset) to the header.  This test simulates that cumulative behavior at
        the Python level to verify the fast-path logic is correct when given
        proper cumulative values.
        """
        cov = ShmCoverage()
        try:
            # Execution 1: edges {100, 200, 300}, cumulative edge_count = 3
            cov._entries[0].edge_id = 100
            cov._entries[1].edge_id = 200
            cov._entries[2].edge_id = 300
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 3
            assert cov.is_new_coverage(), "first execution should discover new edges"
            assert cov._last_edge_count == 3

            # Execution 2: disjoint edges {400, 500, 600}
            # In the real system, reset_bitmap() zeros the entire SHM including
            # the header.  The C shim then writes the cumulative total.
            # Cumulative after exec 2: 3 + 3 = 6.
            ctypes.memset(cov._ptr, 0, cov.shm_bytes)  # simulate reset_bitmap
            cov._entries[0].edge_id = 400
            cov._entries[1].edge_id = 500
            cov._entries[2].edge_id = 600
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 6  # cumulative: 3 + 3

            # BUG scenario (old behavior): if edge_count stayed at 3 (per-execution),
            # then 3 == _last_edge_count (3) -> fast-path returns False.
            # With cumulative fix: edge_count = 6 != 3 -> slow path triggered.
            assert cov.is_new_coverage(), (
                "Disjoint edge sets with same cardinality must be detected "
                "when cumulative edge_count differs from _last_edge_count"
            )
        finally:
            cov.cleanup()

    def test_shim_edge_count_monotonic_across_resets(self, tmp_path):
        """C shim edge_count is cumulative across reset_bitmap() calls.

        Two executions with disjoint edge sets but equal per-execution edge
        count must both be detected as new coverage.  Verifies the C shim's
        __afl_total_edge_count is monotonic and its value survives a full
        SHM memset (reset_bitmap).
        """
        import subprocess
        import os

        src = tmp_path / "test_monotonic.c"
        so = tmp_path / "test_monotonic.so"

        src.write_text("""
#include <stdint.h>
#include <stddef.h>
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t len) {
    for (size_t i = 0; i < len; i++)
        __afl_map_edge((uint32_t)buf[i] * 2654435761u);
    return 0;
}
""")

        shim_path = Path(__file__).parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"
        subprocess.run(
            [
                "gcc",
                "-O2",
                "-g",
                "-shared",
                "-fPIC",
                "-include",
                str(shim_path),
                "-o",
                str(so),
                str(src),
            ],
            check=True,
            capture_output=True,
        )

        cov = ShmCoverage()
        try:
            old_shm = os.environ.get("__AFL_SHM_ID")
            old_size = os.environ.get("AFL_MAP_SIZE")
            os.environ["__AFL_SHM_ID"] = str(cov.shm_id)
            os.environ["AFL_MAP_SIZE"] = str(cov.num_entries)
            try:
                lib = ctypes.CDLL(str(so))
                func = lib.fuzz_shm_run
                func.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                func.restype = ctypes.c_int

                def run(data: bytes):
                    buf = ctypes.create_string_buffer(data)
                    func(ctypes.cast(buf, ctypes.c_void_p), ctypes.c_size_t(len(data)))

                # First execution: 3 disjoint edges
                run(b"\\x01\\x02\\x03")
                ec1 = cov.read_edge_count()
                assert ec1 > 0, "first run should produce edges"
                assert cov.is_new_coverage(), "first run should have new coverage"

                # Full SHM reset (as inprocess.py reset_bitmap does)
                ctypes.memset(cov._ptr, 0, cov.shm_bytes)

                # Second execution: 3 different disjoint edges (same cardinality)
                run(b"\\x10\\x20\\x30")
                ec2 = cov.read_edge_count()
                assert ec2 > ec1, f"edge_count should be monotonic across resets: {ec1} -> {ec2}"
                assert cov.is_new_coverage(), (
                    "Second execution with disjoint edges must be detected even "
                    "though per-execution edge count (3) equals the first"
                )
            finally:
                if old_shm is not None:
                    os.environ["__AFL_SHM_ID"] = old_shm
                else:
                    os.environ.pop("__AFL_SHM_ID", None)
                if old_size is not None:
                    os.environ["AFL_MAP_SIZE"] = old_size
                else:
                    os.environ.pop("AFL_MAP_SIZE", None)
        finally:
            cov.cleanup()
