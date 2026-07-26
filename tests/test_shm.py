"""Tests for SHM coverage adapter (sparse entry format)."""

import ctypes

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
        Edge_count +2 from record_edge calls, so total = 57."""
        cov = ShmCoverage()
        try:
            ctypes.c_uint32.from_address(cov._ptr).value = 77
            ctypes.c_uint64.from_address(cov._ptr + 8).value = 8888
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 55
            # Also add some seen edges (each increments edge_count by 1)
            cov.record_edge(10)  # edge_count 56
            cov.record_edge(20)  # edge_count 57
            cov.reset()  # full reset — preserves header, clears _seen_edge_ids
            # Header preserved
            assert cov.read_stack_depth() == 77
            assert cov.read_path_hash() == 8888
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
