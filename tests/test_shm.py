"""Tests for SHM coverage adapter (sparse entry format)."""

import ctypes

from fuzzer_tool.adapters.shm import SHM_MAP_SIZE, SIZEOF_ENTRY, ShmCoverage


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
            # Set the first entry's edge_id to 1 (1st byte of uint32 LE)
            cov._entries[0].edge_id = 1
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
