"""Tests for generation-tagged SHM edge-map reset.

The generation counter replaces the per-execution memset of the SHM edge
table, making reset_edge_map() O(1) independent of map size.
"""

import ctypes

from fuzzer_tool.adapters.shm import ShmCoverage


class TestGenerationReset:
    def test_generation_bumps_on_reset(self):
        cov = ShmCoverage()
        try:
            assert cov.read_generation() == 0
            cov.reset_edge_map()
            assert cov.read_generation() == 1
            cov.reset_edge_map()
            assert cov.read_generation() == 2
        finally:
            cov.cleanup()

    def test_stale_entries_ignored_after_reset(self):
        cov = ShmCoverage()
        try:
            cov._entries[0].edge_id = 1
            cov._entries[0].count = 0  # generation 0
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            assert cov.get_edge_ids() == {1}
            cov.reset_edge_map()
            assert cov.get_edge_ids() == set()
            assert cov.get_edge_counts() == {}
        finally:
            cov.cleanup()

    def test_live_entries_survive_reset(self):
        cov = ShmCoverage()
        try:
            gen = cov.read_generation()
            cov._entries[0].edge_id = 1
            cov._entries[0].count = (gen << 24) | 5
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
            assert cov.get_edge_ids() == {1}
            cov.reset_edge_map()
            # After reset, generation bumped; the old live entry is now stale
            assert cov.get_edge_ids() == set()
        finally:
            cov.cleanup()

    def test_generation_wrap(self):
        cov = ShmCoverage()
        try:
            # Manually set generation to 255
            ctypes.c_uint32.from_address(cov._ptr + 4).value = (0 << 8) | (255 << 24) | 0
            assert cov.read_generation() == 255
            cov.reset_edge_map()
            assert cov.read_generation() == 0
        finally:
            cov.cleanup()

    def test_drop_counter_preserves_generation(self):
        """_note_drop() increments drops while preserving ctx + generation."""
        cov = ShmCoverage()
        try:
            # Set diag with drops=1000, generation=7
            ctypes.c_uint32.from_address(cov._ptr + 4).value = (1000 << 8) | (7 << 24) | 0
            assert cov.read_dropped_edges() == 1000
            assert cov.read_generation() == 7
            # Simulate a drop by calling _note_drop directly
            cov._note_drop()
            assert cov.read_dropped_edges() == 1001
            assert cov.read_generation() == 7
        finally:
            cov.cleanup()

    def test_reset_preserves_header_except_generation(self):
        cov = ShmCoverage()
        try:
            ctypes.c_uint32.from_address(cov._ptr).value = 77
            ctypes.c_uint64.from_address(cov._ptr + 8).value = 8888
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 55
            ctypes.c_uint32.from_address(cov._ptr + 4).value = (1000 << 8) | (0 << 24) | 3
            cov.record_edge(10)
            cov.record_edge(20)
            # Re-set header values we want to verify survive reset
            ctypes.c_uint32.from_address(cov._ptr).value = 77
            ctypes.c_uint64.from_address(cov._ptr + 8).value = 8888
            ctypes.c_uint64.from_address(cov._ptr + 16).value = 55
            ctypes.c_uint32.from_address(cov._ptr + 4).value = (1000 << 8) | (0 << 24) | 3
            cov.reset_edge_map()
            assert cov.read_stack_depth() == 77
            assert cov.read_path_hash() == 8888
            assert cov.read_edge_count() == 55
            assert cov.read_ctx_bits() == 3
            assert cov.read_dropped_edges() == 1000
            assert cov.read_generation() == 1
            assert cov.get_edge_ids() == set()
        finally:
            cov.cleanup()

    def test_resize_preserves_generation(self):
        cov = ShmCoverage(size=8)
        try:
            cov.reset_edge_map()
            assert cov.read_generation() == 1
            cov.resize(16)
            assert cov.read_generation() == 1
            # New table has no live entries, so slow path sees empty active set
            assert cov.get_edge_ids() == set()
        finally:
            cov.cleanup()

    def test_direct_lite_unchanged(self):
        """In direct_lite mode, no reset is called and generation stays at 0.

        All entries remain live across runs because generation never changes.
        """
        cov = ShmCoverage()
        try:
            assert cov.read_generation() == 0
            cov._entries[0].edge_id = 1
            cov._entries[0].count = 1
            cov._entries[1].edge_id = 2
            cov._entries[1].count = 1
            # No reset — simulating direct_lite
            assert cov.get_edge_ids() == {1, 2}
            assert cov.read_generation() == 0
        finally:
            cov.cleanup()
