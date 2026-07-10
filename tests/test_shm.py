"""Tests for SHM coverage adapter."""

import ctypes

from fuzzer_tool.adapters.shm import SHM_MAP_SIZE, ShmCoverage


class TestShmCoverage:
    def test_alloc_returns_valid_id(self):
        cov = ShmCoverage()
        assert cov.shm_id >= 0
        cov.cleanup()

    def test_map_size(self):
        assert SHM_MAP_SIZE == 65536

    def test_read_bitmap_returns_buffer(self):
        cov = ShmCoverage()
        try:
            buf = cov.read_bitmap()
            assert len(buf) == SHM_MAP_SIZE
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
            ctypes.memset(cov._ptr, 0, cov.size)
            ctypes.memset(cov._ptr, 1, 1)
            assert cov.is_new_coverage()
        finally:
            cov.cleanup()

    def test_cleanup_releases_shm(self):
        cov = ShmCoverage()
        cov.cleanup()
        assert cov.shm_id == -1
        assert cov._ptr is None

    def test_env_id(self):
        cov = ShmCoverage()
        try:
            assert cov.env_id.isdigit()
        finally:
            cov.cleanup()

    def test_commit_snapshot_freezes_state(self):
        cov = ShmCoverage()
        try:
            cov.reset_edge_map()
            ctypes.memset(cov._ptr, 0, cov.size)
            ctypes.memset(cov._ptr, 42, 8)
            assert cov.is_new_coverage()
            cov.commit_snapshot()
            assert not cov.is_new_coverage()
        finally:
            cov.cleanup()

    def test_resize_preserves_cumulative(self):
        cov = ShmCoverage(size=4096)
        try:
            # Simulate edges and detection
            ctypes.memset(cov._ptr, 0, cov.size)
            for i in [10, 20, 30]:
                ctypes.memset(cov._ptr + i, 1, 1)
            cov.is_new_coverage()
            assert cov.cumulative_edges == 3

            # Resize preserves cumulative
            cov.resize(8192)
            assert cov.size == 8192
            assert cov.cumulative_edges == 3
        finally:
            cov.cleanup()

    def test_resize_preserves_after_reset(self):
        cov = ShmCoverage(size=4096)
        try:
            # Edges detected, then reset zeros SHM
            ctypes.memset(cov._ptr, 0, cov.size)
            for i in [10, 20, 30]:
                ctypes.memset(cov._ptr + i, 1, 1)
            cov.is_new_coverage()
            cov.reset_edge_map()  # zeros SHM

            # Resize still preserves cumulative (saves _seen before copy)
            cov.resize(8192)
            assert cov.cumulative_edges == 3
        finally:
            cov.cleanup()

    def test_resize_updates_env_id(self):
        cov = ShmCoverage(size=4096)
        try:
            old_env = cov.env_id
            cov.resize(8192)
            assert cov.env_id != old_env
            assert cov.env_id.isdigit()
        finally:
            cov.cleanup()

    def test_resize_noop_if_smaller(self):
        cov = ShmCoverage(size=8192)
        try:
            old_size = cov.size
            cov.resize(4096)  # smaller — should be no-op
            assert cov.size == old_size
        finally:
            cov.cleanup()
