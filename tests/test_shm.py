"""Tests for SHM coverage adapter."""

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

    def test_reset_clears_snapshot(self):
        cov = ShmCoverage()
        try:
            cov.read_bitmap()
            cov.reset()
            assert cov.is_new_coverage() is False
        finally:
            cov.cleanup()

    def test_is_new_coverage_false_initially(self):
        cov = ShmCoverage()
        try:
            cov.reset()
            assert cov.is_new_coverage() is False
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
