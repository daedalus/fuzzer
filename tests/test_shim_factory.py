"""Unit tests for adapters/shim_factory.py — coverage shim builder."""

import os
import tempfile

import pytest

from fuzzer_tool.adapters.shim_factory import (
    ShimResult,
    _cache_key,
    _inspect_target,
    cleanup_shim,
    load_shim,
)


class TestCacheKey:
    def test_deterministic(self):
        assert _cache_key("/foo/bar", "direct") == _cache_key("/foo/bar", "direct")

    def test_different_targets_differ(self):
        assert _cache_key("/a", "direct") != _cache_key("/b", "direct")

    def test_different_modes_differ(self):
        assert _cache_key("/a", "direct") != _cache_key("/a", "subprocess")

    def test_16char_hex(self):
        import re
        k = _cache_key("test", "mode")
        assert len(k) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", k)


class TestShimResult:
    def test_default_fields(self):
        r = ShimResult()
        assert r.shim_path is None
        assert r.coverage_type == "none"
        assert r.bitmap_size == 0
        assert r.needs_preload is False
        assert r.compile_error is None

    def test_custom_fields(self):
        r = ShimResult(shim_path="/tmp/shim.so", coverage_type="inline_8bit", bitmap_size=65536)
        assert r.shim_path == "/tmp/shim.so"
        assert r.coverage_type == "inline_8bit"
        assert r.bitmap_size == 65536


class TestInspectTarget:
    def test_nonexistent_file(self):
        info = _inspect_target("/nonexistent/binary")
        assert info["is_shared_lib"] is False
        assert info["coverage_type"] == "none"

    def test_shared_lib_extension(self, tmp_path):
        so = tmp_path / "libfoo.so"
        so.write_bytes(b"\x7fELF" + b"\x00" * 100)
        info = _inspect_target(str(so))
        assert info["is_shared_lib"] is True

    def test_not_shared_lib(self, tmp_path):
        exe = tmp_path / "myprog"
        exe.write_bytes(b"\x7fELF" + b"\x00" * 100)
        info = _inspect_target(str(exe))
        assert info["is_shared_lib"] is False

    def test_returns_all_keys(self):
        info = _inspect_target("/nonexistent")
        expected_keys = {
            "is_shared_lib", "coverage_type", "has_sancov_counters",
            "has_undefined_sancov_init", "has_asan",
        }
        assert expected_keys == set(info.keys())


class TestCleanupShim:
    def test_cleanup_existing(self, tmp_path):
        shim = tmp_path / "test.so"
        shim.write_bytes(b"fake")
        assert shim.exists()
        cleanup_shim(str(shim))
        assert not shim.exists()

    def test_cleanup_nonexistent(self):
        cleanup_shim("/nonexistent/shim.so")  # should not raise

    def test_cleanup_none(self):
        cleanup_shim(None)  # should not raise


class TestLoadShim:
    def test_load_none_path(self):
        assert load_shim(None) is None

    def test_load_empty_path(self):
        assert load_shim("") is None

    def test_load_nonexistent_file(self):
        assert load_shim("/nonexistent/shim.so") is None

    def test_load_not_a_library(self, tmp_path):
        fake = tmp_path / "notlib.so"
        fake.write_bytes(b"not a shared library")
        assert load_shim(str(fake)) is None
