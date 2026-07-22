"""Unit tests for adapters/shim_factory.py — coverage shim builder."""

import ctypes
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.adapters.shim_factory import (
    BitmapReader,
    ShimResult,
    _cache_key,
    _compile_source,
    _find_compiler,
    _inspect_target,
    build_minimal_shim,
    build_shim,
    cleanup_shim,
    load_shim,
    read_bitmap,
    reset_bitmap,
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

    def test_compile_error(self):
        r = ShimResult(compile_error="gcc not found")
        assert r.compile_error == "gcc not found"

    def test_elf_offsets(self):
        r = ShimResult(_elf_offsets=(0x100, 0x200))
        assert r._elf_offsets == (0x100, 0x200)


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
            "is_shared_lib",
            "coverage_type",
            "has_sancov_counters",
            "has_undefined_sancov_init",
            "has_asan",
        }
        assert expected_keys == set(info.keys())

    def test_dylib_extension(self, tmp_path):
        dylib = tmp_path / "libfoo.dylib"
        dylib.write_bytes(b"\x7fELF" + b"\x00" * 100)
        info = _inspect_target(str(dylib))
        assert info["is_shared_lib"] is True

    def test_dll_extension(self, tmp_path):
        dll = tmp_path / "foo.dll"
        dll.write_bytes(b"\x7fELF" + b"\x00" * 100)
        info = _inspect_target(str(dll))
        assert info["is_shared_lib"] is True


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

    def test_load_direct_mode(self, tmp_path):
        # Build a real shim to test load
        shim = build_minimal_shim()
        if shim:
            try:
                handle = load_shim(shim, mode="direct")
                assert handle is not None
            finally:
                cleanup_shim(shim)
        else:
            pytest.skip("No compiler available")

    def test_load_subprocess_mode(self, tmp_path):
        shim = build_minimal_shim()
        if shim:
            try:
                handle = load_shim(shim, mode="subprocess")
                assert handle is not None
            finally:
                cleanup_shim(shim)
        else:
            pytest.skip("No compiler available")


class TestFindCompiler:
    def test_finds_available_compiler(self):
        cc = _find_compiler()
        assert cc in ("clang", "gcc", "cc")


class TestCompileSource:
    def test_compile_success(self, tmp_path):
        src = "int main() { return 0; }"
        out = str(tmp_path / "test_bin")
        result = _compile_source(src, out)
        assert result is True
        assert os.path.exists(out)

    def test_compile_invalid_source(self, tmp_path):
        src = "this is not valid C {{{"
        out = str(tmp_path / "test_bin")
        result = _compile_source(src, out)
        assert result is False

    def test_compile_cleanup_on_success(self, tmp_path):
        src = "int main() { return 0; }"
        out = str(tmp_path / "test_bin")
        _compile_source(src, out)
        # Source temp file should be cleaned up
        files = [f for f in os.listdir(tmp_path) if f.endswith(".c")]
        assert len(files) == 0

    def test_compile_cleanup_on_failure(self, tmp_path):
        src = "invalid C code"
        out = str(tmp_path / "test_bin")
        _compile_source(src, out)
        files = [f for f in os.listdir(tmp_path) if f.endswith(".c")]
        assert len(files) == 0


class TestBuildMinimalShim:
    def test_builds_successfully(self):
        shim = build_minimal_shim()
        if shim:
            try:
                assert os.path.exists(shim)
                assert shim.endswith(".so")
            finally:
                cleanup_shim(shim)
        else:
            pytest.skip("No compiler available")

    def test_caches_result(self):
        shim1 = build_minimal_shim()
        shim2 = build_minimal_shim()
        if shim1:
            try:
                assert shim1 == shim2
            finally:
                cleanup_shim(shim1)
        else:
            pytest.skip("No compiler available")

    def test_returns_none_on_compile_failure(self):
        with patch("fuzzer_tool.adapters.shim_factory._compile_source", return_value=False):
            shim = build_minimal_shim()
            assert shim is None

class TestBuildShim:
    def test_no_coverage_returns_none_type(self):
        with patch("fuzzer_tool.adapters.shim_factory._inspect_target") as mock:
            mock.return_value = {
                "coverage_type": "none",
                "is_shared_lib": False,
                "has_sancov_counters": False,
                "has_undefined_sancov_init": False,
                "has_asan": False,
            }
            result = build_shim("/fake/target")
            assert result.coverage_type == "none"

    def test_direct_mode_builds_minimal_shim(self):
        with (
            patch("fuzzer_tool.adapters.shim_factory._inspect_target") as mock,
            patch("fuzzer_tool.adapters.shim_factory.build_minimal_shim") as mock_build,
            patch("fuzzer_tool.adapters.shim_factory.parse_sancov_offsets") as mock_offsets,
        ):
            mock.return_value = {
                "coverage_type": "inline_8bit",
                "is_shared_lib": False,
                "has_sancov_counters": False,
                "has_undefined_sancov_init": True,
                "has_asan": False,
            }
            mock_build.return_value = "/tmp/shim.so"
            mock_offsets.return_value = (0x100, 0x200)
            result = build_shim("/fake/target", mode="direct")
            assert result.shim_path == "/tmp/shim.so"
            assert result.needs_preload is True
            assert result.bitmap_size == 0x100

    def test_direct_mode_compile_failure(self):
        with (
            patch("fuzzer_tool.adapters.shim_factory._inspect_target") as mock,
            patch("fuzzer_tool.adapters.shim_factory.build_minimal_shim", return_value=None),
        ):
            mock.return_value = {
                "coverage_type": "inline_8bit",
                "is_shared_lib": False,
                "has_sancov_counters": False,
                "has_undefined_sancov_init": True,
                "has_asan": False,
            }
            result = build_shim("/fake/target", mode="direct")
            assert result.compile_error is not None

    def test_subprocess_mode(self):
        with patch("fuzzer_tool.adapters.shim_factory._inspect_target") as mock:
            mock.return_value = {
                "coverage_type": "inline_8bit",
                "is_shared_lib": False,
                "has_sancov_counters": False,
                "has_undefined_sancov_init": True,
                "has_asan": False,
            }
            result = build_shim("/fake/target", mode="subprocess")
            assert result.bitmap_size == 65536
            assert result.needs_preload is False

    def test_uses_cache(self):
        with (
            patch("fuzzer_tool.adapters.shim_factory._shim_cache") as mock_cache,
            patch("fuzzer_tool.adapters.shim_factory._inspect_target") as mock_inspect,
        ):
            mock_cache.__contains__ = lambda self, k: True
            mock_cache.__getitem__ = lambda self, k: "/cached/shim.so"
            mock_inspect.return_value = {"coverage_type": "inline_8bit", "bitmap_size": 4096}
            with patch("os.path.exists", return_value=True):
                result = build_shim("/fake/target", mode="direct")
                assert result.shim_path == "/cached/shim.so"

    def test_direct_mode_no_offsets(self):
        with (
            patch("fuzzer_tool.adapters.shim_factory._inspect_target") as mock,
            patch("fuzzer_tool.adapters.shim_factory.build_minimal_shim") as mock_build,
            patch("fuzzer_tool.adapters.shim_factory.parse_sancov_offsets", return_value=None),
        ):
            mock.return_value = {
                "coverage_type": "inline_8bit",
                "is_shared_lib": False,
                "has_sancov_counters": False,
                "has_undefined_sancov_init": True,
                "has_asan": False,
            }
            mock_build.return_value = "/tmp/shim.so"
            result = build_shim("/fake/target", mode="direct")
            assert result.bitmap_size == 0


class TestReadBitmap:
    def test_none_handle(self):
        assert read_bitmap(None) is None

    def test_handle_without_cov_get_bitmap(self):
        mock = MagicMock(spec=[])  # no attributes
        assert read_bitmap(mock) is None

    def test_handle_with_valid_bitmap(self):
        buf = (ctypes.c_uint8 * 4)(1, 2, 3, 4)
        mock = MagicMock()
        mock.cov_get_bitmap.return_value = ctypes.addressof(buf)
        mock.cov_get_size.return_value = 4
        result = read_bitmap(mock)
        assert result == bytes([1, 2, 3, 4])

    def test_handle_returns_zero_pointer(self):
        mock = MagicMock()
        mock.cov_get_bitmap.return_value = 0
        mock.cov_get_size.return_value = 4
        assert read_bitmap(mock) is None

    def test_handle_returns_zero_size(self):
        mock = MagicMock()
        mock.cov_get_bitmap.return_value = 1234
        mock.cov_get_size.return_value = 0
        assert read_bitmap(mock) is None


class TestResetBitmap:
    def test_none_handle(self):
        reset_bitmap(None)  # should not raise

    def test_calls_cov_reset(self):
        mock = MagicMock()
        reset_bitmap(mock)
        mock.cov_reset.assert_called_once()

    def test_suppresses_attribute_error(self):
        mock = MagicMock(spec=[])  # no attributes
        reset_bitmap(mock)  # should not raise


class TestBitmapReader:
    def test_init_no_counters(self, tmp_path):
        fake_target = tmp_path / "target"
        fake_target.write_bytes(b"\x7fELF" + b"\x00" * 200)
        with patch("fuzzer_tool.adapters.shim_factory.parse_sancov_offsets", return_value=None):
            mock_lib = MagicMock()
            reader = BitmapReader(str(fake_target), mock_lib)
            assert reader.bitmap_size == 0
            assert reader.valid is False

    def test_read_bitmap_when_invalid(self, tmp_path):
        fake_target = tmp_path / "target"
        fake_target.write_bytes(b"\x7fELF" + b"\x00" * 200)
        with patch("fuzzer_tool.adapters.shim_factory.parse_sancov_offsets", return_value=None):
            mock_lib = MagicMock()
            reader = BitmapReader(str(fake_target), mock_lib)
            assert reader.read_bitmap() is None

    def test_reset_bitmap_when_invalid(self, tmp_path):
        fake_target = tmp_path / "target"
        fake_target.write_bytes(b"\x7fELF" + b"\x00" * 200)
        with patch("fuzzer_tool.adapters.shim_factory.parse_sancov_offsets", return_value=None):
            mock_lib = MagicMock()
            reader = BitmapReader(str(fake_target), mock_lib)
            reader.reset_bitmap()  # should not raise

    def test_valid_property(self, tmp_path):
        fake_target = tmp_path / "target"
        fake_target.write_bytes(b"\x7fELF" + b"\x00" * 200)
        with patch("fuzzer_tool.adapters.shim_factory.parse_sancov_offsets", return_value=None):
            mock_lib = MagicMock()
            reader = BitmapReader(str(fake_target), mock_lib)
            assert reader.valid is False
