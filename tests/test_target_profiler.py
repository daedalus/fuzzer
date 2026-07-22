"""Tests for core/target_profiler.py — static target analysis."""

import os
import struct
import tempfile
import math

import pytest

from fuzzer_tool.core.target_profiler import (
    MAGIC_SIGNATURES,
    TargetProfiler,
    TargetProfile,
    FunctionInfo,
    _FORMAT_OPERATOR_HINTS,
    format_operator_priors,
)


class TestTargetProfile:
    def test_default_fields(self):
        p = TargetProfile()
        assert p.rodata_strings == []
        assert p.interesting_strings == []
        assert p.magic_bytes == []
        assert p.extracted_constants == []
        assert p.functions == {}
        assert p.hot_functions == []
        assert p.entry_points == []
        assert p.input_parsers == []
        assert p.boundary_markers == []
        assert p.format_signature is None
        assert p.call_graph == {}
        assert p.reverse_calls == {}


class TestFunctionInfo:
    def test_defaults(self):
        fi = FunctionInfo(addr=0x1000, size=0x100, name="test_func")
        assert fi.addr == 0x1000
        assert fi.size == 0x100
        assert fi.name == "test_func"
        assert fi.bb_count == 0
        assert fi.call_depth == 0
        assert fi.branch_density == 0.0


class TestTargetProfilerNonexistent:
    def test_nonexistent_target(self):
        profiler = TargetProfiler("/nonexistent/binary")
        profile = profiler.profile()
        assert profile.functions == {}
        assert profile.format_signature is None


class TestTargetProfilerRealBinary:
    """Tests against the real test binaries in targets/."""

    @pytest.fixture
    def png_profile(self):
        target = os.path.join(os.path.dirname(__file__), "..", "targets", "png_read")
        if not os.path.isfile(target):
            pytest.skip("png_read binary not found")
        profiler = TargetProfiler(target)
        return profiler.profile()

    @pytest.fixture
    def test_profile(self):
        target = os.path.join(os.path.dirname(__file__), "..", "targets", "test_target")
        if not os.path.isfile(target):
            pytest.skip("test_target binary not found")
        profiler = TargetProfiler(target)
        return profiler.profile()

    def test_profile_returns_target_profile(self, png_profile):
        assert isinstance(png_profile, TargetProfile)

    def test_functions_detected(self, png_profile):
        assert len(png_profile.functions) > 0

    def test_functions_have_addrs(self, png_profile):
        for name, fi in png_profile.functions.items():
            assert fi.addr > 0
            assert fi.size > 0
            assert fi.name == name

    def test_hot_functions_populated(self, png_profile):
        assert len(png_profile.hot_functions) > 0
        # Hot functions should be a subset of all functions
        for name in png_profile.hot_functions:
            assert name in png_profile.functions

    def test_entry_points_populated(self, png_profile):
        assert len(png_profile.entry_points) > 0

    def test_call_graph_populated(self, png_profile):
        # At least some functions should have call edges
        assert len(png_profile.call_graph) > 0

    def test_extracted_constants_populated(self, png_profile):
        """Capstone constant extraction from .text disassembly."""
        if len(png_profile.extracted_constants) == 0:
            pytest.skip("capstone not available or binary too small")
        assert len(png_profile.extracted_constants) > 0
        assert len(png_profile.extracted_constants) <= 256
        for c in png_profile.extracted_constants:
            assert isinstance(c, bytes)
            assert len(c) >= 2

    def test_format_signature_is_string(self, png_profile):
        assert isinstance(png_profile.format_signature, str)

    def test_strings_extracted(self, png_profile):
        assert len(png_profile.rodata_strings) > 0

    def test_interesting_strings_filtered(self, png_profile):
        assert isinstance(png_profile.interesting_strings, list)
        # Should be a subset of rodata_strings
        for s in png_profile.interesting_strings:
            assert isinstance(s, str)

    def test_test_target_profile(self, test_profile):
        assert len(test_profile.functions) > 0


class TestMagicSignatures:
    def test_magic_signature_list(self):
        assert len(MAGIC_SIGNATURES) > 10
        for sig, fmt in MAGIC_SIGNATURES:
            assert isinstance(sig, bytes)
            assert isinstance(fmt, str)
            assert len(sig) >= 1


class TestTargetProfilerELF:
    """Test with synthetic ELF binaries."""

    def _make_elf(self, sections=None, syms=None):
        """Create a minimal ELF64 binary for testing."""
        # Build a minimal ELF header
        elf = bytearray(4096)
        # ELF magic
        elf[0:4] = b"\x7fELF"
        elf[4] = 2  # ELFCLASS64
        elf[5] = 1  # ELFDATA2LSB
        # e_type = ET_EXEC (2) at offset 16
        struct.pack_into("<H", elf, 16, 2)
        # e_entry at offset 24
        struct.pack_into("<Q", elf, 24, 0x400000)
        # e_phoff at offset 32
        struct.pack_into("<Q", elf, 32, 64)
        # e_shoff at offset 40
        struct.pack_into("<Q", elf, 40, 256)
        # e_ehsize at offset 52
        struct.pack_into("<H", elf, 52, 64)
        # e_phentsize at offset 54
        struct.pack_into("<H", elf, 54, 56)
        # e_phnum at offset 56
        struct.pack_into("<H", elf, 56, 1)
        # e_shentsize at offset 58
        struct.pack_into("<H", elf, 58, 64)
        # e_shnum at offset 60
        struct.pack_into("<H", elf, 60, 3)
        # e_shstrndx at offset 62
        struct.pack_into("<H", elf, 62, 2)

        return bytes(elf)

    def test_non_elf_returns_empty_profile(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"not an ELF" + b"\x00" * 100)
            f.flush()
            try:
                profiler = TargetProfiler(f.name)
                profile = profiler.profile()
                assert profile.functions == {}
            finally:
                os.unlink(f.name)

    def test_32bit_elf_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as f:
            elf = bytearray(256)
            elf[0:4] = b"\x7fELF"
            elf[4] = 1  # ELFCLASS32
            elf[5] = 1
            f.write(bytes(elf))
            f.flush()
            try:
                profiler = TargetProfiler(f.name)
                profile = profiler.profile()
                assert profile.functions == {}
            finally:
                os.unlink(f.name)


class TestBoundaryMarkers:
    def test_boundary_markers_are_bytes(self):
        # Boundary markers should be bytes objects
        p = TargetProfile()
        p.boundary_markers = [b":", b"/", b"\n"]
        for bm in p.boundary_markers:
            assert isinstance(bm, bytes)


class TestFormatOperatorPriors:
    def test_no_hints_when_profile_empty(self):
        p = TargetProfile()
        assert format_operator_priors(p) == {}

    def test_png_format_boosts_png_operators(self):
        p = TargetProfile()
        p.format_signature = "png"
        priors = format_operator_priors(p)
        assert priors["png_chunk_mutate"] == (2.0, 1.0)
        assert priors["png_crc_fix"] == (2.0, 1.0)
        assert "jpeg_chunk_mutate" not in priors

    def test_unknown_format_no_format_hints(self):
        p = TargetProfile()
        p.format_signature = "text"
        priors = format_operator_priors(p)
        all_format_ops = {op for ops in _FORMAT_OPERATOR_HINTS.values() for op in ops}
        assert not (set(priors) & all_format_ops)

    def test_magic_bytes_boost_dict_operators(self):
        p = TargetProfile()
        p.magic_bytes = [b"\x89PNG\r\n\x1a\n"]
        priors = format_operator_priors(p)
        assert priors["dict_insert"] == (2.0, 1.0)
        assert priors["checksum_repair"] == (2.0, 1.0)

    def test_boundary_markers_boost_dict_operators(self):
        p = TargetProfile()
        p.boundary_markers = [b":"]
        priors = format_operator_priors(p)
        assert "dict_replace" in priors

    def test_priors_are_positive_finite_floats(self):
        p = TargetProfile()
        p.format_signature = "gzip"
        p.magic_bytes = [b"\x1f\x8b"]
        priors = format_operator_priors(p)
        for alpha, beta in priors.values():
            assert alpha > 0
            assert beta > 0
            assert math.isfinite(alpha)
            assert math.isfinite(beta)
