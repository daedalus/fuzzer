"""Tests for core/target_profiler.py — static target analysis."""

import math
import os
import struct
import tempfile

import pytest

from fuzzer_tool.core.target_profiler import (
    _FORMAT_OPERATOR_HINTS,
    MAGIC_SIGNATURES,
    FunctionInfo,
    TargetProfile,
    TargetProfiler,
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
        """Constant extraction from .text disassembly."""
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


class TestSymbolFormatDetection:
    """Symbol-ladder detection for gif/webp/webm/zip/protobuf targets."""

    def _infer(self, func_names):
        profiler = TargetProfiler("/nonexistent")
        profiler._elf = b"\x7fELF"  # non-None so _infer_format proceeds
        p = TargetProfile()
        p.functions = {name: FunctionInfo(addr=0x1000, size=0x10, name=name) for name in func_names}
        profiler._infer_format(p)
        return p.format_signature

    def test_gif_symbols(self):
        assert self._infer(["DGifOpenFileName"]) == "gif"

    def test_webp_symbols(self):
        assert self._infer(["WebPGetInfo"]) == "webp"

    def test_webm_symbols(self):
        assert self._infer(["mkvparser::Segment::ParseStream"]) == "webm"

    def test_zip_symbols(self):
        assert self._infer(["unzOpen"]) == "zip"

    def test_protobuf_symbol(self):
        assert self._infer(["ParseFromArray"]) == "protobuf"

    def test_protobuf_rodata_string(self):
        # Fallback: rodata containing "protobuf" also tags the format
        profiler = TargetProfiler("/nonexistent")
        profiler._elf = b"\x7fELF"
        p = TargetProfile()
        p.rodata_strings = [(0x1000, "protobuf wire type mismatch")]
        profiler._infer_format(p)
        assert p.format_signature == "protobuf"

    def test_unknown_symbols_stay_unknown(self):
        assert self._infer(["frobnicate", "main"]) in (None, "text", "unknown")


class TestParserTokenExtraction:
    """Regression: _extract_parser_tokens used undefined `sec_size` (F821)."""

    def _profiler_with_token_table(self):
        """Build a profiler whose .rodata holds a yytname-style token table.

        Layout mirrors the section tuple (sh_type, sh_offset, sh_addr, sh_size):
        .rodata at file offset 0x100, vaddr 0x400000, size 0x100; the symbol
        yytname lives at vaddr 0x400010, and file offset 0x110 holds an 8-byte
        pointer to a null-terminated string at rodata offset 0x30.
        """
        elf = bytearray(0x200)
        struct.pack_into("<Q", elf, 0x110, 0x130)  # pointer to string at file offset 0x130
        elf[0x130:0x136] = b"TOKEN\x00"
        profiler = TargetProfiler("/nonexistent")
        profiler._elf = bytes(elf)
        profiler._sections = {".rodata": (3, 0x100, 0x400000, 0x100)}
        profiler._symtab = [("yytname", 0x400010, 8, 1)]
        return profiler

    def test_token_table_extraction_no_nameerror(self):
        profiler = self._profiler_with_token_table()
        p = TargetProfile()
        profiler._extract_parser_tokens(p)
        assert b"TOKEN" in p.parser_tokens

    def test_no_rodata_returns_empty(self):
        profiler = TargetProfiler("/nonexistent")
        profiler._elf = b"\x7fELF" + b"\x00" * 128
        profiler._sections = {}
        profiler._symtab = []
        p = TargetProfile()
        profiler._extract_parser_tokens(p)
        assert p.parser_tokens == []


class TestParserTokenArrayWalk:
    """Tier 1.1: a yytname symbol is a pointer *array*, not a single pointer.

    honggfuzz arch_bfdExtractStrArray walks consecutive pointers up to 2048
    entries and stops at a NULL terminator. The single-pointer read that
    preceded this only ever recovered the first token.
    """

    def _profiler_with_array(self):
        elf = bytearray(0x300)
        targets = [0x130, 0x150, 0x170]
        names = [b"TOK_A\x00", b"TOK_B\x00", b"TOK_C\x00"]
        for i, (tgt, nm) in enumerate(zip(targets, names, strict=True)):
            struct.pack_into("<Q", elf, 0x110 + i * 8, tgt)
            elf[tgt : tgt + len(nm)] = nm
        struct.pack_into("<Q", elf, 0x110 + len(targets) * 8, 0)  # NULL terminator
        profiler = TargetProfiler("/nonexistent")
        profiler._elf = bytes(elf)
        profiler._sections = {".rodata": (3, 0x100, 0x400000, 0x200)}
        profiler._symtab = [("yytname", 0x400010, 64, 1)]
        return profiler

    def test_all_three_tokens_walked(self):
        profiler = self._profiler_with_array()
        p = TargetProfile()
        profiler._extract_parser_tokens(p)
        assert b"TOK_A" in p.parser_tokens
        assert b"TOK_B" in p.parser_tokens
        assert b"TOK_C" in p.parser_tokens

    def test_null_terminator_stops_the_walk(self):
        """A NULL entry must halt the walk, not be copied as a token."""
        profiler = self._profiler_with_array()
        p = TargetProfile()
        profiler._extract_parser_tokens(p)
        assert b"\x00" not in [t for t in p.parser_tokens]

    def test_heuristic_scan_budget_past_4096(self):
        """A pointer table above the old 4096-byte cap is recovered once
        the heuristic scan is budget-bounded instead of hard-capped."""
        rodata_off = 0x100
        rodata = bytearray(0x5000)
        table_off = 0x1400  # > 4096, the old scan bound
        targets = [0x4000 + i * 0x100 for i in range(5)]
        for i, tgt in enumerate(targets):
            struct.pack_into("<Q", rodata, table_off + i * 8, rodata_off + tgt)
        for i, tgt in enumerate(targets):
            nm = f"TOK_P{i}\x00".encode()
            rodata[tgt : tgt + len(nm)] = nm

        profiler = TargetProfiler("/nonexistent")
        elf = bytearray(0x6000)
        elf[rodata_off : rodata_off + len(rodata)] = rodata
        profiler._elf = bytes(elf)
        profiler._sections = {".rodata": (3, rodata_off, 0x400000, len(rodata))}
        profiler._symtab = []

        p = TargetProfile()
        profiler._extract_parser_tokens(p)
        for i in range(5):
            assert f"TOK_P{i}".encode() in p.parser_tokens


def _elf_with_rodata(rodata_bytes: bytes) -> bytes:
    """Build a minimal ELF64 with a single SHT_PROGBITS .rodata section."""
    elf = bytearray(0x2000)
    elf[0:4] = b"\x7fELF"
    elf[4] = 2  # ELFCLASS64
    elf[5] = 1  # ELFDATA2LSB
    struct.pack_into("<Q", elf, 40, 0x40)  # e_shoff
    struct.pack_into("<H", elf, 58, 64)  # e_shentsize
    struct.pack_into("<H", elf, 60, 2)  # e_shnum
    struct.pack_into("<H", elf, 62, 0)  # e_shstrndx

    shstrtab = b"\x00.shstrtab\x00.rodata\x00"
    rodata_off = (0x40 + 3 * 64 + 7) & ~7  # 0x100
    shstr_off = rodata_off + len(rodata_bytes)

    def _shdr(sh_type, sh_name, sh_offset, sh_size):
        sh = bytearray(64)
        struct.pack_into("<I", sh, 0, sh_name)
        struct.pack_into("<I", sh, 4, sh_type)
        struct.pack_into("<Q", sh, 16, sh_offset)
        struct.pack_into("<Q", sh, 24, sh_offset)
        struct.pack_into("<Q", sh, 32, sh_size)

        return bytes(sh)

    elf[0x40:0x80] = _shdr(3, 0, shstr_off, len(shstrtab))  # shstrtab
    elf[0x80:0xC0] = _shdr(1, 11, rodata_off, len(rodata_bytes))  # ".rodata" at 11
    elf[rodata_off : rodata_off + len(rodata_bytes)] = rodata_bytes
    elf[shstr_off : shstr_off + len(shstrtab)] = shstrtab
    return bytes(elf)


class TestRoDataWordConstantsProfile:
    """TargetProfile rodata_word_constants field + profiler wiring."""

    def test_default_empty(self):
        p = TargetProfile()
        assert p.rodata_word_constants == []

    def test_to_dict_hex_roundtrip(self):
        words = [b"\x0d\x0a\x1a\x0a", b"\x88\x77\x66\x55\x44\x33\x22\x11"]
        p = TargetProfile(rodata_word_constants=words)
        d = p.to_dict()
        assert d["rodata_word_constants"] == ["0d0a1a0a", "8877665544332211"]
        p2 = TargetProfile.from_dict(d)
        assert p2.rodata_word_constants == words

    def test_old_cache_without_field_loads_empty(self):
        d = TargetProfile(rodata_word_constants=[b"\x0d\x0a\x1a\x0a"]).to_dict()
        del d["rodata_word_constants"]
        p = TargetProfile.from_dict(d)
        assert p.rodata_word_constants == []

    def test_profiler_populates_from_rodata(self, tmp_path):
        elf = _elf_with_rodata(b"\x0d\x0a\x1a\x0a\x0d\x0a\x1a\x0a\x88\x77\x66\x55\x44\x33\x22\x11")
        p_bin = tmp_path / "prof_words.elf"
        p_bin.write_bytes(elf)

        profiler = TargetProfiler(str(p_bin))
        profile = TargetProfile()
        profiler._extract_data_word_constants(profile)

        assert b"\x0d\x0a\x1a\x0a" in profile.rodata_word_constants
        assert b"\x88\x77\x66\x55\x44\x33\x22\x11" in profile.rodata_word_constants

    def test_profiler_absent_target_stays_empty(self):
        profiler = TargetProfiler("/nonexistent")
        profile = TargetProfile()
        profiler._extract_data_word_constants(profile)
        assert profile.rodata_word_constants == []
