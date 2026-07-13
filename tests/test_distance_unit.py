"""Tests for core/distance.py — directed distance computation."""

import struct

import pytest

from fuzzer_tool.core.distance import TargetDistance


def _make_minimal_elf(
    entry_addr=0x400000,
    text_start=0x400000,
    text_end=0x401000,
    base_addr=0x400000,
    functions=None,
    calls=None,
):
    """Build a minimal ELF64 binary for testing distance computation."""
    # Minimal ELF header (64 bytes)
    header = bytearray(64)
    header[0:4] = b"\x7fELF"  # magic
    header[4] = 2  # ELF64
    header[5] = 1  # little-endian
    header[16] = 2  # ET_EXEC
    header[18] = 0x3E  # EM_X86_64
    struct.pack_into("<Q", header, 24, entry_addr)  # e_entry
    # e_phoff, e_shoff, etc. set below

    # Build .text content with CALL instructions
    text_content = bytearray(text_text := b"\x90" * 256)  # NOP sled
    if calls:
        for caller, callee_addr in calls:
            # Place a REL32 CALL at offset 0 of text_content
            disp = callee_addr - (text_start + 0) - 5
            struct.pack_into("<i", text_content, 0, disp)
            text_content[0] = 0xE8  # CALL opcode

    # Program headers (PT_LOAD for text)
    ph_offset = 64
    ph = bytearray(56)  # 56 bytes per phdr
    struct.pack_into("<I", ph, 0, 1)  # PT_LOAD
    struct.pack_into("<I", ph, 4, 5)  # PF_R | PF_X
    struct.pack_into("<Q", ph, 16, text_start)  # p_vaddr
    struct.pack_into("<Q", ph, 24, ph_offset + 56)  # p_offset (data after phdrs)
    struct.pack_into("<Q", ph, 32, len(text_content))  # p_filesz
    struct.pack_into("<Q", ph, 40, len(text_content))  # p_memsz

    struct.pack_into("<Q", header, 32, ph_offset)  # e_phoff
    struct.pack_into("<H", header, 54, 56)  # e_phentsize
    struct.pack_into("<H", header, 56, 1)  # e_phnum

    # Section headers (empty — no symtab)
    sh_offset = ph_offset + 56 + len(text_content)
    struct.pack_into("<Q", header, 40, sh_offset)  # e_shoff
    struct.pack_into("<H", header, 58, 64)  # e_shentsize
    struct.pack_into("<H", header, 60, 0)  # e_shnum (0 sections)
    struct.pack_into("<H", header, 62, 0)  # e_shstrndx

    return bytes(header) + bytes(ph) + bytes(text_content)


class TestTargetDistanceInit:
    def test_init(self):
        td = TargetDistance("/nonexistent", ["main"])
        assert td.target == "/nonexistent"
        assert td.target_names == ["main"]
        assert td._loaded is False

    def test_default_targets(self):
        td = TargetDistance("/nonexistent")
        assert td.target_names == []


class TestLoad:
    def test_nonexistent_file(self):
        td = TargetDistance("/nonexistent/file.elf", ["main"])
        assert td.load() is False

    def test_not_elf(self, tmp_path):
        f = tmp_path / "notelf.bin"
        f.write_bytes(b"not an elf file")
        td = TargetDistance(str(f), ["main"])
        assert td.load() is False

    def test_valid_minimal_elf(self, tmp_path):
        f = tmp_path / "test.elf"
        f.write_bytes(_make_minimal_elf())
        td = TargetDistance(str(f), ["main"])
        # No symtab → returns False
        assert td.load() is False

    def test_too_short(self, tmp_path):
        f = tmp_path / "tiny.elf"
        f.write_bytes(b"\x7fELF")
        td = TargetDistance(str(f))
        assert td.load() is False


class TestResolveTargets:
    def test_resolve_by_hex(self):
        td = TargetDistance("/tmp/x", ["0x400000"])
        td.functions["main"] = (0x400000, 0x400100)
        td._resolve_targets()
        assert 0x400000 in td.target_addrs

    def test_resolve_by_name(self):
        td = TargetDistance("/tmp/x", ["main"])
        td.functions["main"] = (0x400000, 0x400100)
        td._resolve_targets()
        assert 0x400000 in td.target_addrs

    def test_resolve_by_substring(self):
        td = TargetDistance("/tmp/x", ["foo"])
        td.functions["foobar_func"] = (0x400000, 0x400100)
        td._resolve_targets()
        assert 0x400000 in td.target_addrs


class TestBbDistance:
    def test_unknown_address_heuristic(self):
        td = TargetDistance("/tmp/x")
        td.functions["func_a"] = (0x1000, 0x1100)
        td._distances = {"func_a": 3.0}
        # Address near func_a
        dist = td.bb_distance(0x1050)
        assert 0.0 < dist < 10.0

    def test_known_function_distance(self):
        td = TargetDistance("/tmp/x")
        td.functions["main"] = (0x1000, 0x1100)
        td._distances = {"main": 0.0}
        dist = td.bb_distance(0x1050)
        assert dist == 0.0

    def test_caches_result(self):
        td = TargetDistance("/tmp/x")
        td.functions["main"] = (0x1000, 0x1100)
        td._distances = {"main": 2.0}
        d1 = td.bb_distance(0x1050)
        d2 = td.bb_distance(0x1050)
        assert d1 == d2


class TestSeedDistance:
    def test_empty_trace(self):
        td = TargetDistance("/tmp/x")
        assert td.seed_distance(set()) == 20.0

    def test_single_edge(self):
        td = TargetDistance("/tmp/x")
        td.functions["main"] = (0x1000, 0x1100)
        td._distances = {"main": 0.0}
        dist = td.seed_distance({(0x0, 0x1050)})
        assert dist == 0.0

    def test_average_of_multiple_bbs(self):
        td = TargetDistance("/tmp/x")
        td.functions["a"] = (0x1000, 0x1100)
        td.functions["b"] = (0x2000, 0x2100)
        td._distances = {"a": 0.0, "b": 5.0}
        dist = td.seed_distance({(0x0, 0x1050), (0x1050, 0x2050)})
        assert dist == 2.5  # (0 + 5) / 2


class TestMaxDistance:
    def test_empty(self):
        td = TargetDistance("/tmp/x")
        assert td.max_distance == 10.0

    def test_with_distances(self):
        td = TargetDistance("/tmp/x")
        td._distances = {"a": 3.0, "b": 7.0}
        assert td.max_distance == 8.0  # max(3,7) + 1


class TestIsTarget:
    def test_in_target_function(self):
        td = TargetDistance("/tmp/x")
        td.functions["main"] = (0x1000, 0x1100)
        td.target_addrs = {0x1000}
        assert td.is_target(0x1050) is True

    def test_not_in_target(self):
        td = TargetDistance("/tmp/x")
        td.functions["other"] = (0x2000, 0x2100)
        td.target_addrs = {0x1000}
        assert td.is_target(0x2050) is False

    def test_unknown_address(self):
        td = TargetDistance("/tmp/x")
        td.target_addrs = {0x1000}
        assert td.is_target(0x9999) is False


class TestHeuristicDistance:
    def test_inside_function(self):
        td = TargetDistance("/tmp/x")
        td.functions["main"] = (0x1000, 0x1100)
        td._distances = {"main": 3.0}
        assert td._heuristic_distance(0x1050) == 3.0

    def test_outside_function(self):
        td = TargetDistance("/tmp/x")
        td.functions["main"] = (0x1000, 0x1100)
        td._distances = {"main": 3.0}
        # 64 bytes away → 64/64 + 2 = 3.0
        assert td._heuristic_distance(0x1040) == pytest.approx(3.0)
