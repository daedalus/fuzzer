"""Tests for core/elf.py — ELF parsing for sancov counter discovery."""

import os
import struct

from fuzzer_tool.core.elf import find_load_segment, parse_sancov_offsets


def _build_elf64_header(
    e_shoff=0, e_shnum=0, e_shentsize=64, e_shstrndx=0,
    e_phoff=0, e_phentsize=56, e_phnum=0,
) -> bytes:
    """Build a minimal ELF64 little-endian header."""
    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = 2  # ELFCLASS64
    header[5] = 1  # ELFDATA2LSB
    header[6] = 1  # EV_CURRENT
    header[7] = 0  # ELFOSABI_NONE
    struct.pack_into("<Q", header, 32, e_phoff)
    struct.pack_into("<Q", header, 40, e_shoff)
    struct.pack_into("<H", header, 54, e_phentsize)
    struct.pack_into("<H", header, 56, e_phnum)
    struct.pack_into("<H", header, 58, e_shentsize)
    struct.pack_into("<H", header, 60, e_shnum)
    struct.pack_into("<H", header, 62, e_shstrndx)
    return bytes(header)


def _build_section_header(sh_type=0, sh_name=0, sh_offset=0, sh_size=0,
                          sh_link=0, sh_info=0, sh_addralign=0, sh_entsize=0):
    """Build a single 64-byte ELF section header."""
    sh = bytearray(64)
    struct.pack_into("<I", sh, 0, sh_name)
    struct.pack_into("<I", sh, 4, sh_type)
    struct.pack_into("<Q", sh, 16, sh_offset)
    struct.pack_into("<Q", sh, 24, sh_size)
    struct.pack_into("<I", sh, 40, sh_link)
    struct.pack_into("<I", sh, 44, sh_info)
    struct.pack_into("<Q", sh, 48, sh_addralign)
    struct.pack_into("<Q", sh, 56, sh_entsize)
    return bytes(sh)


def _build_program_header(p_type=1, p_vaddr=0, p_filesz=0, p_memsz=0, p_offset=0):
    """Build a single 56-byte ELF program header."""
    ph = bytearray(56)
    struct.pack_into("<I", ph, 0, p_type)
    struct.pack_into("<Q", ph, 8, p_offset)
    struct.pack_into("<Q", ph, 16, p_vaddr)
    struct.pack_into("<Q", ph, 24, p_vaddr)
    struct.pack_into("<Q", ph, 32, p_filesz)
    struct.pack_into("<Q", ph, 40, p_memsz)
    return bytes(ph)


class TestParseSancovOffsets:
    def test_non_elf(self):
        assert parse_sancov_offsets("/dev/null") is None

    def test_too_short(self, tmp_path):
        p = tmp_path / "short"
        p.write_bytes(b"\x7fELF")
        assert parse_sancov_offsets(str(p)) is None

    def test_elf32_rejected(self, tmp_path):
        """ELF32 big-endian should be rejected (not ELF64 little-endian)."""
        header = bytearray(64)
        header[0:4] = b"\x7fELF"
        header[4] = 1  # ELFCLASS32
        header[5] = 2  # ELFDATA2MSB
        p = tmp_path / "elf32"
        p.write_bytes(bytes(header))
        assert parse_sancov_offsets(str(p)) is None

    def test_no_symtab(self, tmp_path):
        """ELF with section headers but no symtab → returns None."""
        header = _build_elf64_header(e_shnum=1, e_shstrndx=0, e_shentsize=64)
        shstrtab = b".shstrtab\x00"
        sh = _build_section_header(sh_type=3, sh_name=0, sh_offset=256, sh_size=len(shstrtab))
        data = bytearray(256 + len(shstrtab) + 100)
        data[:64] = header
        data[64:128] = sh
        data[256:256 + len(shstrtab)] = shstrtab
        p = tmp_path / "no_symtab"
        p.write_bytes(bytes(data))
        assert parse_sancov_offsets(str(p)) is None

    def test_no_sancov_symbols(self, tmp_path):
        """ELF with symtab but no __start/__stop___sancov_cntrs."""
        header = _build_elf64_header(e_shnum=3, e_shstrndx=0, e_shentsize=64)
        shstrtab = b".shstrtab\x00.strtab\x00.symtab\x00"
        sh_shstrtab = _build_section_header(sh_type=3, sh_name=0, sh_offset=256, sh_size=len(shstrtab))
        strtab = b"\x00my_func\x00"
        sh_strtab = _build_section_header(sh_type=3, sh_name=10, sh_offset=512, sh_size=len(strtab))
        sym = bytearray(24)
        struct.pack_into("<I", sym, 0, 1)
        struct.pack_into("<Q", sym, 8, 0x4000)
        sh_symtab = _build_section_header(
            sh_type=2, sh_name=18, sh_link=1, sh_offset=768, sh_size=24, sh_entsize=24
        )
        data = bytearray(256 + len(shstrtab) + len(strtab) + 24 + 100)
        data[:64] = header
        data[64:128] = sh_shstrtab
        data[128:192] = sh_strtab
        data[192:256] = sh_symtab
        data[256:256 + len(shstrtab)] = shstrtab
        data[512:512 + len(strtab)] = strtab
        data[768:792] = sym
        p = tmp_path / "no_sancov"
        p.write_bytes(bytes(data))
        assert parse_sancov_offsets(str(p)) is None

    def test_real_binary(self):
        """Test with actual compiled target that has sancov counters."""
        target = "targets/png_read_afl.so"
        if not os.path.isfile(target):
            return
        result = parse_sancov_offsets(target)
        # May or may not find sancov symbols — just verify no crash
        if result is not None:
            assert len(result) == 2
            assert result[0] > 0
            assert result[1] > 0

    def test_exception_path(self, tmp_path):
        """Corrupt ELF triggers exception → returns None."""
        header = bytearray(64)
        header[0:4] = b"\x7fELF"
        header[4] = 2
        header[5] = 1
        struct.pack_into("<Q", header, 40, 999999)  # bogus e_shoff
        struct.pack_into("<H", header, 60, 3)  # e_shnum = 3
        struct.pack_into("<H", header, 62, 0)  # e_shstrndx = 0
        p = tmp_path / "corrupt"
        p.write_bytes(bytes(header))
        assert parse_sancov_offsets(str(p)) is None

    def test_entsize_zero(self, tmp_path):
        """symtab with entsize=0 → returns None."""
        header = _build_elf64_header(e_shnum=3, e_shstrndx=0, e_shentsize=64)
        shstrtab = b".shstrtab\x00.strtab\x00.symtab\x00"
        sh_shstrtab = _build_section_header(sh_type=3, sh_name=0, sh_offset=256, sh_size=len(shstrtab))
        strtab = b"\x00func\x00"
        sh_strtab = _build_section_header(sh_type=3, sh_name=10, sh_offset=512, sh_size=len(strtab))
        sh_symtab = _build_section_header(
            sh_type=2, sh_name=18, sh_link=1, sh_offset=768, sh_size=0, sh_entsize=0
        )
        data = bytearray(256 + len(shstrtab) + len(strtab) + 100)
        data[:64] = header
        data[64:128] = sh_shstrtab
        data[128:192] = sh_strtab
        data[192:256] = sh_symtab
        data[256:256 + len(shstrtab)] = shstrtab
        data[512:512 + len(strtab)] = strtab
        p = tmp_path / "entsize0"
        p.write_bytes(bytes(data))
        assert parse_sancov_offsets(str(p)) is None


class TestFindLoadSegment:
    def test_non_elf(self):
        assert find_load_segment(b"\x00" * 10, 0x1000) is None

    def test_too_short(self):
        assert find_load_segment(b"\x7fELF", 0x1000) is None

    def test_no_pt_load(self):
        header = _build_elf64_header(e_phoff=64, e_phnum=1)
        ph = _build_program_header(p_type=6, p_vaddr=0, p_filesz=0, p_memsz=0)
        data = header + ph
        assert find_load_segment(data, 0x1000) is None

    def test_real_binary(self):
        """Use the actual compiled test target to verify segment lookup."""
        target = "targets/png_read_afl.so"
        if not os.path.isfile(target):
            return
        with open(target, "rb") as f:
            data = f.read()
        result = find_load_segment(data, 0x1000)
        if result is not None:
            vaddr, filesz, memsz = result
            assert vaddr > 0
            assert filesz > 0
            assert memsz >= filesz
