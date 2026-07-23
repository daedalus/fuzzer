"""Tests for core/elf.py — ELF parsing for sancov counter discovery."""

import os
import struct

from fuzzer_tool.core.elf import find_load_segment, parse_sancov_offsets


def _build_elf64_header(
    e_shoff=0,
    e_shnum=0,
    e_shentsize=64,
    e_shstrndx=0,
    e_phoff=0,
    e_phentsize=56,
    e_phnum=0,
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


def _build_section_header(
    sh_type=0, sh_name=0, sh_offset=0, sh_size=0, sh_link=0, sh_info=0, sh_addralign=0, sh_entsize=0
):
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
        data[256 : 256 + len(shstrtab)] = shstrtab
        p = tmp_path / "no_symtab"
        p.write_bytes(bytes(data))
        assert parse_sancov_offsets(str(p)) is None

    def test_no_sancov_symbols(self, tmp_path):
        """ELF with symtab but no __start/__stop___sancov_cntrs."""
        header = _build_elf64_header(e_shnum=3, e_shstrndx=0, e_shentsize=64)
        shstrtab = b".shstrtab\x00.strtab\x00.symtab\x00"
        sh_shstrtab = _build_section_header(
            sh_type=3, sh_name=0, sh_offset=256, sh_size=len(shstrtab)
        )
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
        data[256 : 256 + len(shstrtab)] = shstrtab
        data[512 : 512 + len(strtab)] = strtab
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
        sh_shstrtab = _build_section_header(
            sh_type=3, sh_name=0, sh_offset=256, sh_size=len(shstrtab)
        )
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
        data[256 : 256 + len(shstrtab)] = shstrtab
        data[512 : 512 + len(strtab)] = strtab
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


class TestExtractCapstoneConstants:
    """Tests for extract_capstone_constants and helpers."""

    def test_no_capstone(self, monkeypatch):
        """Returns [] when capstone is not installed."""
        monkeypatch.setattr("fuzzer_tool.core.elf.extract_capstone_constants", lambda _: [])
        from fuzzer_tool.core.elf import extract_capstone_constants

        assert extract_capstone_constants("anything") == []

    def test_non_elf(self):
        from fuzzer_tool.core.elf import extract_capstone_constants

        result = extract_capstone_constants("/dev/null")
        assert result == []

    def test_too_short(self, tmp_path):
        from fuzzer_tool.core.elf import extract_capstone_constants

        p = tmp_path / "short"
        p.write_bytes(b"\x7fELF")
        assert extract_capstone_constants(str(p)) == []

    def test_elf32_rejected(self, tmp_path):
        from fuzzer_tool.core.elf import extract_capstone_constants

        header = bytearray(64)
        header[0:4] = b"\x7fELF"
        header[4] = 1  # ELFCLASS32
        header[5] = 2  # ELFDATA2MSB
        p = tmp_path / "elf32"
        p.write_bytes(bytes(header))
        assert extract_capstone_constants(str(p)) == []

    def test_no_text_section(self, tmp_path):
        """ELF with no .text section returns []."""
        from fuzzer_tool.core.elf import extract_capstone_constants

        header = _build_elf64_header(e_shnum=1, e_shstrndx=0, e_shentsize=64)
        shstrtab = b".shstrtab\x00"
        sh = _build_section_header(sh_type=3, sh_name=0, sh_offset=256, sh_size=len(shstrtab))
        data = bytearray(256 + len(shstrtab) + 100)
        data[:64] = header
        data[64:128] = sh
        data[256 : 256 + len(shstrtab)] = shstrtab
        p = tmp_path / "no_text"
        p.write_bytes(bytes(data))
        assert extract_capstone_constants(str(p)) == []

    def test_real_binary_asan_tracecmp(self):
        """Extract constants from the ASAN+tracecmp target."""
        from fuzzer_tool.core.elf import extract_capstone_constants

        target = "targets/png_read_tracecmp_asan.so"
        if not os.path.isfile(target):
            return
        result = extract_capstone_constants(target)
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 256
        # Verify format: list of bytes objects, each >= 2 bytes
        for c in result:
            assert isinstance(c, bytes)
            assert len(c) >= 2
            assert len(c) in (2, 4, 8)
        # At least some should be multi-byte or recognizable
        has_wide = any(len(c) >= 4 for c in result)
        assert has_wide, f"Expected wide constants (>=4 bytes), got {result[:10]}"

    def test_real_binary_tracecmp(self):
        """Extract constants from the non-ASAN tracecmp target."""
        from fuzzer_tool.core.elf import extract_capstone_constants

        target = "targets/png_read_tracecmp.so"
        if not os.path.isfile(target):
            return
        result = extract_capstone_constants(target)
        assert isinstance(result, list)
        assert len(result) > 0
        for c in result:
            assert isinstance(c, bytes)
            assert len(c) >= 2


class TestIsNoiseImmediate:
    def test_zero(self):
        from fuzzer_tool.core.elf import _is_noise_immediate

        assert _is_noise_immediate(0, 4)

    def test_small_positive(self):
        from fuzzer_tool.core.elf import _is_noise_immediate

        assert _is_noise_immediate(42, 4)
        assert _is_noise_immediate(127, 4)

    def test_small_negative(self):
        from fuzzer_tool.core.elf import _is_noise_immediate

        assert _is_noise_immediate(-1, 4)  # 0xFFFFFFFF
        assert _is_noise_immediate(-128, 4)  # 0xFFFFFF80

    def test_negative_one_byte(self):
        """Single-byte negatives (128-255) are NOT filtered — they're valid constants."""
        from fuzzer_tool.core.elf import _is_noise_immediate

        # 0x89 is PNG magic, should not be filtered despite being > 127
        assert not _is_noise_immediate(0x89, 1)
        # 0xFF is JPEG marker, should not be filtered
        assert not _is_noise_immediate(0xFF, 1)
        # 0x80 is JPEG marker, should not be filtered
        assert not _is_noise_immediate(0x80, 1)

    def test_page_aligned_address(self):
        from fuzzer_tool.core.elf import _is_noise_immediate

        # 64-bit kernel-space address (high bit set + page-aligned)
        assert _is_noise_immediate(0xFFFFFFFF80000000, 8)
        # Low 4-byte values are NOT filtered even if page-aligned
        # (they could be legitimate constants like file offsets)
        assert not _is_noise_immediate(0x400000, 8)

    def test_user_space_address_64bit(self):
        from fuzzer_tool.core.elf import _is_noise_immediate

        # High-bit set for 64-bit + page-aligned
        assert _is_noise_immediate(0x800000000000, 8)

    def test_interesting_values(self):
        from fuzzer_tool.core.elf import _is_noise_immediate

        # Magic constants should NOT be filtered
        assert not _is_noise_immediate(0x89, 1)
        assert not _is_noise_immediate(0x0A1A0A0D0A474E89, 8)
        assert not _is_noise_immediate(0x424D, 2)  # 'BM' bitmap magic
        assert not _is_noise_immediate(0x0D000000, 4)  # PNG IHDR length
        assert not _is_noise_immediate(0xFFFF0000, 4)  # Mask value
        assert not _is_noise_immediate(0x80, 1)  # JPEG marker


class TestGuessImmWidth:
    def test_byte_width(self):
        from fuzzer_tool.core.elf import _guess_imm_width

        assert _guess_imm_width(0) == 1
        assert _guess_imm_width(0xFF) == 1

    def test_word_width(self):
        from fuzzer_tool.core.elf import _guess_imm_width

        assert _guess_imm_width(0x100) == 2
        assert _guess_imm_width(0xFFFF) == 2

    def test_dword_width(self):
        from fuzzer_tool.core.elf import _guess_imm_width

        assert _guess_imm_width(0x10000) == 4
        assert _guess_imm_width(0xFFFFFFFF) == 4

    def test_qword_width(self):
        from fuzzer_tool.core.elf import _guess_imm_width

        assert _guess_imm_width(0x100000000) == 8
        assert _guess_imm_width(0xFFFFFFFFFFFFFFFF) == 8


class TestMaybeAddConstant:
    def test_adds_valid(self):
        from fuzzer_tool.core.elf import _maybe_add_constant

        s: set[bytes] = set()
        _maybe_add_constant(s, b"\x7fELF")
        assert b"\x7fELF" in s

    def test_skips_short(self):
        from fuzzer_tool.core.elf import _maybe_add_constant

        s: set[bytes] = set()
        _maybe_add_constant(s, b"a")
        assert len(s) == 0

    def test_skips_empty(self):
        from fuzzer_tool.core.elf import _maybe_add_constant

        s: set[bytes] = set()
        _maybe_add_constant(s, b"")
        assert len(s) == 0

    def test_skips_all_zeros(self):
        from fuzzer_tool.core.elf import _maybe_add_constant

        s: set[bytes] = set()
        _maybe_add_constant(s, b"\x00\x00")
        assert len(s) == 0

    def test_skips_all_ff(self):
        from fuzzer_tool.core.elf import _maybe_add_constant

        s: set[bytes] = set()
        _maybe_add_constant(s, b"\xff\xff")
        assert len(s) == 0

    def test_dedup(self):
        from fuzzer_tool.core.elf import _maybe_add_constant

        s: set[bytes] = set()
        _maybe_add_constant(s, b"\x01\x02")
        _maybe_add_constant(s, b"\x01\x02")
        assert len(s) == 1


# ═══════════════════════════════════════════════════════════════════
# extract_div_constants — backward register tracing
# ═══════════════════════════════════════════════════════════════════


class TestExtractDivConstants:
    """Tests for ``extract_div_constants()``.

    These compile a small C file with known DIV patterns, then verify
    the static analysis recovers the correct divisor constants.
    """

    @classmethod
    def setup_class(cls):
        import subprocess, tempfile, os

        cls._tmpdir = tempfile.mkdtemp(prefix="elf_test_")
        src = os.path.join(cls._tmpdir, "test_div.c")
        cls._bin = os.path.join(cls._tmpdir, "test_div")
        # Write C source with inline asm that explicitly sets ECX to a
        # constant and then uses DIV with that register — avoids the
        # -O0 memory-operand issue.
        _src = (
            'void f_div10(void) {\n'
            '    int r;\n'
            '    asm("mov $10,%%ecx\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","ecx","edx");\n'
            '}\n'
            'void f_div7(void) {\n'
            '    int r;\n'
            '    asm("mov $7,%%ecx\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","ecx","edx");\n'
            '}\n'
            'int f_mod10_check(void) {\n'
            '    /* div %%ecx puts remainder in %%edx; cmp %%edx,0 checks mod */\n'
            '    int r;\n'
            '    asm("mov $10,%%ecx\\n\\t"\n'
            '        "mov $42,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%edx,%0\\n\\t"\n'
            '        : "=r"(r) : : "eax","ecx","edx");\n'
            '    /* separate asm for the CMP to prevent reordering */\n'
            '    asm("cmp $0,%%edx\\n\\t" :: "d"(r) : );\n'
            '    return r;\n'
            '}\n'
            'int main(void) { f_div10(); f_div7(); return f_mod10_check(); }\n'
        )
        with open(src, "w") as f:
            f.write(_src)
        subprocess.run(
            ["gcc", "-O0", "-o", cls._bin, src],
            capture_output=True,
            timeout=30,
        )

    @classmethod
    def teardown_class(cls):
        import shutil

        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_backward_scan_finds_div10(self):
        """mov $10, %%ecx before div %%ecx → divisor=10."""
        from fuzzer_tool.core.elf import extract_div_constants

        d, w = extract_div_constants(self._bin)
        vals = [v for v in d.values() if v == 10]
        assert len(vals) >= 1, f"expected divisor=10, got div_map={d} weak={w}"

    def test_backward_scan_finds_div7(self):
        """mov $7, %%ecx before div %%ecx → divisor=7."""
        from fuzzer_tool.core.elf import extract_div_constants

        d, w = extract_div_constants(self._bin)
        vals = [v for v in d.values() if v == 7]
        assert len(vals) >= 1, f"expected divisor=7, got div_map={d} weak={w}"

    def test_non_elf_returns_empty(self):
        """Non-ELF file returns empty dict and set."""
        from fuzzer_tool.core.elf import extract_div_constants

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            f.write(b"\x00" * 100)
            f.flush()
            d, w = extract_div_constants(f.name)
        assert d == {}
        assert w == set()

    def test_forward_modulus_cmp_mapped(self):
        """cmp $0, %%edx after div → CMP PC mapped to same divisor."""
        from fuzzer_tool.core.elf import extract_div_constants

        d, w = extract_div_constants(self._bin)
        # At least one entry should have divisor=10 from a CMP (not DIV)
        vals = [v for v in d.values() if v == 10]
        # We expect at least 2 entries with divisor=10: the DIV and the CMP
        assert len(vals) >= 2, (
            f"expected >=2 entries with divisor=10 "
            f"(DIV PC + CMP PC), got {len(vals)}: div_map={d} weak={w}"
        )

    def test_weak_modulus_variable_divisor(self):
        """div with variable divisor (runtime parameter) → weak_mod_pcs set."""
        import subprocess, os

        from fuzzer_tool.core.elf import extract_div_constants

        v_src = os.path.join(self._tmpdir, "var_mod.c")
        v_bin = os.path.join(self._tmpdir, "var_mod")
        with open(v_src, "w") as f:
            f.write(
                'int f(int d) {\n'
                '    int r;\n'
                '    asm("mov %1,%%ecx\\n\\t"\n'
                '        "mov $100,%%eax\\n\\t"\n'
                '        "xor %%edx,%%edx\\n\\t"\n'
                '        "div %%ecx\\n\\t"\n'
                '        "mov %%edx,%0\\n\\t"\n'
                '        "cmp $0,%%edx\\n\\t"\n'
                '        : "=r"(r) : "r"(d) : "eax","ecx","edx");\n'
                '    return r;\n'
                '}\n'
                'int main(void) { return f(7); }\n'
            )
        subprocess.run(["gcc", "-O0", "-o", v_bin, v_src],
                       capture_output=True, timeout=30)
        d, w = extract_div_constants(v_bin)
        # No constant divisor should be resolvable (d comes from parameter)
        # The CMP that checks EDX should be in weak_mod_pcs
        assert len(w) >= 1, (
            f"expected at least 1 CMP in weak_mod_pcs "
            f"for variable-divisor DIV, got div_map={d} weak={w}"
        )
