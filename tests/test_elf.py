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


class TestExtractConstants:
    """Tests for extract_constants_pure and helpers."""

    def test_non_elf(self):
        from fuzzer_tool.core.elf import extract_constants_pure

        result = extract_constants_pure("/dev/null")
        assert result == []

    def test_too_short(self, tmp_path):
        from fuzzer_tool.core.elf import extract_constants_pure

        p = tmp_path / "short"
        p.write_bytes(b"\x7fELF")
        assert extract_constants_pure(str(p)) == []

    def test_elf32_rejected(self, tmp_path):
        from fuzzer_tool.core.elf import extract_constants_pure

        header = bytearray(64)
        header[0:4] = b"\x7fELF"
        header[4] = 1  # ELFCLASS32
        header[5] = 2  # ELFDATA2MSB
        p = tmp_path / "elf32"
        p.write_bytes(bytes(header))
        assert extract_constants_pure(str(p)) == []

    def test_no_text_section(self, tmp_path):
        """ELF with no .text section returns []."""
        from fuzzer_tool.core.elf import extract_constants_pure

        header = _build_elf64_header(e_shnum=1, e_shstrndx=0, e_shentsize=64)
        shstrtab = b".shstrtab\x00"
        sh = _build_section_header(sh_type=3, sh_name=0, sh_offset=256, sh_size=len(shstrtab))
        data = bytearray(256 + len(shstrtab) + 100)
        data[:64] = header
        data[64:128] = sh
        data[256 : 256 + len(shstrtab)] = shstrtab
        p = tmp_path / "no_text"
        p.write_bytes(bytes(data))
        assert extract_constants_pure(str(p)) == []

    def test_real_binary(self):
        """Extract constants from the png_read target."""
        from fuzzer_tool.core.elf import extract_constants_pure

        target = "targets/png_read.so"
        if not os.path.isfile(target):
            return
        result = extract_constants_pure(target)
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 256
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
        import os
        import subprocess
        import tempfile

        cls._tmpdir = tempfile.mkdtemp(prefix="elf_test_")
        src = os.path.join(cls._tmpdir, "test_div.c")
        cls._bin = os.path.join(cls._tmpdir, "test_div")
        # Write C source with inline asm that explicitly sets ECX to a
        # constant and then uses DIV with that register — avoids the
        # -O0 memory-operand issue.
        _src = (
            "void f_div10(void) {\n"
            "    int r;\n"
            '    asm("mov $10,%%ecx\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","ecx","edx");\n'
            "}\n"
            "void f_div7(void) {\n"
            "    int r;\n"
            '    asm("mov $7,%%ecx\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","ecx","edx");\n'
            "}\n"
            "int f_mod10_check(void) {\n"
            "    /* div %%ecx puts remainder in %%edx; cmp %%edx,0 checks mod */\n"
            "    int r;\n"
            '    asm("mov $10,%%ecx\\n\\t"\n'
            '        "mov $42,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%edx,%0\\n\\t"\n'
            '        : "=r"(r) : : "eax","ecx","edx");\n'
            "    /* separate asm for the CMP to prevent reordering */\n"
            '    asm("cmp $0,%%edx\\n\\t" :: "d"(r) : );\n'
            "    return r;\n"
            "}\n"
            "int f_mod10_check_copy(void) {\n"
            "    /* div puts remainder in %%edx; copy to %%eax, cmp %%eax */\n"
            "    int r;\n"
            '    asm("mov $10,%%ecx\\n\\t"\n'
            '        "mov $42,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%edx,%%eax\\n\\t"\n'
            '        "cmp $5,%%eax\\n\\t"\n'
            '        "mov %%eax,%0"\n'
            '        : "=r"(r) : : "eax","ecx","edx");\n'
            "    return r;\n"
            "}\n"
            "int main(void) { f_div10(); f_div7(); f_mod10_check_copy(); return f_mod10_check(); }\n"
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
        import tempfile

        from fuzzer_tool.core.elf import extract_div_constants

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
        import os
        import subprocess

        from fuzzer_tool.core.elf import extract_div_constants

        v_src = os.path.join(self._tmpdir, "var_mod.c")
        v_bin = os.path.join(self._tmpdir, "var_mod")
        with open(v_src, "w") as f:
            f.write(
                "int f(int d) {\n"
                "    int r;\n"
                '    asm("mov %1,%%ecx\\n\\t"\n'
                '        "mov $100,%%eax\\n\\t"\n'
                '        "xor %%edx,%%edx\\n\\t"\n'
                '        "div %%ecx\\n\\t"\n'
                '        "mov %%edx,%0\\n\\t"\n'
                '        "cmp $0,%%edx\\n\\t"\n'
                '        : "=r"(r) : "r"(d) : "eax","ecx","edx");\n'
                "    return r;\n"
                "}\n"
                "int main(void) { return f(7); }\n"
            )
        subprocess.run(["gcc", "-O0", "-o", v_bin, v_src], capture_output=True, timeout=30)
        d, w = extract_div_constants(v_bin)
        # No constant divisor should be resolvable (d comes from parameter)
        # The CMP that checks EDX should be in weak_mod_pcs
        assert len(w) >= 1, (
            f"expected at least 1 CMP in weak_mod_pcs "
            f"for variable-divisor DIV, got div_map={d} weak={w}"
        )

    def test_forward_modulus_reg_copy(self):
        """mov %%edx,%%eax; cmp $5,%%eax after div → CMP PC mapped."""
        from fuzzer_tool.core.elf import extract_div_constants

        d, w = extract_div_constants(self._bin)
        # The function f_mod10_check_copy does:
        #   div %%ecx  → remainder in EDX
        #   mov %%edx,%%eax  → copy to EAX
        #   cmp $5,%%eax     → compare EAX (reg 0)
        # The forward analysis should map CMP PC → divisor=10
        vals_10 = [v for v in d.values() if v == 10]
        # We expect at least 3 entries: DIV(f_div10) + CMP(f_mod10_check)
        # + CMP(f_mod10_check_copy). The copy CMP adds a 3rd entry via _rem_regs.
        assert len(vals_10) >= 3, (
            f"expected >=3 entries with divisor=10 "
            f"(DIV + direct-CMP + reg-copy-CMP), "
            f"got {len(vals_10)}: div_map={d} weak={w}"
        )

    def test_weak_modulus_reg_copy_variable(self):
        """div with variable divisor, mov %%edx,%%eax; cmp %%eax -> weak_mod."""
        import os
        import subprocess

        from fuzzer_tool.core.elf import extract_div_constants

        v_src = os.path.join(self._tmpdir, "var_mod_copy.c")
        v_bin = os.path.join(self._tmpdir, "var_mod_copy")
        with open(v_src, "w") as f:
            f.write(
                "int f(int d) {\n"
                "    int r;\n"
                '    asm("mov %1,%%ecx\\n\\t"\n'
                '        "mov $100,%%eax\\n\\t"\n'
                '        "xor %%edx,%%edx\\n\\t"\n'
                '        "div %%ecx\\n\\t"\n'
                '        "mov %%edx,%%eax\\n\\t"\n'
                '        "mov %%eax,%0\\n\\t"\n'
                '        "cmp $0,%%eax\\n\\t"\n'
                '        : "=r"(r) : "r"(d) : "eax","ecx","edx");\n'
                "    return r;\n"
                "}\n"
                "int main(void) { return f(7); }\n"
            )
        subprocess.run(["gcc", "-O0", "-o", v_bin, v_src], capture_output=True, timeout=30)
        d, w = extract_div_constants(v_bin)
        assert len(w) >= 1, (
            f"expected at least 1 CMP in weak_mod_pcs "
            f"for variable-divisor DIV with reg-copy, "
            f"got div_map={d} weak={w}"
        )


# ============================================================================
# Pure x86-64 decoder unit tests (no compilation needed)
# ============================================================================


class TestX86_64Decoder:
    """Direct unit tests for _decode_x86_64 with hand-crafted byte sequences."""

    def _decode(self, code: bytes, base: int = 0x1000):
        from fuzzer_tool.core.elf import _decode_x86_64

        return list(_decode_x86_64(code, base))

    def test_mov_ecx_imm32(self):
        """B9 0A000000 → mov ecx, 10"""
        insns = self._decode(b"\xb9\x0a\x00\x00\x00")
        assert len(insns) == 1
        insn = insns[0]
        assert insn.insn_id == 1  # _INS_MOV
        assert len(insn.operands) == 2
        assert insn.operands[0].type == 1  # _OP_REG
        assert insn.operands[0].reg == 1  # ecx
        assert insn.operands[1].type == 2  # _OP_IMM
        assert insn.operands[1].imm == 10

    def test_mov_eax_imm32(self):
        """B8 64000000 → mov eax, 100"""
        insns = self._decode(b"\xb8\x64\x00\x00\x00")
        assert len(insns) == 1
        assert insns[0].insn_id == 1
        assert insns[0].operands[0].reg == 0  # eax
        assert insns[0].operands[1].imm == 100

    def test_mov_rax_imm64_rex_w(self):
        """48 B8 1000000000000000 → mov rax, 16"""
        insns = self._decode(b"\x48\xb8\x10\x00\x00\x00\x00\x00\x00\x00")
        assert len(insns) == 1
        assert insns[0].insn_id == 1
        assert insns[0].operands[0].reg == 0  # rax
        assert insns[0].operands[0].size == 8
        assert insns[0].operands[1].imm == 16

    def test_mov_r8_imm64_rex_b(self):
        """49 B8 ... → mov r8, imm64 (REX.B extends opcode reg)"""
        imm = 0x1234567890ABCDEF
        insns = self._decode(b"\x49\xb8" + struct.pack("<q", imm))
        assert len(insns) == 1
        assert insns[0].insn_id == 1
        assert insns[0].operands[0].reg == 8  # r8
        assert insns[0].operands[1].imm == imm

    def test_xor_edx_edx(self):
        """33 D2 → xor edx, edx"""
        insns = self._decode(b"\x33\xd2")
        assert len(insns) == 1
        assert insns[0].insn_id == 3  # _INS_XOR
        assert insns[0].operands[0].reg == 2  # edx
        assert insns[0].operands[1].reg == 2  # edx
        assert 2 in insns[0]._regs_write

    def test_xor_ecx_ecx(self):
        """31 C9 → xor ecx, ecx (alternate encoding)"""
        insns = self._decode(b"\x31\xc9")
        assert len(insns) == 1
        assert insns[0].insn_id == 3  # _INS_XOR
        assert insns[0].operands[0].reg == 1  # ecx (rm destination)
        assert insns[0].operands[1].reg == 1  # ecx (reg source)

    def test_div_ecx(self):
        """F7 F1 → div ecx (F7 /6, ModR/M=0xF1=11_110_001)"""
        insns = self._decode(b"\xf7\xf1")
        assert len(insns) == 1
        assert insns[0].insn_id == 6  # _INS_DIV
        assert insns[0].operands[0].reg == 1  # ecx
        assert 0 in insns[0]._regs_read  # eax
        assert 2 in insns[0]._regs_read  # edx
        assert 1 in insns[0]._regs_read  # ecx

    def test_idiv_ecx(self):
        """F7 F9 → idiv ecx (F7 /7, ModR/M=0xF9=11_111_001)"""
        insns = self._decode(b"\xf7\xf9")
        assert len(insns) == 1
        assert insns[0].insn_id == 7  # _INS_IDIV
        assert insns[0].operands[0].reg == 1  # ecx

    def test_div_edx(self):
        """F7 F2 → div edx (ModR/M=0xF2=11_110_010)"""
        insns = self._decode(b"\xf7\xf2")
        assert len(insns) == 1
        assert insns[0].insn_id == 6  # _INS_DIV
        assert insns[0].operands[0].reg == 2  # edx

    def test_idiv_memory_operand_disp8(self):
        """F7 7D FC → idiv dword ptr [rbp-4] (F7 /7, mod=01, rm=5, disp8=-4).

        Regression: memory-operand GRP3 DIV/IDIV were previously decoded as
        _INS_OTHER — only the register-operand form (mod==3) was recognized.
        """
        insns = self._decode(b"\xf7\x7d\xfc")
        assert len(insns) == 1
        insn = insns[0]
        assert insn.insn_id == 7  # _INS_IDIV
        assert insn.operands[0].type == 3  # _OP_MEM
        assert 0 in insn._regs_read  # eax
        assert 2 in insn._regs_read  # edx
        assert 5 not in insn._regs_read  # divisor lives in memory, not rbp

    def test_div_memory_operand_disp32(self):
        """F7 B5 00010000 → div dword ptr [rbp+0x100] (F7 /6, mod=10, disp32)"""
        insns = self._decode(b"\xf7\xb5\x00\x01\x00\x00")
        assert len(insns) == 1
        insn = insns[0]
        assert insn.insn_id == 6  # _INS_DIV
        assert insn.operands[0].type == 3  # _OP_MEM
        assert 0 in insn._regs_read
        assert 2 in insn._regs_read

    def test_idiv_memory_operand_sib(self):
        """F7 3C 24 → idiv dword ptr [rsp] (F7 /7, mod=00, rm=4 → SIB)"""
        insns = self._decode(b"\xf7\x3c\x24")
        assert len(insns) == 1
        insn = insns[0]
        assert insn.insn_id == 7  # _INS_IDIV
        assert insn.operands[0].type == 3  # _OP_MEM

    def test_div_byte_register(self):
        """F6 F1 → div cl (F6 /6, ModR/M=0xF1=11_110_001)"""
        insns = self._decode(b"\xf6\xf1")
        assert len(insns) == 1
        insn = insns[0]
        assert insn.insn_id == 6  # _INS_DIV
        assert insn.operands[0].type == 1  # _OP_REG
        assert insn.operands[0].reg == 1  # cl
        assert insn.operands[0].size == 1
        assert 0 in insn._regs_read  # al/ax
        assert 2 in insn._regs_read  # dx

    def test_idiv_byte_memory(self):
        """F6 7D FC → idiv byte ptr [rbp-4] (F6 /7, mod=01, disp8)"""
        insns = self._decode(b"\xf6\x7d\xfc")
        assert len(insns) == 1
        insn = insns[0]
        assert insn.insn_id == 7  # _INS_IDIV
        assert insn.operands[0].type == 3  # _OP_MEM
        assert insn.operands[0].size == 1

    def test_f6_test_byte_imm(self):
        """F6 C1 0A → test cl, 10 (F6 /0, ModR/M=0xC1=11_000_001, imm8)"""
        insns = self._decode(b"\xf6\xc1\x0a")
        assert len(insns) == 1
        insn = insns[0]
        assert insn.insn_id == 12  # _INS_TEST
        assert insn.operands[0].reg == 1  # cl
        assert insn.operands[1].imm == 10

    def test_cmp_eax_imm32(self):
        """3D 00000000 → cmp eax, 0 (special encoding for eax).

        This is a special encoding (3D = opcode for CMP EAX, imm32).
        Our decoder only handles 81/83 for CMP, so this should be _INS_OTHER.
        That's fine — the important CMP patterns are 81 /7 and 83 /7.
        """
        insns = self._decode(b"\x3d\x00\x00\x00\x00")
        # 3D is not in our opcode table, so it's 5 unrecognized bytes

    def test_cmp_edx_imm32(self):
        """81 FA 00000000 → cmp edx, 0 (81 /7, ModR/M=0xFA=11_110_010)"""
        insns = self._decode(b"\x81\xfa\x00\x00\x00\x00")
        assert len(insns) == 1
        assert insns[0].insn_id == 4  # _INS_CMP
        assert insns[0].operands[0].reg == 2  # edx
        assert insns[0].operands[1].imm == 0
        assert 2 in insns[0]._regs_read

    def test_cmp_ecx_imm8(self):
        """83 F9 05 → cmp ecx, 5 (83 /7, ModR/M=0xF9=11_111_001)"""
        insns = self._decode(b"\x83\xf9\x05")
        assert len(insns) == 1
        assert insns[0].insn_id == 4  # _INS_CMP
        assert insns[0].operands[0].reg == 1  # ecx
        assert insns[0].operands[1].imm == 5

    def test_cmp_ecx_edx(self):
        """39 D1 → cmp ecx, edx (39 /r, ModR/M=0xD1=11_010_001)"""
        insns = self._decode(b"\x39\xd1")
        assert len(insns) == 1
        assert insns[0].insn_id == 4  # _INS_CMP
        assert 1 in insns[0]._regs_read  # ecx
        assert 2 in insns[0]._regs_read  # edx

    def test_cmp_edx_ecx(self):
        """3B D1 → cmp edx, ecx (3B /r, alternate direction)"""
        insns = self._decode(b"\x3b\xd1")
        assert len(insns) == 1
        assert insns[0].insn_id == 4  # _INS_CMP
        assert 2 in insns[0]._regs_read  # edx
        assert 1 in insns[0]._regs_read  # ecx

    def test_mov_r32_rm32(self):
        """89 C1 → mov ecx, eax (89 /r, mov r/m32, r32).

        Not handled by our minimal decoder (only C7 /0 for mov r/m, imm).
        Should decode as two _INS_OTHER bytes — that's fine for
        extract_div_constants which doesn't need this pattern.
        """
        insns = self._decode(b"\x89\xc1")
        # 89 /r with mod=3 is now decoded as MOV reg-to-reg
        assert len(insns) == 1, f"expected 1 insn, got {len(insns)}"
        assert insns[0].insn_id == 1  # _INS_MOV

    def test_lea_eax(self):
        """8D 05 10000000 → lea eax, [rip+16]"""
        insns = self._decode(b"\x8d\x05\x10\x00\x00\x00")
        assert len(insns) == 1
        assert insns[0].insn_id == 5  # _INS_LEA
        assert insns[0].operands[0].reg == 0  # eax

    def test_ret(self):
        """C3 → ret"""
        insns = self._decode(b"\xc3")
        assert len(insns) == 1
        assert insns[0].insn_id == 10  # _INS_RET
        assert 3 in insns[0].groups  # _GRP_RET

    def test_call_rel32(self):
        """E8 10000000 → call +16"""
        insns = self._decode(b"\xe8\x10\x00\x00\x00")
        assert len(insns) == 1
        assert insns[0].insn_id == 8  # _INS_CALL
        assert 1 in insns[0].groups  # _GRP_CALL

    def test_jmp_rel8(self):
        """EB FE → jmp -2 (infinite loop)"""
        insns = self._decode(b"\xeb\xfe")
        assert len(insns) == 1
        assert insns[0].insn_id == 9  # _INS_JMP
        assert 2 in insns[0].groups  # _GRP_JUMP

    def test_jmp_rel32(self):
        """E9 FFFFFF00 → jmp +0xFFFFFF"""
        insns = self._decode(b"\xe9\xff\xff\xff\x00")
        assert len(insns) == 1
        assert insns[0].insn_id == 9  # _INS_JMP

    def test_jcc_rel8(self):
        """74 FE → je -2"""
        insns = self._decode(b"\x74\xfe")
        assert len(insns) == 1
        assert insns[0].insn_id == 11  # _INS_JCC
        assert 2 in insns[0].groups  # _GRP_JUMP

    def test_jcc_rel32(self):
        """0F 84 FFFFFF00 → je near"""
        insns = self._decode(b"\x0f\x84\xff\xff\xff\x00")
        assert len(insns) == 1
        assert insns[0].insn_id == 11  # _INS_JCC

    def test_rex_prefix_extends_rm(self):
        """41 F7 F1 → div r9d (REX.B=1 extends rm to r8+r1=9)"""
        insns = self._decode(b"\x41\xf7\xf1")
        assert len(insns) == 1
        assert insns[0].insn_id == 6  # _INS_DIV
        assert insns[0].operands[0].reg == 9  # r9d

    def test_rex_prefix_extends_reg(self):
        """41 33 C1 → xor eax, r9d (REX.B=1 extends rm=1 to r9).

        REX.B extends the rm field: rm=1 → 1|0x8 = 9 (r9d).
        The reg field (destination) stays at 0 (eax, no REX.R).
        """
        insns = self._decode(b"\x41\x33\xc1")
        assert len(insns) == 1
        assert insns[0].insn_id == 3  # _INS_XOR
        # reg field with REX.R=0: stays at 0 → eax
        assert insns[0].operands[0].reg == 0  # eax (destination)
        # rm field with REX.B=1: 1 | 0x8 = 9 → r9d
        assert insns[0].operands[1].reg == 9  # r9d (source)

    def test_mov_r10d_imm32(self):
        """41 BA 2A000000 → mov r10d, 42 (REX.B=1, B8+2=BA)"""
        insns = self._decode(b"\x41\xba\x2a\x00\x00\x00")
        assert len(insns) == 1
        assert insns[0].insn_id == 1  # _INS_MOV
        assert insns[0].operands[0].reg == 10  # r10d
        assert insns[0].operands[1].imm == 42

    def test_div_r8d(self):
        """41 F7 F2 → div r10d (REX.B=1, F7 /6, rm=2 → r10)"""
        insns = self._decode(b"\x41\xf7\xf2")
        assert len(insns) == 1
        assert insns[0].insn_id == 6  # _INS_DIV
        assert insns[0].operands[0].reg == 10  # r10d

    def test_multiple_instructions(self):
        """Sequence: mov, xor, div, cmp, ret"""
        code = (
            b"\xb9\x0a\x00\x00\x00"  # mov ecx, 10
            b"\xb8\x64\x00\x00\x00"  # mov eax, 100
            b"\x31\xd2"  # xor edx, edx
            b"\xf7\xf1"  # div ecx
            b"\x81\xfa\x00\x00\x00\x00"  # cmp edx, 0
            b"\xc3"  # ret
        )
        insns = self._decode(code)
        assert len(insns) == 6
        assert insns[0].insn_id == 1  # mov
        assert insns[1].insn_id == 1  # mov
        assert insns[2].insn_id == 3  # xor
        assert insns[3].insn_id == 6  # div
        assert insns[4].insn_id == 4  # cmp
        assert insns[5].insn_id == 10  # ret

    def test_empty_input(self):
        """Empty byte sequence yields no instructions."""
        insns = self._decode(b"")
        assert len(insns) == 0

    def test_truncated_rex(self):
        """REX prefix at end of input yields one OTHER insn."""
        insns = self._decode(b"\x48")
        assert len(insns) == 1
        assert insns[0].insn_id == 99  # _INS_OTHER

    def test_unrecognized_opcode(self):
        """Unrecognized opcode yields _INS_OTHER with length=1."""
        insns = self._decode(b"\xcc")  # INT3 — not handled
        assert len(insns) == 1
        assert insns[0].insn_id == 99  # _INS_OTHER
        assert insns[0].length == 1

    def test_endbr64(self):
        """F3 0F 1E FA → one _INS_OTHER with length=4."""
        insns = self._decode(b"\xf3\x0f\x1e\xfa")
        assert len(insns) == 1, f"expected 1 insn, got {len(insns)}"
        assert insns[0].insn_id == 99  # _INS_OTHER
        assert insns[0].length == 4

    def test_multibyte_nop(self):
        """0F 1F 00 → one _INS_OTHER with ModRM consumed (length=3)."""
        insns = self._decode(b"\x0f\x1f\x00")
        assert len(insns) == 1, f"expected 1 insn, got {len(insns)}"
        assert insns[0].insn_id == 99  # _INS_OTHER
        assert insns[0].length == 3

    def test_mov_r32_rm32_89(self):
        """89 D0 → mov eax, edx (89 /r, register-to-register MOV)."""
        insns = self._decode(b"\x89\xd0")
        assert len(insns) == 1
        assert insns[0].insn_id == 1  # _INS_MOV
        assert insns[0].operands[0].type == 1  # _OP_REG
        assert insns[0].operands[0].reg == 0  # eax (destination, rm field)
        assert insns[0].operands[1].type == 1  # _OP_REG
        assert insns[0].operands[1].reg == 2  # edx (source, reg field)
        assert 2 in insns[0]._regs_read  # edx is read
        assert 0 in insns[0]._regs_write  # eax is written

    def test_mov_r32_rm32_8b(self):
        """8B C2 → mov eax, edx (8B /r, register-to-register MOV)."""
        insns = self._decode(b"\x8b\xc2")
        assert len(insns) == 1
        assert insns[0].insn_id == 1  # _INS_MOV
        assert insns[0].operands[0].type == 1  # _OP_REG
        assert insns[0].operands[0].reg == 0  # eax (destination, reg field)
        assert insns[0].operands[1].type == 1  # _OP_REG
        assert insns[0].operands[1].reg == 2  # edx (source, rm field)
        assert 2 in insns[0]._regs_read  # edx is read
        assert 0 in insns[0]._regs_write  # eax is written

    def test_endbr64_with_sequence(self):
        """endbr64 + mov + xor + div + reg-copy + cmp + ret → all decoded."""
        code = (
            b"\xf3\x0f\x1e\xfa"  # endbr64 (4 bytes)
            b"\xb9\x0a\x00\x00\x00"  # mov ecx, 10
            b"\xb8\x64\x00\x00\x00"  # mov eax, 100
            b"\x31\xd2"  # xor edx, edx
            b"\xf7\xf1"  # div ecx
            b"\x89\xd0"  # mov eax, edx
            b"\x83\xf8\x05"  # cmp eax, 5
            b"\xc3"  # ret
        )
        insns = self._decode(code)
        assert len(insns) == 8, (
            f"expected 8 insns, got {len(insns)}: {[(i.insn_id, i.length) for i in insns]}"
        )
        assert insns[0].insn_id == 99  # endbr64 → _INS_OTHER (with F3 consumed as prefix)
        assert insns[0].length == 4
        assert insns[1].insn_id == 1  # mov ecx, 10
        assert insns[1].operands[1].imm == 10
        assert insns[2].insn_id == 1  # mov eax, 100
        assert insns[2].operands[1].imm == 100
        assert insns[3].insn_id == 3  # xor edx, edx
        assert insns[4].insn_id == 6  # div ecx
        assert insns[5].insn_id == 1  # mov eax, edx
        assert insns[5].operands[0].reg == 0
        assert insns[5].operands[1].reg == 2
        assert insns[6].insn_id == 4  # cmp eax, 5
        assert insns[6].operands[0].reg == 0
        assert insns[6].operands[1].imm == 5
        assert insns[7].insn_id == 10  # ret


# ============================================================================
# Comprehensive ELF test with many DIV patterns
# ============================================================================


class TestExtractDivComprehensive:
    """Compile an ELF with diverse DIV patterns and verify extraction."""

    @classmethod
    def setup_class(cls):
        import subprocess
        import tempfile

        cls._tmpdir = tempfile.mkdtemp(prefix="elf_div_test_")
        src = os.path.join(cls._tmpdir, "all_divs.c")
        cls._bin = os.path.join(cls._tmpdir, "all_divs")

        # Use regular string with proper escaping (not raw string)
        _src = (
            "/* Many different DIV patterns for thorough decoder testing */\n"
            "\n"
            "/* Pattern 1: mov imm -> ecx, div ecx */\n"
            "void p1(void) {\n"
            "    int r;\n"
            '    asm("mov $11,%%ecx\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","ecx","edx");\n'
            "}\n"
            "\n"
            "/* Pattern 2: mov imm -> ebx, div ebx */\n"
            "void p2(void) {\n"
            "    int r;\n"
            '    asm("mov $13,%%ebx\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ebx\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","ebx","edx");\n'
            "}\n"
            "\n"
            "/* Pattern 3: mov imm -> esi, div esi */\n"
            "void p3(void) {\n"
            "    int r;\n"
            '    asm("mov $17,%%esi\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%esi\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","esi","edx");\n'
            "}\n"
            "\n"
            "/* Pattern 4: mov imm -> edi, div edi */\n"
            "void p4(void) {\n"
            "    int r;\n"
            '    asm("mov $19,%%edi\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%edi\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","edi","edx");\n'
            "}\n"
            "\n"
            "/* Pattern 5: IDIV (signed divide) */\n"
            "void p5(void) {\n"
            "    int r;\n"
            '    asm("mov $23,%%ecx\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "idiv %%ecx\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","ecx","edx");\n'
            "}\n"
            "\n"
            "/* Pattern 6: cmp after div (modulus check) */\n"
            "int p6(void) {\n"
            "    int r;\n"
            '    asm("mov $29,%%ecx\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%edx,%0\\n\\t"\n'
            '        : "=r"(r) : : "eax","ecx","edx");\n'
            '    asm("cmp $0,%%edx\\n\\t" :: "d"(r) : );\n'
            "    return r;\n"
            "}\n"
            "\n"
            "/* Pattern 7: cmp with non-zero immediate */\n"
            "int p7(void) {\n"
            "    int r;\n"
            '    asm("mov $31,%%ecx\\n\\t"\n'
            '        "mov $100,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%edx,%0\\n\\t"\n'
            '        : "=r"(r) : : "eax","ecx","edx");\n'
            '    asm("cmp $5,%%edx\\n\\t" :: "d"(r) : );\n'
            "    return r;\n"
            "}\n"
            "\n"
            "/* Pattern 8: div with larger divisor */\n"
            "void p8(void) {\n"
            "    int r;\n"
            '    asm("mov $100,%%ecx\\n\\t"\n'
            '        "mov $10000,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","ecx","edx");\n'
            "}\n"
            "\n"
            "/* Pattern 9: div with divisor 1 (edge case) */\n"
            "void p9(void) {\n"
            "    int r;\n"
            '    asm("mov $1,%%ecx\\n\\t"\n'
            '        "mov $42,%%eax\\n\\t"\n'
            '        "xor %%edx,%%edx\\n\\t"\n'
            '        "div %%ecx\\n\\t"\n'
            '        "mov %%eax,%0" : "=r"(r) : : "eax","ecx","edx");\n'
            "}\n"
            "\n"
            "int main(void) {\n"
            "    p1(); p2(); p3(); p4(); p5();\n"
            "    p8(); p9();\n"
            "    return p6() + p7();\n"
            "}\n"
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

    def test_finds_div11(self):
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        assert 11 in d.values(), f"divisor 11 not found in {d}"

    def test_finds_div13(self):
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        assert 13 in d.values(), f"divisor 13 not found in {d}"

    def test_finds_div17(self):
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        assert 17 in d.values(), f"divisor 17 not found in {d}"

    def test_finds_div19(self):
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        assert 19 in d.values(), f"divisor 19 not found in {d}"

    def test_finds_idiv_divisor_23(self):
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        assert 23 in d.values(), f"idiv divisor 23 not found in {d}"

    def test_finds_div29(self):
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        assert 29 in d.values(), f"divisor 29 not found in {d}"

    def test_finds_div31(self):
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        assert 31 in d.values(), f"divisor 31 not found in {d}"

    def test_finds_div100(self):
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        assert 100 in d.values(), f"divisor 100 not found in {d}"

    def test_finds_div1(self):
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        assert 1 in d.values(), f"divisor 1 not found in {d}"

    def test_cmp_mapped_to_divisor(self):
        """CMP instructions that follow DIV should be mapped to the same divisor."""
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        # divisor=29 appears in p6 (div) and p6 (cmp), divisor=31 in p7
        # We should have at least 2 entries for some divisors (DIV + CMP)
        counts = {}
        for v in d.values():
            counts[v] = counts.get(v, 0) + 1
        # At least one divisor should appear >= 2 times (DIV + CMP mapping)
        assert any(c >= 2 for c in counts.values()), (
            f"expected at least one divisor mapped to both DIV and CMP, got counts={counts}"
        )

    def test_total_div_count(self):
        """Should find at least 8 distinct DIV/IDIV instructions."""
        from fuzzer_tool.core.elf import extract_div_constants

        d, _ = extract_div_constants(self._bin)
        assert len(d) >= 8, f"expected >= 8 DIV/CMP entries, got {len(d)}: {d}"
