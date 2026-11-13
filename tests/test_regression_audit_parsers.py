"""Regressions from the 2026-11-13 audit: hostile ELF offsets, cmplog counter files, deep grammars."""

import shutil
import struct
import sys
from pathlib import Path

import pytest

from fuzzer_tool.core import elf

_E_PHOFF = 32
_E_SHOFF = 40
_U64_MAX = 2**64 - 1
# Section headers are ignored by the kernel: a binary with these still runs.
_HOSTILE_SHOFF = [_U64_MAX, 2**63, 0xDEADBEEF]


# A PIE (ET_DYN): is_elf_executable reads its program headers.
_PIE = "/bin/true"


def _patched(tmp_path, offset: int, value: int, src: str = sys.executable) -> str:
    """A real, runnable ELF with one 64-bit header field overwritten."""
    out = tmp_path / "t"
    shutil.copy(src, out)
    raw = bytearray(out.read_bytes())
    struct.pack_into("<Q", raw, offset, value)
    out.write_bytes(raw)
    return str(out)


class TestHostileSectionOffset:
    """``e_shoff`` past the file raised ``OverflowError`` / ``struct.error`` at fuzzer startup."""

    @pytest.mark.parametrize("shoff", _HOSTILE_SHOFF)
    def test_regression_symbol_readers_decline(self, tmp_path, shoff):
        target = _patched(tmp_path, _E_SHOFF, shoff)
        raw = Path(target).read_bytes()

        assert elf._sym_sections(raw) is None
        assert elf._has_static_symtab(target) is False
        assert elf._static_symbol_addr(raw, "main") is None
        assert elf.trace_pc_call_sites(target) is None
        assert isinstance(elf.sancov_guard_status(target), str)
        assert elf.estimate_map_size(target) > 0

    def test_intact_binary_still_parsed(self):
        """Falsification: the guard does not decline a well-formed image."""
        raw = Path(sys.executable).read_bytes()

        assert elf._sym_sections(raw) is not None


class TestHostileProgramOffset:
    """``e_phoff = 2**64 - 1`` raised ``ValueError`` out of ``is_elf_executable``."""

    @pytest.mark.parametrize("phoff", [_U64_MAX, 2**63, 2**40])
    def test_regression_declined(self, tmp_path, phoff):
        assert elf.is_elf_executable(_patched(tmp_path, _E_PHOFF, phoff, _PIE)) is False

    def test_intact_binary_is_executable(self):
        assert elf.is_elf_executable(_PIE) is True


class TestCmplogCounterFiles:
    """Counter files come from inside the target: bad UTF-8 and >int64 values raised per exec."""

    _GOOD = "CNS cmp4 0x401000 3 1\n"

    @staticmethod
    def _counters():
        from fuzzer_tool.core import cmplog

        return cmplog._SiteCounters(lambda _n: None)

    @pytest.mark.parametrize("value", [str(2**63), str(2**64), "-" + str(2**64), "9" * 5000])
    def test_regression_oversized_count_folds_nothing(self, value):
        sc = self._counters()
        sc.fold([f"CNS cmp4 0x401000 {value} 1\n", f"CNS cmp4 0x401000 1 {value}\n"])

        assert sc._n_slots == 0

    def test_valid_line_still_folds(self):
        """Falsification: an in-range line still lands in a slot."""
        sc = self._counters()
        sc.fold([self._GOOD])

        assert sc._n_slots == 1

    def test_int64_max_is_accepted(self):
        """Adversarial: the largest representable count is not rejected."""
        sc = self._counters()
        sc.fold([f"CNS cmp4 0x401000 {2**63 - 1} 1\n"])

        assert sc._n_slots == 1

    def test_regression_sites_file_with_non_utf8(self, tmp_path):
        from fuzzer_tool.core import cmplog

        col = cmplog.CmplogCollector.__new__(cmplog.CmplogCollector)
        col.sites_path = str(tmp_path / "sites")
        col._sites_offset = 0
        raw = self._GOOD.encode() + b"\xff\xfe garbage\n"
        (tmp_path / "sites").write_bytes(raw)

        lines = col._read_site_lines()

        assert lines[0] == self._GOOD.rstrip("\n")
        assert len(lines) == raw.count(b"\n")
        assert col._sites_offset == len(raw)


class TestDwarfTypeUnits:
    """A DWARF5 type unit ended the CU walk: no line table with ``-fdebug-types-section``."""

    _SRCS = {
        "c.cc": "struct A { int x; };\nint fa(A *a) {\n  return a->x;\n}\n",
        "d.cc": "struct B { long y; };\nlong fb(B *b) {\n  return b->y;\n}\nint main() {\n  return 0;\n}\n",
    }

    def _build(self, tmp_path, *flags):
        cxx = shutil.which("clang++")
        if not cxx:
            pytest.skip("no clang++")
        for name, body in self._SRCS.items():
            (tmp_path / name).write_text(body)
        out = tmp_path / "tu"
        cmd = [
            cxx,
            "-g",
            "-gdwarf-5",
            *flags,
            "-o",
            str(out),
            *(str(tmp_path / n) for n in self._SRCS),
        ]
        import subprocess

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"build failed: {r.stderr[:300]}")
        return str(out)

    def _files(self, target):
        from fuzzer_tool.core.dwarf import DwarfLineResolver

        res = DwarfLineResolver(target)
        loaded = res.load()
        return loaded, sorted(res._by_basename)

    def test_regression_type_units_are_skipped(self, tmp_path):
        loaded, files = self._files(self._build(tmp_path, "-fdebug-types-section"))

        assert loaded
        assert files == sorted(self._SRCS)

    def test_plain_dwarf5_unchanged(self, tmp_path):
        """Falsification: without type units the same two files resolve."""
        loaded, files = self._files(self._build(tmp_path))

        assert loaded
        assert files == sorted(self._SRCS)


class TestCompressedSectionCap:
    """A ``SHF_COMPRESSED`` section was inflated in full before ``ch_size`` applied."""

    _BOMB = 64 * 1024 * 1024
    _DECLARED = 16
    _PEAK_BUDGET = 8 * 1024 * 1024

    def _elf(self, declared: int, payload: bytes) -> bytes:
        """ELF64 with .shstrtab and one zlib-compressed .debug_info."""
        import zlib

        names = b"\x00.shstrtab\x00.debug_info\x00"
        # Elf64_Chdr: ch_type, ch_reserved, ch_size, ch_addralign
        body = struct.pack("<IIQQ", 1, 0, declared, 1) + zlib.compress(payload, 9)
        shoff = 64 + len(names) + len(body)
        hdr = bytearray(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56)
        struct.pack_into("<Q", hdr, _E_SHOFF, shoff)
        struct.pack_into("<HHH", hdr, 58, 64, 3, 1)  # e_shentsize, e_shnum, e_shstrndx

        def shdr(name, flags, off, size):
            return struct.pack("<IIQQQQIIQQ", name, 1, flags, 0, off, size, 0, 0, 1, 0)

        table = bytes(64) + shdr(1, 0, 64, len(names)) + shdr(11, 0x800, 64 + len(names), len(body))
        return bytes(hdr) + names + body + table

    def test_regression_inflate_is_capped(self):
        import tracemalloc

        from fuzzer_tool.core.dwarf import _elf_sections

        image = self._elf(self._DECLARED, bytes(self._BOMB))
        tracemalloc.start()
        try:
            sections = _elf_sections(image)
            _cur, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert len(sections[b".debug_info"]) == self._DECLARED
        assert peak < self._PEAK_BUDGET

    def test_honest_section_inflates_fully(self):
        """Falsification: a truthful ch_size yields the whole payload."""
        from fuzzer_tool.core.dwarf import _elf_sections

        payload = bytes(range(256)) * 16

        assert _elf_sections(self._elf(len(payload), payload))[b".debug_info"] == payload


class TestTreeParseDepth:
    """``TreeMutator.parse`` recursed once per bracket level: a nested seed raised ``RecursionError``."""

    @staticmethod
    def _mutator():
        from fuzzer_tool.core.grammar import Grammar, TreeMutator

        grammar = Grammar()
        grammar.parse('value = "x"\n')
        return TreeMutator(grammar)

    @staticmethod
    def _flat(node) -> bytes:
        """Leaf bytes in order, walked without recursion."""
        out, stack = [], [node]
        while stack:
            cur = stack.pop()
            if cur.children:
                stack.extend(reversed(cur.children))
                continue
            out.append(cur.data)
        return b"".join(out)

    @pytest.mark.parametrize("depth", [2000, 20000])
    def test_regression_deep_nesting_parses(self, depth):
        data = b"[" * depth + b"]" * depth

        assert self._flat(self._mutator().parse(data)) == data

    def test_shallow_nesting_keeps_its_tree(self):
        """Falsification: ordinary nesting still splits into delimiters and elements."""
        tree = self._mutator().parse(b"[[1,2],[3]]")

        # delim, [1,2], [3], delim; separators are not kept as nodes.
        assert [c.data for c in tree.children if c.rule == "delim"] == [b"[", b"]"]
        assert [self._flat(c) for c in tree.children[1:-1]] == [b"[12]", b"[3]"]
