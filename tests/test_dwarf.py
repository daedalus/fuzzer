"""Tests for core/dwarf.py — pure-Python DWARF line-table resolution.

Expected values are derived independently of the code under test:

  - the line number for each marker is counted from the literal source
    string in this file;
  - the containing function range for each address comes from an inline
    ELF symbol-table parse (a different algorithm than DWARF parsing).

Variants cover DWARF 5 (clang default), DWARF 4, and compressed debug
sections (-gz). Each variant is compiled at test time with clang and
skipped when clang is unavailable.
"""

import shutil
import struct
import subprocess

import pytest

from fuzzer_tool.core.dwarf import DwarfLineResolver

SRC = """\
int helper(int x) {
    return x * 2;
}
int compute(int a) {
    int r = helper(a);
    if (a > 5)
        r += 1;
    return r;
}
int main(void) {
    return compute(3);
}
"""

# marker text -> source line (derived by counting newlines in SRC)
MARKERS = {
    "helper_entry": "int helper(int x) {",
    "helper_body": "return x * 2;",
    "compute_entry": "int compute(int a) {",
    "compute_call": "int r = helper(a);",
    "compute_if": "if (a > 5)",
    "compute_ret": "return r;",
}


def _line(marker: str) -> int:
    return SRC.count("\n", 0, SRC.index(marker)) + 1


def _func_ranges(elf_path: str) -> dict[str, tuple[int, int]]:
    """Independent oracle: parse ELF64 symtab (st_size at offset 16)."""
    with open(elf_path, "rb") as fh:
        elf = fh.read()
    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shentsize = struct.unpack_from("<H", elf, 58)[0]
    e_shnum = struct.unpack_from("<H", elf, 60)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 62)[0]
    shstr = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from("<Q", elf, shstr + 24)[0]
    symtab = strtab = None
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from("<I", elf, sh + 4)[0]
        name = elf[shstr_off + struct.unpack_from("<I", elf, sh)[0] :].split(b"\x00")[0]
        if sh_type == 2:
            symtab = sh
        elif sh_type == 3 and name == b".strtab":
            strtab = sh
    if symtab is None or strtab is None:
        return {}
    sym_off = struct.unpack_from("<Q", elf, symtab + 24)[0]
    sym_size = struct.unpack_from("<Q", elf, symtab + 32)[0]
    sym_entsize = struct.unpack_from("<Q", elf, symtab + 56)[0]
    str_off = struct.unpack_from("<Q", elf, strtab + 24)[0]
    ranges = {}
    for i in range(sym_size // sym_entsize):
        sym = sym_off + i * sym_entsize
        st_info = struct.unpack_from("<B", elf, sym + 4)[0]
        st_value = struct.unpack_from("<Q", elf, sym + 8)[0]
        st_size = struct.unpack_from("<Q", elf, sym + 16)[0]
        name = (
            elf[str_off + struct.unpack_from("<I", elf, sym)[0] :]
            .split(b"\x00")[0]
            .decode(errors="replace")
        )
        if (st_info & 0xF) == 2 and st_size:
            ranges[name] = (st_value, st_value + st_size)
    return ranges


@pytest.fixture(scope="module")
def dwarf_variants(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not available")
    variants = {}
    for label, extra in [
        ("v5", ["-gdwarf-5"]),
        ("v4", ["-gdwarf-4"]),
        ("gz", ["-gz"]),
    ]:
        d = tmp_path_factory.mktemp(label)
        src = d / "dwarf_sanity.c"
        src.write_text(SRC)
        out = d / "dwarf_sanity"
        r = subprocess.run(
            ["clang", "-O0", "-g", *extra, "-o", str(out), str(src)],
            capture_output=True,
        )
        if r.returncode == 0 and out.exists():
            variants[label] = str(out)
    if not variants:
        pytest.skip("no DWARF variants could be built")
    return variants


def _resolver(path):
    r = DwarfLineResolver(path)
    assert r.load(), f"load() failed for {path}"
    return r


class TestDwarfResolution:
    def test_load_all_variants(self, dwarf_variants):
        for label, path in dwarf_variants.items():
            assert DwarfLineResolver(path).load(), label

    def test_line_addresses_inside_function(self, dwarf_variants):
        for label, path in dwarf_variants.items():
            resolver = _resolver(path)
            ranges = _func_ranges(path)
            assert "helper" in ranges and "compute" in ranges, label
            cases = [
                ("helper_entry", "helper"),
                ("helper_body", "helper"),
                ("compute_entry", "compute"),
                ("compute_call", "compute"),
                ("compute_if", "compute"),
                ("compute_ret", "compute"),
            ]
            for marker, func in cases:
                ln = _line(MARKERS[marker])
                addrs = resolver.resolve("dwarf_sanity.c", ln)
                assert addrs, f"{label}: no addresses for {marker} (line {ln})"
                start, end = ranges[func]
                for a in addrs:
                    assert start <= a < end, (
                        f"{label}: {marker} addr {hex(a)} outside {func} [{hex(start)},{hex(end)})"
                    )

    def test_sorted_unique(self, dwarf_variants):
        for label, path in dwarf_variants.items():
            resolver = _resolver(path)
            addrs = resolver.resolve("dwarf_sanity.c", _line(MARKERS["compute_call"]))
            assert addrs == sorted(set(addrs)), label
            assert len(addrs) == len(set(addrs)), label

    def test_basename_match_with_path_spec(self, dwarf_variants):
        resolver = _resolver(next(iter(dwarf_variants.values())))
        by_base = resolver.resolve("dwarf_sanity.c", _line(MARKERS["helper_body"]))
        by_path = resolver.resolve("/some/other/dir/dwarf_sanity.c", _line(MARKERS["helper_body"]))
        assert by_path == by_base

    def test_unknown_file_or_line(self, dwarf_variants):
        resolver = _resolver(next(iter(dwarf_variants.values())))
        assert resolver.resolve("nonexistent.c", 1) == []
        assert resolver.resolve("dwarf_sanity.c", 99999) == []


class TestNoDebugInfo:
    def test_missing_debug_sections(self, tmp_path):
        if not shutil.which("clang"):
            pytest.skip("clang not available")
        src = tmp_path / "plain.c"
        src.write_text("int f(void) { return 1; }\nint main(void) { return f(); }\n")
        out = tmp_path / "plain"
        r = subprocess.run(["clang", "-O0", "-o", str(out), str(src)], capture_output=True)
        if r.returncode != 0 or not out.exists():
            pytest.skip("clang build failed")
        resolver = DwarfLineResolver(str(out))
        assert not resolver.load()
        assert resolver.resolve("plain.c", 1) == []
