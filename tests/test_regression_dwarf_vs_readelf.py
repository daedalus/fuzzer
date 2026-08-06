"""Regression tests: DWARF line-table resolution vs. readelf ground truth.

`DwarfLineResolver` is a hand-rolled DWARF 4/5 line-program parser. Nothing
in the suite previously checked its output against an independent oracle, so
two classes of defect could pass unnoticed:

  1. DWARF5 support silently returning nothing. `gcc -g` with no extra flags
     emits DWARF5 on every current distro (Ubuntu 22.04+, Debian 12+,
     Fedora 34+), so a v5 regression would silently disable AFLGo directed
     fuzzing for file:line targets on the default build of most systems.
  2. The end_sequence row being attributed to a real source line. That row
     marks the address one past the last instruction of a sequence and
     belongs to no line; emitting it with the stale line number made a
     file:line target resolve to an address past the function body.

readelf --debug-dump=decodedline is the oracle. Tests skip if it is absent.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from fuzzer_tool.core.dwarf import DwarfLineResolver

SOURCE = """\
#include <stdio.h>
int helper(int x) { int y = x * 2; y += 1; return y; }
int main(int argc, char **argv) {
    int a = helper(argc);
    printf("%d\\n", a);
    return 0;
}
"""

pytestmark = pytest.mark.skipif(
    shutil.which("readelf") is None, reason="readelf (binutils) not available"
)


def _readelf_truth(binary: str, src_name: str) -> dict[int, list[int]]:
    """line number -> sorted addresses, per readelf's decoded line table.

    Rows whose line column is '-' (end_sequence) are intentionally not
    matched by the regex: they belong to no source line.
    """
    out = subprocess.run(
        ["readelf", "--debug-dump=decodedline", binary],
        capture_output=True,
        text=True,
    ).stdout
    truth: dict[int, list[int]] = {}
    for m in re.finditer(rf"{re.escape(src_name)}\s+(\d+)\s+(0x[0-9a-f]+)", out):
        truth.setdefault(int(m.group(1)), []).append(int(m.group(2), 16))
    return {k: sorted(v) for k, v in truth.items()}


def _compile(tmp_path, cc: str, debug_flag: str):
    src = tmp_path / "dwarf_test.c"
    src.write_text(SOURCE)
    out = tmp_path / f"bin_{cc}_{debug_flag.strip('-').replace('-', '_')}"
    proc = subprocess.run(
        [cc, debug_flag, "-g", "-O0", "-o", str(out), str(src)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"{cc} {debug_flag} unavailable: {proc.stderr[:200]}")
    return out


@pytest.mark.parametrize(
    "cc,debug_flag",
    [
        ("gcc", "-gdwarf-4"),
        ("gcc", "-gdwarf-5"),
        ("gcc", "-g"),  # distro default — DWARF5 on current toolchains
        ("clang", "-gdwarf-4"),
        ("clang", "-gdwarf-5"),
    ],
)
def test_line_table_matches_readelf(tmp_path, cc, debug_flag):
    if shutil.which(cc) is None:
        pytest.skip(f"{cc} not installed")
    binary = _compile(tmp_path, cc, debug_flag)

    truth = _readelf_truth(str(binary), "dwarf_test.c")
    assert truth, "readelf produced no decoded line rows — oracle broken"

    resolver = DwarfLineResolver(str(binary))
    assert resolver.load(), f"{cc} {debug_flag}: DWARF failed to load"

    for line, expected in truth.items():
        got = sorted(resolver.resolve("dwarf_test.c", line))
        assert got == expected, (
            f"{cc} {debug_flag} line {line}: resolver {[hex(a) for a in got]} "
            f"!= readelf {[hex(a) for a in expected]}"
        )


def test_dwarf5_is_actually_supported(tmp_path):
    """Explicit guard: v5 must resolve real addresses, not silently nothing."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not installed")
    binary = _compile(tmp_path, "gcc", "-gdwarf-5")

    resolver = DwarfLineResolver(str(binary))
    assert resolver.load()
    # helper() body is on line 2 of SOURCE.
    assert resolver.resolve("dwarf_test.c", 2), "DWARF5 resolved no addresses"


def test_end_sequence_not_attributed_to_a_line(tmp_path):
    """The end_sequence address must not appear under any source line."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not installed")
    binary = _compile(tmp_path, "gcc", "-gdwarf-5")

    out = subprocess.run(
        ["readelf", "--debug-dump=decodedline", str(binary)],
        capture_output=True,
        text=True,
    ).stdout
    # Rows readelf prints with '-' in the line column are end_sequence rows.
    end_seq = {int(m, 16) for m in re.findall(r"dwarf_test\.c\s+-\s+(0x[0-9a-f]+)", out)}
    if not end_seq:
        pytest.skip("no end_sequence row in this build")

    resolver = DwarfLineResolver(str(binary))
    assert resolver.load()
    resolved = {a for line in range(1, 40) for a in resolver.resolve("dwarf_test.c", line)}
    assert not (end_seq & resolved), (
        f"end_sequence addresses leaked into line resolution: "
        f"{[hex(a) for a in sorted(end_seq & resolved)]}"
    )
