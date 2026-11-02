"""The shim carries no coverage instrumentation of its own under clang.

It is -include'd into the target's translation unit, so before the
``#pragma clang attribute push(no_sanitize("coverage"))`` at the top of
``afl_shim.c`` every shim function got guards (510-552 sites) that numbered
ahead of the target's, fed shim-internal edges into the map and inflated the
``blocks:`` denominator. Moving them moved every clang edge id: scheme 4.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from fuzzer_tool.core.elf import detect_edge_id_scheme, parse_sancov_guard_count
from tests.conftest import requires_clang

SHIM = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"

_SRC = """
int main(int argc, char **argv) {
    volatile int x = argc;
    if (x > 3) { __asm__ volatile("nop" ::: "memory"); return 1; }
    __asm__ volatile("nop; nop" ::: "memory");
    return 0;
}
"""


def _build(tmp_path: Path, name: str, *extra: str) -> str:
    src = tmp_path / f"{name}.c"
    src.write_text(_SRC)
    out = tmp_path / name
    proc = subprocess.run(
        ["clang", "-O1", "-fsanitize-coverage=trace-pc-guard", *extra, "-o", str(out), str(src), "-ldl"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    return str(out)


@requires_clang
def test_shim_adds_no_guard_sites(tmp_path):
    """Same source with and without the -include'd shim: identical guard count."""
    bare = parse_sancov_guard_count(_build(tmp_path, "bare"))
    shimmed = parse_sancov_guard_count(_build(tmp_path, "shimmed", "-include", str(SHIM)))
    assert bare and bare < 16
    assert shimmed == bare


@requires_clang
def test_shim_build_reports_scheme_4(tmp_path):
    assert detect_edge_id_scheme(_build(tmp_path, "s4", "-include", str(SHIM))) == 4


@requires_clang
def test_target_functions_keep_their_guards(tmp_path):
    """Falsification: the pragma must not leak past the shim into the target."""
    shimmed = parse_sancov_guard_count(_build(tmp_path, "keeps", "-include", str(SHIM)))
    assert shimmed and shimmed >= 2
