"""``trace_cmp_call_sites``: direct calls to the trace-cmp callbacks in a binary.

Sizes the cmplog record table per target. The oracle is objdump's
disassembly, independent of the E8 scan under test.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core.elf import trace_cmp_call_sites, trace_pc_call_sites
from tests.conftest import requires_clang

AFL_SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

_SRC = r"""
#include <stdint.h>
#include <stdlib.h>
volatile int sink;
int main(int argc, char **argv) {
    unsigned long v = strtoul(argc > 1 ? argv[1] : "0", NULL, 0);
    volatile uint8_t v1 = (uint8_t)v; volatile uint32_t v4 = (uint32_t)v;
    volatile uint64_t v8 = v;
    sink += v1 == 0x41; sink += v4 == 0x71727374u; sink += v8 == 0x1122334455667788ull;
    sink += v4 > (uint32_t)argc;
    switch (v4) { case 1: sink++; break; case 9: sink--; break; case 77: sink += 3; break; }
    return 0;
}
"""

_CALL = re.compile(r"call\s+[0-9a-f]+ <(__sanitizer_cov_trace_[a-z_0-9]+)>")
_CMP = re.compile(r"__sanitizer_cov_trace_(const_)?cmp[1248]$|__sanitizer_cov_trace_switch$")


def _objdump_calls(exe: Path, pattern: re.Pattern) -> int:
    out = subprocess.run(["objdump", "-d", str(exe)], capture_output=True, text=True).stdout
    return sum(1 for m in _CALL.finditer(out) if pattern.search(m.group(1)))


def _build(tmp: Path, cov: str, name: str) -> Path:
    src = tmp / f"{name}.c"
    src.write_text(_SRC)
    exe = tmp / name
    flags = ["-D__AFL_CMPLOG=1"] if "trace-cmp" in cov else []
    proc = subprocess.run(
        ["clang", "-O2", f"-fsanitize-coverage={cov}", *flags, "-include", str(AFL_SHIM)]
        + ["-o", str(exe), str(src), "-ldl"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"build failed: {proc.stderr[-300:]}")
    return exe


@pytest.fixture(scope="module")
def cmp_exe(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("cmp_sites"), "trace-pc-guard,trace-cmp", "cmp")


@pytest.fixture(scope="module")
def pc_exe(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("pc_sites"), "trace-pc", "pc")


@requires_clang
def test_control_objdump_counts_are_stable(cmp_exe):
    """Rule 46: the oracle agrees with itself and sees compares at all."""
    assert _objdump_calls(cmp_exe, _CMP) == _objdump_calls(cmp_exe, _CMP) > 0


@requires_clang
def test_cmp_sites_match_objdump(cmp_exe):
    assert trace_cmp_call_sites(str(cmp_exe)) == _objdump_calls(cmp_exe, _CMP)


@requires_clang
def test_trace_pc_sites_still_match_objdump(pc_exe):
    """The shared scanner must not change trace_pc_call_sites."""
    pc = re.compile(r"__sanitizer_cov_trace_pc$")
    assert trace_pc_call_sites(str(pc_exe)) == _objdump_calls(pc_exe, pc) > 0


@requires_clang
def test_falsify_defined_but_uncalled_is_zero(pc_exe):
    """The shim defines the callbacks; a trace-pc build never calls them."""
    assert trace_cmp_call_sites(str(pc_exe)) == 0


@requires_clang
def test_no_callbacks_in_symtab_is_none(tmp_path):
    """No shim, no trace-cmp: unknown, not 0."""
    src = tmp_path / "plain.c"
    src.write_text("int main(void) { return 0; }\n")
    exe = tmp_path / "plain"
    subprocess.run(["clang", "-O2", "-o", str(exe), str(src)], check=True)
    assert trace_cmp_call_sites(str(exe)) is None


def test_adversarial_not_elf(tmp_path):
    junk = tmp_path / "junk"
    junk.write_bytes(b"\xe8" * 4096)
    assert trace_cmp_call_sites(str(junk)) is None


def test_adversarial_missing_file(tmp_path):
    assert trace_cmp_call_sites(str(tmp_path / "absent")) is None
