"""Regression: clang -O2 folds value-returning switches before SanitizerCoverage.

SimplifyCFG turns ``switch (x) { case 10: return 7; ... }`` into a range check
plus a table load. SanitizerCoverage runs later, so the cases get no edges and
no ``__sanitizer_cov_trace_switch``; cmplog sees only ``cmp(5, x-10)``, whose
operand never appears in the input. ``-fno-jump-tables`` stops the fold.
``cov_flag_for_cc`` / ``sancov_trace_flags`` must carry it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang

BUILD_SCRIPT = Path(__file__).parent.parent / "tools" / "build_targets.sh"

GUARD = "-fsanitize-coverage=trace-pc-guard"
CMP = "-fsanitize-coverage=trace-cmp"
NO_JT = "-fno-jump-tables"

# Value-returning switch: SimplifyCFG's lookup-table candidate.
LUT_CASES = (10, 11, 12, 13, 14)
LUT_SRC = (
    "int lut(int x){switch(x){"
    + "".join(f"case {c}:return {c * 3 % 7 + 1};" for c in LUT_CASES)
    + "default:return 0;}}\n"
)

# Control: each case calls a function, so no table fold is possible.
CALL_SRC = (
    "int f1(int),f2(int),f3(int);"
    "int calls(int x){switch(x){case 3:return f1(x);case 40:return f2(x);"
    "case 77:return f3(x);default:return 0;}}\n"
)


def _const_line(name: str) -> str:
    """The script's own ``NAME=...`` line, so the test tracks its value."""
    m = re.search(rf"^{name}=.*$", BUILD_SCRIPT.read_text(), re.M)
    assert m, f"{name} not defined in build_targets.sh"
    return m.group(0)


def _fn(name: str) -> str:
    text = BUILD_SCRIPT.read_text()
    m = re.search(rf"^{name}\(\) \{{[^\n]*\}}$", text, re.M)  # one-liner
    m = m or re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert m, f"{name} not found in build_targets.sh"
    return m.group(0)


def _run(body: str, call: str, env_prelude: str = "") -> str:
    script = f"{env_prelude}{_const_line('NO_JUMP_TABLES')}\n{body}\n{call}"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _cov_flag(cc: str, prelude: str = "") -> str:
    return _run(_fn("cov_flag_for_cc"), f'cov_flag_for_cc "{cc}"', "WITH_CMPLOG=0\n" + prelude)


def _ir(tmp_path: Path, src: str, flags: list[str]) -> str:
    c = tmp_path / "sw.c"
    c.write_text(src)
    r = subprocess.run(
        ["clang", "-O2", *flags, "-S", "-emit-llvm", "-o", "-", str(c)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def _switch_cases(ir: str) -> list[int]:
    """Case values handed to trace_switch; [] when the call is absent."""
    if "@__sanitizer_cov_trace_switch(" not in ir.split("declare", 1)[0]:
        return []
    m = re.search(r"@__sancov_gen_cov_switch_values[^=]*= .*?\[\d+ x i64\] \[(.*?)\]", ir)
    assert m, "trace_switch call without its case table"
    vals = [int(v.split()[-1]) for v in m.group(1).split(",")]
    return vals[2:]  # [count, width, cases...]


def _guards(ir: str) -> int:
    return ir.count("call void @__sanitizer_cov_trace_pc_guard(")


@requires_clang
class TestOracle:
    """Hard Rule 46: the oracle must report both outcomes on known inputs."""

    def test_control_sees_unfoldable_switch(self, tmp_path):
        assert _switch_cases(_ir(tmp_path, CALL_SRC, [GUARD, CMP])) == [3, 40, 77]

    def test_control_misses_folded_switch(self, tmp_path):
        assert _switch_cases(_ir(tmp_path, LUT_SRC, [GUARD, CMP])) == []


@requires_clang
def test_regression_lookup_switch_keeps_cases(tmp_path):
    """The build's clang flags keep every case visible to trace_switch."""
    flags = _cov_flag("clang").split() + [CMP]
    assert _switch_cases(_ir(tmp_path, LUT_SRC, flags)) == list(LUT_CASES)


@requires_clang
def test_lookup_switch_regains_case_edges(tmp_path):
    """Falsification: the flag must add per-case edges, not just the call."""
    base = _guards(_ir(tmp_path, LUT_SRC, [GUARD]))
    built = _guards(_ir(tmp_path, LUT_SRC, _cov_flag("clang").split()))
    assert built > base  # measured 3 -> 7


@requires_clang
def test_huge_sparse_switch_keeps_cases(tmp_path):
    """Adversarial: 300 sparse 64-bit cases still reach trace_switch."""
    cases = [(i * 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF for i in range(1, 301)]
    body = "".join(f"case {c}LL:return {i};" for i, c in enumerate(cases))
    src = f"int big(long long x){{switch(x){{{body}default:return -1;}}}}\n"
    flags = _cov_flag("clang").split() + [CMP]
    assert sorted(_switch_cases(_ir(tmp_path, src, flags))) == sorted(cases)


@pytest.mark.parametrize("cc", ["clang", "/usr/bin/clang-18", "zig cc"])
def test_clang_family_carries_flag(cc):
    assert NO_JT in _cov_flag(cc).split()


def test_sancov_override_carries_flag():
    """Adversarial: --sancov replaces the coverage mode, not the switch guard."""
    flag = _cov_flag("clang", 'SANCOV_FLAG="-fsanitize-coverage=inline-8bit-counters"\n')
    assert flag.split() == ["-fsanitize-coverage=inline-8bit-counters", NO_JT]


def test_gcc_opt_out_stays_empty():
    """gcc is out of scope: GCC_TRACE_PC=0 must still print nothing."""
    assert _cov_flag("gcc", "GCC_TRACE_PC=0\n") == ""


def test_tracecmp_flags_carry_flag():
    out = _run(_fn("sancov_cmp_modes") + "\n" + _fn("sancov_trace_flags"), "sancov_trace_flags")
    assert NO_JT in out.split()


def test_ffmpeg_library_carries_flag():
    """Regression: FFmpeg's objects take a hardcoded COV_FLAGS, not cov_flag_for_cc."""
    body = _fn("build_vendored_ffmpeg_sancov")
    m = re.search(r'local COV_FLAGS="(-fsanitize-coverage[^"]*)"', body)
    assert m, "sancov COV_FLAGS not found"
    assert "$NO_JUMP_TABLES" in m.group(1).split()
