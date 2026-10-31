"""Regression: clang -O2 folds value-returning switches before SanitizerCoverage.

SimplifyCFG turns ``switch (x) { case 10: return 7; ... }`` into a range check
plus a table load. SanitizerCoverage runs later, so the cases get no edges and
no ``__sanitizer_cov_trace_switch``; cmplog sees only ``cmp(5, x-10)``, whose
operand never appears in the input. ``-fno-jump-tables`` stops the fold.
Opt-in (``--switch-cases`` / ``SWITCH_CASES=1``; ledger #63-#64): when on,
``cov_flag_for_cc``, ``sancov_trace_flags`` and FFmpeg's ``COV_FLAGS`` carry it;
when off, every flag string is byte-identical to the pre-feature build.
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


def _gate() -> str:
    """The script's opt-in block: flag constant, default, env/CLI switch."""
    text = BUILD_SCRIPT.read_text()
    parts = [
        re.search(r"^SWITCH_CASES_FLAG=.*$", text, re.M),
        re.search(r"^NO_JUMP_TABLES=.*$", text, re.M),
        re.search(
            r'^SWITCH_CASES=.*?^if \[ "\$SWITCH_CASES" = "1" \]; then.*?^fi$', text, re.M | re.S
        ),
    ]
    assert all(parts), "switch-cases gate not found in build_targets.sh"
    return "\n".join(m.group(0) for m in parts)


def _fn(name: str) -> str:
    text = BUILD_SCRIPT.read_text()
    m = re.search(rf"^{name}\(\) \{{[^\n]*\}}$", text, re.M)  # one-liner
    m = m or re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert m, f"{name} not found in build_targets.sh"
    return m.group(0)


def _run(body: str, call: str, env_prelude: str = "", *, on: bool = True) -> str:
    """Run *call* with the gate evaluated; *on* sets SWITCH_CASES=1 in the env."""
    script = f"{env_prelude}{_gate()}\n{body}\nprintf '%s' \"$({call})\""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin"}
    if on:
        env["SWITCH_CASES"] = "1"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _cov_flag(cc: str, prelude: str = "", *, on: bool = True) -> str:
    body = _fn("cov_flag_for_cc")
    return _run(body, f'cov_flag_for_cc "{cc}"', "WITH_CMPLOG=0\n" + prelude, on=on).strip()


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


def _ffmpeg_cov_flags(*, on: bool) -> str:
    """FFmpeg's sancov COV_FLAGS, expanded under the gate."""
    body = _fn("build_vendored_ffmpeg_sancov")
    m = re.search(r'local COV_FLAGS="(-fsanitize-coverage[^"]*)"', body)
    assert m, "sancov COV_FLAGS not found"
    return _run("", f'echo "{m.group(1)}"', on=on)


def test_ffmpeg_library_carries_flag():
    """Regression: FFmpeg's objects take a hardcoded COV_FLAGS, not cov_flag_for_cc."""
    assert NO_JT in _ffmpeg_cov_flags(on=True).split()


# Pre-feature strings: off must not move a byte, or FFmpeg's configure stamp
# and every vendored object rebuild, and saved edge ids shift.
MASTER_FFMPEG = "-fsanitize-coverage=trace-pc-guard -fsanitize-coverage=trace-cmp"
MASTER_TRACE = "-fsanitize-coverage=trace-cmp,trace-div,trace-gep,trace-pc-guard"


def test_off_by_default_byte_identical():
    assert _cov_flag("clang", on=False) == GUARD
    assert _cov_flag("zig cc", on=False) == f"{GUARD} -fno-sanitize=undefined"
    trace = _run(
        _fn("sancov_cmp_modes") + "\n" + _fn("sancov_trace_flags"), "sancov_trace_flags", on=False
    )
    assert trace == MASTER_TRACE
    assert _ffmpeg_cov_flags(on=False) == MASTER_FFMPEG


@requires_clang
def test_off_by_default_still_folds(tmp_path):
    """Falsification: default flags must leave the fold (feature really gated)."""
    flags = _cov_flag("clang", on=False).split() + [CMP]
    assert _switch_cases(_ir(tmp_path, LUT_SRC, flags)) == []


def test_env_other_than_one_stays_off():
    """Adversarial: only SWITCH_CASES=1 enables; 'yes', '0', '' do not."""
    for val in ("yes", "0", "", "true"):
        out = _run("", "echo $NO_JUMP_TABLES", f"SWITCH_CASES={val!r}\n", on=False)
        assert out == "", val


def test_cli_flag_parsed_and_documented():
    text = BUILD_SCRIPT.read_text()
    assert '[ "$arg" = "--switch-cases" ] && SWITCH_CASES=1' in text
    assert "--switch-cases" in text.split("set -e")[0], "documented in the usage header"
    parse = text.index('[ "$arg" = "--switch-cases" ] && SWITCH_CASES=1')
    gate = text.index('if [ "$SWITCH_CASES" = "1" ]; then')
    assert parse < gate, "the gate must run after the argument loop"
