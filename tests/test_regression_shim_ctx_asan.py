"""Regression: the caller-context walk tripped ASAN in instrumented targets.

``__afl_get_caller_ctx`` reads ``caller_fp[1]`` after a range check that an
unlinked frame can still pass. In an ASAN build that load was instrumented
and could land in a stack redzone: ASAN aborted the whole in-process fuzzer
(ffmpeg wmv2, 2026-09-25). ``no_sanitize("address")`` on the helper does not
help: it is ``always_inline``, so ASAN instruments it as part of its caller.

Oracle: the ASAN checks in the guard callback with the walk compiled in
(ctx on) versus out (ctx off). The walk must add none.
"""

import re
import shutil
import subprocess

import pytest

from tests.test_shim_ctx_instrumented import SHIM

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")

_CALLBACK = "__sanitizer_cov_trace_pc_guard"


def _asan_checks(tmp_path, ctx: int) -> int:
    """``__asan_report_load*`` calls in the guard callback's IR."""
    src = tmp_path / "m.c"
    src.write_text("int main(void) { return 0; }\n")
    out = tmp_path / f"m{ctx}.ll"
    proc = subprocess.run(
        [
            "clang",
            "-O2",
            "-fno-omit-frame-pointer",
            "-fsanitize=address",
            "-fsanitize-coverage=trace-pc-guard",
            f"-D__AFL_CTX_SENSITIVE={ctx}",
            "-include",
            SHIM,
            "-S",
            "-emit-llvm",
            str(src),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"shim failed to build under clang: {proc.stderr[:300]}")

    ir = out.read_text()
    m = re.search(rf"^define [^\n]*@{_CALLBACK}\(.*?^}}", ir, re.S | re.M)
    assert m, f"{_CALLBACK} missing from IR"
    return len(re.findall(r"__asan_report_load", m.group(0)))


@needs_clang
def test_control_ctx_off_builds_are_identical(tmp_path):
    """Control: the oracle is stable across two identical builds."""
    second = tmp_path / "b"
    second.mkdir()
    assert _asan_checks(tmp_path, 0) == _asan_checks(second, 0)


@needs_clang
def test_regression_ctx_walk_adds_no_asan_checks(tmp_path):
    assert _asan_checks(tmp_path, 1) == _asan_checks(tmp_path, 0)
