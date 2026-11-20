"""Regression: direct_lite died on the second UBSAN report of a run.

The wrapper set ``halt_on_error=1:abort_on_error=1`` for UBSAN targets. In
direct_lite the first report ran Die() -> abort(); the guard recovered the
SIGABRT, but its siglongjmp left UBSAN's report lock held, so the next report
hit "nested bug in the same thread" and ``_exit(1)`` took the fuzzer with it,
silently (stderr is a capture pipe). Measured with clang 18's
libclang_rt.ubsan_standalone: `fuzzer-tool fuzz` on a cmplog
ubsan_target.so exited 1 during seed calibration, no crash saved.

Recover mode (``halt_on_error=0``, as ASAN already uses) reports and returns;
the classifier reads the report from captured stderr. Same bug class as
tests/test_regression_direct_lite_asan_recover.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.cli import ldpreload_wrapper as wrapper
from tests.conftest import requires_clang

_ROOT = Path(__file__).resolve().parent.parent
SHIM = _ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
NESTED = "nested bug in the same thread"
UB_MARK = "runtime error:"

BUG_SRC = """
#include <limits.h>
#include <stddef.h>
int fuzz_shm_run(const unsigned char *d, size_t n) {
    if (n && d[0] == 'O') { int x = INT_MAX; return x + (int)n; }  /* add overflow */
    if (n && d[0] == 'S') { int s = 40 + (int)n; return 1 << s; }  /* shift exponent */
    return 0;
}
"""

# Two different UB sites, then a clean input: the host must reach all three.
_PROBE = f"""
import sys
from fuzzer_tool.adapters.inprocess import InProcessRunner
r = InProcessRunner(sys.argv[1], function_name="fuzz_shm_run", direct_lite=True,
                    capture_stderr=True)
for d in (b"O1", b"S1", b"q"):
    rc, err = r.run_one(d)
    print(d.decode(), int({UB_MARK!r} in err), flush=True)
r.stop()
"""


def _build(tmp_path: Path, *flags: str) -> Path:
    src = tmp_path / "ub.c"
    src.write_text(BUG_SRC)
    so = tmp_path / "ub.so"
    r = subprocess.run(
        ["clang", "-O1", "-fsanitize=undefined", *flags, "-shared", "-fPIC"]
        + ["-include", str(SHIM), "-o", str(so), str(src)],
        capture_output=True,
    )
    if r.returncode != 0:
        pytest.skip(f"clang cannot build UBSAN .so: {r.stderr[-200:]!r}")
    return so


def _probe(so: Path, ubsan_opts: str) -> subprocess.CompletedProcess:
    runtime = wrapper._resolve_ubsan()
    if runtime is None:
        pytest.skip("clang UBSAN runtime not installed")
    env = dict(os.environ, LD_PRELOAD=runtime, UBSAN_OPTIONS=ubsan_opts)
    return subprocess.run(
        [sys.executable, "-c", _PROBE, str(so)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


# ── runtime: the wrapper's defaults survive repeated reports ────────────


@requires_clang
def test_regression_direct_lite_survives_ubsan(tmp_path):
    proc = _probe(_build(tmp_path), ":".join(wrapper._UBSAN_DEFAULTS))

    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout.split() == ["O1", "1", "S1", "1", "q", "0"]


@requires_clang
def test_halting_ubsan_dies_in_process(tmp_path):
    """Control: the old halting options must reproduce the death, or the
    test above proves nothing."""
    proc = _probe(_build(tmp_path), "halt_on_error=1:abort_on_error=1")

    assert proc.returncode != 0
    assert NESTED in proc.stdout + proc.stderr or proc.stdout.split()[-2:] != ["q", "0"]


# ── wrapper: recover by default, user keys win ──────────────────────────


class _Exec(Exception):
    """Stands in for execvpe so main() returns control to the test."""


def _wrapper_opts(monkeypatch, tmp_path, preset: str) -> list[str]:
    target = tmp_path / "t.so"
    target.write_bytes(b"")
    monkeypatch.setattr(wrapper.sys, "argv", ["fuzzer-tool", "fuzz", str(target)])
    monkeypatch.setattr(wrapper, "_detect_asan", lambda t: False)
    monkeypatch.setattr(wrapper, "_detect_ubsan", lambda t: True)
    monkeypatch.setattr(wrapper, "_resolve_ubsan", lambda: None)
    monkeypatch.setattr(wrapper.os, "execvpe", lambda *a: (_ for _ in ()).throw(_Exec()))
    monkeypatch.setenv("UBSAN_OPTIONS", preset)
    with pytest.raises(_Exec):
        wrapper.main()
    return wrapper.os.environ["UBSAN_OPTIONS"].split(":")


def test_regression_wrapper_ubsan_recovers(monkeypatch, tmp_path):
    assert "halt_on_error=0" in _wrapper_opts(monkeypatch, tmp_path, "")


def test_wrapper_keeps_user_halt_choice(monkeypatch, tmp_path):
    """Falsification: a user-set key wins over the default."""
    opts = _wrapper_opts(monkeypatch, tmp_path, "halt_on_error=1")
    assert "halt_on_error=1" in opts
    assert "halt_on_error=0" not in opts


# ── no-recover builds: *_abort handlers release the lock first ─────────


@requires_clang
def test_no_recover_build_survives(tmp_path):
    """Adversarial: -fno-sanitize-recover sites Die() whatever halt_on_error
    says, but only after the report scope released the lock, so each report
    is a recovered SIGABRT and the host keeps running."""
    so = _build(tmp_path, "-fno-sanitize-recover=undefined")
    proc = _probe(so, ":".join(wrapper._UBSAN_DEFAULTS))

    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout.split() == ["O1", "1", "S1", "1", "q", "0"]
