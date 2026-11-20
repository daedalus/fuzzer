"""abort() in shim-compiled code is a crash the fuzzer sees.

Inside __afl_guarded_call (in-process / persistent harness) the override
raises SIGABRT, which the guard recovers: the fuzzer records the crash and
the campaign continues. A returned abort was invisible -- UBSAN aborts were
never classified. Outside a guard (one-shot, forkserver) the override still
writes a marker and returns unless --abort-is-crash (``__AFL_ABORT_CRASH=1``).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.inprocess import InProcessRunner
from fuzzer_tool.adapters.persistent_subprocess import PersistentLoader
from fuzzer_tool.adapters.process import signal_number
from fuzzer_tool.services import fuzzer as fuzzer_mod

_ROOT = Path(__file__).resolve().parent.parent
SHIM = _ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
ENV = "__AFL_ABORT_CRASH"
_PAST_ABORT_RC = 7

_SO_SRC = rf"""
#include <signal.h>
#include <stddef.h>
#include <stdlib.h>
int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n) {{
    if (n && d[0] == 'B') {{               /* target blocks SIGABRT first */
        sigset_t s;
        sigemptyset(&s);
        sigaddset(&s, SIGABRT);
        sigprocmask(SIG_BLOCK, &s, NULL);
        abort();
    }}
    if (n && d[0] == 'A') abort();
    return n && (d[0] == 'A' || d[0] == 'B') ? {_PAST_ABORT_RC} : 0;
}}
"""


@pytest.fixture
def so(tmp_path) -> str:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / "t.c"
    src.write_text(_SO_SRC)
    out = tmp_path / "t.so"
    cmd = ["clang", "-O1", "-fsanitize-coverage=trace-pc-guard", "-include", str(SHIM)]
    r = subprocess.run(
        [*cmd, "-shared", "-fPIC", "-o", str(out), str(src)], capture_output=True, text=True
    )
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    return str(out)


def _run(so: str, data: bytes) -> int:
    ldr = PersistentLoader(so, timeout=5.0)
    try:
        assert ldr.start()
        rc, _ = ldr.run_one(data)
    finally:
        ldr.stop()
    return rc


def test_regression_abort_is_crash(so, monkeypatch) -> None:
    monkeypatch.setenv(ENV, "1")

    # Persistent mode relays a guarded crash as 128+sig.
    assert signal_number(_run(so, b"A")) == signal.SIGABRT


def test_regression_guarded_abort_is_crash(so, monkeypatch) -> None:
    # Without the flag: a guarded abort() is still a crash, not a return.
    monkeypatch.delenv(ENV, raising=False)

    assert signal_number(_run(so, b"A")) == signal.SIGABRT


def test_inprocess_abort_reported_then_continues(so, monkeypatch) -> None:
    # direct_lite: the abort reaches the fuzzer as -SIGABRT, and the host
    # survives to run the next input.
    monkeypatch.delenv(ENV, raising=False)
    runner = InProcessRunner(so, direct_lite=True, capture_stderr=True)
    try:
        crash_rc, err = runner.run_one(b"A")
        after_rc, _ = runner.run_one(b"q")
    finally:
        runner.stop()

    assert signal_number(crash_rc) == signal.SIGABRT
    assert "[shim] abort() intercepted" in err
    assert after_rc == 0


def test_guarded_abort_with_sigabrt_blocked(so, monkeypatch) -> None:
    # Adversarial: the target blocks SIGABRT before aborting; the guard keeps
    # its signals deliverable, so the abort still cannot run on.
    monkeypatch.delenv(ENV, raising=False)

    assert signal_number(_run(so, b"B")) == signal.SIGABRT


def test_default_abort_returns(so, monkeypatch) -> None:
    # Falsification: outside a guard (one-shot binary) the override still
    # returns without the flag, so only the guard changed.
    monkeypatch.delenv(ENV, raising=False)
    tmp_path = Path(so).parent
    src = tmp_path / "m.c"
    src.write_text(
        "#include <stdlib.h>\nint main(void) { abort(); return " + str(_PAST_ABORT_RC) + "; }\n"
    )
    exe = tmp_path / "m"
    subprocess.run(
        ["clang", "-O1", "-include", str(SHIM), "-o", str(exe), str(src)],
        check=True,
        capture_output=True,
    )

    r = subprocess.run([str(exe)], capture_output=True, timeout=10)

    assert r.returncode == _PAST_ABORT_RC
    assert b"[shim] abort() intercepted" in r.stderr


def test_abort_crash_flag_no_effect_without_abort(so, monkeypatch) -> None:
    # Edge: inputs that never reach abort() are untouched.
    monkeypatch.setenv(ENV, "1")

    assert _run(so, b"q") == 0


@pytest.mark.parametrize(("enabled", "want"), [(True, "1"), (False, None)])
def test_fuzzer_exports_abort_env(monkeypatch, enabled: bool, want: str | None) -> None:
    monkeypatch.setenv(ENV, "x")  # recorded, so the test restores the environment
    monkeypatch.delenv(ENV)
    fuzzer_mod._export_abort_mode(
        fuzzer_mod.AbortMode.CRASH if enabled else fuzzer_mod.AbortMode.RETURN
    )

    assert os.environ.get(ENV) == want
