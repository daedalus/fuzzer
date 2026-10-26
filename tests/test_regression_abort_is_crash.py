"""--abort-is-crash: the shim's abort() override raises SIGABRT instead of returning.

By default the shim turns abort() into a stderr marker and a return (FFmpeg's
~1600 av_assert0 sites would flood the crash set), so an assertion failure in
shim-compiled code was never a crash. The flag (env ``__AFL_ABORT_CRASH=1``)
makes it one.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.persistent_subprocess import PersistentLoader
from fuzzer_tool.adapters.process import signal_number
from fuzzer_tool.services import fuzzer as fuzzer_mod

_ROOT = Path(__file__).resolve().parent.parent
SHIM = _ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
ENV = "__AFL_ABORT_CRASH"
_PAST_ABORT_RC = 7

_SO_SRC = rf"""
#include <stddef.h>
#include <stdlib.h>
int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n) {{
    if (n && d[0] == 'A') abort();
    return n && d[0] == 'A' ? {_PAST_ABORT_RC} : 0;
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


def test_default_abort_returns(so, monkeypatch) -> None:
    # Falsification: without the flag the override still returns.
    monkeypatch.delenv(ENV, raising=False)

    assert _run(so, b"A") == _PAST_ABORT_RC


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
