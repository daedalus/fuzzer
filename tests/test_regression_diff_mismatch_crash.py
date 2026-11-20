"""Differential targets must crash on a mismatch in default runs.

afl_shim.c rewrites ``abort()`` in shim-compiled code into a stderr marker
and a return unless ``--abort-is-crash`` (see
``tests/test_regression_abort_is_crash.py``). Differential targets used
``abort()`` as their oracle, so a mismatch -- the only finding such a
target exists to report -- became a shim-health warning and the run went on.
``raise(SIGABRT)`` is not rewritten and is a crash in every mode.

Planted-crash fixtures (test_target, proto_target, prng_token_read) and
library error hooks (grep_read's dfaerror) keep ``abort()``: they are what
the override and its tests exercise.
"""

from __future__ import annotations

import re
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.persistent_subprocess import PersistentLoader
from fuzzer_tool.adapters.process import signal_number

_ROOT = Path(__file__).resolve().parent.parent
SHIM = _ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
ENV = "__AFL_ABORT_CRASH"

DIFF_TARGETS = (
    "bithacks_diff.c",
    "brainflayer_read.c",
    "ec_pubkey_diff.c",
    "gf256_inv_diff.c",
    "hash_simd_diff.c",
    "libbase64_read.c",
    "openssl_b64_diff.c",
)


def _code(path: Path) -> str:
    """Source with comments removed, so prose mentioning abort() is ignored."""
    text = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


@pytest.mark.parametrize("name", DIFF_TARGETS)
def test_regression_mismatch_not_routed_through_abort(name):
    code = _code(_ROOT / "targets" / name)

    assert "abort()" not in code, f"{name}: abort() is swallowed by the shim"
    assert "raise(SIGABRT)" in code
    assert "#include <signal.h>" in code


_SO_SRC = r"""
#include <signal.h>
#include <stddef.h>
int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n) {
    if (n && d[0] == 'M') raise(SIGABRT);
    return 0;
}
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


def test_raise_sigabrt_is_crash_without_flag(so, monkeypatch):
    """Default mode (no --abort-is-crash): still a SIGABRT crash."""
    monkeypatch.delenv(ENV, raising=False)

    assert signal_number(_run(so, b"M")) == signal.SIGABRT


def test_no_mismatch_no_crash(so, monkeypatch):
    # Falsification: an agreeing input is not a crash.
    monkeypatch.delenv(ENV, raising=False)

    assert _run(so, b"q") == 0
