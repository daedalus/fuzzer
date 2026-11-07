"""``verify`` / ``replay`` executed a ``.so`` target as a program.

A shared object dies with SIGSEGV when exec'd, on every input, so both
commands "confirmed" every crash file: a gzip crash "reproduced" on
``zlib_read_asan.so`` and ``png_read_asan.so`` (2026-11-07). ``.so`` targets
now run through the in-process loader, ASAN runtime preloaded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

CRASH_BYTE = b"C"

# Null write on a 'C' first byte; any other input returns cleanly.
_SO_SRC = """
#include <stddef.h>
int fuzz_shm_run(const unsigned char *d, size_t n) {
    volatile int *p = 0;
    if (n && d[0] == 'C') *p = 1;
    return 0;
}
"""

# Never crashes.
_BENIGN_SRC = """
#include <stddef.h>
int fuzz_shm_run(const unsigned char *d, size_t n) { return n && d[0] == 'Z'; }
"""


def _so(tmp: Path, name: str, src: str) -> Path:
    sub = tmp / name
    sub.mkdir()
    return _build(sub, src, "-fsanitize-coverage=trace-pc-guard", shared=True, asan=True)


def _cli(tmp: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", *argv],
        capture_output=True, text=True, timeout=120, cwd=tmp, env=_clean_env(),
    )  # fmt: skip


@pytest.fixture
def crash_dir(tmp_path) -> Path:
    d = tmp_path / "crashes"
    d.mkdir()
    (d / "crash_1.bin").write_bytes(CRASH_BYTE)
    return d


@requires_clang
def test_regression_verify_so_benign_not_confirmed(tmp_path, crash_dir):
    """Falsification: a .so that never crashes confirms nothing."""
    r = _cli(tmp_path, "verify", str(_so(tmp_path, "benign", _BENIGN_SRC)), str(crash_dir))

    assert "0 confirmed, 1 not reproduced" in r.stdout, r.stdout + r.stderr[-2000:]


@requires_clang
def test_regression_verify_so_crash_confirmed(tmp_path, crash_dir):
    """Adversarial: the fix must not hide a real crash in a .so."""
    r = _cli(tmp_path, "verify", str(_so(tmp_path, "crasher", _SO_SRC)), str(crash_dir))

    assert "1 confirmed, 0 not reproduced" in r.stdout, r.stdout + r.stderr[-2000:]


@requires_clang
def test_regression_replay_so_benign_no_crash(tmp_path, crash_dir):
    """Falsification: replay of a non-crashing input on a .so reports no crash."""
    so = _so(tmp_path, "benign", _BENIGN_SRC)
    r = _cli(tmp_path, "replay", str(so), str(crash_dir / "crash_1.bin"))

    assert "No crash detected" in r.stdout, r.stdout + r.stderr[-2000:]
    assert r.returncode == 1


@requires_clang
def test_regression_replay_so_crash_reproduced(tmp_path, crash_dir):
    """Adversarial: replay still reproduces a real .so crash."""
    so = _so(tmp_path, "crasher", _SO_SRC)
    r = _cli(tmp_path, "replay", str(so), str(crash_dir / "crash_1.bin"))

    assert "Crash reproduced" in r.stdout, r.stdout + r.stderr[-2000:]
    assert r.returncode == 0


@requires_clang
def test_regression_verify_so_names_signal(tmp_path, crash_dir):
    """The loader child reports a SIGSEGV as exit 139; the verdict names signal 11."""
    r = _cli(tmp_path, "verify", str(_so(tmp_path, "crasher", _SO_SRC)), str(crash_dir))

    assert "signal 11" in r.stdout, r.stdout + r.stderr[-2000:]
