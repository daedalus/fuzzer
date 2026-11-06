"""Regression: fuzz_loader reused a grow-only input buffer for .so targets.

After a 64-byte input, an 8-byte one sat in the 64-byte allocation, so an
overread of ``buf[8]`` hit stale bytes instead of ASAN's redzone (AFL++ tip:
never test on the fuzz buffer directly, pass a size-fit copy).
"""

from __future__ import annotations

import os
import subprocess

import pytest

from fuzzer_tool.adapters.forkserver import _ensure_compiled
from tests.support.asan_overread import (
    OVERREAD,
    REPORT,
    asan_env,
    asan_reports,
    build_overread_so,
)

_BIG = 64
_SMALL = 8
_TIMEOUT_S = 30


def _session(so: str, log_dir, *inputs: bytes) -> subprocess.CompletedProcess:
    """Run *inputs* through one fuzz_loader, then QUIT."""
    loader = _ensure_compiled()
    if loader is None:
        pytest.skip("fuzz_loader failed to compile")

    msg = f"INIT {so} fuzz_shm_run /dev/null 5\n".encode()
    for data in inputs:
        msg += f"RUN {len(data)}\n".encode() + data
    msg += b"QUIT\n"

    return subprocess.run(
        [loader],
        input=msg,
        capture_output=True,
        env=asan_env(dict(os.environ), log_dir),
        timeout=_TIMEOUT_S,
    )


def test_regression_shrunk_input_overread_caught(tmp_path) -> None:
    """Falsification: a stale larger buffer must not hide buf[n]."""
    so = build_overread_so(tmp_path)
    _session(so, tmp_path, b"S" * _BIG, bytes([OVERREAD]) * _SMALL)
    assert REPORT in asan_reports(tmp_path)


def test_in_bounds_shrink_clean(tmp_path) -> None:
    """Control: shrinking without overreading reports nothing."""
    so = build_overread_so(tmp_path)
    r = _session(so, tmp_path, b"S" * _BIG, b"S" * _SMALL, b"S" * _BIG)
    assert r.returncode == 0, asan_reports(tmp_path)[-500:]
    assert r.stdout.count(b"RC 0 0\n") == 3


def test_empty_then_overread(tmp_path) -> None:
    """Adversarial: zero-length runs between sizes keep later buffers exact."""
    so = build_overread_so(tmp_path)
    r = _session(so, tmp_path, b"S" * _BIG, b"", bytes([OVERREAD]))
    assert b"RC 0 0\nRC 0 0\n" in r.stdout
    assert REPORT in asan_reports(tmp_path)
