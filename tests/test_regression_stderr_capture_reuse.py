"""Regression: direct modes built a fresh stderr pipe per exec.

``_run_c_direct_lite`` (ASAN/UBSAN targets) and ``_run_c_direct`` did dup,
pipe, dup2, close, dup2, close, two fcntl, read, close every exec. A
``StderrCapture`` opens its pipe once; per exec it is two dup2 and the reads.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters import stderr_capture
from fuzzer_tool.adapters.inprocess import InProcessRunner
from fuzzer_tool.adapters.stderr_capture import StderrCapture

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
EXECS = 5
# Over one 64 KiB read: the drain must loop, and nothing may leak forward.
BIG = 100_000

PROBE_SRC = r"""
#include <stddef.h>
#include <string.h>
#include <unistd.h>
static char big[BIG_SIZE];
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    if (len && buf[0] == 'E') write(2, "report\n", 7);
    if (len && buf[0] == 'B') { memset(big, 'x', sizeof big); write(2, big, sizeof big); }
    return 0;
}
""".replace("BIG_SIZE", str(BIG))


@pytest.fixture(scope="module")
def probe_so(tmp_path_factory):
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    build = tmp_path_factory.mktemp("stderr_reuse")
    src = build / "probe.c"
    src.write_text(PROBE_SRC)
    so = build / "probe_stderr.so"
    proc = subprocess.run(
        ["clang", "-O1", "-fPIC", "-shared", "-o", str(so), str(src), str(SHIM)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"probe target failed to build: {proc.stderr[-400:]}")
    return str(so)


@pytest.fixture(params=["direct_lite", "direct"])
def runner(probe_so, request):
    kw = {request.param: True}
    r = InProcessRunner(
        probe_so, "fuzz_test", shm_size=4096, timeout=5.0, capture_stderr=True, **kw
    )
    r._start()
    yield r
    r.stop()


def _fd2_identity() -> tuple[int, int]:
    st = os.fstat(2)
    return st.st_dev, st.st_ino


def test_regression_stderr_capture_reuse(runner, monkeypatch):
    """One pipe for the runner's life, not one per exec."""
    runner.run_one(b"-")
    pipes = []
    real = stderr_capture.os.pipe
    monkeypatch.setattr(stderr_capture.os, "pipe", lambda: pipes.append(1) or real())
    for _ in range(EXECS):
        runner.run_one(b"-")
    assert pipes == []


def test_capture_exact_and_no_leak(runner):
    """Falsification: what the target wrote is returned once, then nothing."""
    assert runner.run_one(b"E") == (0, "report\n")
    assert runner.run_one(b"-") == (0, "")


def test_big_report_drained(runner):
    """Adversarial: >64 KiB is drained in full and never bleeds into the next exec."""
    rc, err = runner.run_one(b"B")
    assert rc == 0
    assert len(err) == BIG
    assert runner.run_one(b"-") == (0, "")


def test_fd2_restored(runner):
    before = _fd2_identity()
    runner.run_one(b"E")
    assert _fd2_identity() == before


def test_close_releases_fds():
    """stop()/close() returns both pipe ends; a second close is a no-op."""
    cap = StderrCapture()
    cap.begin()
    assert cap.finish() == ""
    fds = cap._fds()
    cap.close()
    cap.close()
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
