"""Regression: loaders spawned a thread per exec to read the reply.

``ForkserverRunner.run_one``/``set_timeout`` and ``PersistentLoader.run_one``
started a ``threading.Thread`` just to run ``readline()`` under a timeout:
~190 us of spawn + join on every exec. ``PipeReader`` polls the raw fd with
a deadline instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import pytest

from fuzzer_tool.adapters import forkserver, persistent_subprocess
from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
from fuzzer_tool.adapters.pipe_reader import PipeReader

# Slack on top of a deadline for scheduler jitter.
_DEADLINE_SLACK = 0.5

_EXE_SRC = r"""
#include <stdio.h>
#include <string.h>
#include <unistd.h>
int main(void) {
    char buf[64];
    size_t n = fread(buf, 1, sizeof(buf), stdin);
    if (n >= 4 && memcmp(buf, "HANG", 4) == 0) for (;;) pause();
    if (n >= 3 && memcmp(buf, "ERR", 3) == 0) fprintf(stderr, "boom\n");
    return 0;
}
"""

_SO_SRC = r"""
#include <stddef.h>
int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n) { return 0; }
"""


class _Pipe:
    """A raw pipe; ``close_write`` simulates the loader exiting."""

    def __init__(self):
        self.r, self.w = os.pipe()
        self._open = {self.r, self.w}

    def close_write(self) -> None:
        os.close(self.w)
        self._open.discard(self.w)

    def close(self) -> None:
        for fd in self._open:
            os.close(fd)
        self._open.clear()


@pytest.fixture
def pipe():
    p = _Pipe()
    yield p
    p.close()


def _deadline(seconds: float) -> float:
    return time.monotonic() + seconds


class TestPipeReader:
    def test_header_and_payload_in_one_chunk(self, pipe):
        """Bytes past the newline stay buffered for the next read."""
        r, w = pipe.r, pipe.w
        os.write(w, b"RC 0 5\nhelloRC 1 0\n")
        rd = PipeReader(r)
        assert rd.readline(_deadline(1)) == b"RC 0 5\n"
        assert rd.read_exact(5, _deadline(1)) == b"hello"
        assert rd.readline(_deadline(1)) == b"RC 1 0\n"

    def test_line_split_across_writes(self, pipe):
        r, w = pipe.r, pipe.w
        rd = PipeReader(r)
        os.write(w, b"REA")
        writer = threading.Timer(0.05, os.write, (w, b"DY\n"))
        writer.start()
        try:
            assert rd.readline(_deadline(2)) == b"READY\n"
        finally:
            writer.join()

    def test_timeout_returns_none(self, pipe):
        """Adversarial: a silent peer must not block past the deadline."""
        r = pipe.r
        rd = PipeReader(r)
        wait = 0.1
        t0 = time.monotonic()
        assert rd.readline(_deadline(wait)) is None
        assert time.monotonic() - t0 < wait + _DEADLINE_SLACK

    def test_partial_line_kept_after_timeout(self, pipe):
        """Falsification: a timed-out read must not drop bytes already read."""
        r, w = pipe.r, pipe.w
        rd = PipeReader(r)
        os.write(w, b"RC 0")
        assert rd.readline(_deadline(0.05)) is None
        os.write(w, b" 0\n")
        assert rd.readline(_deadline(1)) == b"RC 0 0\n"

    def test_eof_returns_empty(self, pipe):
        pipe.close_write()
        assert PipeReader(pipe.r).readline(_deadline(1)) == b""

    def test_short_payload_at_eof(self, pipe):
        os.write(pipe.w, b"abc")
        pipe.close_write()
        assert PipeReader(pipe.r).read_exact(10, _deadline(1)) is None


def _no_threads(monkeypatch, module):
    """Make any Thread construction inside *module* fail the test."""

    def _boom(*_a, **_k):
        raise AssertionError("thread started per exec")

    monkeypatch.setattr(module.threading, "Thread", _boom)


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    d = tmp_path_factory.mktemp("reply_exe")
    src = d / "t.c"
    src.write_text(_EXE_SRC)
    out = d / "t"
    subprocess.run(["clang", "-O0", "-o", str(out), str(src)], check=True)
    return str(out)


@pytest.fixture
def fsrv(exe):
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    r = ForkserverRunner(exe, timeout=0.3)
    if not r.start():
        r.stop()
        pytest.skip("forkserver failed to start")
    yield r
    r.stop()


def test_regression_loader_reply_thread(fsrv, monkeypatch):
    """Forkserver run_one reads its reply without starting a thread."""
    _no_threads(monkeypatch, forkserver)
    assert fsrv.run_one(b"ok") == (0, "")


def test_forkserver_stderr_payload(fsrv, monkeypatch):
    """The RC header's stderr payload is read through the same buffer."""
    _no_threads(monkeypatch, forkserver)
    rc, err = fsrv.run_one(b"ERR")
    assert rc == 0
    assert "boom" in err
    assert fsrv.run_one(b"ok") == (0, "")


def test_forkserver_hang_times_out_and_recovers(fsrv):
    """Adversarial: a hung child still times out and later execs still work."""
    rc, _ = fsrv.run_one(b"HANG")
    assert rc == -1
    assert fsrv.run_one(b"ok") == (0, "")


def test_forkserver_set_timeout_no_thread(fsrv, monkeypatch):
    if "retune" not in fsrv.capabilities:
        pytest.skip("loader lacks retune")
    _no_threads(monkeypatch, forkserver)
    assert fsrv.set_timeout(0.5)
    assert fsrv.timeout == pytest.approx(0.5)


@pytest.fixture
def persistent(tmp_path):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / "t.c"
    src.write_text(_SO_SRC)
    so = tmp_path / "t.so"
    subprocess.run(["clang", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    loader = persistent_subprocess.PersistentLoader(str(so), timeout=5.0)
    if not loader.start():
        loader.stop()
        pytest.skip("persistent loader failed to start")
    yield loader
    loader.stop()


def test_persistent_reply_no_thread(persistent, monkeypatch):
    _no_threads(monkeypatch, persistent_subprocess)
    rc, _ = persistent.run_one(b"x")
    assert rc == 0
    rc, _ = persistent.run_one(b"y")
    assert rc == 0
