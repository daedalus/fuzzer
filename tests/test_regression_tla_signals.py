"""Regressions for TLC findings on PersistentRunner (fork/waitpid/kill) and
StackSampler (signal handlers): `tools/tla_extract.py <machine> --check`."""

import errno
import os
import signal
import sys
import textwrap

import pytest

from fuzzer_tool.adapters import persistent_signal as psig
from fuzzer_tool.adapters import stack_sampler as ss

_REAL_KILL = os.kill
_REAL_WAITPID = os.waitpid

# Parks at the persistent-mode boundary (SIGSTOP), then idles until killed.
_TARGET = textwrap.dedent(
    f"""\
    #!{sys.executable}
    import os, signal, time
    os.kill(os.getpid(), signal.SIGSTOP)
    while True:
        time.sleep(1)
    """
)


@pytest.fixture
def target(tmp_path):
    path = tmp_path / "persistent_target"
    path.write_text(_TARGET)
    path.chmod(0o755)
    return str(path)


class _KillAudit:
    """os.kill / os.waitpid wrappers that record a signal sent to an already-reaped pid."""

    def __init__(self):
        self.reaped: set[int] = set()
        self.stale: list[tuple[int, int]] = []

    def waitpid(self, pid, flags):
        got, status = _REAL_WAITPID(pid, flags)
        if got and not os.WIFSTOPPED(status):
            self.reaped.add(got)
        return got, status

    def kill(self, pid, sig):
        if pid in self.reaped:
            self.stale.append((pid, sig))
        return _REAL_KILL(pid, sig)


def test_regression_stop_stale_kill(target, monkeypatch):
    """stop() must not signal the pid it has just reaped (it may be reused)."""
    audit = _KillAudit()
    monkeypatch.setattr(psig.os, "waitpid", audit.waitpid)
    monkeypatch.setattr(psig.os, "kill", audit.kill)
    runner = psig.PersistentRunner(target, timeout=2.0)
    try:
        assert runner.start() is True
        runner.stop()
        assert audit.reaped and audit.stale == []
    finally:
        runner.stop()


def test_stop_control(target):
    """Control: start/stop reaps the child (the harness is not broken)."""
    runner = psig.PersistentRunner(target, timeout=2.0)
    assert runner.start() is True
    pid = runner.pid
    runner.stop()
    assert runner.pid is None
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_regression_restart_fork_eagain(target, monkeypatch):
    """A fork failure in restart() is a failed start: no exception, no leaked pipe fds."""
    runner = psig.PersistentRunner(target, timeout=2.0)
    try:
        assert runner.start() is True
        runner.stop()
        before = _open_fds()

        def eagain():
            raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

        monkeypatch.setattr(psig.os, "fork", eagain)
        assert runner.restart() is False
        assert _open_fds() == before
    finally:
        runner.stop()


@pytest.fixture
def clean_sigprof():
    """Leave SIGPROF and ITIMER_PROF as this test found them."""
    old = signal.getsignal(signal.SIGPROF)
    old_timer = signal.getitimer(signal.ITIMER_PROF)
    yield
    signal.setitimer(signal.ITIMER_PROF, 0)
    signal.signal(signal.SIGPROF, old if old is not None else signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_PROF, *old_timer)


def test_regression_sampler_c_handler(monkeypatch, clean_sigprof):
    """A previous handler installed from C reads back as None: stop() must not raise."""
    real = signal.signal
    calls = []

    def from_c(sig, handler):
        calls.append(handler)
        prev = real(sig, handler)
        return None if len(calls) == 1 else prev  # the first install displaced a C handler

    monkeypatch.setattr(ss.signal, "signal", from_c)
    sampler = ss.StackSampler(interval_s=1.0)
    sampler.start()
    sampler.stop()
    assert sampler._running is False


def test_sampler_control(clean_sigprof):
    """Control: with a Python previous handler, stop() restores it."""
    sampler = ss.StackSampler(interval_s=1.0)
    before = signal.getsignal(signal.SIGPROF)
    sampler.start()
    sampler.stop()
    assert signal.getsignal(signal.SIGPROF) is before


def test_regression_sampler_reentrant_tick(clean_sigprof):
    """A tick that lands inside the handler (slow handler, e.g. under coverage) is dropped.

    Python does not block a signal while its handler runs, so a handler slower than
    the interval re-entered itself until RecursionError (CI, 2026-11-28).
    """
    sampler = ss.StackSampler(interval_s=1.0)
    nested = []
    record = sampler._record

    def reentering(frames):
        nested.append(1)
        if len(nested) < 50:
            sampler._on_tick(signal.SIGPROF, None)  # the next tick arrives mid-handler
        record(frames)

    sampler._record = reentering
    sampler._on_tick(signal.SIGPROF, None)
    assert len(nested) == 1


def test_regression_sampler_slow_handler(clean_sigprof):
    """Real SIGPROF at 1 ms with a handler burning ~5 ms of CPU: no RecursionError."""
    import time

    sampler = ss.StackSampler(interval_s=0.001)
    record = sampler._record

    def slow(frames):
        end = time.process_time() + 0.005
        while time.process_time() < end:
            pass
        record(frames)

    sampler._record = slow
    try:
        sampler.start()
        end = time.process_time() + 0.1
        while time.process_time() < end:
            pass
    finally:
        sampler.stop()
    assert sampler.samples > 0
