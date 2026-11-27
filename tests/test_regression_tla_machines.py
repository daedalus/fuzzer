"""Regressions for TLC findings on PersistentLoader, GdbJob, PushNotifier and the
forkserver restart path (`tools/tla_extract.py <machine> --check`).

Each test drives the real class against small fake processes.
"""

import errno
import os
import subprocess
import sys
import textwrap
import time

import pytest

from fuzzer_tool.adapters import persistent_subprocess as ps
from fuzzer_tool.core.analyzers.analyzer_trace import GdbJob, TraceReport
from fuzzer_tool.services import push

_REAL_POPEN = subprocess.Popen
_GUARD_S = 10.0  # wait bound for a fake child to reach its checkpoint
_SLEEP_S = 30


def _eagain(*_a, **_k):
    raise OSError(errno.EAGAIN, "Resource temporarily unavailable")


class _Spawned:
    """Popen stand-in recording every real child it starts (for cleanup and asserts)."""

    def __init__(self, script: str):
        self.script = script
        self.procs: list[subprocess.Popen] = []

    def __call__(self, _argv, **kwargs):
        proc = _REAL_POPEN([sys.executable, "-c", self.script], **kwargs)
        self.procs.append(proc)
        return proc

    def reap(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()


def _closed_stdin(sentinel) -> str:
    """Child that closes stdin, signals it, then stays alive: writes to it raise EPIPE."""
    return f"import os, time; os.close(0); open({str(sentinel)!r}, 'w').close(); time.sleep({_SLEEP_S})"


def _await(path) -> None:
    deadline = time.monotonic() + _GUARD_S
    while not os.path.exists(path):
        assert time.monotonic() < deadline, "fake child never reached its checkpoint"
        time.sleep(0.01)


# ── PersistentLoader ────────────────────────────────────────────────


def test_regression_loader_eagain(monkeypatch):
    """A Popen failure in start() is a failed start, not an exception."""
    monkeypatch.setattr(ps.subprocess, "Popen", _eagain)
    loader = ps.PersistentLoader("/fake/target.so")
    try:
        assert loader.start() is False
    finally:
        loader.stop()


def test_regression_loader_init_pipe(monkeypatch, tmp_path):
    """INIT hitting EPIPE: start() returns False and leaves no loader running."""
    sentinel = tmp_path / "closed"
    spawned = _Spawned(_closed_stdin(sentinel))

    def popen(argv, **kwargs):
        proc = spawned(argv, **kwargs)
        _await(sentinel)
        return proc

    monkeypatch.setattr(ps.subprocess, "Popen", popen)
    loader = ps.PersistentLoader("/fake/target.so")
    try:
        assert loader.start() is False
        assert spawned.procs and all(p.poll() is not None for p in spawned.procs)
    finally:
        loader.stop()
        spawned.reap()


_FAKE_LOADER = textwrap.dedent(
    """\
    import os, sys, time
    mode = os.environ["FAKE_MODE"]
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    inp.readline()
    out.write(b"READY\\n"); out.flush()
    if mode == "deaf":
        os.close(0)  # alive, but every write to it now raises EPIPE
        open(os.environ["FAKE_ONCE"], "w").close()
        time.sleep(60)
    n = 0
    while True:
        line = inp.readline()
        if not line or line.startswith(b"QUIT"):
            sys.exit(0)
        inp.read(int(line.split()[1]))
        n += 1
        if mode == "hang":
            time.sleep(60)
        once = os.environ["FAKE_ONCE"]
        if mode == "short" and not os.path.exists(once):
            open(once, "w").close()  # only the first loader sends a short payload
            out.write(b"RC 0 8 - - - -\\nabcd"); out.flush()
            continue
        out.write(b"RC 7 0 - - - -\\n"); out.flush()
    """
)


@pytest.fixture
def fake_persistent(monkeypatch, tmp_path):
    """Run PersistentLoader against _FAKE_LOADER instead of the ctypes loader."""
    monkeypatch.setattr(ps, "_PERSISTENT_LOADER", _FAKE_LOADER)

    def make(mode: str, timeout: float) -> ps.PersistentLoader:
        monkeypatch.setenv("FAKE_MODE", mode)
        monkeypatch.setenv("FAKE_ONCE", str(tmp_path / "short_sent"))
        return ps.PersistentLoader("/fake/target.so", timeout=timeout)

    return make


def test_loader_fake_control(fake_persistent):
    """Control: the fake loader starts and answers (the harness is not broken)."""
    loader = fake_persistent("ok", 2.0)
    try:
        assert loader.start() is True
        assert loader.run_one(b"x")[0] == 7
    finally:
        loader.stop()


def test_regression_loader_respawn(fake_persistent, monkeypatch):
    """A hang whose respawn hits EAGAIN reports the timeout instead of raising."""
    loader = fake_persistent("hang", 0.3)
    try:
        assert loader.start() is True
        monkeypatch.setattr(ps.subprocess, "Popen", _eagain)
        assert loader.run_one(b"x")[0] == -1
    finally:
        loader.stop()


def test_regression_loader_short(fake_persistent):
    """A short bitmap payload must not leave its tail to be read as the next header."""
    loader = fake_persistent("short", 0.5)
    try:
        assert loader.start() is True
        assert loader.run_one(b"x")[1] is None
        assert loader.run_one(b"y")[0] == 7
    finally:
        loader.stop()


def test_regression_loader_deaf(fake_persistent):
    """A write failure on a live loader must not leave it running unready."""
    loader = fake_persistent("deaf", 0.3)
    try:
        assert loader.start() is True
        _await(os.environ["FAKE_ONCE"])
        loader.run_one(b"x")
        proc = loader._proc
        assert proc is None or proc.poll() is not None or loader._ready
    finally:
        loader.stop()


# ── GdbJob ──────────────────────────────────────────────────────────


class _Tracer:
    """CrashTracer stand-in: GdbJob reads .timeout and calls _parse_gdb_output."""

    timeout = 5.0

    def _parse_gdb_output(self, output, report):
        report.raw = output


def _job(tmp_path, argv: list[str]) -> tuple[GdbJob | None, list]:
    temps = [tmp_path / "input.bin", tmp_path / "harness.py"]
    for t in temps:
        t.write_bytes(b"x")
    report = TraceReport(target="t", input_size=1)
    job = GdbJob(_Tracer(), report, str(temps[0]), argv, None, str(temps[1]))
    return job, temps


def test_regression_gdbjob_spawn(tmp_path):
    """A Popen failure in GdbJob() removes its temp files before raising."""
    with pytest.raises(OSError):
        _job(tmp_path, [str(tmp_path / "no-such-gdb")])
    assert not (tmp_path / "input.bin").exists()
    assert not (tmp_path / "harness.py").exists()


def test_regression_gdbjob_repoll(tmp_path):
    """poll() after the report was returned returns it again instead of raising."""
    job, _temps = _job(tmp_path, ["true"])
    first = job.wait()
    assert job.poll() is first


def test_gdbjob_kill_control(tmp_path):
    """Control: kill() then kill() again is already safe, and drops the temps."""
    job, temps = _job(tmp_path, [sys.executable, "-c", f"import time; time.sleep({_SLEEP_S})"])
    job.kill()
    job.kill()
    assert not any(t.exists() for t in temps)


# ── PushNotifier ────────────────────────────────────────────────────


def _target() -> push.PushTarget:
    return push.parse_push_url("https://ntfy.sh/test")


def test_regression_push_epipe(monkeypatch, tmp_path):
    """A sender that died before reading its message is still reaped later."""
    sentinel = tmp_path / "closed"
    spawned = _Spawned(_closed_stdin(sentinel))

    def popen(argv, **kwargs):
        proc = spawned(argv, **kwargs)
        _await(sentinel)
        return proc

    monkeypatch.setattr(push.subprocess, "Popen", popen)
    n = push.PushNotifier([_target()])
    try:
        with pytest.raises(OSError):
            n.notify(push.PushMessage("t", "b"))
        assert [s.proc for s in n._senders] == spawned.procs
    finally:
        spawned.reap()
        n.close(timeout=0)


def test_push_control(monkeypatch):
    """Control: a sender that reads its message is tracked, then drained by close()."""
    spawned = _Spawned("import sys; sys.stdin.buffer.read()")
    monkeypatch.setattr(push.subprocess, "Popen", spawned)
    n = push.PushNotifier([_target()])
    try:
        n.notify(push.PushMessage("t", "b"))
        assert [s.proc for s in n._senders] == spawned.procs
        n.close(timeout=_GUARD_S)
        assert all(p.poll() is not None for p in spawned.procs)
    finally:
        spawned.reap()
