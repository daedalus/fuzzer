"""Regressions for the three violations `tools/tla_extract.py forkserver --check` found.

Each test drives ForkserverRunner against a fake loader script speaking the
fuzz_loader.c line protocol; FAKE_MODE picks its misbehaviour.
"""

import os
import sys
import textwrap
import threading
import time

import pytest

from fuzzer_tool.adapters import forkserver
from fuzzer_tool.adapters.forkserver import ForkserverRunner

_FAKE_RC = 7
_JOIN_GUARD_S = 10.0  # a hung start() fails the test instead of stalling pytest
_SLEEP_FOREVER_S = 60

_FAKE = textwrap.dedent(
    f"""\
    import os, sys, time
    mode = os.environ["FAKE_MODE"]
    with open(os.environ["FAKE_PIDFILE"], "a") as f:
        f.write(f"{{os.getpid()}}\\n")
    inp, out = sys.stdin.buffer, sys.stdout

    def say(line):
        out.write(line + "\\n")
        out.flush()

    line = inp.readline()
    while line.startswith(b"ARG "):
        line = inp.readline()
    if mode == "silent":
        time.sleep({_SLEEP_FOREVER_S})
    if mode == "bogus":
        say("BOGUS")
        time.sleep({_SLEEP_FOREVER_S})
    say("READY exec retune argv")
    while True:
        line = inp.readline()
        if not line or line.startswith(b"QUIT"):
            sys.exit(0)
        if line.startswith(b"RUN "):
            inp.read(int(line.split()[1]))
            say("RC {_FAKE_RC} 0")
        if line.startswith(b"TIMEOUT "):
            if mode == "late":
                time.sleep(float(os.environ["FAKE_LATE_S"]))
            say("TIMEOUT_OK " + line.split()[1].decode())
    """
)


@pytest.fixture
def fake_loader(tmp_path, monkeypatch):
    """Point _ensure_compiled at the fake; yield (env factory, pid list)."""
    script = tmp_path / "fake_loader"
    script.write_text(f"#!{sys.executable}\n{_FAKE}")
    script.chmod(0o755)
    pidfile = tmp_path / "pids"
    monkeypatch.setattr(forkserver, "_ensure_compiled", lambda: str(script))

    def env(mode: str, **extra: str) -> dict[str, str]:
        return {"FAKE_MODE": mode, "FAKE_PIDFILE": str(pidfile), **extra}

    def pids() -> list[int]:
        return [int(p) for p in pidfile.read_text().split()] if pidfile.exists() else []

    yield env, pids

    for pid in pids():
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            continue


def _gone(pid: int) -> bool:
    """True once *pid* has exited and been reaped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _start_guarded(r: ForkserverRunner) -> bool | None:
    """start() in a thread; None if it is still blocked after the guard."""
    box: list[bool] = []
    t = threading.Thread(target=lambda: box.append(r.start()), daemon=True)
    t.start()
    t.join(_JOIN_GUARD_S)
    return box[0] if box else None


def test_regression_start_hang(fake_loader, monkeypatch):
    """NoHang: a loader that never sends READY must not block start() forever."""
    env, pids = fake_loader
    monkeypatch.setattr(forkserver, "_START_DEADLINE", 0.5)
    r = ForkserverRunner("/fake/target", env=env("silent"))
    try:
        assert _start_guarded(r) is False
        assert r._proc is None
        assert all(_gone(p) for p in pids())
    finally:
        r.stop()


def test_regression_start_leak(fake_loader):
    """LiveLoaderReady: a failed handshake must not leave the loader running."""
    env, pids = fake_loader
    r = ForkserverRunner("/fake/target", env=env("bogus"))
    try:
        assert _start_guarded(r) is False
        assert r._proc is None and not r._ready
        assert pids() and all(_gone(p) for p in pids())
    finally:
        r.stop()


def test_start_ok_control(fake_loader):
    """Control: a well-behaved loader starts and runs (the fake is not broken)."""
    env, _pids = fake_loader
    r = ForkserverRunner("/fake/target", env=env("ok"))
    try:
        assert _start_guarded(r) is True
        assert r.run_one(b"x")[0] == _FAKE_RC
    finally:
        r.stop()


def _late_runner(env, late_s: float, monkeypatch) -> ForkserverRunner:
    monkeypatch.setattr(forkserver, "_LOADER_GRACE", 0.1)
    return ForkserverRunner("/fake/target", timeout=0.2, env=env("late", FAKE_LATE_S=str(late_s)))


def test_retune_prompt_control(fake_loader, monkeypatch):
    """Control: a prompt TIMEOUT_OK keeps the same loader."""
    env, pids = fake_loader
    r = _late_runner(env, 0.0, monkeypatch)
    try:
        assert r.start()
        assert r.set_timeout(0.3) is True
        assert len(pids()) == 1
        assert r.run_one(b"x")[0] == _FAKE_RC
    finally:
        r.stop()


def test_regression_retune_desync(fake_loader, monkeypatch):
    """NoDesync: a late TIMEOUT_OK must not be read as the next RC header."""
    env, pids = fake_loader
    late_s = 0.8
    r = _late_runner(env, late_s, monkeypatch)
    try:
        assert r.start()
        assert r.set_timeout(0.3) is False
        time.sleep(late_s)  # the stale reply, if any, is now in the pipe
        assert r.run_one(b"x")[0] == _FAKE_RC
        assert _gone(pids()[0])
    finally:
        r.stop()


class _RecordingBackoff:
    """Backoff double: records failures, never sleeps."""

    def __init__(self):
        self.failures = []

    def started(self):
        return None

    def failed(self, how):
        self.failures.append(how)
        return 0.0


def test_regression_retune_backoff(fake_loader, monkeypatch):
    """The set_timeout() respawn is paced by --respawn-backoff like run_one's."""
    env, _pids = fake_loader
    r = _late_runner(env, 0.8, monkeypatch)
    backoff = _RecordingBackoff()
    try:
        assert r.start()
        r.attach_backoff(backoff)
        assert r.set_timeout(0.3) is False
        assert backoff.failures == [forkserver.Failure.DEATH]
    finally:
        r.stop()


def test_retune_backoff_control(fake_loader, monkeypatch):
    """Control: a prompt reply respawns nothing, so records no failure."""
    env, _pids = fake_loader
    r = _late_runner(env, 0.0, monkeypatch)
    backoff = _RecordingBackoff()
    try:
        assert r.start()
        r.attach_backoff(backoff)
        assert r.set_timeout(0.3) is True
        assert backoff.failures == []
    finally:
        r.stop()
