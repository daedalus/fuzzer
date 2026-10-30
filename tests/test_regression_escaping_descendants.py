"""Regression: target descendants that outlive or escape the per-exec kill.

- a background child holding the stderr pipe kept the drain waiting for
  EOF after the target exited: a clean exit (or crash) was filed as a
  timeout, after the full deadline
- a descendant that called setsid() left the target's process group, so
  the timeout killpg() missed it and it leaked
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from fuzzer_tool.adapters import process

TIMEOUT_S = 5.0
SHORT_TIMEOUT_S = 0.5
# Exit-to-return budget once the target is gone: far below TIMEOUT_S.
FAST_RETURN_S = 2.0
ESCAPEE_SLEEP_S = 30
# A pidfile write can lag the spawn under load.
PIDFILE_WAIT_S = 5.0


def _alive(pid: int) -> bool:
    """True while *pid* runs; a zombie is dead (its reaper may be slow)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


def _dies(pid: int) -> bool:
    """True once *pid* is dead; SIGKILL lands asynchronously."""
    deadline = time.monotonic() + PIDFILE_WAIT_S
    while _alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _read_pid(path: Path) -> int:
    deadline = time.monotonic() + PIDFILE_WAIT_S
    while time.monotonic() < deadline:
        text = path.read_text().strip() if path.exists() else ""
        if text:
            return int(text)
        time.sleep(0.01)
    raise AssertionError(f"{path} never written")


@pytest.fixture
def escapee(tmp_path):
    """Pidfile path; kills whatever pid it names at teardown."""
    pidfile = tmp_path / "escapee.pid"
    yield pidfile
    if not pidfile.exists() or not pidfile.read_text().strip():
        return
    pid = int(pidfile.read_text())
    if _alive(pid):
        os.kill(pid, signal.SIGKILL)


def _script(tmp_path: Path, body: str) -> str:
    path = tmp_path / "target.sh"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


RUNNERS = {
    "stdin": lambda t, timeout: process.run_target_stdin(t, b"x", timeout)[:2],
    "fast": lambda t, timeout: process.run_target_fast(t, b"x", timeout=timeout)[:2],
}


# ── Background child holds stderr ───────────────────────────────────────


@pytest.mark.parametrize("mode", RUNNERS)
def test_regression_bg_child_holding_stderr_not_timeout(tmp_path, escapee, mode):
    target = _script(
        tmp_path,
        f"sleep {ESCAPEE_SLEEP_S} & echo $! > {escapee}\necho REPORT >&2\nexit 3\n",
    )

    t0 = time.monotonic()
    rc, err = RUNNERS[mode](target, TIMEOUT_S)
    elapsed = time.monotonic() - t0

    assert (rc, "REPORT" in err) == (3, True), err
    assert elapsed < FAST_RETURN_S
    assert _dies(_read_pid(escapee)), "background child leaked"


@pytest.mark.parametrize("mode", RUNNERS)
def test_regression_bg_child_does_not_mask_crash(tmp_path, escapee, mode):
    """Adversarial: the target crashes; the held pipe must not hide the signal."""
    target = _script(
        tmp_path,
        f"sleep {ESCAPEE_SLEEP_S} & echo $! > {escapee}\necho ERROR >&2\nkill -SEGV $$\n",
    )

    rc, err = RUNNERS[mode](target, TIMEOUT_S)

    assert rc == process.signal_rc(signal.SIGSEGV), (rc, err)
    assert "ERROR" in err


@pytest.mark.parametrize("mode", RUNNERS)
def test_chatty_exit_keeps_stderr_tail(tmp_path, mode):
    """Falsification: the exit-time drain must not cut the report short."""
    lines = 20000
    target = _script(tmp_path, f"seq 1 {lines} >&2\nexit 0\n")

    rc, err = RUNNERS[mode](target, TIMEOUT_S)

    assert rc == 0
    assert err.rstrip().endswith(str(lines))


# ── setsid() escapes the group kill ─────────────────────────────────────


@pytest.mark.parametrize("mode", RUNNERS)
def test_regression_setsid_escapee_killed_on_timeout(tmp_path, escapee, mode):
    """Escapee closes stderr: only the descendant walk can find it."""
    target = _script(
        tmp_path,
        f"setsid sleep {ESCAPEE_SLEEP_S} 2>/dev/null & echo $! > {escapee}\n"
        f"sleep {ESCAPEE_SLEEP_S}\n",
    )

    rc, _ = RUNNERS[mode](target, SHORT_TIMEOUT_S)

    assert rc == -1
    assert _dies(_read_pid(escapee)), "setsid descendant survived the timeout kill"


@pytest.mark.parametrize("mode", RUNNERS)
def test_regression_orphaned_setsid_pipe_holder_killed(tmp_path, escapee, mode):
    """Adversarial: parent already gone, escapee reparented: the pipe names it."""
    target = _script(
        tmp_path,
        f"setsid sleep {ESCAPEE_SLEEP_S} & echo $! > {escapee}\nexit 0\n",
    )

    t0 = time.monotonic()
    rc, _ = RUNNERS[mode](target, TIMEOUT_S)

    assert rc == 0
    assert time.monotonic() - t0 < FAST_RETURN_S
    assert _dies(_read_pid(escapee))


# ── Pipe-holder identification ──────────────────────────────────────────


def test_pipe_writers_ignores_read_end_holder():
    """Falsification: a process holding only the read end is not a writer."""
    r, w = os.pipe()
    procs = []
    try:
        reader = subprocess.Popen(["sleep", str(ESCAPEE_SLEEP_S)], pass_fds=(r,))
        writer = subprocess.Popen(["sleep", str(ESCAPEE_SLEEP_S)], pass_fds=(w,))
        procs = [reader, writer]

        found = process._pipe_writers(os.fstat(r).st_ino)

        assert writer.pid in found
        assert reader.pid not in found
        assert os.getpid() not in found
    finally:
        for p in procs:
            p.kill()
            p.wait()
        os.close(r)
        os.close(w)


def test_descendants_walks_past_new_session():
    """setsid() keeps the parent link; the walk follows it."""
    outer = subprocess.Popen(
        ["sh", "-c", f"setsid sleep {ESCAPEE_SLEEP_S} & wait"],
    )
    try:
        deadline = time.monotonic() + PIDFILE_WAIT_S
        found: list[int] = []
        while time.monotonic() < deadline and not found:
            found = process._descendants(outer.pid)
            time.sleep(0.01)

        assert found, "no descendant seen"
        assert outer.pid not in found
        assert all(os.getsid(p) != os.getsid(outer.pid) for p in found)
    finally:
        for pid in process._descendants(outer.pid):
            os.kill(pid, signal.SIGKILL)
        outer.kill()
        outer.wait()
