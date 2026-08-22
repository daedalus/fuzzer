"""Regression tests for bug report 2026-08-21, CRITICAL #3.

``run_target_fast`` is the default spawn-fallback path -- ``run_target`` picks
it whenever the run is neither file_mode nor cmplog -- and it carried three
defects that every other backend had already solved:

  * ``os.waitpid(pid, 0)`` with no deadline. One non-terminating input hung the
    whole campaign, silently and forever.
  * stderr read only *after* the reap. The pipe holds 64 KiB; a target that
    writes more blocks in ``write()`` while the parent blocks in ``waitpid()``.
    Neither ever wakes. A chatty target was as fatal as a looping one.
  * ``except Exception: return -2, str(e), 0``. The pid the caller needs for
    crash attribution was replaced with 0, and a child that had already been
    spawned was never killed or reaped.

The targets here are ``/bin/sh`` scripts rather than compiled binaries so the
suite exercises this on a machine without a toolchain -- the path under test is
process handling, not anything the target's code does.
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.adapters.process import _child_pids, run_target_fast


@pytest.fixture
def script(tmp_path):
    """Write an executable /bin/sh target and hand back its path."""
    made: list[str] = []

    def _make(name: str, body: str) -> str:
        p = tmp_path / f"{name}.sh"
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(0o755)
        made.append(str(p))
        return str(p)

    yield _make
    for path in made:
        if os.path.exists(path):
            os.unlink(path)


def _live_pgid_members(pgid: int) -> list[int]:
    """Live (non-zombie) PIDs in *pgid*, read from /proc.

    Zombies are excluded deliberately. A SIGKILLed orphan is reparented to
    PID 1, and inside a container PID 1 frequently does not reap, so a
    correctly-killed grandchild lingers in state Z. Counting it as a survivor
    would fail a working fix on exactly the machines CI runs on.
    """
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                fields = f.read().split(") ")[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                out.append(int(entry))
        except (OSError, IndexError, ValueError):
            pass
    return out


class TestFastPathTimeout:
    def test_non_terminating_target_times_out(self, script):
        """The headline bug: a looping input must not hang the campaign."""
        looper = script("looper", "while true; do :; done\n")
        t0 = time.monotonic()
        rc, _, _ = run_target_fast(looper, b"x", timeout=1.0)
        elapsed = time.monotonic() - t0
        assert rc == -1, "timeout must use the -1 sentinel the other backends use"
        assert elapsed < 5.0, f"deadline not enforced (took {elapsed:.1f}s)"

    def test_timed_out_child_is_killed_and_reaped(self, script):
        """A bounded run that leaks the process it bounded has fixed nothing."""
        looper = script("looper", "while true; do :; done\n")
        before = set(_child_pids())
        rc, _, pid = run_target_fast(looper, b"x", timeout=0.5)
        assert rc == -1
        assert not set(_child_pids()) - before, "pid left in the tracking set"
        assert not os.path.exists(f"/proc/{pid}"), "child still running after timeout"

    def test_timeout_kill_reaches_grandchildren(self, script):
        """Adversarial: the target spawns a child of its own before hanging.

        Killing only the direct child leaves the grandchild holding the CPU
        for the rest of the campaign, which reads as unexplained slowdown
        rather than as a leak.
        """
        gp = script("grandparent", "sleep 288 &\nsleep 288\n")
        rc, _, pid = run_target_fast(gp, b"x", timeout=1.0)
        assert rc == -1
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not _live_pgid_members(pid):
                break
            time.sleep(0.05)
        assert not _live_pgid_members(pid), "grandchild survived the process-group kill"

    def test_chatty_target_does_not_deadlock(self, script):
        """Adversarial: more stderr than the 64 KiB pipe can hold.

        This is the case a naive timeout fix leaves broken. Reading after the
        reap deadlocks regardless of how good the deadline is, and the symptom
        is identical to a hang -- which is why it stayed hidden.
        """
        chatty = script(
            "chatty",
            "i=0\n"
            "while [ $i -lt 400 ]; do head -c 1024 /dev/zero | tr '\\0' 'A' >&2; i=$((i+1)); done\n"
            "exit 3\n",
        )
        t0 = time.monotonic()
        rc, stderr, _ = run_target_fast(chatty, b"x", timeout=30.0)
        elapsed = time.monotonic() - t0
        assert rc == 3, f"target did not run to completion (rc={rc}, {elapsed:.1f}s)"
        assert elapsed < 25.0, "deadlocked on the stderr pipe"
        assert len(stderr) > 0

    def test_stderr_is_capped_not_unbounded(self, script):
        """The drain must not turn a pipe deadlock into unbounded memory."""
        from fuzzer_tool.adapters.process import _STDERR_CAP

        chatty = script(
            "chatty2",
            "i=0\n"
            "while [ $i -lt 400 ]; do head -c 1024 /dev/zero | tr '\\0' 'B' >&2; i=$((i+1)); done\n",
        )
        _, stderr, _ = run_target_fast(chatty, b"x", timeout=30.0)
        assert len(stderr) <= _STDERR_CAP


class TestFastPathUnchangedBehaviour:
    """Falsification: the bound must not change any terminating run."""

    def test_clean_exit(self):
        rc, _, pid = run_target_fast("/bin/true", b"", timeout=10.0)
        assert rc == 0
        assert pid > 0

    def test_exit_status_preserved(self, script):
        rc, _, _ = run_target_fast(script("rc7", "exit 7\n"), b"", timeout=10.0)
        assert rc == 7

    def test_crash_signal_preserved(self, script):
        rc, _, _ = run_target_fast(script("segv", "kill -SEGV $$\n"), b"", timeout=10.0)
        assert rc == -11, "a crash must stay a crash, not become a timeout"

    def test_stdin_is_the_input(self, script):
        """The temp-file/stdin contract this path exists for."""
        rc, _, _ = run_target_fast(
            script("reader", 'read line; [ "$line" = "MARKER" ]\n'), b"MARKER\n", timeout=10.0
        )
        assert rc == 0

    def test_timeout_none_is_still_unbounded(self):
        """Existing callers that pass no timeout keep working."""
        rc, _, _ = run_target_fast("/bin/true", b"")
        assert rc == 0

    def test_spawn_failure_reports_infra_error(self):
        rc, _, pid = run_target_fast("/nonexistent/binary", b"", timeout=10.0)
        assert rc == -2, "a failure to spawn is infrastructure (-2), not a crash"
        assert pid == 0, "no child was spawned, so 0 is the truthful pid"


def test_runner_passes_the_configured_timeout(monkeypatch):
    """Pin the plumbing.

    ``timeout`` defaults to None, so the fix is only live if the production
    call site forwards ``f.timeout``. Without this, a refactor could drop the
    kwarg and restore the original unbounded behaviour with every unit test
    above still green.
    """
    from fuzzer_tool.services.runner import TargetRunner

    f = MagicMock()
    f._target_shm_covs = {}
    f.target = "/bin/true"
    f.multi_targets = None
    f.shm_cov = None
    f._cmplog = None
    f._inprocess_runner = None
    f._persistent_runner = None
    f._network_runner = None
    f.ptrace_cov = None
    f._forkserver = None
    f.use_coverage = True
    f.map_size = 65536
    f.file_mode = False
    f.timeout = 3.5
    f._perf_counters = None
    f._last_child_pid = None

    seen: dict = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return (0, "", 4242)

    monkeypatch.setattr("fuzzer_tool.services.runner.run_target_fast", _spy)
    TargetRunner(f).run_target(b"data")

    assert seen.get("timeout") == 3.5, "runner did not forward f.timeout to the fast path"
