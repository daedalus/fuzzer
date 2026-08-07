"""Tests for the child-teardown switch.

``_kill_children`` SIGKILLs the process group of every recorded child. That
is the right default — a fuzzer exiting with target processes behind it
exhausts the machine over a long campaign — but it is destructive and not
always wanted, so it is switchable.
"""

import os
import subprocess
import time
from unittest.mock import patch

import pytest

from fuzzer_tool.services import fuzzer as F


@pytest.fixture(autouse=True)
def _restore_switch():
    original = F._kill_children_enabled
    yield
    F.set_kill_children_enabled(original)


@pytest.fixture
def child():
    """A short-lived process in its own group, like the real runners spawn."""
    procs = []

    def _spawn(own_group=True):
        proc = subprocess.Popen(["sleep", "30"], preexec_fn=os.setsid if own_group else None)
        procs.append(proc)
        time.sleep(0.2)
        return proc

    yield _spawn
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def _alive(proc):
    time.sleep(0.3)
    return proc.poll() is None


class TestSwitch:
    def test_enabled_by_default(self):
        """Leaving targets running is the worse failure, so default on."""
        assert F._kill_children_enabled is True

    def test_setter_toggles(self):
        F.set_kill_children_enabled(False)
        assert F._kill_children_enabled is False
        F.set_kill_children_enabled(True)
        assert F._kill_children_enabled is True

    def test_setter_coerces_to_bool(self):
        F.set_kill_children_enabled(0)
        assert F._kill_children_enabled is False

    def test_disabled_leaves_the_child_running(self, child):
        proc = child()
        F.set_kill_children_enabled(False)
        with patch.object(F, "_child_pids", lambda: [proc.pid]):
            F._kill_children()
        assert _alive(proc)

    def test_enabled_kills_the_child(self, child):
        proc = child()
        F.set_kill_children_enabled(True)
        with patch.object(F, "_child_pids", lambda: [proc.pid]):
            F._kill_children()
        assert not _alive(proc)

    def test_shutdown_is_signalled_even_when_disabled(self):
        """Only the SIGKILL is suppressed — the loop must still stop."""
        F._shutdown = False
        F.set_kill_children_enabled(False)
        with patch.object(F, "_child_pids", lambda: []):
            F._kill_children()
        assert F._shutdown is True


class TestSelfProcessGroupGuard:
    """Children call os.setsid(), so a child's pgid is its own. A pid whose
    pgid matches ours was recorded before setsid ran, or has been reused —
    killing that group would SIGKILL the fuzzer and everything beside it."""

    def test_pid_sharing_our_group_is_skipped(self, child):
        proc = child(own_group=False)
        assert os.getpgid(proc.pid) == os.getpgrp()
        F.set_kill_children_enabled(True)
        with patch.object(F, "_child_pids", lambda: [proc.pid]):
            F._kill_children()  # must not take us down with it
        assert _alive(proc)

    def test_separate_group_is_still_killed(self, child):
        """The guard must not disable teardown for genuine children."""
        proc = child(own_group=True)
        assert os.getpgid(proc.pid) != os.getpgrp()
        F.set_kill_children_enabled(True)
        with patch.object(F, "_child_pids", lambda: [proc.pid]):
            F._kill_children()
        assert not _alive(proc)

    def test_dead_pid_does_not_raise(self):
        proc = subprocess.Popen(["true"])
        proc.wait()
        F.set_kill_children_enabled(True)
        with patch.object(F, "_child_pids", lambda: [proc.pid, 999_999_999]):
            F._kill_children()  # ProcessLookupError is suppressed


class TestHandlerInstallation:
    def test_install_reports_success_on_the_main_thread(self):
        assert F.install_cleanup_handlers() is True

    def test_install_does_not_raise_off_the_main_thread(self):
        """signal.signal only works on the main thread; importing this module
        from a worker previously raised at import time."""
        import threading

        result = {}

        def run():
            result["ok"] = F.install_cleanup_handlers()

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        assert result["ok"] is False


class TestCliFlag:
    def test_flag_is_registered(self):
        from fuzzer_tool.cli.commands import main

        assert main is not None

    def test_help_lists_the_flag(self):
        out = subprocess.run(
            ["python", "-m", "fuzzer_tool", "fuzz", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--no-kill-children" in out.stdout
