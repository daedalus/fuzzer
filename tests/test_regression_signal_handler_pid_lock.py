"""A signal handler that reads the tracked-pid set must not deadlock on its lock.

``_kill_children`` runs as the SIGTERM/SIGINT handler on the main thread and
calls ``_child_pids()``. If the signal lands while the main thread already
holds ``_TRACKED_PIDS_LOCK`` (inside ``_track``/``_untrack``/``_child_pids``),
a non-reentrant lock makes the handler wait on itself forever.
"""

import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import signal, time
    from fuzzer_tool.adapters import process as p

    p._track(4242)

    def handler(sig, frame):
        # What services.fuzzer._kill_children does first.
        assert 4242 in p._child_pids()
        print("HANDLER_DONE", flush=True)

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    with p._TRACKED_PIDS_LOCK:  # the signal arrives mid critical section
        time.sleep(0.6)
    p._untrack(4242)
    print("MAIN_DONE", flush=True)
    """
)


def test_regression_signal_handler_pid_lock_reentrant():
    # Subprocess + timeout: a regression hangs forever, which must fail, not stall pytest.
    r = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["HANDLER_DONE", "MAIN_DONE"]
