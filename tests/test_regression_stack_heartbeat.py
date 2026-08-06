"""Regression: --stack-heartbeat writes the main-thread stack to a file.

SIGKILL (kill -9) is uncatchable in-process, so the fuzzer provides three
ways to see where Python is executing: SIGTERM/SIGINT dump the stack,
`kill -USR1 <pid>` dumps it on demand (faulthandler), and the heartbeat
thread periodically records the main-thread stack to a small file so a
hard kill still leaves the last known location on disk.
"""

import time
from pathlib import Path

from fuzzer_tool.services.fuzzer import Fuzzer


def test_heartbeat_noop_without_path():
    """No path configured -> no thread started, no file."""
    f = Fuzzer.__new__(Fuzzer)
    f._stack_heartbeat_path = None
    f.exec_count = 0
    f._start_stack_heartbeat(interval=0.05)
    time.sleep(0.15)
    # No exception raised; nothing to assert on disk (no path).


def test_heartbeat_writes_main_thread_stack(tmp_path: Path):
    """The heartbeat file records where the main thread is executing."""
    out = tmp_path / ".fuzz_stack.txt"
    f = Fuzzer.__new__(Fuzzer)
    f._stack_heartbeat_path = out
    f.exec_count = 42
    f._start_stack_heartbeat(interval=0.05)

    # Give the beat thread a few cycles.  The main thread is sleeping here,
    # so the recorded stack should name the sleep() call site.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if out.exists():
            break
        time.sleep(0.1)
    assert out.exists(), "heartbeat file was never written"
    content = out.read_text()
    assert "heartbeat" in content, f"missing heartbeat marker in:\n{content}"
    assert "execs=42" in content, f"missing exec counter in:\n{content}"
