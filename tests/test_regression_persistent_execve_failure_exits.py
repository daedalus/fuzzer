"""Regression test for a failed execve() leaking into the parent's stack.

PersistentRunner.start() (and Runner's ptrace launch path) fork() then
execve() in the child. Previously the execve call in the child wasn't
guarded: on failure (e.g. missing/non-executable target) the raised
exception unwound through the child's *inherited* copy of the caller's
stack instead of exiting the child. Since fork() duplicates the entire
process, that inherited stack includes the pytest test runner itself —
so the child kept running as an orphaned second copy of the whole
process, re-executing every subsequent test concurrently with the real
parent. This produced duplicated test output and, at full-suite scale,
resource contention (e.g. shm ids, temp files) between the two copies
that hung the suite.

This is verified out-of-process: run just the offending test under a
fresh pytest subprocess and confirm it both terminates promptly and
reports the test exactly once. Checking this from inside the same
process wouldn't catch a regression, since a resurrected orphan child
inherits the *same* pytest session object the parent uses for reporting.
"""

import subprocess
import sys

TIMEOUT_SECONDS = 20


def _run_target_test(nodeid: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", nodeid, "-v", "-p", "no:cacheprovider"],
        cwd=__file__.rsplit("/tests/", 1)[0],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )


def test_persistent_execve_failure_does_not_duplicate_session():
    result = _run_target_test(
        "tests/test_persistent_signal.py::TestPersistentRunner::test_start_nonexistent_target"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # An orphaned duplicate child re-runs (and re-reports) the same test.
    assert result.stdout.count("test_start_nonexistent_target") == 1
