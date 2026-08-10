"""Regression: loader teardown must not leak the stderr-drain thread.

Every InProcessRunner in subprocess mode starts a loader child, and each
loader spawns a daemon thread reading the child's stderr.  A runner that is
never stopped leaves that thread blocked on the pipe forever (a forked
grandchild may even keep the write end open, so EOF never arrives), which
makes the whole test process permanently multi-threaded — and every later
``os.fork()`` then runs in a multi-threaded process, the classic
deadlock/heap-corruption hazard.  stop() must terminate the thread
deterministically by closing the pipes, and __del__ must call stop() so a
forgotten runner is cleaned up at scope exit.
"""

import subprocess
import threading
from pathlib import Path

from fuzzer_tool.adapters.inprocess import InProcessRunner
from tests.conftest import requires_clang

pytestmark = requires_clang

TARGETS_DIR = Path(__file__).parent.parent / "targets"


def _compile_shared(src: Path, out: Path):
    subprocess.run(
        ["clang", "-shared", "-fPIC", "-o", str(out), str(src)],
        check=True,
        capture_output=True,
    )


def _drain_threads() -> list[threading.Thread]:
    # Python 3.13 auto-names threads "Thread-N (<target function>)".
    return [t for t in threading.enumerate() if t.name.endswith("(_drain_stderr)")]


def _count_after(db: int) -> int:
    """Drain threads alive now minus the baseline, after the runner is gone."""
    return len(_drain_threads()) - db


def test_stop_terminates_the_drain_thread(tmp_path):
    so = tmp_path / "test_nosan.so"
    _compile_shared(TARGETS_DIR / "test_target.c", so)
    baseline = len(_drain_threads())

    runner = InProcessRunner(
        target=str(so),
        function_name="fuzz_shm_run",
        timeout=2.0,
        shm_size=4096,
        direct_lite=False,
        coverage_env_id=None,
        cov=False,
        debug=False,
    )
    assert len(_drain_threads()) > baseline, "loader should have started a drain thread"

    runner.stop()
    # Closing the pipes unblocks the reader even though the loader child may
    # still hold a copy of the write end at this instant.
    assert _count_after(baseline) == 0, f"drain thread survived stop(): {_drain_threads()}"


def test_del_cleans_up_an_unstopped_runner(tmp_path):
    so = tmp_path / "test_nosan.so"
    _compile_shared(TARGETS_DIR / "test_target.c", so)
    baseline = len(_drain_threads())

    runner = InProcessRunner(
        target=str(so),
        function_name="fuzz_shm_run",
        timeout=2.0,
        shm_size=4096,
        direct_lite=False,
        coverage_env_id=None,
        cov=False,
        debug=False,
    )
    assert len(_drain_threads()) > baseline
    del runner  # refcount drop → __del__ → stop()
    assert _count_after(baseline) == 0, f"drain thread survived del: {_drain_threads()}"
