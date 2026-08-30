"""The benchmark single-process safeguard.

Two campaigns once ran concurrently in one container on a single core and
halved each other's throughput. The cost was not the wall clock: the
Boltzmann arm under test reads its energy from ``effective_fuzz_count``,
which is ``total_time / mean_exec`` -- both measured quantities. A second
process on the machine therefore perturbs the *input* to the thing being
measured, and does it silently, since every cell still completes and still
records a plausible number.

These cover the three properties that make ``--lock-single-thread`` a
safeguard rather than a decoration: it excludes a second holder, it refuses
loudly instead of queueing, and it survives a holder that dies without
cleaning up.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

bench_lock = pytest.importorskip("bench_lock")


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "bench.lock"


def test_acquire_then_release_allows_reacquire(lock_path):
    first = bench_lock.SingleProcessLock(lock_path)
    first.acquire()
    first.release()
    second = bench_lock.SingleProcessLock(lock_path)
    second.acquire()
    second.release()


def test_second_holder_in_same_process_is_refused(lock_path):
    held = bench_lock.SingleProcessLock(lock_path)
    held.acquire()
    try:
        with pytest.raises(SystemExit) as exc:
            bench_lock.SingleProcessLock(lock_path).acquire()
        # Refuses rather than queueing. A lock that blocked would still let
        # an operator start a run believing they had the machine alone.
        assert exc.value.code == 3
    finally:
        held.release()


def test_lock_excludes_a_separate_process(lock_path):
    """The case that matters: two independently launched harnesses."""
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import sys, time
                sys.path.insert(0, {str(TOOLS)!r})
                import bench_lock
                from pathlib import Path
                lock = bench_lock.SingleProcessLock(Path({str(lock_path)!r}))
                lock.acquire()
                print("HELD", flush=True)
                time.sleep(30)
            """),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert "HELD" in holder.stdout.readline()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(f"""
                    import sys
                    sys.path.insert(0, {str(TOOLS)!r})
                    import bench_lock
                    from pathlib import Path
                    bench_lock.SingleProcessLock(Path({str(lock_path)!r})).acquire()
                """),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 3
        # The message has to name the holder, or an operator cannot tell a
        # deliberate concurrent run from a stale file.
        assert "already holds" in result.stderr
    finally:
        holder.kill()
        holder.wait()


def test_lock_survives_a_holder_that_dies_uncleanly(lock_path):
    """A killed holder must not wedge the machine for everyone after it.

    This is why the lock is ``flock`` and not a pid file: the kernel drops
    the lock when the holder dies, so the stale pid left in the file is
    inert. Two VM restarts during the Boltzmann run exercised exactly this.
    """
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import sys, time
                sys.path.insert(0, {str(TOOLS)!r})
                import bench_lock
                from pathlib import Path
                bench_lock.SingleProcessLock(Path({str(lock_path)!r})).acquire()
                print("HELD", flush=True)
                time.sleep(30)
            """),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert "HELD" in holder.stdout.readline()
    holder.kill()
    holder.wait()

    # Stale pid still on disk, but the lock itself is free.
    assert lock_path.read_text().strip()
    recovered = bench_lock.SingleProcessLock(lock_path)
    recovered.acquire()
    recovered.release()


def test_pin_single_thread_caps_thread_env(monkeypatch):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    note = bench_lock.pin_single_thread()
    import os

    # A campaign that quietly starts a BLAS pool per process is the same
    # contention problem one level down, and harder to spot.
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert "threads=1" in note


def test_engage_disabled_is_a_noop():
    assert bench_lock.engage(False) is None
