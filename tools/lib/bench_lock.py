#!/usr/bin/env python3
"""Single-process safeguard for the benchmark harnesses.

Two benchmark campaigns were once started in the same container by two
different operators, neither aware of the other. On a one-core box they
halved each other: cells went from ~22 s to ~50 s. The cost is not the
wall clock. ``mean_exec`` is a *measured* quantity, and the arm under test
in the Boltzmann A/B reads its energy from it via ``effective_fuzz_count``
-- so a second process on the machine perturbs the input to the thing being
measured. Contention does not slow the experiment down, it corrupts it, and
it does so silently: every cell still completes and still records a number.

``--lock-single-thread`` closes that off at two levels.

*Between processes.* An exclusive ``flock`` on a well-known path. A second
harness that asks for the lock does not queue and does not warn-and-proceed
-- it refuses to start and names the process already holding the machine.
Failing loudly is the point: a run that silently waits would still be
started by an operator who believes they are running alone, and a run that
proceeds anyway is the incident this exists to prevent.

*Within the process.* Child campaigns get ``OMP_NUM_THREADS=1`` and the
BLAS equivalents, and are pinned to one CPU where the platform allows it,
so a single campaign cannot fan out across cores and become its own noise
source.

The lock is advisory and opt-in, so it protects only harnesses that ask for
it. It is worth passing on every run that will be quoted.
"""

from __future__ import annotations

import errno
import fcntl
import os
import sys
from pathlib import Path

LOCK_PATH = Path("/tmp/.daedalus_bench.lock")

# Thread caps applied to child campaigns. A campaign that quietly starts a
# BLAS thread pool per process is the same contention problem one level
# down, and harder to spot because it never shows up as a second command.
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _holder(path: Path) -> str:
    """Describe whoever currently holds the lock, best effort."""
    try:
        pid = int(path.read_text().strip() or 0)
    except (OSError, ValueError):
        return "unknown process"
    if pid <= 0:
        return "unknown process"
    try:
        cmd = (Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")).decode(
            errors="replace"
        ).strip()
    except OSError:
        cmd = "(exited or not readable)"
    return f"pid {pid}: {cmd or '(no cmdline)'}"


class SingleProcessLock:
    """Exclusive, non-blocking advisory lock over the benchmark machine.

    Held for the life of the process. The file descriptor is deliberately
    kept open rather than closed after locking, because closing it drops
    the lock -- so it is stored on the instance and released only on
    ``release()`` or process exit.
    """

    def __init__(self, path: Path = LOCK_PATH):
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                os.close(fd)
                raise
            who = _holder(self.path)
            os.close(fd)
            print(
                f"[!] another benchmark already holds {self.path}\n"
                f"    {who}\n"
                "    Refusing to start: a second campaign on this machine perturbs\n"
                "    mean_exec, which is an input to the arm under test. Wait for it\n"
                "    to finish, or stop it deliberately -- do not run both.",
                file=sys.stderr,
            )
            raise SystemExit(3)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> SingleProcessLock:
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


def pin_single_thread() -> str:
    """Cap this process and its children to one thread on one CPU.

    Returns a short description of what was applied, for the run log --
    a safeguard that leaves no trace in the record is one nobody can
    check was in force when a number was produced.
    """
    for k, v in _THREAD_ENV.items():
        os.environ[k] = v
    note = "threads=1"
    if hasattr(os, "sched_setaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {cpus[0]})
            note += f" cpu={cpus[0]} (of {len(cpus)})"
        except OSError:
            note += " cpu=unpinned (setaffinity refused)"
    else:
        note += " cpu=unpinned (no sched_setaffinity)"
    return note


def engage(enabled: bool) -> SingleProcessLock | None:
    """Apply the safeguard if *enabled*, returning the held lock."""
    if not enabled:
        return None
    lock = SingleProcessLock()
    lock.acquire()
    note = pin_single_thread()
    print(f"[*] --lock-single-thread: exclusive on {LOCK_PATH}, {note}", flush=True)
    return lock


def add_argument(parser) -> None:
    """Register the flag on an argparse parser."""
    parser.add_argument(
        "--lock-single-thread",
        action="store_true",
        help=(
            "take an exclusive lock so no second benchmark can run concurrently, "
            "and pin this process and its campaigns to one thread on one CPU. "
            "Refuses to start if another benchmark holds the machine."
        ),
    )
