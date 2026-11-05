"""Run one bench campaign and report its CPU time next to wall time.

Wall time includes time off the CPU: preemption by neighbours, I/O and
scheduler waits (the hugepages A/A drifted -1.0% to -3.7%). User CPU time
of the campaign's process tree drops those. ``os.wait4`` returns the
rusage of exactly this child plus the descendants it reaped, so cells run
in parallel cannot leak into each other the way a ``RUSAGE_CHILDREN``
delta would.

CPU time still moves with clock frequency and cache contention; it is not
an instruction count. Say so in any verdict that rests on it.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass

# rc reported for a campaign killed at its timeout (matches subprocess.run callers).
TIMEOUT_RC = -1

# Metrics where less is better: harnesses flip their delta so a "win" always
# means the test arm did better (cpu_user: 2.0 s -> 1.8 s scores +0.2).
LOWER_IS_BETTER = frozenset({"cpu_user"})


@dataclass(frozen=True)
class TimedRun:
    """Outcome of one child: exit code, merged output, wall and CPU seconds."""

    rc: int
    log: str
    wall: float
    cpu_user: float
    cpu_sys: float


def run_timed(cmd: list[str], timeout: float, **popen_kw) -> TimedRun:
    """Run *cmd* to completion or *timeout*; return output and rusage.

    Output goes to a temp file, not a pipe: ``wait4`` blocks, so nothing
    would drain a pipe and a chatty child would deadlock.
    """
    with tempfile.TemporaryFile() as out:
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, **popen_kw)

        # Kill on timeout from a timer: wait4 itself takes no timeout.
        fired = threading.Event()
        timer = threading.Timer(timeout, lambda: (fired.set(), proc.kill()))
        timer.start()
        try:
            _, status, ru = os.wait4(proc.pid, 0)
        finally:
            timer.cancel()

        # Popen must not wait() on the pid wait4 already reaped.
        rc = TIMEOUT_RC if fired.is_set() else os.waitstatus_to_exitcode(status)
        proc.returncode = rc
        wall = time.monotonic() - t0

        out.seek(0)
        log = out.read().decode(errors="replace")

    return TimedRun(rc=rc, log=log, wall=wall, cpu_user=ru.ru_utime, cpu_sys=ru.ru_stime)
