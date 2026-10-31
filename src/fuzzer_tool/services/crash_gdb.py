"""GDB crash replay beside the fuzz loop.

```
save_crash (novel) ─► report written, GDB section empty
                   └► enqueue(meta, data, rc, base)        (≤ MAX_PENDING)
--trace (default) ─► want_trace(data, rc, base): same job, upgraded to FULL
maintenance tick ──► tick(): poll the live gdb (never waits)
                       exited ─► meta.gdb_replay ─► refresh .txt / .json
                              └► FULL: strace + repro ─► <base>.trace
                       then start the next queued crash
end of run ────────► close(): finish every held job, each within its timeout
```

One gdb run serves both the sidecar and the trace report (they used to run
gdb twice, inline). One job at a time: each is a whole process tree.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from fuzzer_tool.adapters.filesystem import crash_exists, refresh_crash_report
from fuzzer_tool.core.analyzers.analyzer_trace import CrashTracer, GdbJob, TraceReport
from fuzzer_tool.core.crash_metadata import CrashMetadata

log = logging.getLogger(__name__)

# Crashes held, live job included (Hard Rule 54); later ones get no replay.
MAX_PENDING = 16


class TraceMode(Enum):
    """What a finished replay publishes."""

    SIDECAR = "sidecar"  # GDB section of the .txt / .json report
    FULL = "full"  # also <base>.trace: GDB, strace, reproducer (--trace)


@dataclass
class _Job:
    meta: CrashMetadata | None  # None: trace-only, no report to refresh
    data: bytes
    returncode: int
    base: str
    mode: TraceMode
    target: str


class CrashGdbQueue:
    """Background GDB replays for novel crashes.

    Args:
        target: Target path for crashes whose metadata names none.
        crashes_dir: Where save_crash wrote the reports.
        timeout: Seconds per gdb run before it is killed.
    """

    def __init__(self, target: str, crashes_dir: Path, timeout: int) -> None:
        self._target = target
        self._dir = Path(crashes_dir)
        self._timeout = timeout
        # One per target: multi-target campaigns replay what crashed.
        self._tracers: dict[str, CrashTracer] = {}
        self._queue: deque[_Job] = deque()
        self._live: tuple[GdbJob, _Job] | None = None

    @property
    def pending(self) -> int:
        return len(self._queue) + (self._live is not None)

    def enqueue(self, meta: CrashMetadata, data: bytes, returncode: int, base: str) -> None:
        """Hold one saved crash for its report's GDB section."""
        target = meta.target or self._target
        self._add(_Job(meta, data, returncode, base, TraceMode.SIDECAR, target))

    def want_trace(self, data: bytes, returncode: int, base: str) -> None:
        """Also write ``<base>.trace``: upgrades the held job, else adds one."""
        for job in self._held():
            if job.base == base:
                job.mode = TraceMode.FULL
                return
        self._add(_Job(None, data, returncode, base, TraceMode.FULL, self._target))

    def tick(self) -> None:
        """Publish a finished replay and start the next; never blocks on gdb."""
        if self._live is None:
            return
        proc, job = self._live
        report = proc.poll()
        if report is None:
            return
        self._publish(job, report)
        self._start_next()

    def close(self) -> None:
        """Finish every held replay (end of run), each bounded by its timeout."""
        while self._live is not None:
            proc, job = self._live
            self._publish(job, proc.wait())
            self._start_next()

    def _held(self):
        if self._live is not None:
            yield self._live[1]
        yield from self._queue

    def _add(self, job: _Job) -> None:
        if self.pending >= MAX_PENDING:
            return
        self._queue.append(job)
        if self._live is None:
            self._start_next()

    def _start_next(self) -> None:
        self._live = None
        while self._queue:
            job = self._queue.popleft()
            proc = self._start(job)
            if proc is not None:
                self._live = (proc, job)
                return

    def _tracer(self, target: str) -> CrashTracer:
        tracer = self._tracers.get(target)
        if tracer is None:
            tracer = self._tracers[target] = CrashTracer(target, timeout=self._timeout)
        return tracer

    def _start(self, job: _Job) -> GdbJob | None:
        try:
            return self._tracer(job.target).gdb_start(job.data, job.returncode)
        except OSError:
            log.debug("GDB replay could not start", exc_info=True)
            return None

    def _publish(self, job: _Job, report: TraceReport) -> None:
        try:
            self._publish_sidecar(job, report.sidecar_block())
            if job.mode is TraceMode.FULL and crash_exists(self._dir, job.base):
                tracer = self._tracer(job.target)
                tracer.save_report(tracer.finish_trace(job.data, report), str(self._dir), job.base)
        except OSError:
            log.warning("GDB replay publish failed for %s", job.base, exc_info=True)

    def _publish_sidecar(self, job: _Job, block: str) -> None:
        if job.meta is None or not block:
            return
        job.meta.gdb_replay = block
        refresh_crash_report(self._dir, job.base, job.meta)
