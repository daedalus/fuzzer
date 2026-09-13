"""Maintenance-tick job queue (P3-3, step 4 of 6).

Absorbs the three ad-hoc gates identified in
``docs/handover/handover_pending_2026-09-06.md`` §P3-3, each independently
reinventing "run this every so often, cheaply":

* ``_run_crash_replays`` / ``_run_sanitizer_replays`` -- gated on ``i % 500``
  in the caller, each internally budget-capped at ``budget_ms=200``.
* ``_check_memory_and_prune`` -- gated on an internal
  ``self.exec_count - self._last_memory_prune_exec < 1000`` early return.
* ``gc.collect`` -- gated on ``i % 500`` in the caller.

Three mechanisms, three shapes, no shared vocabulary. ``MaintenanceQueue``
gives them one: each is a :class:`MaintenanceJob` with an
``interval_execs`` (how often it is due) and an optional ``active``
predicate (the ``self.replay_n > 0`` / ``self.asan_target or
self.ubsan_target`` / ``self.prune_corpus_max_memory > 0`` guards that used
to live beside each ad-hoc gate). All exec-count bookkeeping moves out of
the individual methods and into the queue, which is the only thing that
tracks "when did this last run".

**Ordering, not just gating.** A tick can find more than one job due at
once. Rather than run them in whatever order they happen to be checked in
(the previous fixed program order in ``services/fuzzer.py``), the queue
orders due jobs with :func:`core.job_scheduling.lawler_order` -- Lawler's
exact 1962 algorithm for ``1|prec|f_max`` -- against the lateness cost
``completion_time - due_date``. This is exact for *any* job set whose
precedence is a DAG, which matters here because a caller-supplied
``predecessors`` set is honored: nothing currently registered needs one
(none of the three absorbed gates depends on another), but
``services/fuzzer.py`` already has real cross-job precedence elsewhere in
the tick (``_cull_queue`` writes ``self._favored`` before the power
schedule reads it; pruning must precede ``_save_state``) that a future
caller of this module may want to express the same way, and the ordering
machinery should not need to change to accommodate that.

A job that has never run is treated as maximally overdue (``due_date =
-inf``) so it always sorts first among the due set -- there is no prior
tick to have been "due since", so waiting for one is wrong.

This module holds no ``Fuzzer`` reference and does no I/O of its own: each
job's ``action`` is the caller's existing method, unmodified. Wiring
(constructing the queue and calling ``tick`` once per stats interval in
``services/fuzzer.py``) is a separate, deliberately small diff, and is
itself gated behind ``--job-scheduler`` (default off, excluded from
``--hail-mary``) -- see ``Fuzzer.__init__``'s comment beside
``self.job_scheduler``. With the flag off, ``services/fuzzer.py`` runs the
original three independent gates byte-for-byte; this module is inert.

**Cadence change, stated explicitly, and opt-in.** The old ``i % 500``
gates for crash/sanitizer replays and ``gc.collect`` ran on the raw
iteration count, independent of the stats-print interval
(``_stats_effective_interval``, 1x-to-10x mean EPS). At high exec rates
that interval can exceed 500 execs, so those two gates could previously
fire several times between stats ticks; wired into
``self._maintenance.tick()`` inside the same
``if self.exec_count - self._last_stats_exec >= effective_interval`` block
as the rest of the tick, under ``--job-scheduler`` they now fire at most
once per stats tick, same as memory pruning already did. This is the
"shared vocabulary" P3-3 asks for, not a hidden side effect, but it is a
real cadence change under fast targets -- which is exactly why it sits
behind a flag instead of replacing the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from fuzzer_tool.core.job_scheduling import Job, lawler_order

__all__ = ["MaintenanceJob", "MaintenanceQueue"]


@dataclass
class MaintenanceJob:
    """One ad-hoc gate, absorbed into shared vocabulary.

    ``id`` must be unique within a :class:`MaintenanceQueue`.

    ``interval_execs`` is how many execs must elapse since this job last ran
    before it is due again -- the number that used to be spelled ``% 500``
    or ``< 1000`` at each call site.

    ``action`` is the existing side-effecting call (e.g.
    ``self._run_crash_replays``), taking no arguments. Any per-call
    parameter the original method had (``budget_ms=200``) stays bound in
    the caller's closure/partial -- this module does not know or care what
    a job does internally, only when it is due and in what order to run it
    relative to its peers.

    ``predecessors`` are ids of other jobs in the *same queue* that must run
    before this one, in a tick where both are due. A predecessor that is
    registered but not due this tick imposes no constraint -- ordering is
    only ever computed over the due set.

    ``active`` gates whether the job is eligible at all (the
    ``self.replay_n > 0`` style guard); it defaults to always-active. A job
    that is due but not active is skipped and does *not* count as run --
    its ``last_run_exec`` is untouched, so it stays maximally overdue until
    activated, rather than silently resetting its clock every tick it
    happens to be checked while inactive.

    ``last_run_exec`` is queue-owned bookkeeping, not caller input: leave it
    at the default. A negative value means "never run".
    """

    id: str
    interval_execs: int
    action: Callable[[], None]
    predecessors: frozenset[str] = frozenset()
    active: Callable[[], bool] = field(default=lambda: True)
    last_run_exec: int = field(default=-1)

    def __post_init__(self) -> None:
        if self.interval_execs <= 0:
            raise ValueError(
                f"job {self.id!r} has non-positive interval_execs {self.interval_execs!r}"
            )


class MaintenanceQueue:
    """Precedence-aware sequencer over interval-gated maintenance jobs.

    Usage::

        queue = MaintenanceQueue()
        queue.register(MaintenanceJob("gc", interval_execs=500, action=gc.collect))
        queue.register(MaintenanceJob(
            "memory_prune", interval_execs=1000, action=self._check_memory_and_prune,
            active=lambda: self.prune_corpus_max_memory > 0,
        ))
        ...
        queue.tick(self.exec_count)  # once per stats interval
    """

    def __init__(self, jobs: list[MaintenanceJob] | None = None) -> None:
        self._jobs: dict[str, MaintenanceJob] = {}
        for j in jobs or []:
            self.register(j)

    def register(self, job: MaintenanceJob) -> None:
        """Add *job* to the queue.

        Raises:
            ValueError: if a job with the same id is already registered, or
                *job* declares a predecessor id that is not (yet)
                registered -- registration order therefore matters:
                predecessors must be registered first.
        """
        if job.id in self._jobs:
            raise ValueError(f"duplicate maintenance job id {job.id!r}")
        unknown = job.predecessors - self._jobs.keys()
        if unknown:
            raise ValueError(f"job {job.id!r} has unregistered predecessors {sorted(unknown)!r}")
        self._jobs[job.id] = job

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._jobs

    def __len__(self) -> int:
        return len(self._jobs)

    def _is_due(self, job: MaintenanceJob, exec_count: int) -> bool:
        if job.last_run_exec < 0:
            return True
        return exec_count - job.last_run_exec >= job.interval_execs

    def due_jobs(self, exec_count: int) -> list[str]:
        """Ids of jobs that are both active and due at *exec_count*.

        Read-only: does not affect ``last_run_exec``. Intended for tests and
        diagnostics; :meth:`tick` computes this internally before running.
        """
        return [
            j.id for j in self._jobs.values() if j.active() and self._is_due(j, exec_count)
        ]

    def tick(self, exec_count: int) -> list[str]:
        """Run every due, active job once, in Lawler-optimal lateness order.

        Returns the ids of jobs actually run, in the order they ran. A job
        due but not active is neither run nor counted -- see
        :class:`MaintenanceJob`'s ``active`` docstring.
        """
        due = [j for j in self._jobs.values() if j.active() and self._is_due(j, exec_count)]
        if not due:
            return []

        due_ids = {j.id for j in due}
        precedence = {j.id: (j.predecessors & due_ids) for j in due}
        due_date_by_id: dict[str, float] = {}
        sched_jobs: list[Job] = []
        for j in due:
            due_date = (
                float(j.last_run_exec + j.interval_execs) if j.last_run_exec >= 0 else float("-inf")
            )
            due_date_by_id[j.id] = due_date
            # processing_time is nominal (1.0, uniform): this module has no
            # measured duration for a caller's action, and none of the
            # ordering guarantees Lawler's algorithm provides depend on the
            # *value* of processing_time being accurate -- only on cost()
            # being non-decreasing in completion time, which lateness is
            # regardless of the time unit each step advances by.
            sched_jobs.append(Job(id=j.id, processing_time=1.0, due_date=due_date))

        def lateness(job: Job, completion_time: float) -> float:
            return completion_time - due_date_by_id[job.id]

        order, _max_lateness = lawler_order(sched_jobs, precedence, lateness)

        ran: list[str] = []
        for sj in order:
            job = self._jobs[sj.id]
            job.action()
            job.last_run_exec = exec_count
            ran.append(sj.id)
        return ran
