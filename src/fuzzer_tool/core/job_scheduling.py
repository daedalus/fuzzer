"""Classical job-scheduling primitives (P3-3, step 1 of 6).

**Status:** pure functions, no wiring.  Nothing in the production fuzz loop
imports this module yet.  It exists to give the maintenance-tick ordering
problem a shared vocabulary before ``services/maintenance.py`` (step 4)
absorbs the three ad-hoc gates currently scattered across
``services/fuzzer.py`` (``_run_crash_replays``/``_run_sanitizer_replays``'s
``budget_ms=200`` + ``i % 500``, ``_check_memory_and_prune``'s internal
1000-exec early return, and ``gc.collect``'s ``i % 500``).

See ``docs/handover/handover_pending_2026-09-06.md`` §P3-3 for the full
design note, including the six-commit sequence this module starts, the
non-monotonicity of Multifit (item 5), and why ``last_picked`` does not yet
exist on ``seed_meta`` (item 6).

Six algorithms, all pure functions over plain data -- no ``Fuzzer``
reference, no I/O, no bandit machinery:

* :func:`edf_order` -- Earliest Due Date, the textbook 1||Lmax optimum with
  no precedence constraints.
* :func:`mdd_order` -- Modified Due Date, a precedence-aware heuristic for
  1|prec|Lmax (not exact -- see its docstring).
* :func:`wmdd_order` -- weighted variant of the above; weight scales urgency
  rather than the due date itself (design choice, stated in its docstring).
* :func:`lawler_order` -- Lawler's 1962 algorithm for 1|prec|f_max: exact,
  O(n^2), for *any* job cost function that is non-decreasing in completion
  time.
* :func:`ffd_pack` -- First Fit Decreasing bin packing.
* :func:`multifit` -- the Multifit algorithm (Coffman, Garey, Johnson 1978)
  for P||Cmax on ``m`` identical parallel machines, built on ``ffd_pack``.

These do **not** belong in ``core/schedulers/``: that package's docstring is
"operator-selection schedulers (bandit algorithms)", every member implements
``select_op``/``record``/``bandit_stats``, and anything armed through
``_register_arms`` is required to declare ``supports_priors`` (Hard Rule 40).
None of the six algorithms here is a bandit -- they are deterministic
sequencers with no arm to reward.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

__all__ = [
    "Job",
    "edf_order",
    "mdd_order",
    "wmdd_order",
    "lawler_order",
    "ffd_pack",
    "multifit",
]


class Job(NamedTuple):
    """One unit of schedulable work.

    ``id`` must be hashable and unique within a call; it is also used to
    break ties deterministically, so callers that care about reproducible
    ordering should use a totally-ordered id (int or str), not e.g. a set.

    ``due_date`` and ``weight`` default to values that make a job inert to
    the algorithms that ignore them: ``float("inf")`` never triggers
    lateness, and a weight of ``1.0`` is neutral in :func:`wmdd_order`.
    """

    id: object
    processing_time: float
    due_date: float = float("inf")
    weight: float = 1.0


def _validate_positive_processing_times(jobs: list[Job]) -> None:
    for j in jobs:
        if j.processing_time < 0:
            raise ValueError(f"job {j.id!r} has negative processing_time {j.processing_time!r}")


def edf_order(jobs: list[Job]) -> list[Job]:
    """Earliest Due Date: sort ascending by due date.

    Exact optimum for 1||Lmax (single machine, no precedence, minimize
    maximum lateness) -- Jackson's rule.  Ties broken by ``id`` so the
    result is deterministic across runs given the same input.
    """
    _validate_positive_processing_times(jobs)
    return sorted(jobs, key=lambda j: (j.due_date, j.id))


def _topo_ready(
    scheduled: set[object], precedence: dict[object, set[object]], remaining: list[Job]
) -> list[Job]:
    """Jobs in *remaining* whose precedence predecessors are all in *scheduled*."""
    return [j for j in remaining if precedence.get(j.id, set()) <= scheduled]


def _check_acyclic(jobs: list[Job], precedence: dict[object, set[object]]) -> None:
    """Raise ValueError if *precedence* is not a DAG over *jobs*' ids, or
    references an id not present in *jobs*."""
    ids = {j.id for j in jobs}
    for job_id, preds in precedence.items():
        if job_id not in ids:
            raise ValueError(f"precedence references unknown job {job_id!r}")
        for p in preds:
            if p not in ids:
                raise ValueError(f"job {job_id!r} has unknown predecessor {p!r}")
    # Kahn's algorithm: if we can't fully order, there's a cycle.
    scheduled: set[object] = set()
    remaining = list(jobs)
    while remaining:
        ready = _topo_ready(scheduled, precedence, remaining)
        if not ready:
            stuck = sorted((j.id for j in remaining), key=repr)
            raise ValueError(f"precedence has a cycle among {stuck!r}")
        for j in ready:
            scheduled.add(j.id)
        remaining = [j for j in remaining if j.id not in scheduled]


def mdd_order(
    jobs: list[Job], precedence: dict[object, set[object]] | None = None
) -> list[Job]:
    """Modified Due Date heuristic for 1|prec|Lmax.

    A greedy, precedence-respecting heuristic -- **not** exact (1|prec|Lmax
    with arbitrary precedence is NP-hard; :func:`lawler_order` is exact but
    needs no precedence-*respecting* greedy step because it builds the
    schedule backward from the sink jobs).

    At each step, among jobs whose precedence predecessors have already been
    placed, pick the one minimizing ``max(t + p_j, d_j)`` where ``t`` is the
    current completion time of the partial schedule -- i.e. among available
    jobs, jobs that are already late by their own due date are prioritized
    by that due date (like EDF), but a job that *isn't* late yet is scored
    by when it would finish if run now, not by how urgent its due date
    looks in isolation. Ties broken by ``id``.

    Raises:
        ValueError: if *precedence* is not acyclic over *jobs*, or a
            negative processing time is present.
    """
    _validate_positive_processing_times(jobs)
    precedence = precedence or {}
    _check_acyclic(jobs, precedence)

    scheduled: set[object] = set()
    remaining = list(jobs)
    order: list[Job] = []
    t = 0.0
    while remaining:
        ready = _topo_ready(scheduled, precedence, remaining)
        assert ready, "acyclicity was checked above; this cannot be empty"
        best = min(ready, key=lambda j: (max(t + j.processing_time, j.due_date), j.id))
        order.append(best)
        scheduled.add(best.id)
        t += best.processing_time
        remaining = [j for j in remaining if j.id != best.id]
    return order


def wmdd_order(
    jobs: list[Job], precedence: dict[object, set[object]] | None = None
) -> list[Job]:
    """Weighted variant of :func:`mdd_order`.

    Design choice, stated explicitly because "weighted MDD" is not a single
    textbook algorithm: weight scales the *urgency margin*, not the due date
    itself.  Score is ``max(t + p_j, d_j) / max(w_j, epsilon)`` -- a job with
    a larger weight is treated as more urgent (sorts earlier) for the same
    completion-time/due-date pair, which matches the usual intent of a
    weight in scheduling literature (opportunity cost of delay), rather than
    inflating or deflating the due date, which would conflate "important"
    with "actually due sooner".  With every weight equal to 1.0 this reduces
    exactly to :func:`mdd_order` -- pin that equivalence with a test before
    relying on it.

    Raises:
        ValueError: same conditions as :func:`mdd_order`, plus a
            non-positive weight.
    """
    _validate_positive_processing_times(jobs)
    for j in jobs:
        if j.weight <= 0:
            raise ValueError(f"job {j.id!r} has non-positive weight {j.weight!r}")
    precedence = precedence or {}
    _check_acyclic(jobs, precedence)

    scheduled: set[object] = set()
    remaining = list(jobs)
    order: list[Job] = []
    t = 0.0
    while remaining:
        ready = _topo_ready(scheduled, precedence, remaining)
        assert ready, "acyclicity was checked above; this cannot be empty"
        best = min(
            ready,
            key=lambda j: (max(t + j.processing_time, j.due_date) / j.weight, j.id),
        )
        order.append(best)
        scheduled.add(best.id)
        t += best.processing_time
        remaining = [j for j in remaining if j.id != best.id]
    return order


def lawler_order(
    jobs: list[Job],
    precedence: dict[object, set[object]] | None,
    cost: Callable[[Job, float], float],
) -> tuple[list[Job], float]:
    """Lawler's exact algorithm for 1|prec|f_max.

    Minimizes the maximum, over all jobs, of an arbitrary per-job cost
    function ``cost(job, completion_time)`` that is non-decreasing in
    completion time -- e.g. lateness (``completion_time - due_date``) is the
    textbook instance, but the algorithm places no restriction beyond
    monotonicity.

    Works **backward**: repeatedly choose, among jobs with no unscheduled
    successors (i.e. every job that lists it as a predecessor has already
    been placed), the one that minimizes ``cost(job, total_processing_time
    remaining)`` -- it is placed *last* among the still-unscheduled jobs,
    then removed, and the process repeats.  This is what makes it exact
    where a forward greedy is not: the last job in any optimal schedule for
    this objective is always some sink of the precedence DAG, and Lawler's
    theorem says it's safe to fix the one minimizing this cost function
    first and recurse on the rest.

    O(n^2): each of n steps scans the remaining sinks, O(n) of them in the
    worst case (a precedence-free instance).

    Args:
        jobs: jobs to sequence.
        precedence: ``{job_id: {predecessor_ids}}``. ``None`` or ``{}``
            means no precedence constraints (all jobs are always
            available).
        cost: ``cost(job, completion_time) -> float``. Must be
            non-decreasing in ``completion_time`` for the exactness
            guarantee to hold; this is the caller's responsibility, it is
            not (and cannot cheaply be) checked here.

    Returns:
        ``(order, max_cost)`` -- the schedule in forward (first-to-run)
        order, and the maximum cost achieved (the objective this algorithm
        minimizes).

    Raises:
        ValueError: if *precedence* is not acyclic over *jobs*, or a
            negative processing time is present.
    """
    _validate_positive_processing_times(jobs)
    precedence = precedence or {}
    _check_acyclic(jobs, precedence)

    # successors[j] = set of job ids that must run after j (j is their
    # predecessor). A job is a "sink" of the remaining DAG when no
    # unscheduled job still lists it as a predecessor.
    successors: dict[object, set[object]] = {j.id: set() for j in jobs}
    for job_id, preds in precedence.items():
        for p in preds:
            successors[p].add(job_id)

    remaining = {j.id: j for j in jobs}
    total_time = sum(j.processing_time for j in jobs)
    tail: list[Job] = []  # built back-to-front; reversed at the end
    max_cost = float("-inf")

    while remaining:
        sinks = [
            j
            for j in remaining.values()
            if not (successors[j.id] & remaining.keys())
        ]
        assert sinks, "acyclicity was checked above; this cannot be empty"
        chosen = min(sinks, key=lambda j: (cost(j, total_time), j.id))
        c = cost(chosen, total_time)
        if c > max_cost:
            max_cost = c
        tail.append(chosen)
        del remaining[chosen.id]
        total_time -= chosen.processing_time

    tail.reverse()
    return tail, max_cost


def ffd_pack(
    items: list[tuple[object, float]], capacity: float
) -> list[list[object]]:
    """First Fit Decreasing bin packing.

    Sort items by size descending, then place each into the first bin (in
    creation order) it fits in; open a new bin if none does.

    Args:
        items: ``[(id, size), ...]``. Sizes must be positive and no larger
            than *capacity* (an item that can never fit is a caller error,
            not something this function can silently resolve).
        capacity: per-bin capacity. Must be positive.

    Returns:
        List of bins, each a list of item ids, in the order bins were
        opened. Deterministic given equal-priority ties, broken by id.

    Raises:
        ValueError: non-positive capacity, non-positive size, or a size
            exceeding capacity.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity!r}")
    for item_id, size in items:
        if size <= 0:
            raise ValueError(f"item {item_id!r} has non-positive size {size!r}")
        if size > capacity:
            raise ValueError(
                f"item {item_id!r} has size {size!r} exceeding capacity {capacity!r}"
            )

    ordered = sorted(items, key=lambda kv: (-kv[1], kv[0]))
    bins: list[list[object]] = []
    remaining_capacity: list[float] = []
    for item_id, size in ordered:
        placed = False
        for i, rem in enumerate(remaining_capacity):
            if size <= rem:
                bins[i].append(item_id)
                remaining_capacity[i] -= size
                placed = True
                break
        if not placed:
            bins.append([item_id])
            remaining_capacity.append(capacity - size)
    return bins


def multifit(
    items: list[tuple[object, float]], m: int, iterations: int = 25
) -> tuple[list[list[object]], float]:
    """Multifit (Coffman, Garey, Johnson 1978) for P||Cmax.

    Binary-searches a candidate makespan ``C`` and repacks with
    :func:`ffd_pack` at each step, tightening the bound until *iterations*
    steps have run or the search interval collapses.  The classic result is
    a worst-case makespan ratio of 1.22 * OPT after roughly log2(n) + a
    small constant steps; ``iterations=25`` is comfortably past that for
    any *m* and item count this fuzzer's maintenance tick will ever see, at
    negligible extra cost (each step is one FFD pass).

    **Non-monotone: an explicit warning, not a footnote.** Multifit's
    packing is not monotone in item size -- shrinking one item's size can
    *increase* the number of bins FFD needs at the same capacity (the
    n=3 example: sizes 17 next to two 25s and a capacity of 50 packs
    differently than 16 in the same slot, because FFD's greedy placement
    order shifts around a threshold). Wherever this feeds partitioning that
    is recomputed on live, EWMA-drifting per-seed costs (as the
    ``--partition=cost`` design in step 5 of the handover proposes), that
    non-monotonicity plus drifting inputs can flip the packing independent
    of any real load change. This function does not smooth or debounce
    that; a caller doing so needs its own hysteresis, deliberately kept out
    of this pure-function module.

    Args:
        items: ``[(id, size), ...]``, sizes strictly positive.
        m: number of identical parallel machines. Must be >= 1.
        iterations: binary-search steps. Higher does not change the
            algorithm's approximation ratio, only how tightly the makespan
            bound is converged upon.

    Returns:
        ``(bins, makespan)`` -- ``bins`` has at most ``m`` non-empty lists
        of item ids (fewer if some machines end up idle), and ``makespan``
        is the maximum bin load achieved by the returned packing.

    Raises:
        ValueError: ``m < 1``, an empty *items* list, or any size
            violation :func:`ffd_pack` would also reject.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m!r}")
    if not items:
        raise ValueError("items must be non-empty")
    for item_id, size in items:
        if size <= 0:
            raise ValueError(f"item {item_id!r} has non-positive size {size!r}")

    sizes = [size for _, size in items]
    total = sum(sizes)
    lo = max(max(sizes), total / m)
    hi = max(sizes) * len(items)  # trivially achievable: everything in one bin's worth of slack

    best_bins: list[list[object]] | None = None
    best_makespan = hi

    for _ in range(iterations):
        if hi - lo < 1e-9:
            break
        mid = (lo + hi) / 2.0
        packed = ffd_pack(items, mid)
        if len(packed) <= m:
            hi = mid
            # Track the best *feasible* packing found so far, not just the
            # bound -- the caller wants an actual packing, not only the
            # makespan estimate.
            capacity_by_id = dict(items)
            makespan = max(
                sum(capacity_by_id[i] for i in b) for b in packed
            )
            if best_bins is None or makespan < best_makespan:
                best_bins = packed
                best_makespan = makespan
        else:
            lo = mid

    if best_bins is None:
        # Fallback: pack at hi, which is feasible by construction (each
        # item is <= max(sizes) <= hi, and there are enough "slots" since
        # hi == max(sizes) * len(items) can always hold every item alone).
        best_bins = ffd_pack(items, hi)
        capacity_by_id = dict(items)
        best_makespan = max(sum(capacity_by_id[i] for i in b) for b in best_bins)

    return best_bins, best_makespan
