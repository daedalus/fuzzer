"""Cost-based parallel-worker partitioning via Multifit (P3-3, step 5 of 6).

``core/parallel_fractal_partition.py`` (Approach C) partitions workers by
the fractal Voronoi root cell of a seed's content hash -- a partition that
knows nothing about how expensive a seed actually is to run, only where its
hash happens to land. This module partitions by **measured or estimated
cost** instead, so that ``-j N`` workers end up with roughly equal total
load rather than roughly equal seed *counts*: :func:`core.job_scheduling.multifit`
(Coffman, Garey, Johnson 1978) bin-packs seeds onto ``m`` identical workers
to balance the sum of per-worker cost, which is a strictly different
objective than the fractal partition's structural stability.

**What "cost" means here is caller-supplied, deliberately.** The natural
source is ``core.cost_ledger``'s measured, persisted per-seed ``p_j`` --
but that requires execution history, which does not exist for a fresh
corpus at campaign start. Callers with no measurement yet (the common
case: ``_distribute_initial_corpus`` handing out a freshly-discovered
corpus before any seed has run) may fall back to a proxy such as input
size; this module does not pick that fallback itself; that decision is
recorded in each function's docstring, not baked into the algorithm here.

**Not monotone, per Multifit's own docstring.** Shrinking one item's cost
can *increase* the makespan Multifit needs at the same worker count. Costs
that are live EWMAs drift every tick; a partition recomputed on every
drift can flip a seed's worker assignment independent of any real load
change, which is exactly the failure mode
``docs/handover/handover_pending_2026-09-06.md`` §P3-3 warns about for this
port. :func:`maybe_repartition` is the hysteresis this module offers
against that: it only repacks when the corpus has drifted past a
threshold since the last packing, and otherwise returns the previous
assignment unchanged, including for items whose measured cost is now
slightly different.

**Best property of fractal partitioning, given up here, stated plainly.**
Fractal partitioning ties a seed to the same worker forever, independent
of discovery order or of anything else in the campaign, because
``root_cell()`` is a pure function of content. A cost-based partition is a
*global* optimization over the current cost vector: adding or removing one
seed, or one seed's cost drifting, can in principle move other seeds
between workers even though nothing about them changed. Hysteresis damps
how often that happens; it does not eliminate it. Callers that need the
"a seed always lands on the same worker" guarantee unconditionally want
fractal partitioning, not this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fuzzer_tool.core.job_scheduling import multifit

__all__ = ["CostPartition", "compute_partition", "maybe_repartition"]


@dataclass
class CostPartition:
    """A cached seed(or job)-id -> worker-index assignment from Multifit.

    ``costs`` is the cost vector this assignment was computed from --
    kept so :func:`maybe_repartition` can measure drift against it later
    without the caller having to keep its own copy in sync.

    An id absent from ``assignment`` was not part of the packing that
    produced this ``CostPartition`` (e.g. it was added to the corpus
    afterward); callers should treat that as "unassigned", not "worker 0",
    and either fall back to another partitioning scheme for it or trigger
    :func:`maybe_repartition`.
    """

    assignment: dict[object, int] = field(default_factory=dict)
    costs: dict[object, float] = field(default_factory=dict)
    makespan: float = 0.0
    m: int = 0


def compute_partition(costs: dict[object, float], m: int) -> CostPartition:
    """Full Multifit repack from the given cost vector. No hysteresis.

    Args:
        costs: ``{id: cost}``, cost strictly positive (Multifit's own
            constraint on item size -- a zero-cost item is not
            representable as a bin-packing size and should be excluded by
            the caller, or given a nominal floor).
        m: Number of workers. Must be >= 1.

    Returns:
        A :class:`CostPartition` with every id from *costs* assigned to
        exactly one worker in ``range(m)``.

    Raises:
        ValueError: propagated from :func:`core.job_scheduling.multifit`
            for ``m < 1``, empty *costs*, or a non-positive cost.
    """
    items = list(costs.items())
    bins, makespan = multifit(items, m)
    assignment: dict[object, int] = {}
    for worker_id, bin_ids in enumerate(bins):
        for item_id in bin_ids:
            assignment[item_id] = worker_id
    return CostPartition(assignment=assignment, costs=dict(costs), makespan=makespan, m=m)


def _relative_drift(old: float, new: float) -> float:
    """Symmetric relative change between two positive costs.

    ``abs(new - old) / max(old, new)``, not ``/ old`` -- the latter blows
    up asymmetrically for a cost that dropped a lot (e.g. old=1, new=0.01
    reads as 99x drift, while the reverse, old=0.01, new=1, reads as
    9900%). The symmetric form bounds drift to ``[0, 1)`` regardless of
    direction, which is what a single ``drift_threshold`` can be compared
    against sensibly in both directions.
    """
    denom = max(old, new)
    if denom <= 0.0:
        return 0.0
    return abs(new - old) / denom


def maybe_repartition(
    previous: CostPartition,
    costs: dict[object, float],
    m: int,
    *,
    drift_threshold: float = 0.25,
) -> CostPartition:
    """Repack only if *costs* has drifted enough from *previous* to justify it.

    Drift is measured id-by-id against ``previous.costs``: an id present in
    *costs* but not in ``previous.costs`` (a new seed) or absent from
    *previous.costs* but present before (a pruned seed) counts as maximal
    drift for that id, since there is nothing to compare against. An id in
    both counts as drift ``_relative_drift(old, new)``. The overall drift
    score is the **maximum** over all ids, not an average -- one seed
    whose cost genuinely quadrupled should trigger a repack even if
    everything else is stable; an average would let that one signal get
    diluted by a large, quiet corpus.

    Args:
        previous: The last :class:`CostPartition` computed, typically from
            an earlier call to this function or to :func:`compute_partition`.
        costs: Current cost vector, same shape as :func:`compute_partition`
            expects.
        m: Number of workers. Must match what *previous* was computed
            with for the drift comparison to mean what it says; a
            different *m* always triggers a full repack regardless of
            *drift_threshold*, since the previous assignment is not
            comparable to a different worker count.
        drift_threshold: Repack only when the maximum per-id drift meets
            or exceeds this. Default 0.25 (25%) is a starting point, not a
            validated value -- ``tools/cost_dispersion.py``'s per-target
            noise floor (see job_scheduling.multifit's docstring reference
            to the Boltzmann result) is the right instrument for tuning it
            per target, not a one-size-fits-all constant.

    Returns:
        *previous* unchanged (same object, not a copy) if drift is below
        *drift_threshold* and *m* is unchanged; otherwise a fresh
        :class:`CostPartition` from :func:`compute_partition`.
    """
    if not previous.assignment:
        return compute_partition(costs, m)
    if previous.m != m:
        return compute_partition(costs, m)

    max_drift = 0.0
    all_ids = set(previous.costs) | set(costs)
    for item_id in all_ids:
        old = previous.costs.get(item_id)
        new = costs.get(item_id)
        if old is None or new is None:
            max_drift = 1.0
            break
        d = _relative_drift(old, new)
        if d > max_drift:
            max_drift = d

    if max_drift >= drift_threshold:
        return compute_partition(costs, m)
    return previous
