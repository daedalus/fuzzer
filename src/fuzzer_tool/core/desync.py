"""Pulse-coupled desynchronization of a periodic fleet-wide action.

``services/parallel.py`` runs ``_sync_corpus_in`` on a fixed wall-clock
period, identically in every worker, with nothing coupling them.  Each sync
lists every sibling's corpus recursively — measured at 0.25 s per worker for
8 workers holding 4000 seeds each, warm cache, and it grows with the corpus
— so when the phases coincide the whole fleet stalls together and hammers
the filesystem in the same instant.  Nothing pushes them apart again:
uncoupled oscillators at the same frequency random-walk against each other
and spend as much time near-collision as anywhere else.

This is the Kuramoto problem run backwards.  The coupling wanted here is
*repulsive*: each node should sit as far as it can from its phase
neighbours, which for N nodes on one period means 1/N spacing.  The rule is
DESYNC (Degesys et al.): on firing, a node looks at the neighbour that fired
just before it and the one due just after, and moves a fraction of the way
to the midpoint between them::

      T=0        prev      own      next            T=period
      |-----------|---------|--------|-----------------|
                  <--d_prev--><-d_next->
                            |--> shift = damping * (d_next - d_prev) / 2

Repeated, that holds even spacing against drift.  Total work is unchanged —
this spreads the cost, it does not reduce it; the O(workers x corpus) rescan
is a separate problem.

Two mechanisms, because the coupling alone does not get there from a cold
start, and this is the part worth being blunt about.  The midpoint rule has
no gradient in the interior of a tight cluster — a node already halfway
between two neighbours 0.1 s away does not move — so a cluster can only peel
from its edges, and under an asynchronous exchange the peeled nodes leapfrog
as a group.  Measured: six workers started inside a 1 s window sit at
coherence 0.62 after 600 firings and no better after 2000
(``test_cold_cluster_does_not_converge`` pins it).  Exactly coincident
phases are worse still — a symmetric equilibrium no deterministic local rule
escapes — and workers are forked together, so a cold fleet starts *at* that
equilibrium.

:func:`initial_offset` therefore stakes out 1/N spacing directly at startup,
using the worker index that a physical oscillator ensemble does not have.
:func:`phase_shift` then has the job it is actually good at: holding that
spacing against drift, restarts and stragglers.

Reference
---------
Degesys, Rose, Patel, Nagpal, "DESYNC: Self-Organizing Desynchronization and
TDMA on Wireless Sensor Networks", IPSN 2007.
"""

from __future__ import annotations

from collections.abc import Sequence

# Fraction of the way to the neighbour midpoint travelled per firing.
# Measured, not inherited from the paper: over 12 seeds x n in {4, 8, 16},
# 600 firings from a staggered start with 1.5 s of per-round drift, final
# coherence by damping was 0.148 (0.2), 0.129 (0.3), **0.113 (0.5)**, 0.117
# (0.8), 0.152 (1.0), 0.210 (1.2) at the median, against 0.250 uncoupled.
# The tails matter more than the medians: uncoupled p90 is 0.848 and the
# worst case 0.973 -- a fully coherent fleet -- while 0.5 caps p90 at 0.204
# and the worst case at 0.237. Above 1.0 the correction overshoots and 1.2
# is already worse than 0.3 at p90 (0.594).
DEFAULT_DAMPING = 0.5

# Upper end of the stable range; at 2.0 the correction overshoots exactly
# onto the mirrored position and the fleet oscillates instead of settling.
MAX_DAMPING = 2.0

# Name of the per-worker file carrying the last firing time, written in the
# worker's own corpus directory. Dotted so it stays out of `seeds/`, which
# is the only subtree corpus sync reads.
PHASE_FILE = ".sync_phase"


def initial_offset(worker_id: int, n_workers: int, period: float) -> float:
    """Startup stagger placing *worker_id* at its share of the period.

    Workers are forked together, so without this every one of them fires
    its first sync at the same instant and stays there.
    """
    if n_workers <= 1:
        return 0.0

    return period * (worker_id % n_workers) / n_workers


def phase_shift(
    own_phase: float,
    neighbor_phases: Sequence[float],
    period: float,
    damping: float = DEFAULT_DAMPING,
) -> float:
    """Signed correction moving *own_phase* toward its neighbour midpoint.

    Only the two phase neighbours matter — the last node before this one and
    the first after it, circularly — so a fleet of any size costs one pass.
    Returns 0.0 when there is nothing to couple to, and for the coincident
    case, which is a genuine fixed point rather than an oversight.

    Raises:
        ValueError: on a non-positive period, or damping outside (0, 2).
    """
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period}")

    if not 0.0 < damping < MAX_DAMPING:
        raise ValueError(f"damping must lie in (0, {MAX_DAMPING}), got {damping}")

    if not neighbor_phases:
        return 0.0

    # Circular distances, measured forward to the next firing and backward
    # to the previous one. A neighbour exactly on top reads as 0 in both
    # directions and contributes nothing, which is the fixed point.
    ahead = [(p - own_phase) % period for p in neighbor_phases]
    behind = [(own_phase - p) % period for p in neighbor_phases]

    d_next = min(ahead)
    d_prev = min(behind)

    return damping * (d_next - d_prev) / 2.0


def next_delay(
    own_phase: float,
    neighbor_phases: Sequence[float],
    period: float,
    damping: float = DEFAULT_DAMPING,
) -> float:
    """Seconds until this node should fire again, after the correction.

    One full period plus the phase shift. Bounded below by
    ``(1 - damping/2) * period``, so it is positive for any damping the
    range check admits.
    """
    return period + phase_shift(own_phase, neighbor_phases, period, damping)
