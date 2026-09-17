"""Marginal-cost estimator: MC_i = Δcost / Δoutput between successive windows.

Why this exists
----------------
``docs/handover/handover_decision_game_theory_survey_2026-09-13.md`` §1
found that every cost-aware mechanism in the tree reasons about **total**
or **average** cost, never the first difference: ``op_replicator.py``'s
fitness is a per-window mean, ``elo.py``'s K-factor/rating decay are
exponential smoothers, ``parallel_cost_partition.py`` balances total load
via Multifit, and ``cost_ledger.py`` exposes point measurements and EWMAs.
None of these can see whether a producer's cost-per-unit-of-output is
rising or falling -- only its level.

The Marginal Cost page's central identity is MC = ΔTotalCost/ΔQuantity,
and its central result is that a rational allocator keeps investing in an
activity only while MC is below the marginal value of what it buys. This
module supplies the missing MC_i signal so a caller can implement that
rule; it does not implement the rule itself, because *what* counts as
"cost" and "output" is caller-specific (see :class:`MarginalCostTracker`).

Deliberately generic in the (cost, output) unit
------------------------------------------------
For operator scheduling (``core/schedulers/op_replicator.py``) the natural
pairing is ``(execs, edges)``. For worker partitioning
(``core/parallel_cost_partition.py``) it might instead be ``(wall_clock,
edges)`` -- execs and wall-clock diverge under ``--hail-mary`` and
job-scheduler-gated maintenance passes, per prior handovers, so which one
is "the" cost is a decision the handover's §4 open questions explicitly
leave to the caller, not something this module guesses at.

Windowed, not per-exec
-----------------------
:meth:`MarginalCostTracker.record_snapshot` is meant to be called once per
fixed window (wherever a caller's existing windowed-fitness reset already
happens), not once per exec. A marginal cost computed over a single unit
is exactly the trap the same handover's §2 finding describes for
``smt_solver.py``'s ``evaluate()``: called on a singleton, an incremental
term can carry no signal about overlap or rate of change at all. Two
window-boundary snapshots are the minimum needed for a first difference
that means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MarginalCostTracker"]


@dataclass
class MarginalCostTracker:
    """Tracks the two most recent window-boundary ``(cost, output)``
    snapshots per key and derives ``MC_i = Δcost / Δoutput``.

    ``cost`` and ``output`` are both expected to be *cumulative* (monotone
    non-decreasing) counters -- e.g. total execs and total edges
    discovered since the key was first seen -- so that the difference
    between two snapshots is the amount of each accrued during that one
    window, not a rate the caller has to compute itself.
    """

    _snapshots: dict[str, tuple[float, float]] = field(default_factory=dict)
    _prev_snapshots: dict[str, tuple[float, float]] = field(default_factory=dict)

    def record_snapshot(self, key: str, cumulative_cost: float, cumulative_output: float) -> None:
        """Record a new window-boundary snapshot for ``key``.

        Shifts the previous "most recent" snapshot into the "prior" slot,
        so :meth:`marginal_cost` always diffs the two latest snapshots.
        Safe to call every window even for a key with no activity that
        window -- the cumulative counters simply won't have moved, and
        :meth:`marginal_cost` returns ``None`` for a zero-output window
        rather than a bogus division.
        """
        prev = self._snapshots.get(key)
        if prev is not None:
            self._prev_snapshots[key] = prev
        self._snapshots[key] = (cumulative_cost, cumulative_output)

    def _delta(self, key: str) -> tuple[float, float] | None:
        """``(Δcost, Δoutput)`` between the two most recent snapshots for
        ``key``, or ``None`` if there aren't yet two of them."""
        prev = self._prev_snapshots.get(key)
        cur = self._snapshots.get(key)
        if prev is None or cur is None:
            return None
        return (cur[0] - prev[0], cur[1] - prev[1])

    def marginal_cost(self, key: str) -> float | None:
        """``MC_i = Δcost / Δoutput`` between the two most recent snapshots.

        Returns ``None`` if ``key`` has fewer than two snapshots yet, or if
        ``Δoutput <= 0`` (no new output since the prior snapshot -- MC is
        undefined/unbounded there, not zero; a value of ``None`` here means
        "cannot express as a finite ratio", not "no cost". :meth:`should_stop`
        handles that case explicitly rather than silently reading ``None``
        as "no signal, keep going" -- see its docstring.
        """
        delta = self._delta(key)
        if delta is None:
            return None
        d_cost, d_output = delta
        if d_output <= 0:
            return None
        return d_cost / d_output

    def population_average_mc(self, keys: list[str] | None = None) -> float | None:
        """Mean of :meth:`marginal_cost` across ``keys`` (default: every
        key with a recorded snapshot). ``None`` if none of them currently
        has a defined marginal cost.
        """
        candidates = keys if keys is not None else list(self._snapshots)
        values = [mc for mc in (self.marginal_cost(k) for k in candidates) if mc is not None]
        if not values:
            return None
        return sum(values) / len(values)

    def should_stop(
        self, key: str, multiplier: float, keys: list[str] | None = None
    ) -> bool:
        """The handover §1 proposal-#2 stopping rule: ``True`` once
        ``key``'s marginal cost exceeds ``multiplier`` times the
        population-average marginal cost across ``keys``.

        ``False`` whenever ``key`` has fewer than two snapshots, or the
        population average is undefined -- absence of evidence (not enough
        windows yet, or nobody in ``keys`` has a defined marginal cost
        either) is not treated as evidence that the key should be stopped.

        A window in which ``key`` incurred cost but produced *no* new
        output (``Δoutput <= 0``, so :meth:`marginal_cost` itself returns
        ``None``) is the one exception: as long as some other key in
        ``keys`` does have a defined average to compare against, this is
        treated as worse than any finite MC and ``should_stop`` returns
        ``True`` -- silently reading that ``None`` as "no signal" would
        let an operator that spent a full window producing nothing pass
        the check for lack of a well-formed ratio, which is the opposite
        of what a marginal-cost stopping rule is for.
        """
        delta = self._delta(key)
        if delta is None:
            return False
        avg = self.population_average_mc(keys)
        if avg is None or avg <= 0:
            return False
        d_cost, d_output = delta
        if d_output <= 0:
            return d_cost > 0
        return (d_cost / d_output) > multiplier * avg

    def reset(self, key: str) -> None:
        """Drop all snapshots for ``key`` (e.g. it was removed from the
        population being tracked)."""
        self._snapshots.pop(key, None)
        self._prev_snapshots.pop(key, None)
