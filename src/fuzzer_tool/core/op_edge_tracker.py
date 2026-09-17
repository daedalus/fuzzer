"""OperatorEdgeTracker: per-operator coverage matrix for the ``op_tang`` arm.

``TangRecommendationScheduler.refit()`` (``core/schedulers/seed_tang.py``) reads
exactly two attributes off whatever it's handed -- ``seed_edges`` (key ->
set of edge ids) and ``seed_hit_counts`` (key -> {edge_id: count}) -- and
does not otherwise care what the keys mean. That means it is directly
reusable for operators with no changes at all: this module supplies an
object with those same two attribute names, populated per-operator instead
of per-seed, and ``core/schedulers/seed_tang.py`` needs zero modification.

The attribute names are kept as ``seed_edges``/``seed_hit_counts`` rather
than renamed, specifically so this stays a drop-in for
``TangRecommendationScheduler.refit(tracker)`` -- renaming them would mean
either forking ``seed_tang.py`` or adding an adapter layer for no benefit.

Empirical note (synthetic paired benchmark, no live target available to
validate against): the op_tang arm did not show a measurable benefit over
Thompson sampling even in a ground truth constructed specifically to favor
low-rank structure (60 paired trials, 30/30 split, Wilcoxon p=0.28), and
lost decisively when there was no structure to exploit (0/60, p<0.001).
That's a second independent negative alongside the one already documented
in ``seed_tang.py``'s own docstring for the seed-side arm. It is landed off by
default and should stay that way pending a real (not synthetic) A/B --
this module exists so that check can be run cheaply, not because the
prior evidence recommends turning it on.
"""

from __future__ import annotations


class OperatorEdgeTracker:
    """Accumulates which edges each operator has contributed to, cumulatively.

    Unlike ``EdgeTracker.seed_edges`` (exclusive per-seed ownership at the
    moment a seed is admitted), an operator's row here can grow across the
    whole campaign: the same operator contributing to a newly-discovered
    edge on exec 10 and again on exec 9000 both count, because the
    question this answers is "what has this operator been good at finding,
    in aggregate," not "what does this one candidate cover."
    """

    def __init__(self) -> None:
        self.seed_edges: dict[str, set[int]] = {}
        self.seed_hit_counts: dict[str, dict[int, int]] = {}

    def record(self, op: str, new_edge_ids) -> None:
        """Attribute ``new_edge_ids`` (an iterable of edge ids) to ``op``.

        Called once per operator that contributed to a round's new-edge
        set, mirroring the existing scalar attribution in
        ``Fuzzer``'s edge-discovery path (``self.op_edges[op] += share``)
        but keeping the actual edge identities instead of collapsing them
        to a float share -- Tang's refit needs the identities, a scalar
        can't reconstruct them.
        """
        if not new_edge_ids:
            return
        edges = self.seed_edges.setdefault(op, set())
        counts = self.seed_hit_counts.setdefault(op, {})
        for edge_id in new_edge_ids:
            edges.add(edge_id)
            counts[edge_id] = counts.get(edge_id, 0) + 1

    def reset(self) -> None:
        self.seed_edges.clear()
        self.seed_hit_counts.clear()
