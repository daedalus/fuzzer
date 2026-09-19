"""Substrate every scheduler should read before it scores anything.

Three pieces, all of them consequences of
``docs/handover/handover_edge_id_axis_2026-09-18.md``:

**Coverage trust.**  A target that carries the shim but no instrumented call
sites records only the harness's own ``__afl_map_edge`` calls, and a campaign
against one reports ``Edges discovered: 2``, ``Total richness: 2 - 2`` and
100% saturation -- indistinguishable from a target that really is exhausted
(F11).  A target whose edge ids differ in every process makes every edge a
singleton owned by exactly one seed, which is maximally *rare* to any
rarity-weighted scheduler (F1).  Both conditions are now detectable, and
``coverage_trust`` is the single call that reports them so a scheduler can
decline to draw conclusions rather than confidently scoring noise.

**Effective edges.**  ``2 ** H`` over the hit-count distribution: how many
edges the execution volume is really spread across.  65 of 445 on fuzzgoat.
Its value is the *trend* -- collapsing while the raw edge count stays flat is
a saturation signal the stall machinery has no equivalent of, and it is the
drift detector the operator side needs, since drift rather than capacity is
what made a learner fall to uniform in the neural-metrics work.

**Canonical edges.**  Edges whose count profile is identical across every
execution so far carry the same information: 126 of 445 on fuzzgoat, of which
79 in 14 classes were context tags of one base edge that never once differed
(F10).  Scoring raw ids counts one branch up to 45 times, so any rarity or
novelty weight is distorted by that multiplicity.  ``EdgeCanonicalizer``
computes the classes; applying them to the weights is deliberately not done
here -- that changes scheduling decisions and is gated on a paired benchmark
(P3-3, P3-4 in the handover).  What ships now is the measurement.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping

log = logging.getLogger(__name__)

__all__ = [
    "EdgeCanonicalizer",
    "coverage_trust",
    "effective_edges",
]


def effective_edges(hits: Mapping[int, int] | Iterable[int]) -> float:
    """Perplexity of the hit-count distribution: ``2 ** H``.

    Args:
        hits: Per-edge hit counts, as a mapping (values are used) or any
            iterable of counts.  Zero and negative entries are ignored.

    Returns:
        The number of edges the execution volume is effectively spread over.
        Equal to the edge count when every edge is hit equally, 1.0 when one
        edge takes all of it, and 0.0 when there is nothing to measure.

    Computed from the counts alone, in one pass.  Note that
    ``EdgeTracker.edge_hit_distribution()`` also carries this information but
    costs O(edges x seeds) to build, because it recounts owners per edge
    rather than reading ``_edge_owner_count``.
    """
    values = list(hits.values()) if isinstance(hits, Mapping) else list(hits)
    positive = [float(v) for v in values if v > 0]
    total = sum(positive)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for v in positive:
        p = v / total
        entropy -= p * math.log2(p)
    return 2.0**entropy


def coverage_trust(
    target: str | None,
    *,
    use_coverage: bool = True,
    ptrace: bool = False,
    id_stability: float | None = None,
) -> tuple[bool, str | None]:
    """Decide whether per-edge statistics from *target* mean anything.

    Args:
        target: Path to the target binary, or None when there is nothing to
            inspect (in-process callable, say).
        use_coverage: False disables the check; the premise does not hold.
        ptrace: True when coverage comes from breakpoints, which need no
            build-time instrumentation -- the same carve-out
            ``_warn_uninstrumented`` makes.
        id_stability: Jaccard of the edge-id sets across repeated executions
            of one input, when it has been measured.  Anything below 1.0 on a
            deterministic target means the ids themselves are moving.

    Returns:
        ``(trusted, reason)``.  ``reason`` is None when trusted, otherwise a
        one-line explanation suitable for a log line.

    Warn-only by design: a scheduler that declines to run is worse than one
    that runs on weak data, so the decision of what to do with a False stays
    with the caller.
    """
    if not use_coverage or ptrace or not target:
        return True, None

    if id_stability is not None and id_stability < 1.0:
        return False, (
            f"edge ids are not reproducible across processes (Jaccard "
            f"{id_stability:.3f}); every per-edge statistic is aggregating ids "
            "that do not denote the same edge"
        )

    try:
        from fuzzer_tool.core.elf import sancov_guard_status
    except ImportError:  # pragma: no cover - core always imports
        return True, None

    if sancov_guard_status(target) == "absent":
        return False, (
            f"{target} carries the shim but no compiler-inserted edge coverage; "
            "only hand-written __afl_map_edge calls register, so edge counts "
            "and saturation describe the harness, not the target"
        )
    return True, None


class EdgeCanonicalizer:
    """Groups edges whose per-seed count profile is identical.

    Two edges in one class are indistinguishable given the corpus: no input
    seen so far separated them.  Classes are recomputed wholesale by
    :meth:`refit` rather than maintained incrementally, because a class splits
    as soon as one input tells its members apart, and a merge-only structure
    would keep them fused forever.
    """

    def __init__(self) -> None:
        self._representative: dict[int, int] = {}
        self._size: dict[int, int] = {}
        self._classes = 0
        self._edges = 0

    def refit(self, seed_hit_counts: Mapping[object, Mapping[int, int]]) -> None:
        """Rebuild the classes from the per-seed hit counts.

        Args:
            seed_hit_counts: ``EdgeTracker.seed_hit_counts`` or the same
                shape -- seed key -> {edge id: count}.
        """
        seeds = list(seed_hit_counts)
        profiles: dict[int, list[int]] = {}
        for index, seed in enumerate(seeds):
            for edge, count in seed_hit_counts[seed].items():
                profiles.setdefault(edge, [0] * len(seeds))[index] = int(count)

        groups: dict[tuple[int, ...], list[int]] = {}
        for edge, profile in profiles.items():
            groups.setdefault(tuple(profile), []).append(edge)

        representative: dict[int, int] = {}
        size: dict[int, int] = {}
        for members in groups.values():
            head = min(members)
            for edge in members:
                representative[edge] = head
                size[edge] = len(members)
        self._representative = representative
        self._size = size
        self._classes = len(groups)
        self._edges = len(profiles)

    def class_of(self, edge: int) -> int:
        """Representative id of *edge*'s class; the edge itself if unseen."""
        return self._representative.get(edge, edge)

    def multiplicity(self, edge: int) -> int:
        """How many edges share *edge*'s profile, at least 1."""
        return self._size.get(edge, 1)

    def stats(self) -> dict:
        """Class counts, and how much of the map is redundant."""
        duplicates = self._edges - self._classes
        return {
            "edges": self._edges,
            "classes": self._classes,
            "duplicate_edges": duplicates,
            "duplicate_fraction": (duplicates / self._edges) if self._edges else 0.0,
            "largest_class": max(self._size.values(), default=0),
        }
