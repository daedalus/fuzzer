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

import itertools
import logging
import math
from collections.abc import Iterable, Mapping

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "EdgeCanonicalizer",
    "ExecutionPerplexity",
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

    Computed from the counts alone, in one pass.  What the counts cover is
    the caller's choice: ``EdgeTracker.effective_edges()`` passes the tracker's
    cumulative hits, which in the fuzz loop are recorded only for inputs with
    new coverage; ``ExecutionPerplexity`` passes sampled executions.  Note that
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


# Sampling cadence and minimum evidence for ExecutionPerplexity.  One scan
# every 32 executions keeps the cost off the hot path (a memoised scan is a
# dict build over the live edges; an unmemoised one was ~0.7-1.2 ms on
# ffmpeg, i.e. <= ~40 us amortised).  16 samples = 512 executions, half the
# default --stall-threshold, so a window is normally measurable by the time a
# stall is declared; the aggressive thresholds (//4, //8) are not, and the
# reason string then says nothing rather than quoting a tiny sample.
EXEC_PERPLEXITY_STRIDE = 32
EXEC_PERPLEXITY_MIN_SAMPLES = 16


class ExecutionPerplexity:
    """``2 ** H`` of the executions between discoveries, not of the campaign.

    P2-1 of the edge-id handover asked for effective edges in the stall
    reason, as a trend: the value when coverage last grew against the value
    now.  ``EdgeTracker.effective_edges()`` cannot give that, for two
    reasons.  It is cumulative, so a stall's concentration is averaged into
    the whole history -- and not merely diluted: mass piling onto edges that
    were in the tail of the history *flattens* the cumulative distribution,
    so its ``2 ** H`` can rise while the executions collapse (simulated:
    32 -> 47 cumulative against 20 in the window).  And in the fuzz loop it
    only sees inputs admitted to the corpus, which is exactly the population
    a stall stops producing.

    This keeps a separate count vector over *executed* inputs, sampled every
    ``stride`` executions, and closes a window at each new-edge discovery:
    the closed window's ``2 ** H`` becomes the reference, and the open one is
    the current value.  A window closed with fewer than ``min_samples``
    samples is not a measurement -- early in a campaign discoveries arrive
    every few executions -- so it is left open and keeps accumulating; the
    open window therefore starts at the last *measured* discovery, which is
    the last new edge whenever discoveries are sparse, i.e. whenever a stall
    is possible at all.

    A drop means the executed inputs spend their hits on fewer edges -- most
    often mutants dying in the same early-reject path.  Reported, not acted
    on: using it to temper exploration is operator-side design (P3).
    """

    __slots__ = ("stride", "min_samples", "_window", "_samples", "_reference", "_seen")

    def __init__(
        self,
        stride: int = EXEC_PERPLEXITY_STRIDE,
        min_samples: int = EXEC_PERPLEXITY_MIN_SAMPLES,
    ) -> None:
        if stride < 1 or min_samples < 1:
            raise ValueError("stride and min_samples must be >= 1")
        self.stride = stride
        self.min_samples = min_samples
        self._window: dict[int, int] = {}
        self._samples = 0
        self._reference: float | None = None
        self._seen = 0

    def due(self) -> bool:
        """Advance the execution clock; True when this execution is sampled."""
        self._seen += 1
        return self._seen % self.stride == 0

    def observe(self, counts: Mapping[int, int]) -> None:
        """Add one sampled execution's ``{edge_id: hit count}``."""
        if not counts:
            return
        w = self._window
        for edge, c in counts.items():
            if c > 0:
                w[edge] = w.get(edge, 0) + c
        self._samples += 1

    def note_new_edge(self) -> None:
        """Close the open window at a discovery, if it holds a measurement."""
        if self._samples < self.min_samples:
            return
        self._reference = effective_edges(self._window)
        self._window = {}
        self._samples = 0

    @property
    def reference(self) -> float | None:
        """``2 ** H`` of the last measured window that ended in a discovery."""
        return self._reference

    @property
    def current(self) -> float | None:
        """``2 ** H`` of the open window, or None below ``min_samples``."""
        if self._samples < self.min_samples:
            return None
        return effective_edges(self._window)

    def reason_suffix(self) -> str:
        """``" + effective edges R->C"`` when both ends are measured, else ``""``."""
        ref, cur = self._reference, self.current
        if ref is None or cur is None:
            return ""
        return f" + effective edges {ref:.0f}->{cur:.0f}"


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


#: Independent weight families in the profile fingerprint. A false merge needs
#: every family's 64-bit hash to collide at once.
_FAMILIES = 2
_FAMILY_STRIDE = 1 << 32  # keeps the families' splitmix inputs disjoint


def _seed_weights(n: int, family: int) -> np.ndarray:
    """*n* odd 64-bit weights: splitmix64 of (family, index).

    Deterministic and stateless, so a refit does not draw from the campaign's
    RNG stream (Hard Rule 16 governs randomness; this is a fixed mixing
    function). Odd means invertible mod 2^64, so no weight zeroes a count.
    """
    z = np.arange(1, n + 1, dtype=np.uint64) + np.uint64(family * _FAMILY_STRIDE)
    z *= np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return (z ^ (z >> np.uint64(31))) | np.uint64(1)


def _flatten(seed_hit_counts: Mapping[object, Mapping[int, int]]):
    """(edges, counts, seed index) as flat arrays, one entry per nonzero cell."""
    rows = list(seed_hit_counts.values())
    total = sum(len(hc) for hc in rows)
    edges = np.fromiter(itertools.chain.from_iterable(rows), dtype=np.int64, count=total)
    counts = np.fromiter(
        itertools.chain.from_iterable(hc.values() for hc in rows), dtype=np.uint64, count=total
    )
    seed = np.repeat(np.arange(len(rows)), [len(hc) for hc in rows])
    return edges, counts, seed


class EdgeCanonicalizer:
    """Groups edges whose per-seed count profile is identical.

    Two edges in one class are indistinguishable given the corpus: no input
    seen so far separated them.  Classes are recomputed wholesale by
    :meth:`refit` rather than maintained incrementally, because a class splits
    as soon as one input tells its members apart, and a merge-only structure
    would keep them fused forever.

    A profile is never materialised. Each edge gets the fingerprint
    ``h(e) = sum_s count(s, e) * w_s mod 2^64`` per weight family: linear in
    the column, so equal columns hash equal, and unequal ones collide only
    when their difference is orthogonal to w (Schwartz-Zippel). Cost is
    O(nonzero cells) instead of O(seeds x edges): measured 9x at 500 seeds
    and 53x at 8000 on an 8189-edge map, identical partitions throughout.

        seed:      s0   s1   s2          h(e) = 3*w0 + 7*w1 + 0*w2
        edge 1:     3    7    .   -->    h(1) == h(2)  -> one class
        edge 2:     3    7    .
    """

    def __init__(self) -> None:
        self._representative: dict[int, int] = {}
        self._size: dict[int, int] = {}
        self._classes = 0
        self._edges = 0
        self._keys = np.empty(0, dtype=np.int64)  # seen edges, ascending
        self._heads = np.empty(0, dtype=np.int64)  # their representatives

    def refit(self, seed_hit_counts: Mapping[object, Mapping[int, int]]) -> None:
        """Rebuild the classes from the per-seed hit counts.

        Args:
            seed_hit_counts: ``EdgeTracker.seed_hit_counts`` or the same
                shape -- seed key -> {edge id: count}.
        """
        edges, counts, seed = _flatten(seed_hit_counts)
        if not len(edges):
            self.__init__()
            return

        # One hash per family per edge: sort cells by edge, sum each run.
        # uint64 products and sums wrap, which is the mod 2^64 we want.
        order = np.argsort(edges, kind="stable")
        edges, counts, seed = edges[order], counts[order], seed[order]
        starts = np.flatnonzero(np.r_[True, edges[1:] != edges[:-1]])
        n = len(seed_hit_counts)
        hashes = np.column_stack(
            [np.add.reduceat(counts * _seed_weights(n, f)[seed], starts) for f in range(_FAMILIES)]
        )

        # Group equal fingerprints. Edges are ascending, so each group's first
        # occurrence is its smallest edge -- the same head the dense version chose.
        _, first, group, size = np.unique(
            hashes, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        group = group.ravel()
        self._keys = edges[starts]
        self._heads = self._keys[first][group]
        unique_edges = self._keys.tolist()
        self._representative = dict(zip(unique_edges, self._heads.tolist(), strict=True))
        self._size = dict(zip(unique_edges, size[group].tolist(), strict=True))
        self._classes = len(first)
        self._edges = len(unique_edges)

    def class_of(self, edge: int) -> int:
        """Representative id of *edge*'s class; the edge itself if unseen."""
        return self._representative.get(edge, edge)

    def class_array(self, edges: np.ndarray) -> np.ndarray:
        """:meth:`class_of` over an int64 array, for folds over every nonzero cell."""
        if not len(self._keys):
            return edges.copy()
        at = np.searchsorted(self._keys, edges).clip(max=len(self._keys) - 1)
        return np.where(self._keys[at] == edges, self._heads[at], edges)

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
