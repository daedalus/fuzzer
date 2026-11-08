"""Igor crash graphs: WL subtree kernel over reduced-trace edge graphs (E4).

Igor (CCS '21) clusters crashes by root cause, not by stack: each crash is
coverage-reduced (``services/crash_cluster.py``), its trace becomes a graph,
graphs are compared with the Weisfeiler-Lehman subtree kernel, and similar
graphs are linked::

    fires 1 2 1 3        nodes = distinct edge ids, arcs = consecutive fires
        |                      1 -> 2 -> 1 -> 3
        v
    WL round 0: {1, 2, 3}   round k: label = hash(label, succ labels, pred labels)
        |
        v
    count vectors -> cosine matrix -> threshold single linkage

Shim edge ids are hashes, so (src, dst) blocks are not recoverable from the
coverage map; arcs come only from the ordered fire log of a
``-D__AFL_TRACE_FIRES=1`` build. Without it the graph has no arcs and the
kernel reduces to edge-set cosine.
"""

from collections import Counter
from collections.abc import Iterable
from itertools import islice

import numpy as np

WL_ITERS = 3
# Fire log lines read per exec: one line per edge hit, so a loop floods it.
MAX_FIRES = 1 << 22
# Pair-product elements materialized per np.add.at batch (memory bound).
_PAIR_CHUNK = 1 << 20


class CrashGraph:
    """Directed graph over edge ids: ``labels[i]`` is node i's edge id."""

    __slots__ = ("labels", "succ", "pred")

    def __init__(self, labels: list[int], arcs: Iterable[tuple[int, int]]):
        n = len(labels)
        succ: list[list[int]] = [[] for _ in range(n)]
        pred: list[list[int]] = [[] for _ in range(n)]
        for s, d in arcs:
            succ[s].append(d)
            pred[d].append(s)

        self.labels = labels
        self.succ = [tuple(x) for x in succ]
        self.pred = [tuple(x) for x in pred]

    @property
    def size(self) -> int:
        """Distinct edges in the trace (the coverage-reduction objective)."""
        return len(self.labels)


def graph_from_fires(fires: Iterable[int], cap: int = MAX_FIRES) -> CrashGraph:
    """Graph of an ordered fire log; reads at most *cap* fires."""
    node: dict[int, int] = {}
    arcs: set[tuple[int, int]] = set()
    prev = -1

    # One pass: number nodes by first sight, record each transition once.
    for edge in islice(fires, cap):
        cur = node.setdefault(edge, len(node))
        if prev >= 0:
            arcs.add((prev, cur))
        prev = cur

    return CrashGraph(list(node), arcs)


def graph_from_edges(edge_ids: Iterable[int]) -> CrashGraph:
    """Arc-less graph of an unordered edge set (no fire log available)."""
    return CrashGraph(list(dict.fromkeys(edge_ids)), ())


def wl_features(g: CrashGraph, iters: int = WL_ITERS) -> dict[int, int]:
    """WL subtree features: count of each (round, label) over *iters* rounds."""
    labels = g.labels
    feats: Counter[int] = Counter()

    for rnd in range(iters + 1):
        # Round tag keeps round-0 edge ids apart from later compressed labels.
        feats.update(hash((rnd, lab)) for lab in labels)
        if rnd == iters:
            break

        # Relabel: own label + sorted multisets of successor / predecessor labels.
        labels = [
            hash(
                (
                    lab,
                    tuple(sorted(labels[j] for j in g.succ[i])),
                    tuple(sorted(labels[j] for j in g.pred[i])),
                )
            )
            for i, lab in enumerate(labels)
        ]

    return dict(feats)


def _accumulate(flat: np.ndarray, n: int, rows: np.ndarray, vals: np.ndarray) -> None:
    """Add every within-group outer product of (*rows*, *vals*) into *flat*.

    *rows*/*vals* are (groups, m): each row is one feature shared by m graphs.
    """
    per = rows.shape[1] ** 2
    step = max(1, _PAIR_CHUNK // per)

    for lo in range(0, rows.shape[0], step):
        r = rows[lo : lo + step]
        v = vals[lo : lo + step]
        idx = (r[:, :, None] * n + r[:, None, :]).ravel()
        np.add.at(flat, idx, (v[:, :, None] * v[:, None, :]).ravel())


def kernel_matrix(feats: list[dict[int, int]]) -> np.ndarray:
    """Normalized WL kernel (cosine) between every pair of feature dicts.

    Inverted index: only graphs sharing a feature interact, so cost is the
    sum of squared posting lengths rather than n^2 dict intersections.
    Empty graphs are similar to nothing; the diagonal is 1.
    """
    n = len(feats)
    flat = np.zeros(n * n)

    sizes = [len(f) for f in feats]
    if sum(sizes):
        rows = np.repeat(np.arange(n), sizes)
        cols = np.fromiter((k for f in feats for k in f), np.int64, sum(sizes))
        vals = np.fromiter((c for f in feats for c in f.values()), np.float64, sum(sizes))

        # Group postings by feature, then batch groups of equal length.
        order = np.argsort(cols, kind="stable")
        rows, cols, vals = rows[order], cols[order], vals[order]
        starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]])
        lens = np.diff(np.r_[starts, cols.size])

        for m in np.unique(lens):
            at = starts[lens == m][:, None] + np.arange(m)
            _accumulate(flat, n, rows[at], vals[at])

    k = flat.reshape(n, n)
    norm = np.sqrt(np.diag(k))
    denom = np.outer(norm, norm)
    sim = np.divide(k, denom, out=np.zeros_like(k), where=denom > 0)
    np.fill_diagonal(sim, 1.0)
    return sim


def cluster_kernel(sim: np.ndarray, threshold: float) -> list[list[int]]:
    """Single-linkage clusters of pairs with ``sim >= threshold``.

    Returns clusters ordered by first member, members ascending.
    """
    n = sim.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    # Union every linked pair; the smaller root wins, so roots are minima.
    for i, j in np.argwhere(np.triu(sim >= threshold, 1)).tolist():
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
