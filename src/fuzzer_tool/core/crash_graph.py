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
                                   +-> spectral: Laplacian embedding, k-means, silhouette k

Shim edge ids are hashes, so (src, dst) blocks are not recoverable from the
coverage map; arcs come only from the ordered fire log of a
``-D__AFL_TRACE_FIRES=1`` build. Without it the graph has no arcs and the
kernel reduces to edge-set cosine.
"""

from collections import Counter
from collections.abc import Iterable
from enum import Enum
from itertools import islice

import numpy as np

WL_ITERS = 3
# Fire log lines read per exec: one line per edge hit, so a loop floods it.
MAX_FIRES = 1 << 22
# Pair-product elements materialized per np.add.at batch (memory bound).
_PAIR_CHUNK = 1 << 20
# Laplacian eigenvalue below which a mode is a connected component.
_ZERO_EIG = 1e-9
# Lloyd iterations cap; well-separated embeddings settle in a few.
KMEANS_ITERS = 100
# Largest eigengaps tried as k; each costs one k-means + one O(n^2) silhouette.
SPECTRAL_CANDIDATES = 8


class Linkage(Enum):
    """How crashes are grouped from the kernel matrix (``--linkage``).

    THRESHOLD  single linkage at ``--threshold``; one bridge pair chains groups.
    SPECTRAL   Igor's choice: normalized-Laplacian embedding, k-means, k by silhouette.
    """

    THRESHOLD = "threshold"
    SPECTRAL = "spectral"


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

    return _groups([find(i) for i in range(n)])


def _groups(keys: list[int]) -> list[list[int]]:
    """Indices grouped by key, groups ordered by first member."""
    groups: dict[int, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def _candidates(lam: np.ndarray) -> tuple[int, list[int]]:
    """(components, candidate k) from ascending Laplacian eigenvalues.

    Zero modes count connected components: k never goes below them. The
    largest eigengaps above that propose k; the silhouette decides. Eigengap
    alone fails on a uniform background: 10 groups at 0.9 inside, 0.1
    across give 0, .549 x9, .989, ... -- the k=1 gap beats the k=10 one.
    """
    n = lam.size
    comps = max(1, int((lam < _ZERO_EIG).sum()))
    lo = max(2, comps)
    if lo >= n:
        return comps, []

    # gaps[j] = lam[j] - lam[j-1] is the gap that ends k = j clusters.
    gaps = np.diff(lam)[lo - 1 :]
    top = np.argsort(-gaps, kind="stable")[:SPECTRAL_CANDIDATES] + lo
    return comps, sorted({comps, *top.tolist()} - {1})


def _kmeans(x: np.ndarray, k: int) -> np.ndarray:
    """Lloyd k-means on rows of *x*; deterministic farthest-first seeding.

    Rows are unit vectors (NJW embedding), so the farthest point is never
    an outlier far off the sphere.
    """
    n = x.shape[0]
    idx = [0]
    d2 = ((x - x[0]) ** 2).sum(axis=1)

    # Farthest-first: each new center is the point worst served so far.
    for _ in range(1, k):
        far = int(np.argmax(d2))
        if d2[far] <= 0:
            break  # fewer distinct points than k
        idx.append(far)
        d2 = np.minimum(d2, ((x - x[far]) ** 2).sum(axis=1))

    c = x[idx].copy()
    labels = np.full(n, -1)
    for _ in range(KMEANS_ITERS):
        # |x - c|^2 up to the per-row |x|^2 constant: n x k, no n x k x d tensor.
        new = ((c**2).sum(axis=1) - 2 * x @ c.T).argmin(axis=1)
        if (new == labels).all():
            break
        labels = new

        counts = np.bincount(labels, minlength=len(c))
        sums = np.zeros_like(c)
        np.add.at(sums, labels, x)
        live = counts > 0  # an emptied center keeps its place
        c[live] = sums[live] / counts[live, None]

    return labels


def _silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette of *labels* under *dist*; singletons score 0."""
    k = int(labels.max()) + 1
    onehot = np.zeros((labels.size, k))
    onehot[np.arange(labels.size), labels] = 1.0
    sums = dist @ onehot
    counts = onehot.sum(axis=0)
    rows = np.arange(labels.size)

    # a: mean distance to the own cluster's others; b: nearest other cluster.
    own = counts[labels] - 1
    a = np.divide(sums[rows, labels], own, out=np.zeros(labels.size), where=own > 0)
    other = np.divide(sums, counts, out=np.full_like(sums, np.inf), where=counts > 0)
    other[rows, labels] = np.inf
    b = other.min(axis=1)

    top = np.maximum(a, b)
    s = np.divide(b - a, top, out=np.zeros(labels.size), where=(top > 0) & (own > 0))
    return float(s.mean())


def cluster_spectral(sim: np.ndarray) -> list[list[int]]:
    """Spectral clusters (Ng-Jordan-Weiss) of a similarity matrix.

    Unlike threshold linkage, one similar pair does not merge two dense
    groups: the bridge barely moves the Laplacian's low eigenvectors.
    k: eigengap candidates scored by silhouette on ``1 - sim``; one cluster
    only when no split scores above 0. Ties take the larger k: merging
    hides distinct bugs. Returns clusters ordered by first member.
    """
    n = sim.shape[0]
    if n == 0:
        return []

    # L = I - D^-1/2 S D^-1/2; the diagonal of S is 1, so degrees are >= 1.
    inv = 1.0 / np.sqrt(sim.sum(axis=1))
    lap = np.eye(n) - inv[:, None] * sim * inv[None, :]
    lam, vec = np.linalg.eigh(lap)

    comps, ks = _candidates(lam)
    if comps >= n:
        return [[i] for i in range(n)]

    dist = np.clip(1.0 - sim, 0.0, None)
    best, best_s = np.zeros(n, dtype=int), 0.0 if comps == 1 else -np.inf

    # Rows of the k low eigenvectors, unit length, are the points to split.
    for k in ks:
        emb = vec[:, :k] / np.linalg.norm(vec[:, :k], axis=1, keepdims=True)
        labels = _kmeans(emb, k)
        score = _silhouette(dist, labels)
        if score >= best_s and (score > 0 or comps > 1):
            best, best_s = labels, score

    return _groups(best.tolist())
