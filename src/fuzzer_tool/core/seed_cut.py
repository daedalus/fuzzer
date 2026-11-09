"""Max-k-cut of the seed overlap graph B = A.A^T (``minimize -c --seed-cut K``).

Seeds i, j weigh B_ij = number of shared edges, once near-universal edges
(held by > strip_frac of the seeds) are stripped. Putting overlapping seeds in
*different* groups leaves every group internally diverse, which is the MaxCut
objective. B is never built: with N[e, g] = seeds of group g holding edge e,

    within  = sum_e sum_g C(N[e, g], 2)        total = sum_e C(N[e], 2)
    cut     = total - within

A uniformly random K-split cuts (1 - 1/K) of the total in expectation
(linearity of expectation, one term per pair). Placing each seed in the group
it overlaps least keeps that bound deterministically (the minimum is at most
the average), and a few 1-move passes only ever raise the cut.

    a={1,2} b={1,2} c={3,4} d={3,4}   k=2   ->   {a,c} | {b,d}, cut 4 of 4
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.edge_components import STRIP_FRAC, _incidence

TITLE = "Seed Cut"
SLACK = 0.1  # groups may exceed the even share n/k by this fraction
PASSES = 2


class _Graph:
    """Seed-major CSR of the kept (seed, edge) incidences."""

    def __init__(self, edges: dict[str, np.ndarray], strip_frac: float):
        self.keys, seed_idx, edge_ids = _incidence(edges)
        n = len(self.keys)
        uniq, dense, held = np.unique(edge_ids, return_inverse=True, return_counts=True)
        keep = held[dense] <= strip_frac * n
        self.n_edges = len(uniq)
        self.stripped = int((held > strip_frac * n).sum())

        # One incidence per (seed, edge); seed-major order for the CSR.
        combo = np.unique(seed_idx[keep] * max(self.n_edges, 1) + dense[keep])
        self.seed = combo // max(self.n_edges, 1)
        self.edge = combo % max(self.n_edges, 1)
        self.off = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.seed, minlength=n), out=self.off[1:])

    def row(self, s: int) -> np.ndarray:
        return self.edge[self.off[s] : self.off[s + 1]]


def _check(k: int, strip_frac: float, slack: float | None) -> None:
    if k < 2:
        raise ValueError("k must be >= 2")
    if not 0.0 < strip_frac <= 1.0:
        raise ValueError("strip_frac must be in (0, 1]")
    if slack is not None and slack < 0.0:
        raise ValueError("slack must be >= 0 or None")


def _pick(cost: np.ndarray, size: np.ndarray, cap: int) -> int:
    """Group with the least overlap; ties go to the smaller group."""
    score = cost.astype(np.int64) * (len(size) * cap + 1) + size
    score[size >= cap] = np.iinfo(np.int64).max
    return int(score.argmin())


def _cap(n: int, k: int, slack: float | None) -> int:
    return n if slack is None else max(1, math.ceil(n / k * (1.0 + slack)))


def _greedy(g: _Graph, k: int, cap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heaviest seeds first, each into the group it overlaps least."""
    n = len(g.keys)
    table = np.zeros((g.n_edges, k), dtype=np.int32)
    size = np.zeros(k, dtype=np.int64)
    lab = np.zeros(n, dtype=np.int64)

    deg = np.diff(g.off)
    order = np.argsort(-deg, kind="stable")
    for s in order.tolist():
        row = g.row(s)
        grp = _pick(table[row].sum(axis=0), size, cap)
        table[row, grp] += 1
        size[grp] += 1
        lab[s] = grp
    return lab, table, order


def _refine(g: _Graph, lab: np.ndarray, table: np.ndarray, order: np.ndarray, cap: int) -> int:
    """One pass of single-seed moves, each strictly lowering the overlap."""
    size = np.bincount(lab, minlength=table.shape[1]).astype(np.int64)
    moved = 0
    for s in order.tolist():
        row, cur = g.row(s), int(lab[s])
        table[row, cur] -= 1
        size[cur] -= 1

        cost = table[row].sum(axis=0)
        new = _pick(cost, size, cap)
        if cost[new] >= cost[cur]:
            new = cur

        table[row, new] += 1
        size[new] += 1
        lab[s] = new
        moved += new != cur
    return moved


def partition(
    edges: dict[str, np.ndarray],
    k: int,
    strip_frac: float = STRIP_FRAC,
    passes: int = PASSES,
    slack: float | None = SLACK,
) -> dict[str, int]:
    """Group in [0, k) per seed key; overlapping seeds are pushed apart.

    *slack* caps a group at ceil(n/k * (1 + slack)) seeds (None = no cap, which
    is the setting the (1 - 1/k) guarantee needs). *passes* bounds refinement.
    """
    _check(k, strip_frac, slack)
    if not edges:
        return {}

    g = _Graph(edges, strip_frac)
    cap = _cap(len(g.keys), k, slack)
    lab, table, order = _greedy(g, k, cap)
    for _ in range(passes):
        if not _refine(g, lab, table, order, cap):
            break
    return dict(zip(g.keys, lab.tolist(), strict=True))


def cut_weight(
    edges: dict[str, np.ndarray], labels: dict[str, int], strip_frac: float = STRIP_FRAC
) -> tuple[int, int]:
    """(cut weight, total weight) of *labels* over the kept edges."""
    if not edges:
        return 0, 0
    g = _Graph(edges, strip_frac)
    lab = np.fromiter((labels[k] for k in g.keys), dtype=np.int64, count=len(g.keys))

    # Pairs inside edge e: C(n_e, 2); inside (e, group): C(n_eg, 2).
    n_e = np.bincount(g.edge, minlength=g.n_edges).astype(np.int64)
    pair = np.unique(g.edge * (int(lab.max()) + 1) + lab[g.seed], return_counts=True)[1]
    pair = pair.astype(np.int64)
    total = int((n_e * (n_e - 1) // 2).sum())
    within = int((pair * (pair - 1) // 2).sum())
    return total - within, total


def seed_cut_report(
    edges: dict[str, np.ndarray],
    k: int,
    strip_frac: float = STRIP_FRAC,
    passes: int = PASSES,
    slack: float | None = SLACK,
) -> str:
    """Report text for ``minimize -c --seed-cut K``."""
    lines = ["", f"--- {TITLE} (k={k}, strip edges held by > {strip_frac:.0%} of seeds) ---"]
    labels = partition(edges, k, strip_frac, passes, slack)
    if not labels:
        lines.append("  No seeds with edge data.")
        return "\n".join(lines)

    cut, total = cut_weight(edges, labels, strip_frac)
    frac = cut / total if total else 1.0
    sizes = np.bincount(list(labels.values()), minlength=k).tolist()
    lines += [
        f"  {'Seeds:':<16s}{len(labels)}",
        f"  {'Total overlap:':<16s}{total}",
        f"  {'Cut weight:':<16s}{cut} ({frac:.1%}; random split expects {1 - 1 / k:.1%})",
        f"  {'Group sizes:':<16s}{sizes}",
    ]
    return "\n".join(lines)
