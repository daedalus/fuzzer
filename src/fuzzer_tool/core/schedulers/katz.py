"""Katz centrality over the horizon graph (K-Scheduler W4).

Fixed point (paper §5, out-degree variant, α from their Table XI):

    c[u] = alpha * sum(c[v] for v in successors(u)) + beta[u]

One SpMV per round via ``np.bincount``; on a DAG the iteration converges
in ≤ graph-depth rounds, and ``build_horizon_graph`` guarantees
acyclicity — re-asserted here because silently diverging centrality
would poison every seed energy downstream.

β is the paper's non-uniform injection: βᵢ = 1 − Rᵢ/T with Rᵢ counting
executions that reached node i. Rarely-hit nodes get the larger baseline,
which is what pushes seed energy toward unvisited horizons. Seed nodes
(append after the U block) carry β=0: they are pure sources whose score
is α × the centrality of the horizons they attach to.
"""

import numpy as np

from fuzzer_tool.core.horizon import HorizonGraph

DEFAULT_ALPHA = 0.5  # paper Table XI
DEFAULT_MAX_ITER = 30


def _assert_dag(src: np.ndarray, dst: np.ndarray, n: int) -> None:
    """Kahn's algorithm; raises when a cycle would trap the fixed point."""
    indeg = np.zeros(n, dtype=np.int64)
    for d in dst.tolist():
        indeg[d] += 1
    children: dict[int, list[int]] = {}
    for s, d in zip(src.tolist(), dst.tolist(), strict=False):
        children.setdefault(s, []).append(d)
    queue = [v for v in range(n) if indeg[v] == 0]
    seen = 0
    while queue:
        v = queue.pop()
        seen += 1
        for w in children.get(v, ()):
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    if seen != n:
        raise ValueError(
            f"horizon graph must be a DAG before Katz "
            f"(found {n - seen} nodes in cycles); run build_horizon_graph"
        )


class KatzResult:
    """Centrality over [U nodes | seed nodes], plus diagnostics."""

    def __init__(self, scores: np.ndarray, iterations: int, seed_names: list[str]):
        self.scores = scores
        self.iterations = iterations
        self.seed_names = seed_names

    def seed_score(self, name: str) -> float:
        return float(
            self.scores[len(self.scores) - len(self.seed_names) + self.seed_names.index(name)]
        )


def katz_scores(
    horizon: HorizonGraph,
    hit_counts: np.ndarray | None = None,
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = 1e-10,
) -> KatzResult:
    """Solve the fixed point over U + seeds.

    Args:
        horizon: W3 output.
        hit_counts: per-U-node execution counts for non-uniform β;
            None → uniform β=1.
    """
    n_u = horizon.n_u
    n_seeds = len(horizon.seed_names)
    n_total = n_u + n_seeds

    seed_edges = [
        (n_u + si, t)
        for si, name in enumerate(horizon.seed_names)
        for t in sorted(horizon._seed_edges.get(name, ()))
    ]
    src = np.concatenate(
        [horizon.src.astype(np.int64), np.array([e[0] for e in seed_edges], dtype=np.int64)]
    )
    dst = np.concatenate(
        [horizon.dst.astype(np.int64), np.array([e[1] for e in seed_edges], dtype=np.int64)]
    )
    if n_total == 0:
        return KatzResult(np.zeros(0, dtype=np.float64), 0, [])

    _assert_dag(src, dst, n_total)

    if hit_counts is None:
        beta_u = np.ones(n_u, dtype=np.float64)
    else:
        hits = np.asarray(hit_counts, dtype=np.float64)
        total = hits.sum()
        if total <= 0:
            beta_u = np.ones(n_u, dtype=np.float64)
        else:
            beta_u = 1.0 - hits / total
            # All-equal nonzero hits collapse to uniform, matching the
            # no-information case exactly.
            if np.allclose(beta_u, beta_u[0]):
                beta_u = np.ones(n_u, dtype=np.float64)
    beta = np.concatenate([beta_u, np.zeros(n_seeds, dtype=np.float64)])

    c = beta.copy()
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        nxt = beta.copy()
        # Successor-summing (paper's out-degree form): node u receives
        # alpha * c[v] for each edge u->v.
        contrib = np.bincount(src, weights=c[dst], minlength=n_total)
        nxt[: len(contrib)] += alpha * contrib
        delta = np.abs(nxt - c).max()
        c = nxt
        if delta < tol:
            break
    return KatzResult(c, iterations, list(horizon.seed_names))
