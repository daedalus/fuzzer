"""Katz centrality over the horizon graph (K-Scheduler W4).

Fixed point (paper §5, out-degree variant, α from their Table XI):

    c[u] = alpha * sum(c[v] for v in successors(u)) + beta[u]

One SpMV per round via ``np.bincount``; on a DAG the fixed point is
exact after ``depth`` rounds, and ``build_horizon_graph`` guarantees
acyclicity — re-asserted here because silently diverging centrality
would poison every seed energy downstream. ``DEFAULT_MAX_ITER`` cuts the
sum off well before that depth; because α < 1 the tail decays
geometrically, so the truncation is a bounded-path-length variant rather
than an error. ``KatzResult.converged`` reports which one you got.

β is the paper's non-uniform injection: βᵢ = 1 − Rᵢ/T. Three things
about it are easy to get wrong and all three silently flatten it to
uniform, which costs nothing visible and disables the whole term:

* **Rᵢ counts executions reaching node i's visited parents**, not node i.
  U nodes are unvisited by construction, so their own counts are ~0 and a
  β built from them is 1 everywhere. ``build_beta`` takes the parent sets
  from the horizon graph for this reason.
* **T is the execution count**, not the sum of per-node counts. The
  latter is larger by the average trace length — 60x on a mid-size CFG —
  and compresses β into a band far narrower than α's dynamic range.
* **hit_counts is indexed by U position**, not by ICFG node. The caller
  holds ICFG-indexed arrays; ``horizon.u_icfg_index`` translates. A
  length mismatch is rejected rather than broadcast.

Seed nodes (appended after the U block) carry β=0: they are pure sources
whose score is α × the centrality of the horizons they attach to.
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


def build_beta(
    horizon: HorizonGraph,
    icfg_hit_counts: np.ndarray,
    total_execs: float,
) -> np.ndarray:
    """Per-U-node β from ICFG-indexed hit counts (paper §5).

    βᵢ = 1 − Rᵢ/T, Rᵢ = executions that reached *any* visited parent of U
    node i, T = executions performed. U nodes off the horizon have no
    visited parent and keep β=1: nothing has come near them yet, so they
    carry the full baseline.

    Args:
        horizon: W3 output; supplies the visited-parent sets.
        icfg_hit_counts: per-ICFG-node execution counts, length
            ``icfg.n_nodes``.
        total_execs: executions performed (the paper's T). Values <= 0
            yield uniform β — no information yet.

    Returns:
        float64 array of length ``horizon.n_u``, values in [0, 1].
    """
    n_u = horizon.n_u
    if total_execs <= 0:
        return np.ones(n_u, dtype=np.float64)
    hits = np.asarray(icfg_hit_counts, dtype=np.float64)
    reach = np.zeros(n_u, dtype=np.float64)
    for u_idx, parents in horizon.visited_parents.items():
        if not parents:
            continue
        # max, not sum: parents of one horizon node are usually alternative
        # predecessors of the same branch, so summing double-counts the same
        # approach and can push R past T.
        reach[u_idx] = max(hits[p] for p in parents if p < len(hits))
    beta = 1.0 - np.clip(reach / float(total_execs), 0.0, 1.0)
    if beta.size and np.allclose(beta, beta[0]):
        # All-equal β carries no information; hand back the exact uniform
        # case so downstream comparisons against it are bit-identical.
        return np.ones(n_u, dtype=np.float64)
    return beta


class KatzResult:
    """Centrality over [U nodes | seed nodes], plus diagnostics."""

    def __init__(
        self,
        scores: np.ndarray,
        iterations: int,
        seed_names: list[str],
        n_u: int = 0,
        converged: bool = True,
    ):
        self.scores = scores
        self.iterations = iterations
        self.seed_names = seed_names
        self.n_u = n_u
        self.converged = converged

    def seed_score(self, name: str) -> float:
        """Centrality of a seed's graph node.

        Indexed from ``n_u``, the start of the seed block. Deriving it from
        ``len(scores)`` instead is only equivalent when the score vector is
        exactly ``n_u + n_seeds`` long, which is precisely the invariant a
        mis-sized β used to break — and it broke it silently, returning the
        zero padding for every seed.
        """
        return float(self.scores[self.n_u + self.seed_names.index(name)])


def katz_scores(
    horizon: HorizonGraph,
    hit_counts: np.ndarray | None = None,
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = 1e-10,
    beta: np.ndarray | None = None,
) -> KatzResult:
    """Solve the fixed point over U + seeds.

    Args:
        horizon: W3 output.
        hit_counts: per-U-node β *source*, length ``horizon.n_u``. Callers
            holding ICFG-indexed counts want :func:`build_beta` instead;
            passing them straight through is the mis-indexing this
            signature now rejects. None → uniform β=1.
        beta: pre-built per-U-node β, length ``horizon.n_u``; takes
            precedence over ``hit_counts``. This is the production path —
            :func:`build_beta` produces it.
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

    if beta is not None:
        beta_u = np.asarray(beta, dtype=np.float64)
        if beta_u.shape != (n_u,):
            raise ValueError(f"beta must be per-U-node (length {n_u}), got {beta_u.shape}")
    elif hit_counts is None:
        beta_u = np.ones(n_u, dtype=np.float64)
    else:
        hits = np.asarray(hit_counts, dtype=np.float64)
        if hits.shape != (n_u,):
            raise ValueError(
                f"hit_counts must be per-U-node (length {n_u}), got "
                f"{hits.shape}; ICFG-indexed counts need build_beta() to "
                f"translate through horizon.u_icfg_index"
            )
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
    converged = False
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
            converged = True
            break
    return KatzResult(c, iterations, list(horizon.seed_names), n_u=n_u, converged=converged)
