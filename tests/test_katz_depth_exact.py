"""Katz over a DAG is exact at the graph's depth, not at a fixed 30 rounds.

The oracle here is a direct back-substitution in reverse topological order:

    c[u] = alpha * sum(c[v] for v in successors(u)) + beta[u]

On a DAG that recurrence has a closed evaluation order, so the oracle is
the exact fixed point with no iteration and no tolerance. The iterative
solver must agree with it whenever it reports ``converged``.

The interesting case is a graph deeper than the old ``DEFAULT_MAX_ITER``:
that is where the truncation used to be invisible, because ``converged``
only ever reported the ``tol`` test and had no way to say "I stopped
early".
"""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.core.horizon import HorizonGraph
from fuzzer_tool.core.schedulers.katz import DEFAULT_MAX_ITER, _dag_depth, katz_scores


def exact_fixed_point(src, dst, beta, alpha, n):
    """Back-substitution in reverse topological order — no iteration."""
    children = {}
    indeg = [0] * n
    for s, d in zip(src.tolist(), dst.tolist(), strict=True):
        children.setdefault(s, []).append(d)
        indeg[d] += 1
    queue = [v for v in range(n) if indeg[v] == 0]
    order = []
    while queue:
        v = queue.pop()
        order.append(v)
        for w in children.get(v, ()):
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    c = np.array(beta, dtype=np.float64)
    for u in reversed(order):
        c[u] = alpha * sum(c[v] for v in children.get(u, ())) + beta[u]
    return c


def _graph(u_nodes: list[int], src: np.ndarray, dst: np.ndarray) -> HorizonGraph:
    return HorizonGraph(
        u_nodes=u_nodes,
        src=src,
        dst=dst,
        horizon_set=set(),
        seed_names=[],
        seed_edges={},
    )


def chain_horizon(length: int) -> HorizonGraph:
    """A path graph 0 -> 1 -> ... -> length-1. Depth is length-1."""
    u_nodes = list(range(length))
    src = np.array(list(range(length - 1)), dtype=np.int64)
    dst = np.array(list(range(1, length)), dtype=np.int64)
    return _graph(u_nodes, src, dst)


def random_dag_horizon(n: int, seed: int, avg_deg: int = 3) -> HorizonGraph:
    rng = np.random.default_rng(seed)
    src, dst = [], []
    for u in range(n - 1):
        for _ in range(int(rng.integers(0, 2 * avg_deg))):
            src.append(u)
            dst.append(int(rng.integers(u + 1, n)))
    return _graph(
        list(range(n)),
        np.array(src, dtype=np.int64),
        np.array(dst, dtype=np.int64),
    )


class TestDagDepth:
    def test_chain_depth_is_its_length(self):
        g = chain_horizon(40)
        assert _dag_depth(g.src, g.dst, g.n_u) == 39

    def test_edgeless_graph_has_depth_zero(self):
        assert _dag_depth(np.zeros(0, np.int64), np.zeros(0, np.int64), 5) == 0

    def test_cycle_still_raises(self):
        """The assertion the depth walk replaced must not have been lost."""
        src = np.array([0, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 0], dtype=np.int64)
        with pytest.raises(ValueError, match="must be a DAG"):
            _dag_depth(src, dst, 3)


class TestExactnessAtDepth:
    @pytest.mark.parametrize("length", [5, 31, 60, 120])
    def test_chain_matches_back_substitution(self, length):
        """A chain of length L needs L-1 rounds; the old cap gave 30."""
        g = chain_horizon(length)
        r = katz_scores(g, alpha=0.5, tol=0.0)
        want = exact_fixed_point(g.src, g.dst, np.ones(length), 0.5, length)
        assert r.converged
        assert np.abs(r.scores - want).max() == pytest.approx(0.0, abs=1e-12)

    def test_deep_graph_beats_the_old_fixed_cap(self):
        """Falsifies the fix: with the old cap the same graph is wrong.

        Fan-in is what makes the truncation bite. On a bare chain each
        round carries one predecessor's delta and ``tol`` fires before the
        depth does; on a graph with real branching the per-round delta is a
        sum over in-edges, so it stays above ``tol`` well past round 30.
        """
        g = random_dag_horizon(3000, seed=17, avg_deg=4)
        n = g.n_u
        assert _dag_depth(g.src, g.dst, n) > DEFAULT_MAX_ITER
        want = exact_fixed_point(g.src, g.dst, np.ones(n), 0.5, n)
        truncated = katz_scores(g, alpha=0.5, max_iter=DEFAULT_MAX_ITER)
        exact = katz_scores(g, alpha=0.5)
        assert truncated.converged is False
        assert np.abs(truncated.scores - want).max() > 1e-6
        assert exact.converged is True
        assert np.abs(exact.scores - want).max() < 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_dags_match_back_substitution(self, seed):
        g = random_dag_horizon(400, seed)
        r = katz_scores(g, alpha=0.5)
        want = exact_fixed_point(g.src, g.dst, np.ones(400), 0.5, 400)
        assert np.abs(r.scores - want).max() == pytest.approx(0.0, abs=1e-9)

    def test_non_uniform_beta_matches_back_substitution(self):
        """beta large enough that tol never fires — depth is the only proof."""
        g = random_dag_horizon(300, 9)
        rng = np.random.default_rng(4)
        beta = rng.random(300) * 1000.0
        r = katz_scores(g, alpha=0.5, beta=beta)
        want = exact_fixed_point(g.src, g.dst, beta, 0.5, 300)
        assert r.converged
        assert np.abs(r.scores - want).max() == pytest.approx(0.0, abs=1e-6)


class TestExplicitMaxIterStillHonoured:
    def test_below_depth_reports_not_converged(self):
        g = chain_horizon(50)
        r = katz_scores(g, max_iter=3)
        assert r.iterations == 3
        assert r.converged is False

    def test_above_depth_stops_at_depth(self):
        """An oversized budget must not spin out the extra no-op rounds."""
        g = chain_horizon(20)
        r = katz_scores(g, max_iter=500)
        assert r.converged is True
        assert r.iterations <= 19

    def test_zero_depth_graph_is_converged_without_iterating(self):
        g = _graph([0, 1, 2], np.zeros(0, np.int64), np.zeros(0, np.int64))
        r = katz_scores(g)
        assert r.converged is True
        assert r.iterations == 0
        assert np.abs(r.scores - 1.0).max() == pytest.approx(0.0, abs=1e-12)
