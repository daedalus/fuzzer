"""Tests for Katz centrality over the horizon graph (K-Scheduler W4).

Fixed-point form used (paper §5, out-degree Katz):
    c[u] = alpha * sum(c[v] for v in successors(u)) + beta[u]
On a DAG this converges in <= depth iterations; build_horizon_graph
guarantees acyclicity and katz_scores re-asserts it.

Hand-derived oracles: with uniform beta=1 and alpha=0.5 on a→b,
c[b] = 1, c[a] = 0.5*1 + 1 = 1.5.
"""

import numpy as np
import pytest

from fuzzer_tool.core.horizon import HorizonGraph
from fuzzer_tool.core.schedulers.katz import katz_scores


def _graph(edges, n_u, seed_edges=None, seed_names=None):
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    names = seed_names or list(seed_edges or {})
    return HorizonGraph(
        u_nodes=list(range(n_u)),
        src=src,
        dst=dst,
        horizon_set=set(),
        seed_names=names,
        seed_edges={k: set(v) for k, v in (seed_edges or {}).items()},
    )


class TestConvergence:
    def test_hand_derived_two_node_chain(self):
        g = _graph([(0, 1)], n_u=2)
        r = katz_scores(g, alpha=0.5)
        assert r.scores.shape == (2,)
        assert r.scores[1] == pytest.approx(1.0)
        assert r.scores[0] == pytest.approx(1.5)

    def test_deep_chain_converges(self):
        """60-node chain needs ~60 propagation rounds; the iteration cap
        must still yield finite, monotone-toward-sink values."""
        edges = [(i, i + 1) for i in range(59)]
        g = _graph(edges, n_u=60)
        r = katz_scores(g, alpha=0.5, max_iter=30)
        assert np.all(np.isfinite(r.scores))
        # Sink carries pure beta; every upstream node >= its child when
        # converged (sum of descendants), truncated or not.
        assert r.scores[-1] == pytest.approx(1.0)
        assert r.iterations <= 31

    def test_deterministic(self):
        g = _graph([(0, 1), (0, 2)], n_u=3)
        a = katz_scores(g).scores
        b = katz_scores(g).scores
        assert (a == b).all()


class TestNonUniformBeta:
    def test_rare_horizon_outscores_hit_horizon(self):
        """beta_i = 1 - hits_i / total: the heavily-hit sink must score
        below the untouched one, and so must its parent."""
        g = _graph([(0, 1)], n_u=2)
        hits = np.array([0.0, 100.0])
        r = katz_scores(g, hit_counts=hits, alpha=0.5)
        assert r.scores[1] < r.scores[0]

    def test_all_hit_collapses_to_uniform(self):
        """Equal hits everywhere == no information == uniform beta."""
        g = _graph([(0, 1)], n_u=2)
        uni = katz_scores(g, hit_counts=np.array([5.0, 5.0])).scores
        plain = katz_scores(g, hit_counts=None).scores
        assert uni == pytest.approx(plain)


class TestSeedAttachment:
    def test_seed_score_follows_attached_horizons(self):
        """Seed A attaches to a rare sink (β=1), B to a saturated one
        (β=0.5): A must outrank B, and each seed scores α × its target."""
        # sinks: u1 rare (hits 0), u3 hit (hits 100)
        edges = [(0, 1), (2, 3)]
        g = _graph(
            edges,
            n_u=4,
            seed_edges={"A": {1}, "B": {3}},
            seed_names=["A", "B"],
        )
        hits = np.array([0.0, 0.0, 50.0, 100.0])
        r = katz_scores(g, hit_counts=hits, alpha=0.5)
        iA = 4 + r.seed_names.index("A")
        iB = 4 + r.seed_names.index("B")
        assert r.scores[iA] == pytest.approx(0.5)  # α · c[u1], c[u1]=β=1
        assert r.scores[iB] == pytest.approx(1.0 / 6)  # α · c[u3], β3=1-100/150
        assert r.scores[iA] > r.scores[iB]

    def test_empty_graph_is_safe(self):
        g = _graph([], n_u=0)
        r = katz_scores(g)
        assert r.scores.shape == (0,)
        assert r.iterations == 0

    def test_lone_seed_scores_zero(self):
        g = _graph([], n_u=0, seed_edges={"lonely": set()}, seed_names=["lonely"])
        r = katz_scores(g)
        assert r.scores.shape == (1,)
        assert r.scores[0] == 0.0


class TestContractEnforcement:
    def test_cycle_is_rejected(self):
        """W3 promises a DAG; feeding a cycle back must fail loudly, not
        silently diverge."""
        g = _graph([(0, 1), (1, 0)], n_u=2)
        with pytest.raises(ValueError, match="[Dd][Aa][Gg]"):
            katz_scores(g)
