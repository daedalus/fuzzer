"""Regression tests for the K-Scheduler beta injection.

Three defects made beta a no-op while leaving every existing test green,
because the tests exercised ``katz_scores`` with already-correct inputs and
the mistakes all lived at the ``KatzChannel`` boundary:

1. R was read from the U node's own hit count. U nodes are unvisited by
   construction, so R was ~0 and beta was 1 everywhere.
2. T was ``hits.sum()`` (a sum over nodes) rather than the execution count,
   compressing beta's range by the average trace length.
3. The array was ICFG-indexed while ``katz_scores`` wanted U-indexed, which
   silently produced a score vector longer than the graph.

Each is pinned below, plus the ``seed_score`` accessor that (3) broke.
"""

import numpy as np
import pytest

from fuzzer_tool.core.horizon import HorizonGraph, build_horizon_graph
from fuzzer_tool.core.icfg import InterproceduralCFG
from fuzzer_tool.core.schedulers.katz import build_beta, katz_scores


def _icfg(n, edges):
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    return InterproceduralCFG(
        node_addrs=[0x1000 + 4 * i for i in range(n)],
        node_funcs=["f"] * n,
        src=src,
        dst=dst,
        cfgs={},
    )


def _mask(n, visited):
    bits = np.zeros(n, dtype=bool)
    bits[list(visited)] = True
    return np.packbits(bits, bitorder="little").tobytes()


class TestIndexSpaces:
    def test_u_icfg_index_maps_back_to_the_icfg(self):
        """U index i is ICFG node u_icfg_index[i], not ICFG node i."""
        icfg = _icfg(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        h = build_horizon_graph(icfg, {"s": _mask(6, [0, 1, 2])})
        assert h.n_u == 3
        assert h.u_icfg_index == [3, 4, 5]
        assert h.u_nodes == [icfg.node_addrs[i] for i in (3, 4, 5)]

    def test_visited_parents_are_icfg_indices(self):
        icfg = _icfg(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        h = build_horizon_graph(icfg, {"s": _mask(6, [0, 1, 2])})
        # node 3 is the horizon: unvisited, parent 2 visited.
        assert h.visited_parents == {0: {2}}

    def test_icfg_indexed_counts_are_rejected(self):
        """The mis-indexing that used to pass silently now raises."""
        icfg = _icfg(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        h = build_horizon_graph(icfg, {"s": _mask(6, [0, 1, 2])})
        with pytest.raises(ValueError, match="per-U-node"):
            katz_scores(h, hit_counts=np.zeros(icfg.n_nodes))

    def test_score_vector_matches_the_graph(self):
        icfg = _icfg(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        h = build_horizon_graph(icfg, {"s": _mask(6, [0, 1, 2])})
        r = katz_scores(h, beta=build_beta(h, np.zeros(icfg.n_nodes), 10.0))
        assert len(r.scores) == h.n_u + len(h.seed_names)
        assert r.n_u == h.n_u


class TestBuildBeta:
    def test_beta_reads_visited_parents_not_the_node_itself(self):
        """A horizon node's own count is 0; its parent's count is what
        distinguishes a well-trodden approach from an untouched one."""
        icfg = _icfg(5, [(0, 2), (1, 3), (2, 4), (3, 4)])
        # 0 and 1 visited; 2 and 3 are horizons with parents 0 and 1.
        h = build_horizon_graph(icfg, {"s": _mask(5, [0, 1])})
        hits = np.zeros(5)
        hits[0] = 900.0  # parent of node 2: hammered
        hits[1] = 10.0  # parent of node 3: barely touched
        beta = build_beta(h, hits, 1000.0)
        i2 = h.u_icfg_index.index(2)
        i3 = h.u_icfg_index.index(3)
        assert beta[i2] == pytest.approx(0.1)
        assert beta[i3] == pytest.approx(0.99)
        assert beta[i3] > beta[i2]

    def test_total_is_executions_not_the_node_count_sum(self):
        """T is the execution count. Using the sum of per-node counts makes
        beta depend on how long a trace is, so covering more of the program
        without running it more times flattens the term."""
        icfg = _icfg(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        h = build_horizon_graph(icfg, {"s": _mask(4, [0])})
        execs = 100.0
        hits = np.array([50.0, 0.0, 0.0, 0.0])
        beta = build_beta(h, hits, execs)
        # Nodes 1 and 2 are horizons off visited node 0: R = 50, T = 100.
        i1 = h.u_icfg_index.index(1)
        assert beta[i1] == pytest.approx(0.5)

        # Same 100 executions, same 50 reaching node 0's children, but the
        # traces are longer. The per-node sum grows; beta must not move.
        longer = hits.copy()
        longer[3] = 4000.0
        assert build_beta(h, longer, execs)[i1] == pytest.approx(0.5)
        # Whereas T = hits.sum() would have moved it to nearly 1.
        assert 1.0 - longer[0] / longer.sum() > 0.98

    def test_beta_is_bounded_and_u_length(self):
        icfg = _icfg(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        h = build_horizon_graph(icfg, {"s": _mask(5, [0, 1])})
        beta = build_beta(h, np.array([1e9, 1e9, 0.0, 0.0, 0.0]), 10.0)
        assert beta.shape == (h.n_u,)
        assert beta.min() >= 0.0 and beta.max() <= 1.0

    def test_no_executions_yet_is_uniform(self):
        icfg = _icfg(4, [(0, 1), (1, 2), (2, 3)])
        h = build_horizon_graph(icfg, {"s": _mask(4, [0])})
        assert build_beta(h, np.zeros(4), 0.0) == pytest.approx(np.ones(h.n_u))

    def test_untouched_nodes_keep_the_full_baseline(self):
        icfg = _icfg(4, [(0, 1), (1, 2), (2, 3)])
        h = build_horizon_graph(icfg, {"s": _mask(4, [0])})
        beta = build_beta(h, np.array([0.0, 0.0, 0.0, 0.0]), 500.0)
        assert beta == pytest.approx(np.ones(h.n_u))

    def test_beta_changes_the_ranking(self):
        """The point of the whole term: two seeds whose horizons differ only
        in how hard their approaches were hammered must not tie."""
        icfg = _icfg(5, [(0, 2), (1, 3), (2, 4), (3, 4)])
        h = build_horizon_graph(icfg, {"A": _mask(5, [0]), "B": _mask(5, [1])})
        hits = np.zeros(5)
        hits[0] = 950.0
        hits[1] = 5.0
        weighted = katz_scores(h, beta=build_beta(h, hits, 1000.0))
        uniform = katz_scores(h)
        assert weighted.seed_score("B") > weighted.seed_score("A")
        assert uniform.seed_score("A") == pytest.approx(uniform.seed_score("B"))


class TestSeedScoreAccessor:
    def test_seed_score_indexes_from_n_u(self):
        icfg = _icfg(5, [(0, 2), (1, 3), (2, 4), (3, 4)])
        h = build_horizon_graph(icfg, {"A": _mask(5, [0]), "B": _mask(5, [1])})
        r = katz_scores(h)
        for i, name in enumerate(r.seed_names):
            assert r.seed_score(name) == pytest.approx(r.scores[h.n_u + i])
        assert all(r.seed_score(n) > 0 for n in r.seed_names)


class TestConvergenceFlag:
    def test_short_cap_reports_truncation(self):
        g = HorizonGraph(
            u_nodes=list(range(40)),
            src=np.arange(39, dtype=np.int64),
            dst=np.arange(1, 40, dtype=np.int64),
            horizon_set=set(),
            seed_names=[],
            seed_edges={},
        )
        assert katz_scores(g, max_iter=3).converged is False
        assert katz_scores(g, max_iter=500).converged is True


class TestChannelIntegration:
    """The boundary where all three defects lived: the channel holds
    ICFG-indexed counts, the scorer wants U-indexed beta."""

    def _channel(self, n=8):
        from fuzzer_tool.services.katz_channel import KatzChannel

        icfg = _icfg(n, [(i, i + 1) for i in range(n - 1)])
        return KatzChannel(icfg, {k: k for k in range(n)})

    def test_scores_are_sized_to_the_graph(self):
        ch = self._channel()
        for _ in range(5):
            ch.record(np.array([True] * 3 + [False] * 5), seed_key="s1")
        res = ch.ensure_scores(force=True)
        assert len(res.scores) == ch._horizon.n_u + len(ch._horizon.seed_names)
        assert res.n_u == ch._horizon.n_u

    def test_seed_energy_is_in_range(self):
        ch = self._channel()
        ch.record(np.array([True] * 3 + [False] * 5), seed_key="s1")
        ch.record(np.array([True] * 2 + [False] * 6), seed_key="s2")
        for key in ("s1", "s2"):
            assert 0.0 <= ch.seed_energy(key) <= 1.0
        assert ch.seed_energy("never-seen") == 0.0

    def test_beta_is_not_flat(self):
        """Distinct approach frequencies must survive to the beta vector."""
        ch = self._channel()
        hot = np.array([True, True, True] + [False] * 5)
        for _ in range(200):
            ch.record(hot, seed_key="hot")
        ch.record(np.array([True] + [False] * 7), seed_key="cold")
        ch.ensure_scores(force=True)
        beta = build_beta(ch._horizon, ch.hit_counts, float(ch.exec_count))
        assert beta.shape == (ch._horizon.n_u,)
        assert beta.max() - beta.min() > 0.1

    def test_expensive_recompute_is_rate_limited(self):
        """The exec gate alone let a costly recompute run every 50 execs."""
        ch = self._channel()
        ch.record(np.array([True] * 3 + [False] * 5), seed_key="s1")
        ch.ensure_scores(force=True)
        ch._last_cost = 10.0  # pretend the last rebuild took 10s
        ch._dirty = True
        ch.exec_count += 10_000  # exec gate wide open
        before = ch._last_recompute_exec
        ch.ensure_scores()
        assert ch._last_recompute_exec == before  # cost gate held it back

        ch._last_cost = 1e-9  # cheap rebuild: gate opens again
        ch.ensure_scores()
        assert ch._last_recompute_exec > before
