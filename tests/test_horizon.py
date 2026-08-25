"""Tests for the K-Scheduler edge-horizon graph (W3).

Paper semantics (She et al., S&P'22 §4): delete visited nodes while
preserving connectivity, turn the unvisited remainder into a DAG by
dropping intra-SCC edges, then attach one node per seed with edges to
the horizon nodes whose visited parent lies on that seed's path.

Oracles are derived by hand on tiny graphs; nothing echoes production
code. Graphs are constructed directly as ICFG arrays — no disassembly,
no compiler.
"""

import numpy as np

from fuzzer_tool.core.horizon import build_horizon_graph


def _icfg(n, edges):
    from fuzzer_tool.core.icfg import InterproceduralCFG

    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    addrs = [0x1000 * i for i in range(n)]
    funcs = [f"f{i}" for i in range(n)]
    return InterproceduralCFG(addrs, funcs, src, dst, {})


def _mask(n, visited):
    m = bytearray(n)
    for i in visited:
        m[i >> 3] |= 1 << (i & 7)
    return bytes(m)


def _is_dag(src, dst, n):
    """Kahn's algorithm over the full edge list."""
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for s, d in zip(src.tolist(), dst.tolist(), strict=False):
        adj[s].append(d)
        indeg[d] += 1
    queue = [v for v in range(n) if indeg[v] == 0]
    seen = 0
    while queue:
        v = queue.pop()
        seen += 1
        for w in adj[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    return seen == n


class TestHorizonIdentification:
    def test_chain_horizon_is_first_unvisited_child(self):
        """0→1→2→3, visit {0,1}: horizon = {2}."""
        g = _icfg(4, [(0, 1), (1, 2), (2, 3)])
        h = build_horizon_graph(g, {"s": _mask(4, {0, 1})})
        assert {icfg_addr for icfg_addr in h.u_nodes} == {0x2000, 0x3000}
        assert h.horizon_set == {0x2000}

    def test_empty_visits_gives_empty_horizon_and_no_crash(self):
        g = _icfg(3, [(0, 1), (1, 2)])
        h = build_horizon_graph(g, {"s": _mask(3, set())})
        assert h.horizon_set == set()
        assert h.n_seed_edges == 0


class TestConnectivityPreservingDeletion:
    def test_u_to_v_to_u_shortcut_exists(self):
        """u1→v→u2 with NO direct u1→u2 edge: deleting v must reconnect
        them or downstream reachability silently collapses."""
        g = _icfg(3, [(0, 1), (1, 2)])
        h = build_horizon_graph(g, {"s": _mask(3, {1})})
        addr_pairs = {
            (h.u_nodes[s], h.u_nodes[d])
            for s, d in zip(h.src.tolist(), h.dst.tolist(), strict=False)
        }
        assert addr_pairs == {(0x0000, 0x2000)}

    def test_multi_hop_visited_interior_contracts(self):
        """u→v1→v2→x: both interiors visited, x unvisited — one shortcut,
        not zero."""
        g = _icfg(4, [(0, 1), (1, 2), (2, 3)])
        h = build_horizon_graph(g, {"s": _mask(4, {1, 2})})
        addr_pairs = {
            (h.u_nodes[s], h.u_nodes[d])
            for s, d in zip(h.src.tolist(), h.dst.tolist(), strict=False)
        }
        assert addr_pairs == {(0x0000, 0x3000)}

    def test_fully_visited_graph_leaves_nothing(self):
        g = _icfg(3, [(0, 1), (1, 2)])
        h = build_horizon_graph(g, {"s": _mask(3, {0, 1, 2})})
        assert h.n_u == 0
        assert h.n_seed_edges == 0


class TestDAGConversion:
    def test_cycle_in_unvisited_subgraph_becomes_dag(self):
        """a→b→a plus exit c: the SCC's internal edges are dropped, the
        exit edge survives — this is what makes Katz safe at any alpha."""
        g = _icfg(3, [(0, 1), (1, 0), (1, 2)])
        h = build_horizon_graph(g, {"s": _mask(3, set())})
        assert _is_dag(h.src, h.dst, h.n_u)

    def test_self_loop_dropped(self):
        g = _icfg(2, [(0, 0), (0, 1)])
        h = build_horizon_graph(g, {"s": _mask(2, set())})
        pairs = set(zip(h.src.tolist(), h.dst.tolist(), strict=False))
        assert (0, 0) not in pairs
        assert (0, 1) in pairs


class TestSeedEdges:
    def test_seed_reaches_horizon_through_its_own_parent(self):
        """Two visited parents of the same horizon node: each seed gets its
        own edge to it. Graph 0→2, 1→2; a visits {0}, b visits {1}. Both
        parents are visited so U={node2} alone and every seed whose path
        holds ANY parent connects to it (index 0 in u_nodes)."""
        g = _icfg(3, [(0, 2), (1, 2)])
        h = build_horizon_graph(g, {"a": _mask(3, {0}), "b": _mask(3, {1})})
        assert h.horizon_set == {0x2000}
        by_seed = h.seed_edges_by_name()
        assert by_seed["a"] == {0}
        assert by_seed["b"] == {0}

    def test_attribution_follows_each_seeds_parents(self):
        """Disjoint parent sets on DIFFERENT horizon nodes: seed edges must
        split accordingly, not fan out globally."""
        # 0→2, 0→3, 1→3: seed a={0} reaches both horizons; seed b={1}
        # reaches only node 3 (its sole visited parent).
        g = _icfg(4, [(0, 2), (0, 3), (1, 3)])
        h = build_horizon_graph(g, {"a": _mask(4, {0}), "b": _mask(4, {1})})
        by_seed = h.seed_edges_by_name()
        assert h.horizon_set == {0x2000, 0x3000}
        assert by_seed["a"] == {0, 1}  # indices of nodes 2 and 3
        assert by_seed["b"] == {1}

    def test_seed_missing_parent_gets_no_edge(self):
        """Seed visited nothing relevant: contributes no edges."""
        g = _icfg(3, [(0, 1), (1, 2)])
        h = build_horizon_graph(g, {"hit": _mask(3, {0, 1}), "miss": _mask(3, set())})
        by_seed = h.seed_edges_by_name()
        assert by_seed["miss"] == set()
        assert by_seed["hit"] == {0}

    def test_duplicate_visits_are_idempotent(self):
        g = _icfg(4, [(0, 1), (1, 2), (2, 3)])
        h1 = build_horizon_graph(g, {"s": _mask(4, {0, 1})})
        h2 = build_horizon_graph(g, {"s": _mask(4, {0, 1}), "t": _mask(4, {0, 1})})
        assert (
            h1.seed_edges_by_name()["s"]
            == h2.seed_edges_by_name()["s"]
            == h2.seed_edges_by_name()["t"]
        )


class TestAdversarialShapes:
    def test_all_nodes_visited_with_dangling_edges(self):
        """Edges into a fully visited region must not resurrect anything."""
        g = _icfg(3, [(0, 1), (0, 2)])
        h = build_horizon_graph(g, {"s": _mask(3, {1, 2})})
        assert h.horizon_set == set()

    def test_index_round_trip(self):
        """u_nodes ↔ node_index must agree both ways."""
        g = _icfg(4, [(0, 1), (1, 2), (2, 3)])
        h = build_horizon_graph(g, {"s": _mask(4, {0})})
        for i, addr in enumerate(h.u_nodes):
            assert h.node_index[addr] == i
