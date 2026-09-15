"""Tests for Brandes' directed betweenness centrality (core/centrality.py).

Every expected score is hand-derived from the definition (sum over ordered
pairs (s, t) of the fraction of shortest s->t paths through v), not
asserted against the implementation's own output -- same discipline as
``test_mincut.py``'s Menger's-theorem-derived expectations.
"""

from fuzzer_tool.core.centrality import betweenness_centrality


class TestDirectedPath:
    """0 -> 1 -> 2 -> 3. Every reachable pair has exactly one shortest path.

    Ordered reachable pairs and their intermediate nodes:
      (0,1): none   (0,2): {1}   (0,3): {1,2}
      (1,2): none   (1,3): {2}
      (2,3): none
    Raw betweenness: node1 = 2 (from (0,2),(0,3)); node2 = 2 (from
    (0,3),(1,3)); node0 = node3 = 0.
    Normalized by (n-1)(n-2) = 3*2 = 6: node1 = node2 = 2/6 = 1/3.
    """

    EDGES = [(0, 1), (1, 2), (2, 3)]

    def test_raw(self):
        c = betweenness_centrality(4, self.EDGES, normalized=False)
        assert c == [0.0, 2.0, 2.0, 0.0]

    def test_normalized(self):
        c = betweenness_centrality(4, self.EDGES, normalized=True)
        assert c[0] == 0.0
        assert abs(c[1] - 1 / 3) < 1e-9
        assert abs(c[2] - 1 / 3) < 1e-9
        assert c[3] == 0.0


class TestDiamondSplitsCreditAcrossParallelShortestPaths:
    """0 -> {1,2} -> 3 (diamond). (0,3) has two equally-short paths.

    Ordered reachable pairs: (0,1),(0,2),(0,3),(1,3),(2,3) -- all direct
    except (0,3), which has two shortest paths of length 2 (via 1, via 2),
    splitting credit 0.5/0.5 between them.
    Raw betweenness: node1 = 0.5, node2 = 0.5, node0 = node3 = 0.
    Normalized by (4-1)(4-2) = 6: node1 = node2 = 0.5/6 = 1/12.
    """

    EDGES = [(0, 1), (0, 2), (1, 3), (2, 3)]

    def test_raw(self):
        c = betweenness_centrality(4, self.EDGES, normalized=False)
        assert c[0] == 0.0
        assert abs(c[1] - 0.5) < 1e-9
        assert abs(c[2] - 0.5) < 1e-9
        assert c[3] == 0.0

    def test_normalized(self):
        c = betweenness_centrality(4, self.EDGES, normalized=True)
        assert abs(c[1] - 1 / 12) < 1e-9
        assert abs(c[2] - 1 / 12) < 1e-9


class TestStarGraphHubGetsAllCredit:
    """Directed star: center 0 -> {1,2,3}. No path passes through a leaf.

    Only pairs (0,1),(0,2),(0,3) are reachable, all direct (no
    intermediate node). Every score is 0 -- there simply is no pair whose
    shortest path has an intermediate hop in a graph this shallow. This
    guards against an implementation bug that would (wrongly) credit the
    hub for its own outgoing edges.
    """

    EDGES = [(0, 1), (0, 2), (0, 3)]

    def test_all_zero(self):
        c = betweenness_centrality(4, self.EDGES, normalized=False)
        assert c == [0.0, 0.0, 0.0, 0.0]


class TestDisconnectedNodesContributeNothing:
    """0 -> 1 -> 2, plus an isolated node 3 with no edges at all.

    node3 is never on any shortest path (nothing reaches it, it reaches
    nothing) and must score exactly 0, with no crash from its BFS never
    discovering any other node.
    """

    EDGES = [(0, 1), (1, 2)]

    def test_isolated_node_scores_zero(self):
        c = betweenness_centrality(4, self.EDGES, normalized=False)
        assert c[3] == 0.0
        # node1 still gets credit for gating (0,2), same as TestDirectedPath.
        assert c[1] == 1.0


class TestEdgeCases:
    def test_empty_graph(self):
        assert betweenness_centrality(0, []) == []

    def test_single_node(self):
        # n <= 2: normalization is skipped (nothing to divide by), and
        # there are no ordered pairs at all, so the raw score is 0.
        assert betweenness_centrality(1, [], normalized=True) == [0.0]

    def test_two_nodes_no_normalization_division_by_zero(self):
        # (n-1)(n-2) = 1*0 = 0 for n=2 -- normalization must be skipped,
        # not attempted (which would divide by zero).
        c = betweenness_centrality(2, [(0, 1)], normalized=True)
        assert c == [0.0, 0.0]

    def test_self_loop_ignored_by_construction(self):
        # A self-loop never appears as an intermediate on any s!=t path;
        # it should simply not crash or distort other nodes' scores.
        c = betweenness_centrality(3, [(0, 0), (0, 1), (1, 2)], normalized=False)
        assert c[1] == 1.0
