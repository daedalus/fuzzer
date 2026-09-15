"""Unit tests for core/mincut.py.

Every expected cut size/edge set below is hand-derived from the diagram
in each test's docstring via Menger's theorem (min edge cut == max number
of edge-disjoint paths), independent of the implementation under test.
"""

import pytest

from fuzzer_tool.core.mincut import min_cut


class TestSinglePath:
    """0 -> 1 -> 2 -> 3. Exactly one edge-disjoint path -> cut size 1,
    and it must be the edge leaving the source (BFS-residual reachability
    from the source stops at the first saturated edge)."""

    EDGES = [(0, 1), (1, 2), (2, 3)]

    def test_cut_size_and_edge(self):
        flow, cut = min_cut(4, self.EDGES, sources={0}, sinks={3})
        assert flow == 1
        assert cut == {(0, 1)}


class TestDiamondSharedBottleneck:
    """0 -> {1, 2} -> 3 -> 4. Both arms of the diamond funnel through the
    single edge (3, 4) before reaching the sink -- that's the only edge
    whose removal disconnects 0 from 4, regardless of which of the two
    parallel sub-paths (0-1-3, 0-2-3) carries flow."""

    EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]

    def test_cut_is_the_shared_edge(self):
        flow, cut = min_cut(5, self.EDGES, sources={0}, sinks={4})
        assert flow == 1
        assert cut == {(3, 4)}


class TestTwoEdgeDisjointPaths:
    """0 -> 1 -> 3 and 0 -> 2 -> 3: two fully independent paths, no shared
    edge anywhere -- Menger says the min cut must have size 2 (a size-1
    cut always leaves the other path intact)."""

    EDGES = [(0, 1), (1, 3), (0, 2), (2, 3)]

    def test_cut_size_two(self):
        flow, cut = min_cut(4, self.EDGES, sources={0}, sinks={3})
        assert flow == 2
        assert cut == {(0, 1), (0, 2)}


class TestParallelEdges:
    """Two parallel edges 0->1 (a real multigraph case a decoder could
    produce, e.g. a switch with two case labels branching to the same
    block) plus 1->2. Cutting 1->2 alone (size 1) suffices -- both
    parallel copies of 0->1 become irrelevant once the single downstream
    edge is gone -- so the min cut is 1, not 2."""

    EDGES = [(0, 1), (0, 1), (1, 2)]

    def test_downstream_bottleneck_beats_parallel_upstream(self):
        flow, cut = min_cut(3, self.EDGES, sources={0}, sinks={2})
        assert flow == 1
        assert cut == {(1, 2)}


class TestDisconnected:
    """0 -> 1, and 2 -> 3 entirely separately: 0 can never reach 3, so
    zero edges are needed to keep it that way."""

    EDGES = [(0, 1), (2, 3)]

    def test_zero_cut_when_already_disconnected(self):
        flow, cut = min_cut(4, self.EDGES, sources={0}, sinks={3})
        assert flow == 0
        assert cut == set()


class TestMultiSourceMultiSink:
    """Two independent chains sharing no nodes: 0->1->2 and 10->11->12.
    Sources={0,10}, sinks={2,12} -- the super-source/super-sink must not
    let flow "leak" between the two unrelated chains, and the total cut
    must be the union of each chain's own bottleneck."""

    EDGES = [(0, 1), (1, 2), (10, 11), (11, 12)]

    def test_union_of_independent_cuts(self):
        flow, cut = min_cut(13, self.EDGES, sources={0, 10}, sinks={2, 12})
        assert flow == 2
        assert cut == {(0, 1), (10, 11)}


class TestEmptySourcesOrSinks:
    def test_empty_sources(self):
        assert min_cut(2, [(0, 1)], sources=set(), sinks={1}) == (0, set())

    def test_empty_sinks(self):
        assert min_cut(2, [(0, 1)], sources={0}, sinks=set()) == (0, set())


class TestOverlapRejected:
    def test_raises_on_shared_node(self):
        with pytest.raises(ValueError):
            min_cut(2, [(0, 1)], sources={0, 1}, sinks={1})
