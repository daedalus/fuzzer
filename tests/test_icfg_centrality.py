"""Tests for InterproceduralCFG.centrality_scores() (core/icfg.py).

Same hand-built-structure style as test_icfg_bottleneck.py -- reuses its
exact diamond-plus-tail graph (0x10 -> {0x20,0x30} -> 0x40 -> 0x50) so the
two wrappers are directly comparable, addressed at the block-address
level rather than dense node indices.

Hand-derivation (see test_centrality.py for the underlying algorithm
tests; this just re-derives the specific numbers for this 5-node graph):
reachable ordered pairs and their credit to intermediate nodes --
  (0x10,0x40): 2 shortest paths (via 0x20, via 0x30) -> 0.5 to each
  (0x10,0x50): 2 shortest paths, both through 0x40 -> 0.5 each to
    0x20/0x30, full 1.0 to 0x40 (present on *both* paths)
  (0x20,0x50): 1 path through 0x40 -> 1.0 to 0x40
  (0x30,0x50): 1 path through 0x40 -> 1.0 to 0x40
  (all direct-edge pairs contribute nothing -- no intermediate node)
Raw totals: 0x20 = 1.0, 0x30 = 1.0, 0x40 = 3.0, 0x10 = 0x50 = 0.0.
Normalized by (5-1)(5-2) = 12: 0x20 = 0x30 = 1/12, 0x40 = 3/12 = 0.25.
"""

import numpy as np

from fuzzer_tool.core.icfg import InterproceduralCFG


def _make_icfg(edges: list[tuple[int, int]]) -> InterproceduralCFG:
    addrs = sorted({a for e in edges for a in e})
    idx = {a: i for i, a in enumerate(addrs)}
    src = np.array([idx[a] for a, _ in edges], dtype=np.int64)
    dst = np.array([idx[b] for _, b in edges], dtype=np.int64)
    return InterproceduralCFG(addrs, ["f"] * len(addrs), src, dst, cfgs={})


class TestCentralityScores:
    ICFG = _make_icfg([(0x10, 0x20), (0x10, 0x30), (0x20, 0x40), (0x30, 0x40), (0x40, 0x50)])

    def test_raw_scores(self):
        scores = self.ICFG.centrality_scores(normalized=False)
        assert scores[0x10] == 0.0
        assert scores[0x20] == 1.0
        assert scores[0x30] == 1.0
        assert scores[0x40] == 3.0
        assert scores[0x50] == 0.0

    def test_normalized_scores(self):
        scores = self.ICFG.centrality_scores(normalized=True)
        assert scores[0x10] == 0.0
        assert abs(scores[0x20] - 1 / 12) < 1e-9
        assert abs(scores[0x30] - 1 / 12) < 1e-9
        assert abs(scores[0x40] - 0.25) < 1e-9
        assert scores[0x50] == 0.0

    def test_every_node_has_an_entry(self):
        # Unlike bottleneck_edges (which silently drops unmapped
        # addresses), centrality_scores needs no address set at all --
        # every node in the ICFG gets a score, including 0.0 ones.
        scores = self.ICFG.centrality_scores()
        assert set(scores) == {0x10, 0x20, 0x30, 0x40, 0x50}

    def test_bottleneck_and_centrality_agree_on_the_choke_point(self):
        # Cross-check against the sibling wrapper: the min-cut bottleneck
        # for hit={0x10} -> target={0x50} is the single edge (0x40,
        # 0x50), and 0x40 is exactly the node with the highest
        # betweenness score in this graph -- both signals independently
        # identify the same structural choke point.
        cut = self.ICFG.bottleneck_edges(hit_addrs={0x10}, target_addrs={0x50})
        assert cut == {(0x40, 0x50)}
        scores = self.ICFG.centrality_scores(normalized=False)
        assert max(scores, key=scores.get) == 0x40
