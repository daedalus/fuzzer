"""Tests for InterproceduralCFG.bottleneck_edges() (core/icfg.py).

Builds an InterproceduralCFG directly (node_addrs/src/dst arrays), the
same hand-built-structure style already used for HorizonGraph in
tests/test_horizon.py -- no ELF or disassembly needed since this is a
pure graph-structure question one level up from tests/test_mincut.py.
"""

import numpy as np

from fuzzer_tool.core.icfg import InterproceduralCFG


def _make_icfg(edges: list[tuple[int, int]]) -> InterproceduralCFG:
    addrs = sorted({a for e in edges for a in e})
    idx = {a: i for i, a in enumerate(addrs)}
    src = np.array([idx[a] for a, _ in edges], dtype=np.int64)
    dst = np.array([idx[b] for _, b in edges], dtype=np.int64)
    return InterproceduralCFG(addrs, ["f"] * len(addrs), src, dst, cfgs={})


class TestBottleneckEdges:
    """Diamond funnelling through a single downstream edge, addressed at
    the block-address level rather than dense node indices -- exercises
    the node_index translation in both directions."""

    ICFG = _make_icfg([(0x10, 0x20), (0x10, 0x30), (0x20, 0x40), (0x30, 0x40), (0x40, 0x50)])

    def test_bottleneck_is_the_shared_edge(self):
        cut = self.ICFG.bottleneck_edges(hit_addrs={0x10}, target_addrs={0x50})
        assert cut == {(0x40, 0x50)}

    def test_unmapped_addresses_are_ignored(self):
        # 0xdead isn't a node in this ICFG at all -- must not raise or
        # otherwise disrupt resolution of the real address alongside it.
        cut = self.ICFG.bottleneck_edges(
            hit_addrs={0x10, 0xDEAD}, target_addrs={0x50}
        )
        assert cut == {(0x40, 0x50)}

    def test_empty_when_target_unmapped(self):
        assert self.ICFG.bottleneck_edges(hit_addrs={0x10}, target_addrs={0xDEAD}) == set()

    def test_empty_when_hit_and_target_are_the_same_node(self):
        # After the overlap is subtracted there's nothing left to
        # separate a node from itself.
        assert self.ICFG.bottleneck_edges(hit_addrs={0x50}, target_addrs={0x50}) == set()
