"""Tests for directed out-closeness (core/centrality.py).

Wasserman-Faust form: C(v) = ((r-1)/(n-1)) * ((r-1)/sum_d), r = nodes
reachable from v including v, sum_d = sum of BFS distances to them.
Expected values hand-derived from that formula.
"""

import pytest

from fuzzer_tool.core.centrality import betweenness_centrality, closeness_centrality
from fuzzer_tool.core.icfg import InterproceduralCFG
from tests.test_icfg_centrality import _make_icfg


def test_directed_path():
    # 0 -> 1 -> 2 -> 3
    # v0: r-1=3, sum=6 -> 1 * 1/2;  v1: r-1=2, sum=3 -> 2/3 * 2/3
    # v2: r-1=1, sum=1 -> 1/3;      v3: nothing reachable -> 0
    c = closeness_centrality(4, [(0, 1), (1, 2), (2, 3)])
    assert c == pytest.approx([0.5, 4 / 9, 1 / 3, 0.0])


def test_empty_and_single():
    assert closeness_centrality(0, []) == []
    assert closeness_centrality(1, []) == [0.0]


def test_cycle_is_uniform():
    """Adversarial: a directed 3-cycle; every node reaches 2 at sum 3."""
    c = closeness_centrality(3, [(0, 1), (1, 2), (2, 0)])
    assert c == pytest.approx([2 / 3] * 3)


def test_parallel_edges_and_self_loops_ignored():
    """Adversarial: multigraph noise must not change distances."""
    base = closeness_centrality(4, [(0, 1), (1, 2), (2, 3)])
    noisy = closeness_centrality(4, [(0, 0), (0, 1), (0, 1), (1, 2), (2, 2), (2, 3)])
    assert noisy == pytest.approx(base)


def test_disconnected_penalized():
    """Falsification: a node reaching 1 of 3 others at d=1 must score
    below one reaching all 3, although its raw 1/mean_d is higher."""
    # 0 -> 1, 0 -> 2, 0 -> 3 (hub); 4 -> 5 isolated pair
    c = closeness_centrality(6, [(0, 1), (0, 2), (0, 3), (4, 5)])
    assert c[0] == pytest.approx((3 / 5) * 1.0)
    assert c[4] == pytest.approx((1 / 5) * 1.0)
    assert c[4] < c[0]


def test_betweenness_unchanged_by_shared_bfs():
    """The BFS extraction must not move betweenness (test_centrality.py)."""
    assert betweenness_centrality(4, [(0, 1), (1, 2), (2, 3)], normalized=False) == [
        0.0,
        2.0,
        2.0,
        0.0,
    ]


class TestICFGCloseness:
    ICFG: InterproceduralCFG = _make_icfg(
        [(0x10, 0x20), (0x10, 0x30), (0x20, 0x40), (0x30, 0x40), (0x40, 0x50)]
    )

    def test_scores(self):
        # n=5. 0x10: r-1=4, sum 1+1+2+3=7 -> 4/7. 0x20/0x30: r-1=2, sum 3
        # -> 2/4 * 2/3. 0x40: r-1=1, sum 1 -> 1/4. 0x50: 0.
        s = self.ICFG.closeness_scores()
        assert s[0x10] == pytest.approx(4 / 7)
        assert s[0x20] == pytest.approx(1 / 3)
        assert s[0x30] == pytest.approx(1 / 3)
        assert s[0x40] == pytest.approx(1 / 4)
        assert s[0x50] == 0.0
