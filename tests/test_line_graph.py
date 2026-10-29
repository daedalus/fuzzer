"""core/line_graph.LineGraph and its InterproceduralCFG wiring."""

from collections import deque

import numpy as np
import pytest

from fuzzer_tool.core.icfg import InterproceduralCFG
from fuzzer_tool.core.line_graph import UNREACHED, LineGraph
from fuzzer_tool.core.rand_pool import RandPool

NUM_RANDOM_CASES = 60
MAX_NODES = 8
MAX_EDGES = 16
HUB = 1000


def _arr(xs) -> np.ndarray:
    return np.array(xs, dtype=np.int64)


def _brute_arcs(src, dst) -> set[tuple[int, int]]:
    """L(G) by definition: e -> f when head(e) == tail(f)."""
    return {(i, j) for i in range(len(src)) for j in range(len(src)) if dst[i] == src[j]}


def _brute_dist(src, dst, targets) -> list[int]:
    """Hops in L(G) to the nearest target edge: BFS over the explicit arc set."""
    pred: dict[int, list[int]] = {}
    for a, b in _brute_arcs(src, dst):
        pred.setdefault(b, []).append(a)
    dist = [UNREACHED] * len(src)
    q = deque()
    for t in targets:
        if dist[t] == UNREACHED:
            dist[t] = 0
            q.append(t)
    while q:
        f = q.popleft()
        for e in pred.get(f, []):
            if dist[e] == UNREACHED:
                dist[e] = dist[f] + 1
                q.append(e)
    return dist


def _floyd_dist(n, src, dst, targets) -> list[int]:
    """Independent oracle: node APSP, then d(e) = 1 + d(head(e), tail(t))."""
    inf = 1 << 30
    d = [[0 if i == j else inf for j in range(n)] for i in range(n)]
    for a, b in zip(src, dst, strict=True):
        d[a][b] = min(d[a][b], 1)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    out = []
    for e in range(len(src)):
        best = 0 if e in targets else min((1 + d[dst[e]][src[t]] for t in targets), default=inf)
        out.append(UNREACHED if best >= inf else best)
    return out


def _random_graph(rng: RandPool) -> tuple[int, list[int], list[int]]:
    n = rng.randint(1, MAX_NODES)
    m = rng.randint(0, MAX_EDGES)
    return n, [rng.randint(0, n - 1) for _ in range(m)], [rng.randint(0, n - 1) for _ in range(m)]


def test_oracle_control_bfs_matches_floyd():
    # The two oracles share no code: they must agree before either judges LineGraph.
    rng = RandPool(seed=11)
    for _ in range(NUM_RANDOM_CASES):
        n, src, dst = _random_graph(rng)
        if not src:
            continue
        targets = {rng.randint(0, len(src) - 1)}
        assert _brute_dist(src, dst, targets) == _floyd_dist(n, src, dst, targets)


def test_successors_are_out_edges_of_head():
    # 0 -> 1 -> 2, 1 -> 3: edge 0 (0->1) feeds edges 1 and 2.
    lg = LineGraph(4, _arr([0, 1, 1]), _arr([1, 2, 3]))
    assert sorted(lg.successors(0).tolist()) == [1, 2]
    assert lg.successors(1).size == 0


def test_arcs_match_definition_on_random():
    rng = RandPool(seed=1234)
    for _ in range(NUM_RANDOM_CASES):
        n, src, dst = _random_graph(rng)
        lg = LineGraph(n, _arr(src), _arr(dst))
        a, b = lg.arcs()
        got = set(zip(a.tolist(), b.tolist(), strict=True))
        assert got == _brute_arcs(src, dst)
        assert lg.n_arcs == len(got) == a.size


def test_distances_match_bfs_on_random():
    rng = RandPool(seed=4321)
    for _ in range(NUM_RANDOM_CASES):
        n, src, dst = _random_graph(rng)
        if not src:
            continue
        targets = sorted({rng.randint(0, len(src) - 1) for _ in range(rng.randint(1, 3))})
        got = LineGraph(n, _arr(src), _arr(dst)).distances(_arr(targets)).tolist()
        assert got == _brute_dist(src, dst, set(targets))


def test_branch_target_is_not_block_target():
    # Diamond 0->1, 0->2, 1->3, 2->3. Target edge 1->3 (id 2): edge 0->2
    # reaches block 3 but never takes branch 1->3.
    lg = LineGraph(4, _arr([0, 0, 1, 2]), _arr([1, 2, 3, 3]))
    assert lg.distances(_arr([2])).tolist() == [1, UNREACHED, 0, UNREACHED]


def test_frontier_ranks_by_distance():
    # Visited {0}; frontier edges 0->1 (d 1 to target 1->3) and 0->2 (unreached).
    lg = LineGraph(4, _arr([0, 0, 1, 2]), _arr([1, 2, 3, 3]))
    visited = np.array([True, False, False, False])
    edges, dist = lg.frontier(visited, _arr([2]))
    assert edges.tolist() == [0, 1]
    assert dist.tolist() == [1, UNREACHED]


# --- adversarial ---------------------------------------------------------


def test_empty_graph():
    lg = LineGraph(3, _arr([]), _arr([]))
    assert lg.n_arcs == 0
    assert lg.distances(_arr([])).size == 0
    a, b = lg.arcs()
    assert a.size == b.size == 0


def test_self_loop_is_its_own_successor():
    lg = LineGraph(1, _arr([0]), _arr([0]))
    assert lg.successors(0).tolist() == [0]
    assert lg.distances(_arr([0])).tolist() == [0]


def test_parallel_edges_distinct():
    lg = LineGraph(2, _arr([0, 0]), _arr([1, 1]))
    assert lg.n_arcs == 0
    assert lg.distances(_arr([1])).tolist() == [UNREACHED, 0]


def test_hub_counts_arcs_without_materializing():
    # HUB in-edges and HUB out-edges on node 0: HUB^2 arcs.
    src = _arr(list(range(1, HUB + 1)) + [0] * HUB)
    dst = _arr([0] * HUB + list(range(HUB + 1, 2 * HUB + 1)))
    lg = LineGraph(2 * HUB + 1, src, dst)
    assert lg.n_arcs == HUB * HUB
    d = lg.distances(_arr([HUB]))  # first out-edge of the hub
    assert (d[:HUB] == 1).all() and d[HUB] == 0 and (d[HUB + 1 :] == UNREACHED).all()


def test_target_out_of_range_raises():
    lg = LineGraph(2, _arr([0]), _arr([1]))
    with pytest.raises(ValueError):
        lg.distances(_arr([1]))
    with pytest.raises(ValueError):
        lg.distances(_arr([-1]))


def test_node_out_of_range_raises():
    with pytest.raises(ValueError):
        LineGraph(1, _arr([0]), _arr([1]))


# --- ICFG wiring -----------------------------------------------------------

ADDRS = [0x10, 0x20, 0x30, 0x40]


def _icfg() -> InterproceduralCFG:
    return InterproceduralCFG(ADDRS, ["f"] * 4, _arr([0, 0, 1, 2]), _arr([1, 2, 3, 3]), None)


def test_icfg_edge_distances_by_address():
    out = _icfg().edge_distances({(0x20, 0x40)})
    assert out == {(0x10, 0x20): 1, (0x20, 0x40): 0}


def test_icfg_frontier_edges_ranked():
    out = _icfg().frontier_edges({0x10}, {(0x20, 0x40)})
    assert out == [((0x10, 0x20), 1), ((0x10, 0x30), UNREACHED)]


def test_icfg_unknown_target_ignored():
    assert _icfg().edge_distances({(0x10, 0x40)}) == {}
    assert _icfg().edge_pair_space() == 2
