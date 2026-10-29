"""core/mincut.min_vertex_cut (vertex Menger) and InterproceduralCFG.bottleneck_blocks."""

from collections import deque
from itertools import combinations

import numpy as np

from fuzzer_tool.core.icfg import InterproceduralCFG
from fuzzer_tool.core.mincut import UNSEPARABLE, min_cut, min_vertex_cut
from fuzzer_tool.core.rand_pool import RandPool

NUM_RANDOM_CASES = 80
MAX_NODES = 7
MAX_EDGES = 14
CHAIN_LEN = 2000


def _reaches(n, edges, sources, sinks, removed) -> bool:
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
    seen = set(sources)
    q = deque(sources)
    while q:
        u = q.popleft()
        if u in sinks:
            return True
        for v in adj.get(u, []):
            if v not in seen and v not in removed:
                seen.add(v)
                q.append(v)
    return False


def _brute_vertex_cut(n, edges, sources, sinks) -> int:
    """Fewest non-terminal nodes whose removal disconnects; UNSEPARABLE if none does."""
    inner = [v for v in range(n) if v not in sources and v not in sinks]
    for k in range(len(inner) + 1):
        for cut in combinations(inner, k):
            if not _reaches(n, edges, sources, sinks, set(cut)):
                return k
    return UNSEPARABLE


def test_oracle_control_on_hand_derived_instances():
    # Diamond 0->{1,2}->3: both arms must go, so 2; a chain needs 1.
    diamond = [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert _brute_vertex_cut(4, diamond, {0}, {3}) == 2
    assert _brute_vertex_cut(3, [(0, 1), (1, 2)], {0}, {2}) == 1
    assert _brute_vertex_cut(2, [(0, 1)], {0}, {1}) == UNSEPARABLE


def test_funnel_block_is_the_cut():
    # 0->{1,2}->3->4: block 3 gates both arms. Edge cut is also 1 here.
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    assert min_vertex_cut(5, edges, {0}, {4}) == (1, {3})


def test_vertex_cut_smaller_than_edge_cut():
    # 0 -> {1,2,3} -> 4 -> {5,6,7} -> 8: every edge layer is 3 wide, block 4 is 1.
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)]
    edges += [(4, 5), (4, 6), (4, 7), (5, 8), (6, 8), (7, 8)]
    assert min_vertex_cut(9, edges, {0}, {8}) == (1, {4})
    assert min_cut(9, edges, {0}, {8})[0] == 3


def test_parallel_edges_do_not_raise_vertex_cut():
    flow, cut = min_vertex_cut(3, [(0, 1), (0, 1), (1, 2), (1, 2)], {0}, {2})
    assert (flow, cut) == (1, {1})


def test_matches_brute_force_on_random():
    rng = RandPool(seed=1234)
    for _ in range(NUM_RANDOM_CASES):
        n = rng.randint(2, MAX_NODES)
        edges = [
            (rng.randint(0, n - 1), rng.randint(0, n - 1)) for _ in range(rng.randint(0, MAX_EDGES))
        ]
        s = rng.randint(0, n - 1)
        t = rng.randint(0, n - 1)
        if s == t:
            continue
        flow, cut = min_vertex_cut(n, edges, {s}, {t})
        assert flow == _brute_vertex_cut(n, edges, {s}, {t})
        if flow != UNSEPARABLE:
            assert len(cut) == flow and not _reaches(n, edges, {s}, {t}, cut)


# --- adversarial ---------------------------------------------------------


def test_adjacent_source_and_sink_unseparable():
    assert min_vertex_cut(3, [(0, 1), (0, 2), (2, 1)], {0}, {1}) == (UNSEPARABLE, set())


def test_unreachable_sink_needs_no_cut():
    assert min_vertex_cut(3, [(0, 1)], {0}, {2}) == (0, set())


def test_terminals_never_cut():
    # Source 1 and sink 3 sit on the only path through 2.
    flow, cut = min_vertex_cut(5, [(0, 1), (1, 2), (2, 3), (3, 4)], {0, 1}, {3, 4})
    assert (flow, cut) == (1, {2})


def test_overlap_raises():
    import pytest

    with pytest.raises(ValueError):
        min_vertex_cut(2, [(0, 1)], {0}, {0, 1})


def test_long_chain():
    edges = [(i, i + 1) for i in range(CHAIN_LEN)]
    flow, cut = min_vertex_cut(CHAIN_LEN + 1, edges, {0}, {CHAIN_LEN})
    assert flow == 1 and len(cut) == 1


# --- ICFG wiring -----------------------------------------------------------


def _icfg(edges: list[tuple[int, int]]) -> InterproceduralCFG:
    addrs = sorted({a for e in edges for a in e})
    idx = {a: i for i, a in enumerate(addrs)}
    src = np.array([idx[a] for a, _ in edges], dtype=np.int64)
    dst = np.array([idx[b] for _, b in edges], dtype=np.int64)
    return InterproceduralCFG(addrs, ["f"] * len(addrs), src, dst, cfgs={})


ICFG = _icfg([(0x10, 0x20), (0x10, 0x30), (0x20, 0x40), (0x30, 0x40), (0x40, 0x50), (0x50, 0x60)])


def test_icfg_bottleneck_blocks_by_address():
    assert ICFG.bottleneck_blocks(hit_addrs={0x10}, target_addrs={0x60}) in ({0x40}, {0x50})


def test_icfg_bottleneck_blocks_adjacent_is_empty():
    assert ICFG.bottleneck_blocks(hit_addrs={0x50}, target_addrs={0x60}) == set()


def test_icfg_bottleneck_blocks_unmapped_ignored():
    assert ICFG.bottleneck_blocks(hit_addrs={0x10}, target_addrs={0xDEAD}) == set()
