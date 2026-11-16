"""core/articulation.cut_damage: articulation pieces and cut damage vs brute-force removal."""

import numpy as np
import pytest

from fuzzer_tool.core.articulation import cut_damage
from fuzzer_tool.core.rand_pool import RandPool

NUM_RANDOM_CASES = 200

# Deep enough that a recursive DFS would blow Python's default limit (1000).
DEEP_CHAIN = 50_000


def _arrays(edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    src = np.array([u for u, _ in edges], dtype=np.int64)
    dst = np.array([v for _, v in edges], dtype=np.int64)
    return src, dst


def _parts(n: int, edges, removed: int | None) -> list[set[int]]:
    """Undirected components of the graph minus *removed*, by repeated flood."""
    adj: list[set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    seen = {removed} if removed is not None else set()
    parts = []
    for s in range(n):
        if s in seen:
            continue
        part, todo = {s}, [s]
        seen.add(s)
        while todo:
            x = todo.pop()
            for y in adj[x] - seen:
                seen.add(y)
                part.add(y)
                todo.append(y)
        parts.append(part)
    return parts


def _brute(n: int, edges, entry: int) -> tuple[list[int], list[int]]:
    """(pieces, damage) by deleting each node and flooding what is left."""
    whole = _parts(n, edges, None)
    comp_of = {v: i for i, p in enumerate(whole) for v in p}
    pieces, damage = [], []
    for v in range(n):
        comp = whole[comp_of[v]]
        split = [p for p in _parts(n, edges, v) if p <= comp]
        pieces.append(len(split))
        if entry not in comp:
            damage.append(0)
            continue
        cut = [len(p) for p in split if entry not in p]
        damage.append(max(cut, default=0))
    return pieces, damage


def _random_graph(rng: RandPool) -> tuple[int, list[tuple[int, int]]]:
    n = rng.randint(1, 9)
    m = rng.randint(0, 2 * n)
    return n, [(rng.randint(0, n - 1), rng.randint(0, n - 1)) for _ in range(m)]


def test_matches_brute_force_on_random():
    rng = RandPool(seed=10)
    for _ in range(NUM_RANDOM_CASES):
        n, edges = _random_graph(rng)
        entry = rng.randint(0, n - 1)
        pieces, damage = cut_damage(n, *_arrays(edges), entry)
        want_p, want_d = _brute(n, edges, entry)
        assert pieces.tolist() == want_p, (n, edges, entry)
        assert damage.tolist() == want_d, (n, edges, entry)


def test_brute_force_oracle_agrees_with_itself_on_relabel():
    # A/A control (Hard Rule 46): the oracle must be invariant under node
    # relabelling, or comparing against it would judge the oracle, not the code.
    rng = RandPool(seed=11)
    for _ in range(NUM_RANDOM_CASES // 4):
        n, edges = _random_graph(rng)
        entry = rng.randint(0, n - 1)
        perm = list(range(n))[::-1]
        moved = [(perm[u], perm[v]) for u, v in edges]
        p0, d0 = _brute(n, edges, entry)
        p1, d1 = _brute(n, moved, perm[entry])
        assert [p1[perm[v]] for v in range(n)] == p0
        assert [d1[perm[v]] for v in range(n)] == d0


def test_chain_hand_computed():
    # 0-1-2-3, entry 0: cutting 1 loses {2,3}, cutting 2 loses {3}.
    pieces, damage = cut_damage(4, *_arrays([(0, 1), (1, 2), (2, 3)]), 0)
    assert pieces.tolist() == [1, 2, 2, 1]
    assert damage.tolist() == [3, 2, 1, 0]


def test_falsification_cycle_has_no_cut():
    # A ring: no single block separates anything from the entry.
    ring = [(i, (i + 1) % 5) for i in range(5)]
    pieces, damage = cut_damage(5, *_arrays(ring), 0)
    assert pieces.tolist() == [1] * 5
    assert damage.tolist() == [4, 0, 0, 0, 0]  # the entry itself loses the rest


def test_direction_is_ignored():
    # 1->0 and 1->2: undirected 0-1-2, so 1 still cuts 2 off from 0.
    _, damage = cut_damage(3, *_arrays([(1, 0), (1, 2)]), 0)
    assert damage.tolist() == [2, 1, 0]


# --- adversarial -----------------------------------------------------------


def test_self_loops_and_duplicates_ignored():
    edges = [(0, 0), (0, 1), (1, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
    pieces, damage = cut_damage(3, *_arrays(edges), 0)
    assert pieces.tolist() == [1, 2, 1]
    assert damage.tolist() == [2, 1, 0]


def test_back_edge_to_parent_twice_is_not_a_cycle():
    # u->v and v->u collapse to one undirected edge; a multigraph-naive
    # parent check would read the pair as a cycle and hide the cut at 1.
    pieces, damage = cut_damage(3, *_arrays([(0, 1), (1, 0), (1, 2), (2, 1)]), 0)
    assert pieces.tolist() == [1, 2, 1]
    assert damage.tolist() == [2, 1, 0]


def test_disconnected_components_score_no_damage():
    # {0,1} holds the entry; {2,3,4} is a separate chain.
    edges = [(0, 1), (2, 3), (3, 4)]
    pieces, damage = cut_damage(5, *_arrays(edges), 0)
    assert pieces.tolist() == [1, 1, 1, 2, 1]
    assert damage.tolist() == [1, 0, 0, 0, 0]


def test_single_node():
    pieces, damage = cut_damage(1, *_arrays([]), 0)
    assert pieces.tolist() == [0]
    assert damage.tolist() == [0]


def test_empty_graph():
    pieces, damage = cut_damage(0, *_arrays([]), 0)
    assert pieces.size == 0
    assert damage.size == 0


def test_entry_out_of_range_raises():
    with pytest.raises(ValueError):
        cut_damage(2, *_arrays([(0, 1)]), 2)


def test_deep_chain_no_recursion_limit():
    n = DEEP_CHAIN
    edges = [(i, i + 1) for i in range(n - 1)]
    pieces, damage = cut_damage(n, *_arrays(edges), 0)
    assert pieces[0] == 1 and pieces[n // 2] == 2
    # Cutting block i loses the n - 1 - i blocks after it.
    assert damage.tolist() == list(range(n - 1, -1, -1))


def test_icfg_method_maps_entry_address():
    from fuzzer_tool.core.icfg import InterproceduralCFG

    edges = [(0, 1), (1, 2), (2, 3)]
    icfg = InterproceduralCFG([0x100 * i for i in range(4)], ["f"] * 4, *_arrays(edges), {})
    # Entry 0x100 is node 1: cutting it strands {2, 3}; cutting 2 strands {3}.
    pieces, damage = icfg.cut_damage(0x100)
    assert damage.tolist() == [0, 2, 1, 0]
    assert pieces.tolist() == [1, 2, 2, 1]
    with pytest.raises(ValueError):
        icfg.cut_damage(0x180)  # not a block start
