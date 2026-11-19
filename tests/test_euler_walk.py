"""Euler-walk seeds over a ``format_fsm`` (Hierholzer + Chinese postman).

Oracles are independent of the module: ``FormatFsm.accepts`` for validity,
a brute-force Dijkstra over (state, covered-edge mask) for the minimum
single-walk length, and hand arithmetic for the postman duplicates.
"""

import heapq

import pytest

from fuzzer_tool.core.euler_walk import MAX_EDGES, euler_seeds, euler_walks
from fuzzer_tool.core.format_fsm import parse_fsm
from fuzzer_tool.core.rand_pool import RandPool

# (a|b)+c: already a path, no duplicates needed.
AB_C = """
start A
final D
A -> B : 'a' | 'b'
B -> B : 'a' | 'b'
B -> D : 'c'
"""

# B has in 2 / out 1, A in 1 / out 2: one 'z' must be walked twice.
POSTMAN = """
start A
final A
A -> B : "x"
A -> B : "yy"
B -> A : "z"
"""

# Two parallel A -> B transitions in a DAG: no single message holds both.
FORK = """
start A
final C
A -> B : "a"
A -> B : "b"
B -> C : "c"
"""


def _width(t) -> int:
    return t.label.width


def _brute_min(fsm) -> int | None:
    """Fewest bytes of one accepted walk using every live edge, or None."""
    live = [i for i, _ in enumerate(fsm.edges) if _is_live(fsm, i)]
    full = 0
    for i in live:
        full |= 1 << i
    heap = [(0, s, 0) for s in fsm.starts]
    seen = set()
    while heap:
        c, s, m = heapq.heappop(heap)
        if (s, m) in seen:
            continue
        seen.add((s, m))
        if m == full and s in fsm.finals:
            return c
        for i, t in enumerate(fsm.edges):
            if t.src == s and i in live:
                heapq.heappush(heap, (c + _width(t), t.dst, m | (1 << i)))
    return None


def _is_live(fsm, i) -> bool:
    """Edge i lies on some start -> final path (plain fixed points)."""
    fwd = set(fsm.starts)
    back = set(fsm.finals)
    for _ in fsm.edges:
        fwd |= {t.dst for t in fsm.edges if t.src in fwd}
        back |= {t.src for t in fsm.edges if t.dst in back}
    t = fsm.edges[i]
    return t.src in fwd and t.dst in back


def _covered(fsm, walks) -> bool:
    used = {id(t) for w in walks for t in w}
    return all(id(t) in used for i, t in enumerate(fsm.edges) if _is_live(fsm, i))


class TestOracle:
    def test_control_brute_on_eulerian_path(self):
        # Rule 46: the oracle must give sum of widths on a graph that needs no
        # duplicates, before it is trusted on one that does.
        assert _brute_min(parse_fsm(AB_C)) == 3

    def test_control_brute_postman(self):
        assert _brute_min(parse_fsm(POSTMAN)) == 5


class TestWalks:
    def test_path_single_walk(self):
        fsm = parse_fsm(AB_C)
        walks = euler_walks(fsm)
        assert len(walks) == 1
        assert [t.dst for t in walks[0]] == ["B", "B", "D"]

    def test_postman_duplicates_cheapest_edge(self):
        fsm = parse_fsm(POSTMAN)
        (walk,) = euler_walks(fsm)
        labels = sorted(t.label.value for t in walk)
        assert labels == [b"x", b"yy", b"z", b"z"]
        assert sum(map(_width, walk)) == _brute_min(fsm)

    def test_falsification_fork_needs_two_walks(self):
        fsm = parse_fsm(FORK)
        walks = euler_walks(fsm)
        assert len(walks) == 2
        assert _covered(fsm, walks)

    def test_walks_start_and_end_right(self):
        fsm = parse_fsm(FORK)
        for w in euler_walks(fsm):
            assert w[0].src in fsm.starts
            assert w[-1].dst in fsm.finals
            assert all(a.dst == b.src for a, b in zip(w, w[1:], strict=False))

    def test_matches_brute_force_on_random_fsms(self):
        rng = RandPool(seed=5)
        checked = 0
        for _ in range(60):
            fsm = _random_fsm(rng)
            if fsm is None:
                continue
            walks = euler_walks(fsm)
            assert _covered(fsm, walks)
            best = _brute_min(fsm)
            if best is None:
                assert len(walks) > 1
                continue
            assert len(walks) == 1
            assert sum(map(_width, walks[0])) == best
            checked += 1
        assert checked >= 10


def _random_fsm(rng):
    n = rng.randint(2, 4)
    lines = ["start S0", f"final S{n - 1}"]
    for _ in range(rng.randint(2, 6)):
        a, b = rng.randint(0, n - 1), rng.randint(0, n - 1)
        lines.append(f'S{a} -> S{b} : "{"x" * rng.randint(1, 3)}"')
    try:
        return parse_fsm("\n".join(lines))
    except ValueError:
        return None


class TestSeeds:
    def test_every_seed_accepted(self):
        for spec in (AB_C, POSTMAN, FORK):
            fsm = parse_fsm(spec)
            for seed in euler_seeds(fsm, RandPool(seed=1)):
                assert fsm.accepts(seed)

    def test_postman_seed_exact(self):
        fsm = parse_fsm(POSTMAN)
        (seed,) = euler_seeds(fsm, RandPool(seed=1))
        assert seed in (b"xzyyz", b"yyzxz")


class TestAdversarial:
    def test_dead_and_unreachable_edges_skipped(self):
        fsm = parse_fsm(AB_C + "X -> B : 'q'\nB -> Y : 'r'\n")
        (walk,) = euler_walks(fsm)
        assert {t.src for t in walk} == {"A", "B"}
        assert all(t.dst in ("B", "D") for t in walk)

    def test_no_live_edges_no_walks(self):
        fsm = parse_fsm("start A\nfinal A\nB -> A : 'x'\n")
        assert euler_walks(fsm) == []
        assert euler_seeds(fsm, RandPool(seed=1)) == []

    def test_final_mid_walk_continues(self):
        # A is final and has a loop: the loop must still be walked.
        fsm = parse_fsm("start A\nfinal A\nA -> A : 'x'\nA -> A : 'y'\n")
        (seed,) = euler_seeds(fsm, RandPool(seed=1))
        assert sorted(seed) == sorted(b"xy")

    def test_edge_cap(self):
        lines = ["start A", "final A"] + ["A -> A : 'x'"] * (MAX_EDGES + 1)
        with pytest.raises(ValueError):
            euler_walks(parse_fsm("\n".join(lines)))
