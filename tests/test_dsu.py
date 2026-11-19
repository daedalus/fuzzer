"""Tests for the shared disjoint-set forest (core/dsu.py)."""

import math
import random

import pytest

from fuzzer_tool.core.dsu import NP_LABELS_MIN, DisjointSet


def _bfs_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Oracle: components by BFS, ordered by first member, members ascending."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    seen = [False] * n
    out: list[list[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        comp, queue = [s], [s]
        while queue:
            for v in adj[queue.pop()]:
                if not seen[v]:
                    seen[v] = True
                    comp.append(v)
                    queue.append(v)
        out.append(sorted(comp))
    return out


def _random_edges(n: int, m: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randrange(n), rng.randrange(n)) for _ in range(m)]


class TestDisjointSet:
    def test_oracle_control(self):
        """Hard Rule 46: the oracle must agree with itself under edge reordering."""
        edges = _random_edges(300, 250, seed=1)
        assert _bfs_components(300, edges) == _bfs_components(300, edges[::-1])

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_groups_match_bfs(self, seed):
        n = 300
        edges = _random_edges(n, 250, seed)
        uf = DisjointSet(n)
        for a, b in edges:
            uf.union(a, b)

        assert uf.groups() == _bfs_components(n, edges)

    def test_union_returns_merge_flag(self):
        uf = DisjointSet(3)
        assert uf.union(0, 1) is True
        assert uf.union(1, 0) is False
        assert uf.union(2, 2) is False

    def test_labels_agree_with_find(self):
        uf = DisjointSet(100)
        for a, b in _random_edges(100, 80, seed=4):
            uf.union(a, b)

        assert uf.labels() == [uf.find(i) for i in range(100)]

    def test_size_invariant_after_random_unions(self):
        """Moved from test_overlap_density (was its private _UnionFind)."""
        n = 200
        uf = DisjointSet(n)
        for a, b in _random_edges(n, 500, seed=7):
            uf.union(a, b)

        counts: dict[int, int] = {}
        for i in range(n):
            root = uf.find(i)
            counts[root] = counts.get(root, 0) + 1
        for root, count in counts.items():
            assert uf.size(root) == count

    def test_falsification_no_spurious_merge(self):
        """Disjoint halves never share a root, whatever the union order."""
        n = 64
        uf = DisjointSet(n)
        half = n // 2
        for i in range(half - 1, 0, -1):
            uf.union(i, i - 1)
            uf.union(half + i, half + i - 1)

        assert uf.find(0) != uf.find(half)
        assert uf.groups() == [list(range(half)), list(range(half, n))]

    def test_adversarial_chain_depth_bounded(self):
        """Chain unions must stay O(log n) deep (union by size)."""
        n = 512
        uf = DisjointSet(n)
        for i in range(n - 1):
            uf.union(i, i + 1)

        assert max(uf.depth(i) for i in range(n)) <= 2 * math.log2(n) + 2
        assert uf.size(0) == n

    def test_chain_is_adversarial_for_naive_attach(self):
        """Control for the depth bound: naive root attach goes O(n) deep."""
        n = 512
        parent = list(range(n))

        def root(x: int) -> int:
            while parent[x] != x:
                x = parent[x]
            return x

        for i in range(n - 1):
            parent[root(i)] = root(i + 1)

        def depth(x: int) -> int:
            d = 0
            while parent[x] != x:
                x, d = parent[x], d + 1
            return d

        assert max(depth(i) for i in range(n)) >= n - 2

    def test_adversarial_bad_index(self):
        uf = DisjointSet(4)
        with pytest.raises(IndexError):
            uf.union(0, 4)

    def test_empty(self):
        uf = DisjointSet(0)
        assert uf.groups() == []
        assert uf.labels() == []


def test_labels_both_paths_agree():
    """Python (small n) and numpy (large n) labels() match find()."""
    for n in (NP_LABELS_MIN - 1, NP_LABELS_MIN, 4 * NP_LABELS_MIN):
        uf = DisjointSet(n)
        for a, b in _random_edges(n, n, seed=n):
            uf.union(a, b)
        expect = [uf.find(i) for i in range(n)]
        assert uf.labels() == expect
