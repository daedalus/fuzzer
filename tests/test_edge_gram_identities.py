"""Seed x edge Gram identities in EdgeTracker's pairwise queries.

With B the binary seed x edge matrix, G = B B^T counts shared edges per seed
pair and C = B^T B shared seeds per edge pair. Two identities replace pair
loops:

* ``coverage_dominance_tree``: S_a is a subset of S_b iff G[a, b] = |S_a|,
  i.e. b is in the intersection of owners(e) over e in S_a. Exact at every
  size; the pair loop fell back to "MinHash Jaccard > 0.8" above 100 edges,
  which misses small-inside-large subsets and reports near-twins.
* ``edge_cooccurrence``: Jaccard(a, b) = C[a, b] / (o_a + o_b - C[a, b]) with
  o the owner counts, so the pair scores are one matrix product.
"""

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker


def _tracker(seeds: dict[str, set[int]]) -> EdgeTracker:
    et = EdgeTracker(map_size=1 << 16)
    for key, edges in seeds.items():
        et.seed_edges[key] = set(edges)
        et._minhash.add(key, et._minhash.compute_signature(edges))
    return et


def _dominance_oracle(seed_edges):
    """Exact subset over every ordered pair, the pair loop's tie rule kept.

    Seeds sorted by size (stable); a earlier than b is dominated by b iff
    S_a <= S_b. Equal sets therefore dominate forward only.
    """
    order = sorted(seed_edges.items(), key=lambda x: len(x[1]))
    tree = {k: [] for k in seed_edges}
    for i, (a, sa) in enumerate(order):
        if not sa:
            continue
        for b, sb in order[i + 1 :]:
            if sb and sa <= sb:
                tree[b].append(a)
    return {k: v for k, v in tree.items() if v}


def _nested_corpus():
    """Chains of nested seeds from 3 to 600 edges, plus disjoint and empty ones."""
    seeds = {}
    for chain in range(4):
        base = chain * 10_000
        for depth, size in enumerate((3, 40, 150, 600)):
            seeds[f"c{chain}_{depth}"] = set(range(base, base + size))
        seeds[f"c{chain}_side"] = set(range(base + 5, base + 45)) | {base + 9_999}
    seeds["loner"] = set(range(90_000, 90_120))
    seeds["empty"] = set()
    return seeds


class TestDominance:
    def test_oracle_is_order_stable_on_distinct_sets(self):
        seeds = _nested_corpus()
        flipped = dict(reversed(list(seeds.items())))
        a, b = _dominance_oracle(seeds), _dominance_oracle(flipped)
        assert {k: sorted(v) for k, v in a.items()} == {k: sorted(v) for k, v in b.items()}

    def test_matches_exact_subsets_at_every_size(self):
        seeds = _nested_corpus()
        assert _tracker(seeds).coverage_dominance_tree() == _dominance_oracle(seeds)

    def test_small_set_inside_a_large_one_is_found(self):
        # Jaccard 150/600 = 0.25: the MinHash branch required > 0.8 and missed it.
        et = _tracker({"small": set(range(150)), "large": set(range(600))})
        assert et.coverage_dominance_tree() == {"large": ["small"]}
        assert et.find_redundant_seeds() == ["small"]

    def test_near_twins_are_not_subsets(self):
        # Jaccard 295/305 = 0.97 (MinHash 0.94 here), neither contains the other.
        et = _tracker({"a": set(range(300)), "b": set(range(5, 305))})
        assert et.coverage_dominance_tree() == {}
        assert et.find_redundant_seeds() == []

    def test_equal_sets_dominate_forward_only(self):
        et = _tracker({"first": {1, 2, 3}, "second": {1, 2, 3}})
        assert et.coverage_dominance_tree() == {"second": ["first"]}

    def test_empty_seeds_neither_dominate_nor_are_dominated(self):
        et = _tracker({"e": set(), "x": {1}, "y": {1, 2}})
        assert et.coverage_dominance_tree() == {"y": ["x"]}


def _cooccurrence_oracle(seed_edges, top_k):
    """The pair loop as it was: first 200 multi-owner edges, set Jaccard."""
    edge_to_seeds = {}
    for seed_key, edges in seed_edges.items():
        for e in edges:
            edge_to_seeds.setdefault(e, set()).add(seed_key)
    common = {e: s for e, s in edge_to_seeds.items() if len(s) >= 2}
    edges = list(common)
    pairs = []
    for i in range(min(len(edges), 200)):
        for j in range(i + 1, min(len(edges), 200)):
            a, b = edges[i], edges[j]
            inter = len(common[a] & common[b])
            union = len(common[a] | common[b])
            if union > 0 and inter / union > 0.1:
                pairs.append((a, b, inter / union))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def _cooccurrence_corpus(n_seeds=120, n_edges=500):
    """Deterministic overlaps with many exact ties in Jaccard."""
    return {
        f"s{s}": {e for e in range(n_edges) if (e * 7 + s * 3) % 11 < 4 or (e // 50) == s % 10}
        for s in range(n_seeds)
    }


class TestCooccurrence:
    def test_oracle_agrees_with_itself(self):
        seeds = _cooccurrence_corpus()
        assert _cooccurrence_oracle(seeds, 50) == _cooccurrence_oracle(dict(seeds), 50)

    @pytest.mark.parametrize("top_k", [1, 10, 64, 10_000])
    def test_equals_the_pair_loop_including_tie_order(self, top_k):
        seeds = _cooccurrence_corpus()
        got = _tracker(seeds).edge_cooccurrence(top_k=top_k)
        want = _cooccurrence_oracle(seeds, top_k)
        assert got == want
        assert all(type(j) is float for _, _, j in got)

    def test_fewer_than_two_shared_edges(self):
        assert _tracker({"a": {1, 2}, "b": {3}}).edge_cooccurrence() == []
        assert _tracker({"a": {1, 2}, "b": {1}}).edge_cooccurrence() == []
        assert _tracker({}).edge_cooccurrence() == []

    def test_threshold_is_strict(self):
        # 10 seeds share edge 0; edge 1 shares one of them: Jaccard exactly 0.1.
        seeds = {f"s{i}": {0} for i in range(10)}
        seeds["s0"] = {0, 1}
        seeds["t"] = {1, 2}
        seeds["u"] = {2}
        got = _tracker(seeds).edge_cooccurrence(top_k=10)
        assert got == _cooccurrence_oracle(seeds, 10)
        assert all(j > 0.1 for _, _, j in got)
