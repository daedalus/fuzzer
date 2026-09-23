"""Tests for core/mds_local_search.py: weighted MDS local search."""

from fuzzer_tool.core.mds_local_search import (
    disk_radius,
    local_search_mds,
)


def _jaccard_table(pairs: dict[tuple[str, str], float]):
    """Build a symmetric jaccard_fn from an explicit pairwise table.

    Any unlisted pair defaults to 0.0 (fully dissimilar / never conflicts,
    matching MinHashLSH.approximate_jaccard's behavior for unknown keys).
    """

    def jaccard_fn(a: str, b: str) -> float:
        if a == b:
            return 1.0
        return pairs.get((a, b), pairs.get((b, a), 0.0))

    return jaccard_fn


class TestDiskRadius:
    def test_equal_scores_get_midpoint_radius(self):
        r = disk_radius(5.0, 5.0, 5.0)
        assert r == (0.12 + 0.40) / 2.0

    def test_highest_score_gets_smallest_radius(self):
        r = disk_radius(10.0, 0.0, 10.0)
        assert r == 0.12

    def test_lowest_score_gets_largest_radius(self):
        r = disk_radius(0.0, 0.0, 10.0)
        assert r == 0.40

    def test_monotonic_in_score(self):
        lo = disk_radius(2.0, 0.0, 10.0)
        hi = disk_radius(8.0, 0.0, 10.0)
        assert hi < lo  # higher score -> smaller radius

    def test_custom_bounds_respected(self):
        r_min = disk_radius(10.0, 0.0, 10.0, r_min=0.05, r_max=0.5)
        assert abs(r_min - 0.05) < 1e-9


class TestLocalSearchMDS:
    def test_no_conflicts_keeps_everyone(self):
        keys = ["a", "b", "c"]
        weight = {"a": 1.0, "b": 1.0, "c": 1.0}
        radius = {k: 0.1 for k in keys}
        jaccard_fn = _jaccard_table({})  # nothing similar to anything
        result = local_search_mds(keys, weight, radius, jaccard_fn)
        assert set(result.selected) == set(keys)

    def test_greedy_start_picks_higher_weight_from_conflicting_pair(self):
        keys = ["a", "b"]
        weight = {"a": 5.0, "b": 1.0}
        radius = {"a": 0.3, "b": 0.3}
        # distance = 1 - 0.9 = 0.1 < 0.3 + 0.3 -> conflict
        jaccard_fn = _jaccard_table({("a", "b"): 0.9})
        result = local_search_mds(keys, weight, radius, jaccard_fn)
        assert result.selected == ["a"]

    def test_swap_improves_over_greedy_single_vs_pair(self):
        """Classic local-search win: one high-weight seed conflicts with two
        lower-weight seeds that don't conflict with each other, but whose
        combined weight beats the single seed. Greedy alone (pick highest
        weight first) would keep only 'hub'; local search should swap in
        the pair once it notices the combined gain.
        """
        keys = ["hub", "leaf1", "leaf2"]
        weight = {"hub": 5.0, "leaf1": 3.0, "leaf2": 3.0}
        radius = {"hub": 0.3, "leaf1": 0.1, "leaf2": 0.1}
        # hub conflicts with both leaves (distance 0.1 < 0.3+0.1=0.4).
        # leaf1/leaf2 don't conflict with each other (distance 1.0).
        jaccard_fn = _jaccard_table(
            {
                ("hub", "leaf1"): 0.9,
                ("hub", "leaf2"): 0.9,
                ("leaf1", "leaf2"): 0.0,
            }
        )
        result = local_search_mds(keys, weight, radius, jaccard_fn, c=2)
        assert set(result.selected) == {"leaf1", "leaf2"}
        assert sum(weight[k] for k in result.selected) == 6.0
        assert result.swaps >= 1

    def test_result_is_always_conflict_free(self):
        keys = [f"s{i}" for i in range(12)]
        weight = {k: float((i * 7) % 5 + 1) for i, k in enumerate(keys)}
        radius = {k: 0.25 for k in keys}
        # Chain of overlaps: s0~s1~s2~... each adjacent pair conflicts.
        pairs = {}
        for i in range(len(keys) - 1):
            pairs[(keys[i], keys[i + 1])] = 0.9
        jaccard_fn = _jaccard_table(pairs)
        result = local_search_mds(keys, weight, radius, jaccard_fn, c=2)
        selected = result.selected
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                assert jaccard_fn(selected[i], selected[j]) < 1.0 - (
                    radius[selected[i]] + radius[selected[j]]
                ), f"{selected[i]} and {selected[j]} should not both be selected"

    def test_empty_input(self):
        result = local_search_mds([], {}, {}, _jaccard_table({}))
        assert result.selected == []
        assert result.swaps == 0

    def test_candidate_limit_does_not_crash_on_large_input(self):
        keys = [f"s{i}" for i in range(50)]
        weight = {k: 1.0 for k in keys}
        radius = {k: 0.15 for k in keys}
        jaccard_fn = _jaccard_table({})
        result = local_search_mds(keys, weight, radius, jaccard_fn, c=2, candidate_limit=10)
        assert set(result.selected) == set(keys)
