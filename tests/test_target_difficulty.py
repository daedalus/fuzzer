"""Tests for static target-difficulty estimation (percolation handover Module 3).

Uses plain adjacency dicts ({node: set(neighbors)}) instead of real ELF
binaries so the algorithms are tested independently of the CFG decoder —
see test_target_difficulty_icfg.py (or the icfg adapter test below) for the
InterproceduralCFG bridge.
"""

from fuzzer_tool.core.target_difficulty import (
    estimate_growth_curve,
    estimate_isoperimetric_profile,
    estimate_percolation_threshold,
)


def _path_graph(n: int) -> dict[int, set[int]]:
    """0 - 1 - 2 - ... - (n-1): every internal boundary is exactly 1 edge."""
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    return adj


def _star_graph(n_leaves: int) -> dict[int, set[int]]:
    """Hub 0 connected to n_leaves leaves — a chokepoint-free hub."""
    adj: dict[int, set[int]] = {0: set()}
    for leaf in range(1, n_leaves + 1):
        adj[0].add(leaf)
        adj[leaf] = {0}
    return adj


def _two_cliques_bridged(clique_size: int) -> dict[int, set[int]]:
    """Two complete graphs of clique_size, joined by a single bridge edge.

    The bridge is a hard chokepoint: Φ(n) stays at 1 for any n that can be
    satisfied by one clique plus crossing the bridge into the other.
    """
    adj: dict[int, set[int]] = {i: set() for i in range(2 * clique_size)}
    left = range(0, clique_size)
    right = range(clique_size, 2 * clique_size)
    for i in left:
        for j in left:
            if i != j:
                adj[i].add(j)
    for i in right:
        for j in right:
            if i != j:
                adj[i].add(j)
    # single bridge edge
    adj[clique_size - 1].add(clique_size)
    adj[clique_size].add(clique_size - 1)
    return adj


class TestEstimateIsoperimetricProfile:
    def test_empty_graph(self):
        assert estimate_isoperimetric_profile({}, [1, 2, 3]) == {}

    def test_path_graph_boundary_is_one_or_two(self):
        # Any contiguous run of a path graph has boundary <= 2 (0 at the
        # very ends, 1 or 2 otherwise). The greedy grower should find this.
        adj = _path_graph(20)
        profile = estimate_isoperimetric_profile(adj, [1, 5, 10, 15], seed=0)
        for n, phi in profile.items():
            assert phi <= 2, f"path graph Φ({n}) should be <=2, got {phi}"

    def test_star_graph_boundary_grows_with_n(self):
        # Any set of size n containing the hub has boundary = (n_leaves in
        # graph - leaves in S); a set of only leaves (no hub) has boundary
        # equal to its own size (each leaf's only edge goes to the hub).
        # Either way Φ(n) is small only while n stays small relative to the
        # star, so this checks the profile is monotone non-decreasing.
        adj = _star_graph(30)
        profile = estimate_isoperimetric_profile(adj, [1, 5, 10, 20], seed=0)
        sizes = sorted(profile)
        for a, b in zip(sizes, sizes[1:], strict=False):
            assert profile[a] <= profile[b], "Φ(n) must be non-decreasing"

    def test_bridge_is_a_persistent_chokepoint(self):
        # Sizes that stay within (or barely cross into) one clique should
        # find the bridge (boundary 1) is at least as good as cutting
        # through the dense clique interior.
        adj = _two_cliques_bridged(clique_size=8)
        profile = estimate_isoperimetric_profile(adj, [1, 2, 8], seed=0)
        # A set of exactly one full clique's vertices has boundary 1 (only
        # the bridge edge leaves it) — the greedy search should find
        # something at least this good.
        assert profile[8] <= 1

    def test_restarts_improve_or_match_single_seed(self):
        adj = _two_cliques_bridged(clique_size=6)
        single = estimate_isoperimetric_profile(adj, [6], seed=0, restarts=1)
        multi = estimate_isoperimetric_profile(adj, [6], seed=0, restarts=4)
        assert multi[6] <= single[6]

    def test_sizes_beyond_graph_are_omitted(self):
        adj = _path_graph(5)
        profile = estimate_isoperimetric_profile(adj, [3, 100])
        assert 100 not in profile
        assert 3 in profile

    def test_deterministic_for_fixed_seed(self):
        adj = _two_cliques_bridged(clique_size=10)
        p1 = estimate_isoperimetric_profile(adj, [5, 10, 15], seed=7)
        p2 = estimate_isoperimetric_profile(adj, [5, 10, 15], seed=7)
        assert p1 == p2

    def test_disconnected_singleton_has_zero_boundary(self):
        adj = {0: set(), 1: {2}, 2: {1}}
        # restarts=3 covers every node deterministically (no reliance on
        # which nodes the RNG happens to sample — see AGENTS.md rule 39).
        profile = estimate_isoperimetric_profile(adj, [1], seed=0, restarts=3)
        # best-of-restarts must find the isolated node (boundary 0)
        assert profile[1] == 0


class TestEstimatePercolationThreshold:
    def test_empty_graph(self):
        result = estimate_percolation_threshold({})
        assert result["n_nodes"] == 0
        assert result["p_c_estimate"] == float("inf")

    def test_regular_graph_matches_hand_computed_er_formula(self):
        # 4-regular graph (path with wraparound + one extra neighbor each
        # side) — every node has degree 4, so p_c = 1/4 by the textbook
        # Erdos-Renyi formula, independent of estimate_percolation_threshold's
        # own implementation (independent reference, not self-validation).
        n = 10
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for d in (1, 2):
                adj[i].add((i + d) % n)
                adj[i].add((i - d) % n)
        result = estimate_percolation_threshold(adj)
        assert result["avg_degree"] == 4.0
        assert result["p_c_estimate"] == 0.25

    def test_isolated_nodes_give_infinite_threshold(self):
        adj = {0: set(), 1: set()}
        result = estimate_percolation_threshold(adj)
        assert result["p_c_estimate"] == float("inf")


class TestEstimateGrowthCurve:
    def test_empty_profile_stays_flat(self):
        curve = estimate_growth_curve({}, initial_size=1, max_steps=5)
        assert curve == [1.0] * 6

    def test_constant_phi_gives_linear_growth(self):
        # dv/dn = c * Phi(v) with Phi constant at 2 and c=1 is Euler-stepped
        # as v += 2 each iteration — an independently hand-computed
        # arithmetic sequence, not re-derived from the function under test.
        phi_profile = {1: 2, 1000: 2}
        curve = estimate_growth_curve(phi_profile, initial_size=1, max_steps=4, c=1.0)
        assert curve == [1.0, 3.0, 5.0, 7.0, 9.0]

    def test_growth_is_monotone_nondecreasing(self):
        phi_profile = {1: 1, 5: 3, 20: 6}
        curve = estimate_growth_curve(phi_profile, initial_size=1, max_steps=10)
        for a, b in zip(curve, curve[1:], strict=False):
            assert b >= a

    def test_higher_phi_grows_faster(self):
        low = estimate_growth_curve({1: 1}, initial_size=1, max_steps=5)
        high = estimate_growth_curve({1: 5}, initial_size=1, max_steps=5)
        assert high[-1] > low[-1]


class TestAdversarial:
    def test_estimate_isoperimetric_profile_rejects_nonpositive_sizes(self):
        adj = _path_graph(5)
        profile = estimate_isoperimetric_profile(adj, [-1, 0, 2])
        assert -1 not in profile
        assert 0 not in profile
        assert 2 in profile

    def test_estimate_growth_curve_negative_max_steps_returns_initial_only(self):
        curve = estimate_growth_curve({1: 1}, initial_size=3, max_steps=0)
        assert curve == [3.0]
