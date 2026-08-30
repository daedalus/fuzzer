"""Regression: EdgeTracker._maybe_prune is bounded, affordable, and correct.

The tracked-seed ceiling was raised from 200 to 200,000 in `fe8fd42`, a perf
commit whose message does not mention it, and `_maybe_prune` has not fired
since: measured, 500 seeds recorded and zero pruned, while campaigns reach
221-403 seeds. Two things went with it — the memory bound on `seed_edges` and
its eight companion maps, and the `_edge_owner_count` rebuild that lives inside
the same function and exists to stop stale owner counts from degrading the
rarity signal the schedule is steered by.

The raise was a real fix for a real cost, though. Pruning back to the ceiling
exactly leaves `excess == 1` in steady state, so the O(seeds x edges) owner
count pass ran on every `record_edges` call to evict one seed — 14.78s against
0.81s over 600 insertions of 800 edges. Restoring the bound therefore needs the
cost removed rather than paid: prune in batches, so the pass amortises.

Batching in turn makes two latent correctness bugs reachable, both of which
lose coverage silently and neither of which could fire while only one seed was
evicted at a time. See the docstring on `_maybe_prune`.
"""

from __future__ import annotations

import random
import time

from fuzzer_tool.core.edge_tracker import MAX_TRACKED_SEEDS, EdgeTracker


class TestCeilingIsABound:
    def test_default_ceiling_binds_at_realistic_corpus_sizes(self):
        """The measured campaigns reached 221-403 tracked seeds."""
        assert MAX_TRACKED_SEEDS <= 10_000

    def test_prune_fires_and_holds_the_ceiling(self):
        et = EdgeTracker(max_tracked_seeds=100)
        rng = random.Random(11)
        for i in range(400):
            edges = {rng.randrange(600) for _ in range(50)}
            edges.add(90_000 + i)
            et.record_edges(f"s{i}", edges)
        assert len(et.seed_edges) <= 100
        assert len(et.seed_edges) > 0

    def test_companion_maps_stay_in_sync_with_seed_edges(self):
        """The eight per-seed maps are the memory the ceiling exists to bound."""
        et = EdgeTracker(max_tracked_seeds=50)
        for i in range(200):
            et.record_edges(
                seed_key=f"s{i}",
                hit_edges={i, 999},
                stack_depth=i,
                path_hash=0x1000 + i,
                hw_instructions=i,
                hw_branches=i,
                hw_branch_misses=i,
            )
        n = len(et.seed_edges)
        assert n <= 50
        for m in (
            et.seed_stack_depth,
            et.seed_path_hash,
            et.seed_hw_instructions,
            et.seed_hw_branches,
            et.seed_hw_branch_misses,
        ):
            assert len(m) == n
            assert set(m) == set(et.seed_edges)


class TestHysteresis:
    def test_a_prune_drops_below_the_ceiling(self):
        """Otherwise the next insertion is over budget again immediately."""
        et = EdgeTracker(max_tracked_seeds=100)
        for i in range(101):
            et.record_edges(f"s{i}", {i, 999})
        # Literal, not 100 - int(100 * PRUNE_BATCH_FRAC): deriving the bound
        # from the constant under test makes the assertion vacuous the moment
        # the constant is zeroed, which is exactly the regression to catch.
        assert len(et.seed_edges) <= 90

    def test_small_ceilings_keep_the_exact_prune_to_ceiling_behaviour(self):
        """A ceiling under 10 truncates the batch to zero, by design."""
        et = EdgeTracker(max_tracked_seeds=2)
        for i in range(5):
            et.record_edges(f"s{i}", {i})
        assert len(et.seed_edges) == 2

    def test_pruning_costs_far_less_than_once_per_insertion(self):
        """The regression that got the ceiling raised instead of the cost fixed."""

        def run(cap):
            rng = random.Random(7)
            et = EdgeTracker(max_tracked_seeds=cap)
            t0 = time.monotonic()
            for i in range(400):
                edges = {rng.randrange(5000) for _ in range(800)}
                edges.add(100_000 + i)
                et.record_edges(f"s{i}", edges)
            return time.monotonic() - t0

        pruning = run(200)
        never_prunes = run(10**9)
        # Was ~18x before batching. Generous bound: this is a timing test and
        # the point is the order of magnitude, not the constant.
        assert pruning < never_prunes * 6, f"{pruning:.2f}s vs {never_prunes:.2f}s"


class TestEvictionPreservesCoverage:
    def test_two_seeds_jointly_owning_an_edge_are_not_both_evicted(self):
        """The snapshot bug: an owner count of 2 marks neither seed protected.

        A and B jointly own edge 42, so both look unprotected; C is fully
        subsumed. With an excess of 2, evicting A and C loses nothing while
        evicting A and B loses edge 42. Only revalidating at the moment of
        eviction can tell those apart.
        """
        et = EdgeTracker(max_tracked_seeds=100)
        et.record_edges("A", {42, 1})
        et.record_edges("B", {42, 1})
        et.record_edges("C", {1})
        for i in range(7):
            et.record_edges(f"P{i}", {1, 500 + i})
        et.max_tracked_seeds = 10
        et.record_edges("Z", {1, 999})

        covered: set[int] = set()
        for edges in et.seed_edges.values():
            covered |= edges
        assert 42 in covered, "evicted both owners of a jointly-held edge"
        assert "C" not in et.seed_edges, "kept a fully-subsumed seed over a unique one"

    def test_subsumed_seeds_are_evicted_before_seeds_owning_unique_edges(self):
        et = EdgeTracker(max_tracked_seeds=100)
        et.record_edges("unique_a", {1, 2, 10})
        et.record_edges("subsumed", {1, 2})
        et.record_edges("unique_b", {1, 2, 20})
        # Ceiling 3 against 4 seeds: exactly one eviction, so the assertion is
        # about which seed goes, not about how many. At ceiling 2 two seeds
        # must go and losing a unique edge is forced, which tests nothing.
        et.max_tracked_seeds = 3
        et.record_edges("unique_c", {1, 2, 30})
        assert "subsumed" not in et.seed_edges
        assert {10, 20, 30} <= set().union(*et.seed_edges.values())

    def test_evicting_a_subsumed_seed_does_not_strand_its_coverer(self):
        """Phase ordering: evict A as subsumed, then evict B that covered for it.

        Neither eviction loses edge 42 alone; in sequence against a stale
        snapshot they do.
        """
        et = EdgeTracker(max_tracked_seeds=100)
        et.record_edges("A", {42, 1})
        et.record_edges("B", {42, 1})
        for i in range(6):
            et.record_edges(f"F{i}", {1})  # subsumed filler, loss 0
        et.max_tracked_seeds = 10
        et.record_edges("Z", {1, 999})
        et.max_tracked_seeds = 5
        et.record_edges("Y", {1, 998})

        covered: set[int] = set()
        for edges in et.seed_edges.values():
            covered |= edges
        assert 42 in covered


class TestOwnerCountsAfterPrune:
    def test_counts_match_the_survivors_exactly(self):
        """The rebuild is the reason _maybe_prune running at all matters."""
        et = EdgeTracker(max_tracked_seeds=20)
        rng = random.Random(5)
        for i in range(120):
            edges = {rng.randrange(80) for _ in range(10)}
            edges.add(70_000 + i)
            et.record_edges(f"s{i}", edges)

        expected: dict[int, int] = {}
        for edges in et.seed_edges.values():
            for e in edges:
                expected[e] = expected.get(e, 0) + 1
        actual = {e: n for e, n in et._edge_owner_count.items() if n > 0}
        assert actual == expected

    def test_no_zero_entries_are_retained(self):
        """_edge_owner_count is read by bare subscript on a defaultdict."""
        et = EdgeTracker(max_tracked_seeds=20)
        for i in range(120):
            et.record_edges(f"s{i}", {i, 999})
        assert all(n > 0 for n in et._edge_owner_count.values())
