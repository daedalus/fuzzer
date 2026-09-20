"""``compute_subsumption_weight`` must measure inclusion, not coverage size.

Every test here is written against the *property*, not against an
implementation. The four marked ``# FALSIFIES`` fail against the MinHash
union-Jaccard version this replaced, which returned ``1 - |S|/|corpus union|``
and so could not distinguish a fully-subsumed seed from a disjoint one of the
same size. Verified failing before the fix; a test that passes against both
implementations would not be pinning anything (Hard Rule 39).
"""

from fuzzer_tool.core.edge_tracker import EdgeTracker


def _tracker(**seed_sets: set[int]) -> EdgeTracker:
    et = EdgeTracker()
    for key, edges in seed_sets.items():
        et.record_edges(key, set(edges))
    return et


class TestInclusionNotSize:
    def test_disjoint_beats_contained_at_equal_size(self):
        # FALSIFIES: both seeds have |S| = 200, so the size-based weight gave
        # them the same value up to MinHash noise and the ordering between
        # them was not stable across corpora.
        big = set(range(1000))
        et = _tracker(
            contained=set(range(200)),  # every edge also covered by `big`
            disjoint=set(range(5000, 5200)),  # owns all 200 of its edges
            big=big,
        )
        assert et.compute_subsumption_weight("disjoint") > et.compute_subsumption_weight(
            "contained"
        )

    def test_fully_subsumed_is_the_floor(self):
        for n in (1, 5, 50, 500):
            et = _tracker(sub=set(range(n)), cover=set(range(n + 10)))
            assert et.compute_subsumption_weight("sub") == 0.1

    def test_fully_unique_is_exactly_one(self):
        # Pinned as an exact equality, not approx: 1.0 is the endpoint of the
        # range and a floating-point near-miss there is a real defect, the
        # same reasoning as the H == 0 boundary in the entropy identities.
        for n in (1, 5, 50, 500):
            et = _tracker(a=set(range(n)), b=set(range(10_000, 10_000 + n)))
            assert et.compute_subsumption_weight("a") == 1.0
            assert et.compute_subsumption_weight("b") == 1.0

    def test_half_shared_is_half(self):
        et = _tracker(
            a=set(range(100)),  # 0..49 shared with b, 50..99 its own
            b=set(range(50)),
        )
        assert et.compute_subsumption_weight("a") == 0.5

    def test_duplicating_a_seeds_coverage_lowers_its_weight(self):
        # FALSIFIES: under 1 - |S|/|U| a new seed grows |U| while |S| is
        # fixed, so adding a duplicate *raised* the original's weight -- the
        # opposite of the documented behaviour.
        et = _tracker(a=set(range(100)), other=set(range(10_000, 10_100)))
        before = et.compute_subsumption_weight("a")
        et.record_edges("clone", set(range(100)))
        et._corpus_sig = None
        after = et.compute_subsumption_weight("a")
        assert after < before
        assert after == 0.1

    def test_scaling_every_seed_leaves_weights_unchanged(self):
        # FALSIFIES: this is the cleanest single statement of the old defect.
        # Multiplying every seed's edge count by a constant leaves the
        # inclusion structure identical, so the weights must not move; a
        # size-based weight moves with |S|/|U| only if the ratio changes, and
        # the shared/private split here makes it change.
        def weights(scale: int) -> dict[str, float]:
            et = _tracker(
                shared_a=set(range(10 * scale)),
                shared_b=set(range(10 * scale)),
                private=set(range(10_000, 10_000 + 10 * scale)),
            )
            return {k: et.compute_subsumption_weight(k) for k in ("shared_a", "shared_b", "private")}

        assert weights(1) == weights(10) == weights(50)


class TestGuardsPreserved:
    def test_unknown_seed_key(self):
        assert _tracker(a={1, 2}).compute_subsumption_weight("missing") == 1.0

    def test_empty_edge_set(self):
        et = _tracker(a={1, 2})
        et.seed_edges["empty"] = set()
        assert et.compute_subsumption_weight("empty") == 0.5

    def test_single_seed_corpus(self):
        assert _tracker(only=set(range(100))).compute_subsumption_weight("only") == 1.0


class TestOwnerMapDiscipline:
    def test_read_does_not_grow_the_owner_map(self):
        # _edge_owner_count is a defaultdict; a bare subscript would insert an
        # entry per queried edge and turn this accessor into a mutating one.
        et = _tracker(a=set(range(500)), b=set(range(400, 900)))
        before = len(et._edge_owner_count)
        for _ in range(3):
            et.compute_subsumption_weight("a")
            et.compute_subsumption_weight("b")
        assert len(et._edge_owner_count) == before

    def test_absent_owner_map_reports_novel_not_subsumed(self):
        # A state snapshot written before edge_owner_count was persisted
        # restores seed_edges in full against an empty owner map. Defaulting
        # the per-edge count to 0 would floor every seed in the corpus at
        # once; defaulting to 1 reports "all novel", which is the same value
        # a cold single-seed corpus already returns.
        et = _tracker(a=set(range(100)), b=set(range(100)))
        et._edge_owner_count.clear()
        assert et.compute_subsumption_weight("a") == 1.0


class TestClassificationConsistency:
    def test_parasitic_is_reachable(self):
        # The weight is clamped at 0.1, so classify_seeds' old ``weight <
        # 0.1`` test could never fire and "parasitic" was an unreachable
        # label. Under inclusion semantics "no singleton edges" and "weight at
        # the floor" are the same condition, so the two tests agree.
        et = _tracker(sub=set(range(50)), cover=set(range(200)))
        classes = et.classify_seeds()
        assert classes["sub"]["classification"] == "parasitic"
        assert classes["cover"]["classification"] == "keystone"

    def test_classification_does_not_depend_on_minhash_cache_state(self):
        et = _tracker(sub=set(range(50)), cover=set(range(200)))
        et._corpus_sig = None
        cold = {k: v["classification"] for k, v in et.classify_seeds().items()}
        et._corpus_sig = et._minhash.corpus_minhash()
        warm = {k: v["classification"] for k, v in et.classify_seeds().items()}
        assert cold == warm
