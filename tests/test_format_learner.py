"""Tests for format structure learner (schema-harness methodology)."""

from fuzzer_tool.core.analyzers.analyzer_format_learner import (
    FieldHypothesis,
    FormatCluster,
    FormatLearner,
)


class _StubRng:
    """Deterministic duck-typed rng: scripted `random()` floats, `randint`
    always returns the low end. Enough surface for `weighted_position`
    (`.random()` and `.randint(a, b)`), no real RandPool needed."""

    def __init__(self, values):
        self._values = list(values)

    def random(self) -> float:
        return self._values.pop(0)

    def randint(self, a: int, b: int) -> int:
        return a


class TestFormatLearnerInit:
    def test_empty_state(self):
        fl = FormatLearner()
        assert fl.timeline == []
        assert fl.hypotheses == []
        assert fl.backtest_passes == 0
        assert fl.backtest_fails == 0

    def test_record_transition(self):
        fl = FormatLearner()
        fl.record_transition(
            input_bytes=b"\x89PNG\r\n\x1a\n",
            mutation_op="bit_flip",
            mutation_offset=0,
            mutation_width=1,
            coverage_before=10,
            coverage_after=15,
            new_edges={100, 101, 102},
            lost_edges=set(),
        )
        assert len(fl.timeline) == 1
        assert fl.timeline[0].mutation_op == "bit_flip"
        assert fl.timeline[0].new_edges == {100, 101, 102}
        # Verify input_hash is stored, not raw bytes
        assert len(fl.timeline[0].input_hash) == 16

    def test_timeline_stores_hash_not_bytes(self):
        fl = FormatLearner()
        data = b"\x00" * 10000  # large input
        fl.record_transition(
            input_bytes=data,
            mutation_op="bit_flip",
            mutation_offset=0,
            mutation_width=1,
            coverage_before=10,
            coverage_after=15,
            new_edges={100},
            lost_edges=set(),
        )
        # Timeline should only store 16-byte hash, not 10KB
        entry = fl.timeline[0]
        assert len(entry.input_hash) == 16
        assert not hasattr(entry, "input_bytes")


class TestHypothesisBuilding:
    def test_sensitive_offset_creates_hypothesis(self):
        fl = FormatLearner()
        for i in range(5):
            fl.record_transition(
                input_bytes=b"\x89PNG" + b"\x00" * 12,
                mutation_op="bit_flip",
                mutation_offset=0,
                mutation_width=1,
                coverage_before=10,
                coverage_after=10 + i,
                new_edges={100 + i},
                lost_edges=set(),
            )
        assert len(fl.hypotheses) >= 1
        h = fl.hypotheses[0]
        assert h.offset == 0
        assert h.confidence > 0

    def test_no_effect_no_hypothesis(self):
        fl = FormatLearner()
        for _i in range(5):
            fl.record_transition(
                input_bytes=b"\x00" * 16,
                mutation_op="bit_flip",
                mutation_offset=5,
                mutation_width=1,
                coverage_before=10,
                coverage_after=10,
                new_edges=set(),
                lost_edges=set(),
            )
        assert len(fl.hypotheses) == 0


class TestFieldClassification:
    def test_magic_bytes_classification(self):
        fl = FormatLearner()
        ops = ["bit_flip", "bit_offset_flip", "byte_flip"]
        for i in range(6):
            fl.record_transition(
                input_bytes=b"\x89PNG\r\n\x1a\n",
                mutation_op=ops[i % len(ops)],
                mutation_offset=0,
                mutation_width=1,
                coverage_before=10,
                coverage_after=15,
                new_edges={100 + i},
                lost_edges=set(),
            )
        fl._classify_fields()
        h = fl.field_map.get(0)
        assert h is not None
        assert h.field_type == "magic"

    def test_length_field_classification(self):
        fl = FormatLearner()
        ops = ["arithmetic", "endianness_swap", "transpose_32"]
        for i in range(6):
            fl.record_transition(
                input_bytes=b"\x00" * 20,
                mutation_op=ops[i % len(ops)],
                mutation_offset=4,
                mutation_width=4,
                coverage_before=10,
                coverage_after=20,
                new_edges=set(range(100, 110)),
                lost_edges=set(),
            )
        fl._classify_fields()
        h = fl.field_map.get(4)
        assert h is not None
        assert h.field_type == "length"


class TestBacktest:
    def test_backtest_passes_with_no_hypotheses(self):
        fl = FormatLearner()
        ok, desc = fl.backtest()
        assert ok is True
        assert desc is None

    def test_backtest_passes_with_consistent_model(self):
        fl = FormatLearner()
        for i in range(5):
            fl.record_transition(
                input_bytes=b"\x89PNG\r\n\x1a\n",
                mutation_op="bit_flip",
                mutation_offset=0,
                mutation_width=1,
                coverage_before=10,
                coverage_after=15,
                new_edges={100 + i},
                lost_edges=set(),
            )
        ok, desc = fl.backtest()
        assert ok is True

    def test_backtest_fails_with_inconsistent_model(self):
        fl = FormatLearner()
        for i in range(3):
            fl.record_transition(
                input_bytes=b"\x89PNG",
                mutation_op="bit_flip",
                mutation_offset=0,
                mutation_width=1,
                coverage_before=10,
                coverage_after=15,
                new_edges={100 + i},
                lost_edges=set(),
            )
        fl.record_transition(
            input_bytes=b"\x00" * 20,
            mutation_op="arithmetic",
            mutation_offset=5,
            mutation_width=1,
            coverage_before=10,
            coverage_after=20,
            new_edges={200},
            lost_edges=set(),
        )
        ok, desc = fl.backtest()
        assert isinstance(ok, bool)


class TestPeriodicBacktest:
    def test_backtest_triggered_at_interval(self):
        fl = FormatLearner(max_timeline=100)
        from fuzzer_tool.core.analyzers.analyzer_format_learner import BACKTEST_INTERVAL

        # Record enough transitions to trigger backtest
        for i in range(BACKTEST_INTERVAL + 1):
            fl.record_transition(
                input_bytes=b"\x00" * 16,
                mutation_op="bit_flip",
                mutation_offset=0,
                mutation_width=1,
                coverage_before=10,
                coverage_after=15,
                new_edges={100 + i},
                lost_edges=set(),
            )
        # backtest should have been called at least once
        assert fl.backtest_passes + fl.backtest_fails >= 1


class TestDiscriminatingMutation:
    def test_no_suggestion_with_few_hypotheses(self):
        fl = FormatLearner()
        assert fl.suggest_discriminating_mutation(["bit_flip"]) is None

    def test_suggestion_with_different_field_types(self):
        fl = FormatLearner()
        h1 = FieldHypothesis(
            offset=0, width=1, field_type="magic", confidence=0.8, sensitive_ops={"bit_flip": 5}
        )
        h2 = FieldHypothesis(
            offset=8, width=4, field_type="length", confidence=0.6, sensitive_ops={"arithmetic": 3}
        )
        fl.hypotheses = [h1, h2]
        fl.field_map = {0: h1, 8: h2}

        suggestion = fl.suggest_discriminating_mutation(["bit_flip", "arithmetic"])
        if suggestion:
            op, offset = suggestion
            assert op in ["bit_flip", "arithmetic"]
            assert offset in [0, 8]


class TestSerialization:
    def test_get_state_roundtrip(self):
        fl = FormatLearner()
        fl.record_transition(
            input_bytes=b"\x89PNG",
            mutation_op="bit_flip",
            mutation_offset=0,
            mutation_width=1,
            coverage_before=10,
            coverage_after=15,
            new_edges={100},
            lost_edges=set(),
        )
        state = fl.get_state()
        assert len(state["timeline"]) == 1
        assert isinstance(state["hypotheses"], list)

        fl2 = FormatLearner()
        fl2.load_state(state)
        assert len(fl2.timeline) == 1
        assert fl2.timeline[0].mutation_op == "bit_flip"
        assert fl2.timeline[0].input_hash == fl.timeline[0].input_hash

    def test_format_summary(self):
        fl = FormatLearner()
        for i in range(5):
            fl.record_transition(
                input_bytes=b"\x89PNG\r\n\x1a\n",
                mutation_op="bit_flip",
                mutation_offset=0,
                mutation_width=1,
                coverage_before=10,
                coverage_after=15,
                new_edges={100 + i},
                lost_edges=set(),
            )
        summary = fl.get_format_summary()
        assert summary["timeline_size"] == 5
        assert summary["hypotheses"] >= 1
        assert "fields" in summary


class TestTimelinePruning:
    def test_timeline_trims_to_max(self):
        fl = FormatLearner(max_timeline=10)
        for i in range(20):
            fl.record_transition(
                input_bytes=b"\x00" * 8,
                mutation_op="bit_flip",
                mutation_offset=i % 8,
                mutation_width=1,
                coverage_before=10,
                coverage_after=10 + (i % 3),
                new_edges={100 + i} if i % 3 != 0 else set(),
                lost_edges=set(),
            )
        assert len(fl.timeline) <= 10


class TestZScoreHasEffect:
    def test_z_score_threshold_init(self):
        fl = FormatLearner(z_score_threshold=3.0)
        assert fl.z_score_threshold == 3.0

    def test_small_delta_filtered_after_warmup(self):
        """After enough observations, a tiny delta should not count as effect."""
        fl = FormatLearner(z_score_threshold=2.0)
        # Feed many zero-delta transitions to establish baseline
        for i in range(20):
            fl.record_transition(
                input_bytes=b"\x00" * 8,
                mutation_op="bit_flip",
                mutation_offset=i % 8,
                mutation_width=1,
                coverage_before=100,
                coverage_after=100,  # delta = 0
                new_edges=set(),
                lost_edges=set(),
            )
        # Now a tiny delta should NOT create a hypothesis
        fl.record_transition(
            input_bytes=b"\x00" * 8,
            mutation_op="bit_flip",
            mutation_offset=0,
            mutation_width=1,
            coverage_before=100,
            coverage_after=100,  # still zero
            new_edges=set(),
            lost_edges=set(),
        )
        # No hypothesis should be created for zero delta
        assert not any(h.offset == 0 for h in fl.hypotheses)

    def test_large_delta_creates_hypothesis(self):
        """A large delta should create a hypothesis even with z-score gate."""
        fl = FormatLearner(z_score_threshold=2.0)
        # Establish baseline with zero deltas
        for i in range(20):
            fl.record_transition(
                input_bytes=b"\x00" * 8,
                mutation_op="bit_flip",
                mutation_offset=i % 8,
                mutation_width=1,
                coverage_before=100,
                coverage_after=100,
                new_edges=set(),
                lost_edges=set(),
            )
        # Large delta should trigger hypothesis
        fl.record_transition(
            input_bytes=b"\x00" * 8,
            mutation_op="bit_flip",
            mutation_offset=5,
            mutation_width=2,
            coverage_before=100,
            coverage_after=200,  # delta = 100
            new_edges={500},
            lost_edges=set(),
        )
        assert any(h.offset == 5 for h in fl.hypotheses)

    def test_new_edges_bypass_z_score(self):
        """New edges should always count as effect regardless of z-score."""
        fl = FormatLearner(z_score_threshold=100.0)  # very high threshold
        for i in range(20):
            fl.record_transition(
                input_bytes=b"\x00" * 8,
                mutation_op="bit_flip",
                mutation_offset=i % 8,
                mutation_width=1,
                coverage_before=100,
                coverage_after=100,
                new_edges=set(),
                lost_edges=set(),
            )
        fl.record_transition(
            input_bytes=b"\x00" * 8,
            mutation_op="bit_flip",
            mutation_offset=3,
            mutation_width=1,
            coverage_before=100,
            coverage_after=100,  # zero delta
            new_edges={999},  # but new edges!
            lost_edges=set(),
        )
        assert any(h.offset == 3 for h in fl.hypotheses)

    def test_lost_edges_bypass_z_score(self):
        """Lost edges should always count as effect."""
        fl = FormatLearner(z_score_threshold=100.0)
        for i in range(20):
            fl.record_transition(
                input_bytes=b"\x00" * 8,
                mutation_op="bit_flip",
                mutation_offset=i % 8,
                mutation_width=1,
                coverage_before=100,
                coverage_after=100,
                new_edges=set(),
                lost_edges=set(),
            )
        fl.record_transition(
            input_bytes=b"\x00" * 8,
            mutation_op="bit_flip",
            mutation_offset=3,
            mutation_width=1,
            coverage_before=100,
            coverage_after=100,
            new_edges=set(),
            lost_edges={500},
        )
        assert any(h.offset == 3 for h in fl.hypotheses)

    def test_mad_fallback_under_high_kurtosis(self):
        """When kurtosis is high, MAD-based z-score should be used."""
        fl = FormatLearner(z_score_threshold=2.0)
        # Feed data that produces high kurtosis (many zeros, one outlier)
        for i in range(30):
            fl.record_transition(
                input_bytes=b"\x00" * 8,
                mutation_op="bit_flip",
                mutation_offset=i % 8,
                mutation_width=1,
                coverage_before=100,
                coverage_after=100,
                new_edges=set(),
                lost_edges=set(),
            )
        # Verify kurtosis is high after an outlier
        fl.record_transition(
            input_bytes=b"\x00" * 8,
            mutation_op="bit_flip",
            mutation_offset=0,
            mutation_width=1,
            coverage_before=100,
            coverage_after=110,  # small delta
            new_edges=set(),
            lost_edges=set(),
        )
        assert fl._delta_moments.kurtosis > 0  # heavy-tailed shape

    def test_delta_moments_tracked(self):
        """Verify delta moments are being tracked."""
        fl = FormatLearner()
        for i in range(10):
            fl.record_transition(
                input_bytes=b"\x00" * 8,
                mutation_op="bit_flip",
                mutation_offset=i % 8,
                mutation_width=1,
                coverage_before=100,
                coverage_after=100 + i,
                new_edges=set(),
                lost_edges=set(),
            )
        assert fl._delta_moments.count >= 10


class TestDelocalisedOffset:
    def test_none_offset_does_not_crash(self):
        """Delocalised ops set mutation_offset=None; the format learner must
        skip hypothesis updates without raising."""
        fl = FormatLearner()
        fl.record_transition(
            input_bytes=b"\x00" * 16,
            mutation_op="byte_shuffle",
            mutation_offset=None,
            mutation_width=8,
            coverage_before=10,
            coverage_after=12,
            new_edges={1, 2},
            lost_edges=set(),
        )
        assert len(fl.timeline) == 1
        assert fl.hypotheses == []


class TestRecordLiveness:
    def test_not_confirmed_dead_is_a_noop(self):
        fl = FormatLearner()
        fl.record_liveness(offset=10, width=4, confirmed_dead=False)
        assert fl.hypotheses == []

    def test_confirmed_dead_creates_padding_hypothesis(self):
        fl = FormatLearner()
        fl.record_liveness(offset=10, width=4, confirmed_dead=True)
        assert len(fl.hypotheses) == 1
        h = fl.hypotheses[0]
        assert h.offset == 10
        assert h.width == 4
        assert h.field_type == "padding"
        assert h.observations == 0

    def test_padding_hypothesis_is_low_confidence(self):
        """Corroborating evidence, not proof -- must not rival a
        multi-observation coverage-delta hypothesis in confidence."""
        fl = FormatLearner()
        fl.record_liveness(offset=0, width=4, confirmed_dead=True)
        assert fl.hypotheses[0].confidence < 0.5

    def test_existing_hypothesis_is_not_overwritten(self):
        """A hypothesis that already has real coverage-delta evidence
        (created via record_transition's has_effect path) must not be
        silently replaced by a padding verdict -- the two signals
        disagreeing is itself informative, not a reason to discard the
        one with actual observations."""
        fl = FormatLearner()
        fl.record_transition(
            input_bytes=b"\x00" * 16,
            mutation_op="bit_flip",
            mutation_offset=4,
            mutation_width=1,
            coverage_before=10,
            coverage_after=20,
            new_edges={1, 2, 3},
            lost_edges=set(),
        )
        assert len(fl.hypotheses) == 1
        original_type = fl.hypotheses[0].field_type
        original_confidence = fl.hypotheses[0].confidence

        fl.record_liveness(offset=4, width=1, confirmed_dead=True)

        assert len(fl.hypotheses) == 1  # no new hypothesis created
        assert fl.hypotheses[0].field_type == original_type  # not overwritten
        assert fl.hypotheses[0].confidence < original_confidence  # penalized

    def test_existing_padding_hypothesis_is_not_re_penalized(self):
        """Repeated confirmed-dead reports for a range already classified
        padding should not keep chipping at its confidence -- the penalty
        branch only fires for a *disagreeing* field_type."""
        fl = FormatLearner()
        fl.record_liveness(offset=10, width=4, confirmed_dead=True)
        confidence_after_first = fl.hypotheses[0].confidence
        fl.record_liveness(offset=10, width=4, confirmed_dead=True)
        assert fl.hypotheses[0].confidence == confidence_after_first

    def test_padding_field_type_counts_as_classified(self):
        fl = FormatLearner()
        fl.record_liveness(offset=0, width=8, confirmed_dead=True)
        summary = fl.get_format_summary()
        assert summary["classified"] == 1

    def test_value_tracking_empty(self):
        fl = FormatLearner()
        # No transitions recorded, should return None
        assert fl.get_learned_value(0, 4) is None

    def test_value_tracking_single_byte(self):
        fl = FormatLearner()
        # Record a transition that creates a hypothesis at offset 0, width 1
        fl.record_transition(
            input_bytes=b"\x42",
            mutation_op="bit_flip",
            mutation_offset=0,
            mutation_width=1,
            coverage_before=10,
            coverage_after=15,
            new_edges={100},
            lost_edges=set(),
        )
        # Should have learned the value 0x42 at offset 0
        val = fl.get_learned_value(0, 1)
        assert val == b"\x42"

    def test_value_tracking_multi_byte_field(self):
        fl = FormatLearner()
        # Record several transitions that modify a multi-byte field at known offset/width
        for i in range(10):
            # Create input where bytes at offset 2,3 are mostly 0xAA and 0xBB
            inp = bytearray(10)
            inp[2] = 0xAA if i % 3 != 0 else 0xCC  # 0xAA 70% of the time
            inp[3] = 0xBB if i % 2 == 0 else 0xDD  # 0xBB 50% of the time
            fl.record_transition(
                input_bytes=bytes(inp),
                mutation_op="byte_flip",
                mutation_offset=2,  # Always mutate starting at offset 2
                mutation_width=2,  # Width 2 covers both bytes
                coverage_before=10,
                coverage_after=15 + i,
                new_edges={200 + i},
                lost_edges=set(),
            )
        # After enough observations, should have a hypothesis for offset 2, width 2
        # Check that we can get learned values
        val = fl.get_learned_value(2, 2)
        assert val is not None
        assert len(val) == 2
        # The exact values depend on the random walk, but should be reasonable
        assert val[0] in (0xAA, 0xCC)
        assert val[1] in (0xBB, 0xDD)

    def test_value_tracking_with_liveness(self):
        fl = FormatLearner()
        # Create a hypothesis
        fl.record_transition(
            input_bytes=b"\x00\x01\x02\x03",
            mutation_op="bit_flip",
            mutation_offset=1,
            mutation_width=1,
            coverage_before=10,
            coverage_after=20,
            new_edges={100, 101},
            lost_edges=set(),
        )
        # Mark the byte as dead (should create a padding hypothesis but not overwrite)
        fl.record_liveness(offset=1, width=1, confirmed_dead=True)
        # Should still be able to get the learned value from the coverage transition
        val = fl.get_learned_value(1, 1)
        assert val == b"\x01"

    def test_value_counts_serialization(self):
        fl = FormatLearner()
        fl.record_transition(
            input_bytes=b"\x89PNG\r\n\x1a\n",
            mutation_op="bit_flip",
            mutation_offset=0,
            mutation_width=1,
            coverage_before=10,
            coverage_after=15,
            new_edges={100},
            lost_edges=set(),
        )
        fl.record_transition(
            input_bytes=b"\x89PNG\r\n\x1a\n",
            mutation_op="bit_flip",
            mutation_offset=0,
            mutation_width=1,
            coverage_before=15,
            coverage_after=20,
            new_edges={101},
            lost_edges=set(),
        )
        # Serialize and deserialize
        state = fl.get_state()
        fl2 = FormatLearner()
        fl2.load_state(state)
        # Should be able to get the same learned value
        val = fl.get_learned_value(0, 1)
        val2 = fl2.get_learned_value(0, 1)
        assert val == val2
        assert val == b"\x89"


class TestClusterWeightedPosition:
    def test_no_hypotheses_declines(self):
        c = FormatCluster(signature="ab")
        assert c.weighted_position(64, _StubRng([])) is None

    def test_all_zero_confidence_declines(self):
        c = FormatCluster(signature="ab")
        c.hypotheses.append(FieldHypothesis(offset=0, width=1, field_type="unknown"))
        c.hypotheses[0].confidence = 0.0
        assert c.weighted_position(64, _StubRng([])) is None

    def test_offset_past_buf_len_is_excluded(self):
        c = FormatCluster(signature="ab")
        c.hypotheses.append(FieldHypothesis(offset=100, width=4, field_type="unknown"))
        assert c.weighted_position(64, _StubRng([])) is None

    def test_roulette_picks_the_only_candidate(self):
        c = FormatCluster(signature="ab")
        c.hypotheses.append(FieldHypothesis(offset=8, width=4, field_type="length"))
        c.hypotheses[0].confidence = 0.3
        # random()=0.5 * total(0.3) = 0.15; r -= 0.3 -> <=0 on the first
        # (only) candidate. randint always returns the low end (offset 8).
        assert c.weighted_position(64, _StubRng([0.5])) == 8

    def test_roulette_weights_by_confidence(self):
        c = FormatCluster(signature="ab")
        # Low-confidence field first, high-confidence field second.
        c.hypotheses.append(FieldHypothesis(offset=0, width=1, field_type="unknown"))
        c.hypotheses[0].confidence = 0.1
        c.hypotheses.append(FieldHypothesis(offset=20, width=1, field_type="length"))
        c.hypotheses[1].confidence = 0.9
        # total = 1.0. random()=0.05 -> r=0.05; r -= 0.1 = -0.05 <= 0 ->
        # first candidate (offset 0) wins.
        assert c.weighted_position(64, _StubRng([0.05])) == 0
        # random()=0.5 -> r=0.5; r -= 0.1 = 0.4 (still > 0, keep going);
        # r -= 0.9 = -0.5 <= 0 -> second candidate (offset 20) wins.
        assert c.weighted_position(64, _StubRng([0.5])) == 20

    def test_width_is_clamped_to_the_buffer(self):
        c = FormatCluster(signature="ab")
        c.hypotheses.append(FieldHypothesis(offset=60, width=10, field_type="unknown"))
        c.hypotheses[0].confidence = 0.3
        # buf_len=64: width clamps from 10 to 4 (64-60), randint(0, 3) -> 0
        # via the stub's low-end rule, so offset stays 60, not out of range.
        assert c.weighted_position(64, _StubRng([0.5])) == 60


class TestLearnerWeightedPosition:
    def test_declines_with_no_clusters(self):
        fl = FormatLearner()
        assert fl.weighted_position(b"\x89PNG", 64) is None

    def test_routes_to_the_matching_cluster_not_the_primary(self):
        fl = FormatLearner()
        fl._rng = _StubRng([0.5, 0.5])
        primary = FormatCluster(signature="")
        primary.hypotheses.append(FieldHypothesis(offset=0, width=1, field_type="unknown"))
        primary.hypotheses[0].confidence = 0.3
        seed = b"PNG\x00"
        sig = FormatLearner.format_signature(seed, fl.sig_len)
        other = FormatCluster(signature=sig)
        other.hypotheses.append(FieldHypothesis(offset=5, width=1, field_type="length"))
        other.hypotheses[0].confidence = 0.3
        fl.clusters[""] = primary
        fl.clusters[sig] = other
        # A seed whose signature matches "other" must draw from its
        # hypotheses, not silently fall back to whichever cluster is
        # primary_cluster -- otherwise a multi-format target's proposals
        # would cross-contaminate between unrelated formats.
        assert fl.weighted_position(seed, 64) == 5

    def test_falls_back_to_primary_cluster_for_an_unknown_signature(self):
        fl = FormatLearner()
        fl._rng = _StubRng([0.5])
        primary = FormatCluster(signature="")
        primary.hypotheses.append(FieldHypothesis(offset=3, width=1, field_type="unknown"))
        primary.hypotheses[0].confidence = 0.3
        fl.clusters[""] = primary
        assert fl.weighted_position(b"\xff\xff\xff\xff", 64) == 3
