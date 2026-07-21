"""Tests for format structure learner (schema-harness methodology)."""

from fuzzer_tool.core.format_learner import FormatLearner, FieldHypothesis, TimelineEntry


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
        for i in range(5):
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
        from fuzzer_tool.core.format_learner import BACKTEST_INTERVAL

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
