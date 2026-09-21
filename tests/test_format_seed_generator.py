"""Tests for FormatSeedGenerator (core/format_seed_generator.py)."""

import random

from fuzzer_tool.core.analyzers.analyzer_format_learner import FieldHypothesis, FormatLearner
from fuzzer_tool.core.format_seed_generator import FormatSeedGenerator, generate_seeds


def _hyp(**kw):
    defaults = dict(
        offset=0,
        width=1,
        field_type="unknown",
        confidence=0.5,
        observations=5,
        sensitive_ops={},
        controlled_edges=set(),
    )
    defaults.update(kw)
    return FieldHypothesis(**defaults)


class TestFieldTypeSkipping:
    def test_magic_field_never_mutated(self):
        base = b"MAGICxxxx"
        fields = [_hyp(offset=0, width=5, field_type="magic", confidence=0.9, observations=10)]
        out = FormatSeedGenerator(fields).generate(base, n_seeds=20)
        assert out == []

    def test_padding_field_never_mutated(self):
        base = b"\x00" * 10
        fields = [_hyp(offset=2, width=4, field_type="padding", confidence=0.2, observations=1)]
        out = FormatSeedGenerator(fields).generate(base, n_seeds=20)
        assert out == []

    def test_mixed_fields_only_targets_non_skipped(self):
        base = b"MAGICLLLLdata......"
        fields = [
            _hyp(offset=0, width=5, field_type="magic", confidence=0.9, observations=10),
            _hyp(offset=5, width=4, field_type="length", confidence=0.6, observations=8),
        ]
        out = FormatSeedGenerator(fields).generate(base, n_seeds=20)
        assert out
        assert all(g.offset == 5 for g in out)
        assert all(g.field_type == "length" for g in out)


class TestLengthField:
    def test_generates_le_and_be_variants(self):
        base = b"HDR" + b"\x00\x00" + b"payload-body"
        fields = [_hyp(offset=3, width=2, field_type="length", confidence=0.7, observations=6)]
        out = FormatSeedGenerator(fields, rng=random.Random(1)).generate(base, n_seeds=50)
        strategies = {g.strategy for g in out}
        assert any(s.startswith("len_le_") for s in strategies)
        assert any(s.startswith("len_be_") for s in strategies)

    def test_patched_bytes_land_at_correct_offset_and_preserve_rest(self):
        base = b"HDR" + b"\x00\x00" + b"TAIL"
        fields = [_hyp(offset=3, width=2, field_type="length", confidence=0.7, observations=6)]
        out = FormatSeedGenerator(fields).generate(base, n_seeds=50)
        for g in out:
            assert len(g.data) == len(base)
            assert g.data[:3] == b"HDR"
            assert g.data[5:] == b"TAIL"

    def test_zero_and_max_boundary_present(self):
        base = b"AB" + b"\x00" * 1 + b"C"
        fields = [_hyp(offset=2, width=1, field_type="length", confidence=0.8, observations=6)]
        out = FormatSeedGenerator(fields).generate(base, n_seeds=50)
        values = {g.data[2] for g in out}
        assert 0x00 in values
        assert 0xFF in values


class TestCrcField:
    def test_stress_values_cover_zero_and_all_ones(self):
        base = b"XX" + b"\x12\x34\x56\x78" + b"YY"
        fields = [_hyp(offset=2, width=4, field_type="crc", confidence=0.9, observations=10)]
        out = FormatSeedGenerator(fields).generate(base, n_seeds=50)
        patches = {g.data[2:6] for g in out}
        assert b"\x00\x00\x00\x00" in patches
        assert b"\xff\xff\xff\xff" in patches

    def test_crc_never_touches_outside_its_width(self):
        base = b"XX" + b"\x12\x34\x56\x78" + b"YY"
        fields = [_hyp(offset=2, width=4, field_type="crc", confidence=0.9, observations=10)]
        out = FormatSeedGenerator(fields).generate(base, n_seeds=50)
        for g in out:
            assert g.data[:2] == b"XX"
            assert g.data[6:] == b"YY"


class TestDataField:
    def test_uses_sensitive_ops_when_mapped(self):
        base = b"H" * 3 + b"D" * 8
        fields = [
            _hyp(
                offset=3,
                width=8,
                field_type="data",
                confidence=0.4,
                observations=15,
                sensitive_ops={"byte_flip": 5, "bit_flip": 2},
            )
        ]
        out = FormatSeedGenerator(fields, rng=random.Random(0)).generate(base, n_seeds=20)
        strategies = {g.strategy for g in out}
        assert strategies <= {"byte_flip", "bit_flip"}

    def test_falls_back_when_no_op_mapped(self):
        base = b"H" * 3 + b"D" * 4
        fields = [
            _hyp(
                offset=3,
                width=4,
                field_type="data",
                confidence=0.4,
                observations=15,
                sensitive_ops={"totally_made_up_op": 9},
            )
        ]
        out = FormatSeedGenerator(fields, rng=random.Random(0)).generate(base, n_seeds=10)
        assert out
        assert all(g.strategy == "byte_stress" for g in out)

    def test_unknown_field_type_also_generates(self):
        base = b"H" * 3 + b"?" * 4
        fields = [_hyp(offset=3, width=4, field_type="unknown", confidence=0.3, observations=3)]
        out = FormatSeedGenerator(fields).generate(base, n_seeds=10)
        assert out


class TestScoreOrderingAndBudget:
    def test_higher_score_field_prioritized_when_budget_tight(self):
        base = b"AA" + b"\x00\x00" + b"BB" + b"\x00\x00"
        low = _hyp(
            offset=2,
            width=2,
            field_type="length",
            confidence=0.1,
            observations=3,
            controlled_edges=set(),
        )
        high = _hyp(
            offset=6,
            width=2,
            field_type="length",
            confidence=0.9,
            observations=50,
            controlled_edges={1, 2, 3, 4, 5},
        )
        out = FormatSeedGenerator([low, high]).generate(base, n_seeds=1)
        assert out[0].offset == 6

    def test_respects_n_seeds_budget(self):
        base = b"\x00" * 4 + b"data" * 10
        fields = [_hyp(offset=4, width=4, field_type="length", confidence=0.7, observations=5)]
        out = FormatSeedGenerator(fields).generate(base, n_seeds=3)
        assert len(out) == 3

    def test_out_of_range_field_is_skipped_not_crashing(self):
        base = b"short"
        fields = [_hyp(offset=100, width=4, field_type="length", confidence=0.7, observations=5)]
        out = FormatSeedGenerator(fields).generate(base, n_seeds=10)
        assert out == []


class TestInputCoercion:
    def test_accepts_field_hypothesis_objects(self):
        fields = [_hyp(offset=0, width=2, field_type="length", confidence=0.5, observations=4)]
        out = FormatSeedGenerator(fields).generate(b"\x00\x00tail", n_seeds=5)
        assert out

    def test_accepts_get_format_summary_dicts(self):
        fl = FormatLearner()
        fl.hypotheses.append(
            _hyp(
                offset=0,
                width=2,
                field_type="length",
                confidence=0.6,
                observations=6,
                controlled_edges={1, 2},
            )
        )
        summary = fl.get_format_summary()
        out = FormatSeedGenerator(summary["fields"]).generate(b"\x00\x00tail", n_seeds=5)
        assert out

    def test_accepts_get_state_hypothesis_dicts(self):
        fl = FormatLearner()
        fl.hypotheses.append(
            _hyp(
                offset=0,
                width=2,
                field_type="length",
                confidence=0.6,
                observations=6,
                controlled_edges={1, 2},
            )
        )
        state = fl.get_state()
        out = FormatSeedGenerator(state["hypotheses"]).generate(b"\x00\x00tail", n_seeds=5)
        assert out

    def test_empty_fields_yields_no_seeds(self):
        assert FormatSeedGenerator([]).generate(b"abcdef", n_seeds=10) == []

    def test_generate_seeds_convenience_function(self):
        fields = [_hyp(offset=0, width=2, field_type="length", confidence=0.5, observations=4)]
        out = generate_seeds(fields, b"\x00\x00tail", n_seeds=5)
        assert out


class TestEdgeCases:
    def test_zero_n_seeds_returns_empty(self):
        fields = [_hyp(offset=0, width=1, field_type="length", confidence=0.5, observations=4)]
        assert FormatSeedGenerator(fields).generate(b"abc", n_seeds=0) == []

    def test_empty_base_seed_returns_empty(self):
        fields = [_hyp(offset=0, width=1, field_type="length", confidence=0.5, observations=4)]
        assert FormatSeedGenerator(fields).generate(b"", n_seeds=10) == []

    def test_deterministic_with_seeded_rng(self):
        fields = [
            _hyp(
                offset=0,
                width=4,
                field_type="data",
                confidence=0.4,
                observations=8,
                sensitive_ops={"bit_flip": 3},
            )
        ]
        base = b"\x01\x02\x03\x04tail"
        out1 = FormatSeedGenerator(fields, rng=random.Random(42)).generate(base, n_seeds=5)
        out2 = FormatSeedGenerator(fields, rng=random.Random(42)).generate(base, n_seeds=5)
        assert [g.data for g in out1] == [g.data for g in out2]
