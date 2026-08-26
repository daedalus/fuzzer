"""Tests for the structure-aware FLV mutator (`flv_chunk_mutate`)."""

from __future__ import annotations

import random

from fuzzer_tool.core.mutations.flv import FlvMutator, parse_flv, serialize_flv
from fuzzer_tool.core.operator_registry import format_gate_matches


def _flv_bytes(rng=None) -> bytes:
    return FlvMutator()._generate_random_flv(rng=rng or random.Random(1))


class TestSniffer:
    def test_generated_flv_detected(self):
        assert format_gate_matches("flv_chunk_mutate", _flv_bytes()) is True

    def test_non_flv_not_detected(self):
        assert format_gate_matches("flv_chunk_mutate", b"\x89PNG\r\n\x1a\n" + bytes(20)) is False


class TestParseSerializeRoundTrip:
    def test_round_trip_preserves_tags(self):
        data = _flv_bytes(random.Random(1))
        parsed = parse_flv(data)
        assert parsed is not None
        header, tags, trailing = parsed
        assert serialize_flv(header, tags, trailing) == data

    def test_non_flv_returns_none(self):
        assert parse_flv(bytes(40)) is None


class TestFlvMutator:
    def test_mutate_changes_output_over_many_trials(self):
        data = _flv_bytes(random.Random(1))
        mutator = FlvMutator()
        changed = False
        for i in range(50):
            out = mutator.mutate(data, max_len=len(data) * 2, rng=random.Random(i))
            if out != data:
                changed = True
                break
        assert changed

    def test_mutate_output_stays_within_max_len(self):
        data = _flv_bytes(random.Random(1))
        mutator = FlvMutator()
        for i in range(20):
            out = mutator.mutate(data, max_len=30, rng=random.Random(i))
            assert len(out) <= 30

    def test_no_valid_flv_generates_random_flv(self):
        mutator = FlvMutator()
        out = mutator.mutate(b"not an flv stream at all", max_len=65536, rng=random.Random(3))
        assert parse_flv(out) is not None

    def test_delete_tag_never_errors_on_single_tag(self):
        data = _flv_bytes(random.Random(1))
        _header, tags, _trailing = parse_flv(data)
        mutator = FlvMutator()
        for i in range(10):
            mutator._delete_tag(tags, random.Random(i))
        assert len(tags) >= 1
