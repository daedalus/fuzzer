"""Tests for the structure-aware RIFF mutator (`riff_chunk_mutate`)."""

from __future__ import annotations

import random

from fuzzer_tool.core.mutations.riff import RiffMutator, parse_riff_chunks, serialize_riff
from fuzzer_tool.core.operator_registry import format_gate_matches


def _riff_bytes(rng=None) -> bytes:
    return RiffMutator()._generate_random_riff(rng=rng or random.Random(1))


class TestSniffer:
    def test_generated_wave_detected(self):
        assert format_gate_matches("riff_chunk_mutate", _riff_bytes()) is True

    def test_non_riff_not_detected(self):
        assert format_gate_matches("riff_chunk_mutate", b"\x89PNG\r\n\x1a\n" + bytes(20)) is False

    def test_webp_declined_to_avoid_operator_overlap(self):
        webp = b"RIFF" + (16).to_bytes(4, "little") + b"WEBP" + bytes(16)
        assert format_gate_matches("riff_chunk_mutate", webp) is False


class TestParseSerializeRoundTrip:
    def test_round_trip_preserves_chunks(self):
        data = _riff_bytes(random.Random(1))
        parsed = parse_riff_chunks(data)
        assert parsed is not None
        form_type, chunks = parsed
        assert form_type == b"WAVE"
        assert serialize_riff(form_type, chunks) == data

    def test_non_riff_returns_none(self):
        assert parse_riff_chunks(bytes(40)) is None

    def test_webp_returns_none(self):
        webp = b"RIFF" + (16).to_bytes(4, "little") + b"WEBP" + bytes(16)
        assert parse_riff_chunks(webp) is None


class TestRiffMutator:
    def test_mutate_changes_output_over_many_trials(self):
        data = _riff_bytes(random.Random(1))
        mutator = RiffMutator()
        changed = False
        for i in range(50):
            out = mutator.mutate(data, max_len=len(data) * 2, rng=random.Random(i))
            if out != data:
                changed = True
                break
        assert changed

    def test_mutate_output_stays_within_max_len(self):
        data = _riff_bytes(random.Random(1))
        mutator = RiffMutator()
        for i in range(20):
            out = mutator.mutate(data, max_len=40, rng=random.Random(i))
            assert len(out) <= 40

    def test_no_valid_chunks_generates_random_riff(self):
        mutator = RiffMutator()
        out = mutator.mutate(b"not a riff stream at all", max_len=65536, rng=random.Random(3))
        assert parse_riff_chunks(out) is not None

    def test_delete_chunk_never_drops_below_one(self):
        data = _riff_bytes(random.Random(1))
        _form_type, chunks = parse_riff_chunks(data)
        mutator = RiffMutator()
        for i in range(10):
            mutator._delete_chunk(chunks, random.Random(i))
        assert len(chunks) >= 1
