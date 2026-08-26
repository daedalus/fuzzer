"""Tests for the structure-aware Ogg mutator (`ogg_chunk_mutate`)."""

from __future__ import annotations

import random

from fuzzer_tool.core.mutations.ogg import OggMutator, parse_ogg_pages, serialize_ogg_pages
from fuzzer_tool.core.operator_registry import format_gate_matches


def _page_bytes(rng=None) -> bytes:
    return OggMutator()._generate_random_ogg(rng=rng or random.Random(1))


class TestSniffer:
    def test_generated_page_detected(self):
        assert format_gate_matches("ogg_chunk_mutate", _page_bytes()) is True

    def test_non_ogg_not_detected(self):
        assert format_gate_matches("ogg_chunk_mutate", b"\x89PNG\r\n\x1a\n" + bytes(20)) is False


class TestParseSerializeRoundTrip:
    def test_round_trip_preserves_page_count(self):
        data = _page_bytes(random.Random(1)) + _page_bytes(random.Random(2))
        pages = parse_ogg_pages(data)
        assert pages is not None
        assert len(pages) == 2
        assert serialize_ogg_pages(pages) == data

    def test_unsynced_data_returns_none(self):
        assert parse_ogg_pages(bytes(40)) is None


class TestOggMutator:
    def test_mutate_changes_output_over_many_trials(self):
        data = _page_bytes(random.Random(1)) + _page_bytes(random.Random(2))
        mutator = OggMutator()
        changed = False
        for i in range(50):
            out = mutator.mutate(data, max_len=len(data) * 2, rng=random.Random(i))
            if out != data:
                changed = True
                break
        assert changed

    def test_mutate_output_stays_within_max_len(self):
        data = _page_bytes(random.Random(1)) + _page_bytes(random.Random(2))
        mutator = OggMutator()
        for i in range(20):
            out = mutator.mutate(data, max_len=40, rng=random.Random(i))
            assert len(out) <= 40

    def test_no_valid_pages_generates_random_ogg(self):
        mutator = OggMutator()
        out = mutator.mutate(b"not an ogg stream at all", max_len=65536, rng=random.Random(3))
        assert parse_ogg_pages(out) is not None

    def test_delete_page_never_drops_below_one(self):
        data = _page_bytes(random.Random(1)) + _page_bytes(random.Random(2))
        pages = parse_ogg_pages(data)
        mutator = OggMutator()
        for i in range(10):
            mutator._delete_page(pages, random.Random(i))
        assert len(pages) >= 1
