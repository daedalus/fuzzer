"""Tests for the structure-aware ASF mutator (`asf_chunk_mutate`)."""

from __future__ import annotations

import random

from fuzzer_tool.core.mutations.asf import (
    HEADER_OBJECT_GUID,
    AsfMutator,
    parse_asf_objects,
    serialize_asf_objects,
)
from fuzzer_tool.core.operator_registry import format_gate_matches


def _asf_bytes(rng=None) -> bytes:
    return AsfMutator()._generate_random_asf(rng=rng or random.Random(1))


class TestSniffer:
    def test_generated_asf_detected(self):
        assert format_gate_matches("asf_chunk_mutate", _asf_bytes()) is True

    def test_non_asf_not_detected(self):
        assert format_gate_matches("asf_chunk_mutate", b"\x89PNG\r\n\x1a\n" + bytes(40)) is False


class TestParseSerializeRoundTrip:
    def test_round_trip_preserves_objects(self):
        data = _asf_bytes(random.Random(1))
        objs = parse_asf_objects(data)
        assert objs is not None
        assert len(objs) == 2  # header + data
        assert objs[0].guid == HEADER_OBJECT_GUID
        assert serialize_asf_objects(objs) == data

    def test_missing_header_guid_returns_none(self):
        assert parse_asf_objects(bytes(40)) is None

    def test_single_object_returns_none(self):
        # Header-object-only stream, no Data object: must not parse.
        header_only = HEADER_OBJECT_GUID + (24).to_bytes(8, "little")
        assert parse_asf_objects(header_only) is None


class TestAsfMutator:
    def test_mutate_changes_output_over_many_trials(self):
        data = _asf_bytes(random.Random(1))
        mutator = AsfMutator()
        changed = False
        for i in range(50):
            out = mutator.mutate(data, max_len=len(data) * 2, rng=random.Random(i))
            if out != data:
                changed = True
                break
        assert changed

    def test_mutate_output_stays_within_max_len(self):
        data = _asf_bytes(random.Random(1))
        mutator = AsfMutator()
        for i in range(20):
            out = mutator.mutate(data, max_len=60, rng=random.Random(i))
            assert len(out) <= 60

    def test_no_valid_objects_generates_random_asf(self):
        mutator = AsfMutator()
        out = mutator.mutate(b"not an asf stream at all", max_len=65536, rng=random.Random(3))
        objs = parse_asf_objects(out)
        assert objs is not None
        assert objs[0].guid == HEADER_OBJECT_GUID

    def test_delete_object_never_drops_below_two(self):
        data = _asf_bytes(random.Random(1))
        objs = parse_asf_objects(data)
        mutator = AsfMutator()
        for i in range(10):
            mutator._delete_object(objs, random.Random(i))
        assert len(objs) >= 2
