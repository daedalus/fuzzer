"""Tests for the structure-aware ADTS AAC mutator (`adts_chunk_mutate`)."""

from __future__ import annotations

import random

from fuzzer_tool.core.mutations.adts import (
    ADTS_HEADER_LEN,
    AdtsMutator,
    parse_adts_frames,
    serialize_adts_frames,
)
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng


def _frame_bytes(rng=None) -> bytes:
    return AdtsMutator()._generate_random_adts(rng=rng or random.Random(1))


class TestSniffer:
    def test_generated_frame_detected(self):
        data = _frame_bytes()
        assert format_gate_matches("adts_chunk_mutate", data) is True

    def test_too_short_not_detected(self):
        assert format_gate_matches("adts_chunk_mutate", b"\xff\xf1\x00") is False

    def test_non_adts_not_detected(self):
        assert format_gate_matches("adts_chunk_mutate", b"\x89PNG\r\n\x1a\n" + bytes(20)) is False

    def test_mp3_shaped_layer_bits_not_detected_as_adts(self):
        # MP3 header with non-zero layer bits must not sniff as ADTS,
        # which fixes those bits to 0.
        mp3_hdr = bytes([0xFF, 0xFB, 0x90, 0x00]) + bytes(20)
        assert format_gate_matches("adts_chunk_mutate", mp3_hdr) is False


class TestParseSerializeRoundTrip:
    def test_header_len_is_7(self):
        assert len(_frame_bytes()) >= ADTS_HEADER_LEN

    def test_round_trip_preserves_frame_count(self):
        data = (
            _frame_bytes(random.Random(1))
            + _frame_bytes(random.Random(2))
            + _frame_bytes(random.Random(3))
        )
        frames = parse_adts_frames(data)
        assert frames is not None
        assert len(frames) == 3
        assert serialize_adts_frames(frames) == data

    def test_frame_length_field_matches_actual_size(self):
        data = _frame_bytes(random.Random(5))
        frames = parse_adts_frames(data)
        assert frames[0].frame_length == len(data)

    def test_short_data_returns_none(self):
        assert parse_adts_frames(b"\xff\xf1\x00\x00") is None

    def test_unsynced_data_returns_none(self):
        assert parse_adts_frames(bytes(20)) is None


class TestAdtsMutator:
    def _frames_and_data(self):
        data = (
            _frame_bytes(random.Random(1))
            + _frame_bytes(random.Random(2))
            + _frame_bytes(random.Random(3))
        )
        return parse_adts_frames(data), data

    def test_mutate_changes_output_over_many_trials(self):
        _frames_unused, data = self._frames_and_data()
        mutator = AdtsMutator()
        protection_op = 4
        # Op 4 flips frame 0's protection_absent (low bit of header byte 1).
        rng = ScriptedRng(randints=[protection_op], choice_idxs=[0])
        out = mutator.mutate(data, max_len=len(data) * 2, rng=rng)

        expected = bytearray(data)
        expected[1] ^= 0x01
        assert out == bytes(expected)

    def test_mutate_output_stays_within_max_len(self):
        _frames_unused, data = self._frames_and_data()
        mutator = AdtsMutator()
        for i in range(20):
            out = mutator.mutate(data, max_len=50, rng=random.Random(i))
            assert len(out) <= 50

    def test_no_valid_frames_generates_random_adts(self):
        mutator = AdtsMutator()
        out = mutator.mutate(b"not an aac stream at all", max_len=65536, rng=random.Random(3))
        frames = parse_adts_frames(out)
        assert frames is not None
        assert frames[0].header[0] == 0xFF

    def test_delete_frame_never_drops_below_one(self):
        frames, _data = self._frames_and_data()
        mutator = AdtsMutator()
        for i in range(10):
            mutator._delete_frame(frames, random.Random(i))
        assert len(frames) >= 1

    def test_mutate_frame_length_changes_header(self):
        # Hard Rule 39 (2026-11-24 review): this used to loop up to 20
        # times until a random draw happened to change the header. One
        # call, scripted draws in evaluation order: choose the only
        # frame, roll the randint inside the bogus-length list (its
        # value is never selected), then choose index 1 = 0x1FFF -- a
        # 13-bit maximum the small generated frame (declared length 56)
        # can never already hold.
        data = _frame_bytes(random.Random(1))
        frames = parse_adts_frames(data)
        assert frames is not None and len(frames) == 1
        before = bytes(frames[0].header)
        AdtsMutator()._mutate_frame_length(frames, ScriptedRng(choice_idxs=[0, 1], randints=[0]))
        after = bytes(frames[0].header)
        assert after != before
        # 0x1FFF across the 13-bit aac_frame_length field: top 2 bits in
        # byte 3, middle 8 in byte 4, low 3 in byte 5's high bits.
        assert after[3] & 0x03 == 0x03
        assert after[4] == 0xFF
        assert after[5] & 0xE0 == 0xE0
