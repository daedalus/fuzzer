"""Tests for the structure-aware MP3 mutator (`mp3_chunk_mutate`)."""

from __future__ import annotations

import random

from fuzzer_tool.core.mutations.mp3 import (
    MP3_HEADER_LEN,
    Mp3Mutator,
    parse_mp3_frames,
    serialize_mp3_frames,
)
from fuzzer_tool.core.operator_registry import format_gate_matches


def _frame_bytes(rng=None) -> bytes:
    return Mp3Mutator()._generate_random_mp3(rng=rng or random.Random(1))


class TestSniffer:
    def test_generated_frame_detected(self):
        data = _frame_bytes()
        assert format_gate_matches("mp3_chunk_mutate", data) is True

    def test_too_short_not_detected(self):
        assert format_gate_matches("mp3_chunk_mutate", b"\xff\xfb\x90") is False

    def test_non_mp3_not_detected(self):
        assert format_gate_matches("mp3_chunk_mutate", b"\x89PNG\r\n\x1a\n" + bytes(20)) is False

    def test_adts_shaped_layer_bits_not_detected_as_mp3(self):
        # ADTS's layer-equivalent field is fixed at 0, which MP3 reserves;
        # an ADTS frame must not sniff as MP3.
        adts_hdr = bytes([0xFF, 0xF1, 0x50, 0x80, 0x00, 0x1F, 0xFC]) + bytes(20)
        assert format_gate_matches("mp3_chunk_mutate", adts_hdr) is False

    def test_reserved_version_not_detected(self):
        data = bytearray(_frame_bytes())
        data[1] = (data[1] & 0xE7) | (0x01 << 3)  # force reserved version
        assert format_gate_matches("mp3_chunk_mutate", bytes(data)) is False

    def test_reserved_layer_not_detected(self):
        data = bytearray(_frame_bytes())
        data[1] = data[1] & 0xF9  # force reserved layer (00)
        assert format_gate_matches("mp3_chunk_mutate", bytes(data)) is False


class TestParseSerializeRoundTrip:
    def test_header_len_is_4(self):
        assert len(_frame_bytes()) >= MP3_HEADER_LEN

    def test_round_trip_preserves_frame_count(self):
        data = _frame_bytes(random.Random(1)) + _frame_bytes(random.Random(2)) + _frame_bytes(random.Random(3))
        frames = parse_mp3_frames(data)
        assert frames is not None
        assert len(frames) == 3
        assert serialize_mp3_frames(frames) == data

    def test_short_data_returns_none(self):
        assert parse_mp3_frames(b"\xff\xfb\x90") is None

    def test_unsynced_data_returns_none(self):
        assert parse_mp3_frames(bytes(20)) is None


class TestMp3Mutator:
    def _frames_and_data(self):
        data = _frame_bytes(random.Random(1)) + _frame_bytes(random.Random(2)) + _frame_bytes(random.Random(3))
        return parse_mp3_frames(data), data

    def test_mutate_changes_output_over_many_trials(self):
        _frames_unused, data = self._frames_and_data()
        mutator = Mp3Mutator()
        changed = False
        for i in range(50):
            out = mutator.mutate(data, max_len=len(data) * 2, rng=random.Random(i))
            if out != data:
                changed = True
                break
        assert changed

    def test_mutate_output_stays_within_max_len(self):
        _frames_unused, data = self._frames_and_data()
        mutator = Mp3Mutator()
        for i in range(20):
            out = mutator.mutate(data, max_len=50, rng=random.Random(i))
            assert len(out) <= 50

    def test_no_valid_frames_generates_random_mp3(self):
        mutator = Mp3Mutator()
        out = mutator.mutate(b"not an mp3 stream at all", max_len=65536, rng=random.Random(3))
        frames = parse_mp3_frames(out)
        assert frames is not None
        assert frames[0].header[0] == 0xFF

    def test_delete_frame_never_drops_below_one(self):
        frames, _data = self._frames_and_data()
        mutator = Mp3Mutator()
        for i in range(10):
            mutator._delete_frame(frames, random.Random(i))
        assert len(frames) >= 1

    def test_mutate_bitrate_index_changes_header(self):
        data = _frame_bytes(random.Random(1))
        frames = parse_mp3_frames(data)
        before = bytes(frames[0].header)
        rng = random.Random(9)
        mutator = Mp3Mutator()
        for _ in range(20):
            mutator._mutate_bitrate_index(frames, rng)
            if bytes(frames[0].header) != before:
                break
        assert bytes(frames[0].header) != before
