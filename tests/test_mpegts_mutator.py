"""Tests for the structure-aware MPEG-TS mutator (`mpegts_chunk_mutate`)."""

from __future__ import annotations

import random

from fuzzer_tool.core.mutations.mpegts import (
    PACKET_SIZE,
    PAT_PID,
    SYNC_BYTE,
    MpegtsMutator,
    TsPacket,
    parse_ts_packets,
    serialize_ts_packets,
)
from fuzzer_tool.core.operator_registry import format_gate_matches


def _pat_packet(pid: int = PAT_PID, cc: int = 0) -> bytes:
    payload = bytes([0x00, 0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00])
    payload += bytes([0x00, 0x01, 0xE1, 0x00]) + bytes(4)
    pkt = TsPacket(
        transport_error_indicator=0,
        payload_unit_start_indicator=1,
        transport_priority=0,
        pid=pid,
        scrambling_control=0,
        adaptation_field_control=0x1,
        continuity_counter=cc,
        payload=payload,
    )
    return pkt.to_bytes()


def _adaptation_field_packet(pid: int = 0x0101, af_len: int = 1) -> bytes:
    pkt = TsPacket(
        transport_error_indicator=0,
        payload_unit_start_indicator=0,
        transport_priority=0,
        pid=pid,
        scrambling_control=0,
        adaptation_field_control=0x3,
        continuity_counter=0,
        adaptation_field=bytes([af_len]) + bytes(af_len),
        payload=bytes(50),
    )
    return pkt.to_bytes()


class TestSniffer:
    def test_two_synced_packets_detected(self):
        data = _pat_packet() + _adaptation_field_packet()
        assert format_gate_matches("mpegts_chunk_mutate", data) is True

    def test_single_sync_byte_not_enough(self):
        # A lone 0x47 with garbage at offset 188 must not sniff as TS.
        data = bytes([SYNC_BYTE]) + bytes(187) + bytes(188)
        assert format_gate_matches("mpegts_chunk_mutate", data) is False

    def test_too_short_not_detected(self):
        assert format_gate_matches("mpegts_chunk_mutate", _pat_packet()) is False

    def test_non_ts_not_detected(self):
        assert (
            format_gate_matches("mpegts_chunk_mutate", b"\x89PNG\r\n\x1a\n" + bytes(400)) is False
        )


class TestParseSerializeRoundTrip:
    def test_packet_size_is_188(self):
        assert len(_pat_packet()) == PACKET_SIZE

    def test_round_trip_preserves_packet_count(self):
        data = _pat_packet() + _adaptation_field_packet() + _pat_packet(pid=0x0102, cc=1)
        packets = parse_ts_packets(data)
        assert packets is not None
        assert len(packets) == 3
        assert serialize_ts_packets(packets) == data

    def test_pid_extracted_correctly(self):
        data = _pat_packet(pid=0x0123) + _pat_packet(pid=0x0123)
        packets = parse_ts_packets(data)
        assert packets[0].pid == 0x0123

    def test_adaptation_field_parsed(self):
        data = _adaptation_field_packet(af_len=3) + _pat_packet()
        packets = parse_ts_packets(data)
        assert packets[0].adaptation_field_control == 0x3
        assert len(packets[0].adaptation_field) == 4  # length byte + 3 body bytes

    def test_short_data_returns_none(self):
        assert parse_ts_packets(_pat_packet()) is None  # only one packet

    def test_unsynced_data_returns_none(self):
        assert parse_ts_packets(bytes(400)) is None


class TestMpegtsMutator:
    def _packets(self):
        data = _pat_packet() + _adaptation_field_packet() + _pat_packet(pid=0x0102, cc=1)
        return parse_ts_packets(data), data

    def test_mutate_changes_output_over_many_trials(self):
        _packets_unused, data = self._packets()
        mutator = MpegtsMutator()
        rng = random.Random(1234)
        changed = False
        for i in range(50):
            rng.seed(i)
            out = mutator.mutate(data, max_len=len(data) * 2, rng=rng)
            if out != data:
                changed = True
                break
        assert changed

    def test_mutate_output_stays_within_max_len(self):
        _packets_unused, data = self._packets()
        mutator = MpegtsMutator()
        rng = random.Random(7)
        for i in range(20):
            rng.seed(i)
            out = mutator.mutate(data, max_len=300, rng=rng)
            assert len(out) <= 300

    def test_no_valid_packets_generates_random_ts(self):
        mutator = MpegtsMutator()
        rng = random.Random(3)
        out = mutator.mutate(b"not a transport stream at all", max_len=65536, rng=rng)
        packets = parse_ts_packets(out)
        assert packets is not None
        assert packets[0].pid == PAT_PID

    def test_generated_stream_is_188_aligned(self):
        mutator = MpegtsMutator()
        rng = random.Random(5)
        out = mutator._generate_random_ts(max_len=65536, rng=rng)
        assert len(out) % PACKET_SIZE == 0

    def test_mutate_pat_program_pid_edits_pmt_mapping(self):
        packets, _data = self._packets()
        mutator = MpegtsMutator()
        rng = random.Random(11)
        before = bytes(packets[0].payload)
        mutator._rng = rng
        mutator._mutate_pat_program_pid(packets, 65536)
        assert bytes(packets[0].payload) != before

    def test_delete_packet_never_drops_below_two(self):
        packets, _data = self._packets()
        mutator = MpegtsMutator()
        mutator._rng = random.Random(2)
        for _ in range(10):
            mutator._delete_packet(packets, 65536)
        assert len(packets) >= 2
