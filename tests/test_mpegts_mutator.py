"""Tests for the structure-aware MPEG-TS mutator (`mpegts_chunk_mutate`)."""

from __future__ import annotations

import random

from fuzzer_tool.core.mutations.mpegts import (
    PACKET_SIZE,
    PAT_PID,
    SYNC_BYTE,
    MpegtsMutator,
    TsPacket,
    fix_psi_crc,
    parse_ts_packets,
    serialize_ts_packets,
)
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

# pointer(1) + table_id(1) + section_length(2) + tsid(2) + version(1) + sec(2).
PAT_ENTRY_OFF = 9
CRC_MPEG2_CHECK = 0x0376E6E7  # catalogue check value of b"123456789"


def _crc_mpeg2(data: bytes) -> int:
    """Bitwise CRC-32/MPEG-2: independent oracle, not the code under test."""
    reg = 0xFFFFFFFF
    for byte in data:
        reg ^= byte << 24
        for _ in range(8):
            reg = ((reg << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if reg & 0x80000000 else reg << 1
    return reg


def _section_crc_ok(payload: bytes) -> bool:
    # CRC over the whole section, CRC included, is zero (as FFmpeg checks it).
    start = 1 + payload[0]
    end = start + 3 + (((payload[start + 1] & 0x0F) << 8) | payload[start + 2])
    return _crc_mpeg2(payload[start:end]) == 0


def test_crc_oracle_control():
    # Hard Rule 46: the oracle must reproduce the catalogue value first.
    assert _crc_mpeg2(b"123456789") == CRC_MPEG2_CHECK


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
    fix_psi_crc(pkt)  # real PATs carry a valid CRC
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
        tei_op = 2
        # Op 2 flips packet 0's transport_error_indicator (top bit of byte 1).
        rng = ScriptedRng(randints=[tei_op], choice_idxs=[0])
        out = mutator.mutate(data, max_len=len(data) * 2, rng=rng)

        expected = bytearray(data)
        expected[1] ^= 0x80
        assert out == bytes(expected)

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

    def test_regression_pat_mutation_keeps_crc_valid(self):
        # Op 6 = PAT edit; choice 0 = first PAT packet, choice 1 = PMT PID 0x1FFF.
        _packets_unused, data = self._packets()
        pat_op = 6
        rng = ScriptedRng(randints=[pat_op, 0], choice_idxs=[0, 1])
        out = MpegtsMutator().mutate(data, max_len=len(data), rng=rng)

        payload = out[4:PACKET_SIZE]
        assert payload[PAT_ENTRY_OFF : PAT_ENTRY_OFF + 2] == b"\x00\x01"  # program kept
        pmt_pid = int.from_bytes(payload[PAT_ENTRY_OFF + 2 : PAT_ENTRY_OFF + 4], "big")
        assert pmt_pid & 0x1FFF == 0x1FFF
        assert _section_crc_ok(payload)

    def test_generated_psi_crcs_valid(self):
        out = MpegtsMutator()._generate_random_ts(max_len=65536, rng=random.Random(5))
        pat, pmt = out[4:PACKET_SIZE], out[PACKET_SIZE + 4 : 2 * PACKET_SIZE]
        assert _section_crc_ok(pat)
        assert _section_crc_ok(pmt)

    def test_crc_fix_skips_pes(self):
        # Falsification: a PES payload (no section-syntax bit) must stay byte-identical.
        pes = TsPacket(0, 1, 0, 0x0101, 0, 0x1, 0, payload=b"\x00\x00\x01\xe0" + bytes(20))
        before = pes.payload
        fix_psi_crc(pes)
        assert pes.payload == before

    def test_crc_fix_adversarial_lengths(self):
        # section_length and pointer_field past the packet end: leave untouched, no raise.
        long_sec = bytes([0x00, 0x00, 0xBF, 0xFF]) + bytes(20)
        far_ptr = bytes([0xFF]) + bytes(20)
        for payload in (long_sec, far_ptr, b"", b"\x00"):
            pkt = TsPacket(0, 1, 0, PAT_PID, 0, 0x1, 0, payload=payload)
            fix_psi_crc(pkt)
            assert pkt.payload == payload

    def test_delete_packet_never_drops_below_two(self):
        packets, _data = self._packets()
        mutator = MpegtsMutator()
        mutator._rng = random.Random(2)
        for _ in range(10):
            mutator._delete_packet(packets, 65536)
        assert len(packets) >= 2
