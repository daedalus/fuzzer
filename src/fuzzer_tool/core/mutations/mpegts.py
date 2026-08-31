"""Structure-aware MPEG Transport Stream (M2TS/.ts) mutator.

MPEG-TS packet structure (ISO/IEC 13818-1), fixed 188-byte packets:

  [sync_byte: 0x47]
  [transport_error_indicator: 1 bit]
  [payload_unit_start_indicator: 1 bit]
  [transport_priority: 1 bit]
  [PID: 13 bits]
  [transport_scrambling_control: 2 bits]
  [adaptation_field_control: 2 bits]
  [continuity_counter: 4 bits]
  [adaptation_field: optional, present if adaptation_field_control has bit 0x2]
  [payload: remaining bytes]

PID 0x0000 carries the PAT (Program Association Table), which maps
program_number -> PMT PID. Each PMT then maps stream PIDs to their
codec (stream_type). This PAT/PMT indirection is the part of the
demuxer FFmpeg's other container mutators (isobmff, webm, nal) don't
exercise at all, and it is a distinct, heavily-hit bug surface in
libavformat/mpegts.c: PID remapping, continuity-counter discontinuity
handling, and adaptation-field stuffing/PCR parsing.

This mutator is a distinct addition from `nal.py`, which mutates the
H.264/H.265 *elementary* stream once demuxed; TS packets here still
carry PES-wrapped payloads, so a TS-level mutation exercises the
demux stage that unwraps them before nal.py's target code ever runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

PACKET_SIZE = 188
SYNC_BYTE = 0x47
PAT_PID = 0x0000
NULL_PID = 0x1FFF

# Common stream_type values usable for PMT corruption (ISO/IEC 13818-1
# Table 2-34 plus the private/registered ranges FFmpeg recognizes).
STREAM_TYPES = [
    0x01,  # MPEG-1 video
    0x02,  # MPEG-2 video
    0x03,  # MPEG-1 audio
    0x04,  # MPEG-2 audio
    0x0F,  # AAC (ADTS)
    0x10,  # MPEG-4 video
    0x1B,  # H.264
    0x24,  # H.265
    0x81,  # AC-3 (private/registered)
    0x86,  # SCTE-35
]


@dataclass
class TsPacket:
    """A single 188-byte MPEG-TS packet, header fields split out."""

    transport_error_indicator: int
    payload_unit_start_indicator: int
    transport_priority: int
    pid: int
    scrambling_control: int
    adaptation_field_control: int
    continuity_counter: int
    adaptation_field: bytes = field(default_factory=bytes)
    payload: bytes = field(default_factory=bytes)

    def to_bytes(self) -> bytes:
        b0 = SYNC_BYTE
        b1 = (
            (self.transport_error_indicator & 0x1) << 7
            | (self.payload_unit_start_indicator & 0x1) << 6
            | (self.transport_priority & 0x1) << 5
            | (self.pid >> 8) & 0x1F
        )
        b2 = self.pid & 0xFF
        b3 = (
            (self.scrambling_control & 0x3) << 6
            | (self.adaptation_field_control & 0x3) << 4
            | (self.continuity_counter & 0xF)
        )
        header = bytes([b0, b1, b2, b3])
        body = self.adaptation_field + self.payload
        # Pad/truncate to a fixed 188-byte packet with stuffing (0xFF) —
        # matches how real muxers pad adaptation-field stuffing bytes.
        body = body[: PACKET_SIZE - 4]
        if len(body) < PACKET_SIZE - 4:
            body = body + b"\xff" * (PACKET_SIZE - 4 - len(body))
        return header + body


def parse_ts_packets(data: bytes) -> list[TsPacket] | None:
    """Parse fixed-188-byte-aligned MPEG-TS packets from *data*.

    Requires at least two consecutive synced packets (sync byte 0x47 at
    both offset 0 and offset 188) so a coincidental single 0x47 in
    otherwise-random data doesn't misparse. Returns None on failure.
    """
    if len(data) < PACKET_SIZE * 2:
        return None
    if data[0] != SYNC_BYTE or data[PACKET_SIZE] != SYNC_BYTE:
        return None

    packets: list[TsPacket] = []
    pos = 0
    n = len(data)
    while pos + PACKET_SIZE <= n:
        if data[pos] != SYNC_BYTE:
            break
        pkt = data[pos : pos + PACKET_SIZE]
        b1, b2, b3 = pkt[1], pkt[2], pkt[3]
        pid = ((b1 & 0x1F) << 8) | b2
        adaptation_field_control = (b3 >> 4) & 0x3
        cursor = 4
        adaptation_field = b""
        payload = b""
        if adaptation_field_control in (0x2, 0x3) and cursor < len(pkt):
            af_len = pkt[cursor]
            af_end = min(cursor + 1 + af_len, len(pkt))
            adaptation_field = pkt[cursor:af_end]
            cursor = af_end
        if adaptation_field_control in (0x1, 0x3):
            payload = pkt[cursor:]
        packets.append(
            TsPacket(
                transport_error_indicator=(b1 >> 7) & 0x1,
                payload_unit_start_indicator=(b1 >> 6) & 0x1,
                transport_priority=(b1 >> 5) & 0x1,
                pid=pid,
                scrambling_control=(b3 >> 6) & 0x3,
                adaptation_field_control=adaptation_field_control,
                continuity_counter=b3 & 0xF,
                adaptation_field=adaptation_field,
                payload=payload,
            )
        )
        pos += PACKET_SIZE

    return packets if len(packets) >= 2 else None


def serialize_ts_packets(packets: list[TsPacket]) -> bytes:
    return b"".join(p.to_bytes() for p in packets)


def _pat_pids(packets: list[TsPacket]) -> list[int]:
    """Return the indices of packets on PID 0 (PAT)."""
    return [i for i, p in enumerate(packets) if p.pid == PAT_PID]


class MpegtsMutator:
    """Structure-aware MPEG-TS packet-level mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        self._rng = rng or random
        packets = parse_ts_packets(data)
        if not packets:
            return self._generate_random_ts(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 9)
        mutators = [
            self._mutate_pid,
            self._mutate_continuity_counter,
            self._toggle_error_indicator,
            self._toggle_pusi,
            self._mutate_adaptation_field_length,
            self._mutate_scrambling_control,
            self._mutate_pat_program_pid,
            self._duplicate_packet,
            self._delete_packet,
            self._reorder_packets,
        ]
        result = mutators[op](packets, max_len)
        if isinstance(result, list):
            return serialize_ts_packets(result)[:max_len]
        return result[:max_len]

    def _mutate_pid(self, packets: list[TsPacket], max_len: int) -> list[TsPacket]:
        """Remap a random packet's PID — exercises PID-table lookups and
        stream/program (mis)association in the demuxer."""
        target = self._rng.choice(packets)
        target.pid = self._rng.choice(
            [PAT_PID, NULL_PID, 0x0100, 0x1FFE, self._rng.randint(0, 0x1FFF)]
        )
        return packets

    def _mutate_continuity_counter(self, packets: list[TsPacket], max_len: int) -> list[TsPacket]:
        """Force a continuity-counter discontinuity on a random packet."""
        target = self._rng.choice(packets)
        target.continuity_counter = self._rng.randint(0, 15)
        return packets

    def _toggle_error_indicator(self, packets: list[TsPacket], max_len: int) -> list[TsPacket]:
        """Flip transport_error_indicator — tests the demuxer's TEI-drop path."""
        target = self._rng.choice(packets)
        target.transport_error_indicator ^= 1
        return packets

    def _toggle_pusi(self, packets: list[TsPacket], max_len: int) -> list[TsPacket]:
        """Flip payload_unit_start_indicator — corrupts PES/PSI framing without
        touching the payload bytes themselves."""
        target = self._rng.choice(packets)
        target.payload_unit_start_indicator ^= 1
        return packets

    def _mutate_adaptation_field_length(
        self, packets: list[TsPacket], max_len: int
    ) -> list[TsPacket]:
        """Corrupt the declared adaptation_field_length so it disagrees with
        the actual bytes present — a classic TS-parser OOB-read trigger."""
        candidates = [p for p in packets if p.adaptation_field]
        if not candidates:
            # No packet currently carries one: manufacture a bogus one so the
            # operator still does something on payload-only streams.
            target = self._rng.choice(packets)
            target.adaptation_field_control |= 0x2
            target.adaptation_field = bytes([self._rng.choice([0, 1, 183, 254, 255])])
            return packets
        target = self._rng.choice(candidates)
        af = bytearray(target.adaptation_field)
        af[0] = self._rng.choice([0, 1, 183, 254, 255, self._rng.randint(0, 255)])
        target.adaptation_field = bytes(af)
        return packets

    def _mutate_scrambling_control(self, packets: list[TsPacket], max_len: int) -> list[TsPacket]:
        """Set transport_scrambling_control to a non-zero value without
        actually scrambling the payload — decoder should reject, not crash."""
        target = self._rng.choice(packets)
        target.scrambling_control = self._rng.choice([1, 2, 3])
        return packets

    def _mutate_pat_program_pid(self, packets: list[TsPacket], max_len: int) -> list[TsPacket]:
        """Corrupt a program->PMT-PID mapping inside a PAT packet's payload.

        PAT payload (after the 1-byte pointer_field on a PUSI packet):
          table_id(1) section_length hi/lo(2) ... then repeated
          program_number(2) + reserved/PMT_PID(2) entries, 4 bytes each,
          starting at payload offset 8 and running to length-4 (CRC32 tail).
        Best-effort: falls through to a byte flip if the PAT doesn't parse
        cleanly, since malformed PATs are exactly what this operator wants
        to feed the demuxer.
        """
        pat_indices = _pat_pids(packets)
        if not pat_indices:
            target = self._rng.choice(packets)
            if target.payload:
                data = bytearray(target.payload)
                idx = self._rng.randint(0, len(data) - 1)
                data[idx] ^= 1 << self._rng.randint(0, 7)
                target.payload = bytes(data)
            return packets
        target = packets[self._rng.choice(pat_indices)]
        data = bytearray(target.payload)
        if len(data) >= 12:
            # First entry starts at offset 8 (after pointer_field + PAT
            # header); PMT PID occupies the low 13 bits of the 2nd u16.
            entry_off = 8
            pmt_pid = self._rng.choice([PAT_PID, 0x1FFF, self._rng.randint(0, 0x1FFF)])
            hi = 0xE0 | ((pmt_pid >> 8) & 0x1F)
            lo = pmt_pid & 0xFF
            if entry_off + 4 <= len(data):
                data[entry_off + 2] = hi
                data[entry_off + 3] = lo
                target.payload = bytes(data)
                return packets
        if data:
            idx = self._rng.randint(0, len(data) - 1)
            data[idx] ^= 1 << self._rng.randint(0, 7)
            target.payload = bytes(data)
        return packets

    def _duplicate_packet(self, packets: list[TsPacket], max_len: int) -> list[TsPacket]:
        """Clone a packet in place — exercises duplicate-continuity-counter
        handling, which real muxers use deliberately for redundancy."""
        idx = self._rng.randint(0, len(packets) - 1)
        orig = packets[idx]
        dup = TsPacket(
            transport_error_indicator=orig.transport_error_indicator,
            payload_unit_start_indicator=orig.payload_unit_start_indicator,
            transport_priority=orig.transport_priority,
            pid=orig.pid,
            scrambling_control=orig.scrambling_control,
            adaptation_field_control=orig.adaptation_field_control,
            continuity_counter=orig.continuity_counter,
            adaptation_field=orig.adaptation_field,
            payload=orig.payload,
        )
        packets.insert(idx + 1, dup)
        return packets

    def _delete_packet(self, packets: list[TsPacket], max_len: int) -> list[TsPacket]:
        """Drop a random packet (never the first, so a PAT-anchored stream
        keeps at least one lead-in packet to stay parseable next round)."""
        if len(packets) > 2:
            idx = self._rng.randint(1, len(packets) - 1)
            packets.pop(idx)
        return packets

    def _reorder_packets(self, packets: list[TsPacket], max_len: int) -> list[TsPacket]:
        """Swap two packets — out-of-order PES continuation is a real
        broadcast-noise condition FFmpeg's TS demuxer must tolerate."""
        if len(packets) >= 2:
            i, j = self._rng.sample(list(range(len(packets))), 2)
            packets[i], packets[j] = packets[j], packets[i]
        return packets

    def _generate_random_ts(self, max_len: int = 65536, rng=None) -> bytes:
        """Generate a minimal PAT + PMT + one PES-payload packet stream."""
        self._rng = rng or self._rng

        # PAT: program 1 -> PMT PID 0x1000.
        pat_section = bytes([0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00])
        pat_section += bytes([0x00, 0x01, 0xE1, 0x00])  # program 1 -> PMT PID 0x0100
        pat_section += bytes(4)  # CRC32 stand-in (not validated by this generator)
        pat_payload = bytes([0x00]) + pat_section  # pointer_field
        pat = TsPacket(
            transport_error_indicator=0,
            payload_unit_start_indicator=1,
            transport_priority=0,
            pid=PAT_PID,
            scrambling_control=0,
            adaptation_field_control=0x1,
            continuity_counter=0,
            payload=pat_payload,
        )

        # PMT on PID 0x0100: one elementary stream, stream_type chosen at
        # random from the FFmpeg-recognized set above.
        stream_type = self._rng.choice(STREAM_TYPES)
        pmt_section = bytes([0x02, 0xB0, 0x12, 0x00, 0x01, 0xC1, 0x00, 0x00])
        pmt_section += bytes([0xE1, 0x01, 0xF0, 0x00])  # PCR_PID=0x101, no program info
        pmt_section += bytes([stream_type, 0xE1, 0x01, 0xF0, 0x00])  # ES on PID 0x101
        pmt_section += bytes(4)
        pmt_payload = bytes([0x00]) + pmt_section
        pmt = TsPacket(
            transport_error_indicator=0,
            payload_unit_start_indicator=1,
            transport_priority=0,
            pid=0x0100,
            scrambling_control=0,
            adaptation_field_control=0x1,
            continuity_counter=0,
            payload=pmt_payload,
        )

        # One PES-wrapped elementary-stream packet on PID 0x0101.
        pes_payload = b"\x00\x00\x01\xe0\x00\x00\x80\x80\x05" + bytes(
            self._rng.randint(0, 255) for _ in range(160)
        )
        es = TsPacket(
            transport_error_indicator=0,
            payload_unit_start_indicator=1,
            transport_priority=0,
            pid=0x0101,
            scrambling_control=0,
            adaptation_field_control=0x1,
            continuity_counter=0,
            payload=pes_payload,
        )

        result = serialize_ts_packets([pat, pmt, es])
        return result[:max_len]
