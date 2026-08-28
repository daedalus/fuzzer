"""Structure-aware NAL (Network Abstraction Layer) unit bitstream mutator.

NAL units are the basic building blocks of H.264 (AVC) and H.265 (HEVC)
elementary bitstreams. This mutator parses NAL units by their start-code
delimiter and applies targeted mutations at the NAL level — a different
bug class (decoder-internal) from the container/demuxer bugs the other
mutators target.

NAL unit structure (H.264):
  [start_code: 0x00000001 or 0x000001]
  [header: 1 byte — forbidden_bit | nal_ref_idc(2) | nal_unit_type(5)]
  [payload: RBSP data]

NAL unit structure (H.265):
  [start_code: 0x00000001 or 0x000001]
  [header: 2 bytes]
  [payload: RBSP data]

NAL types (H.264): 1=non-IDR slice, 5=IDR slice, 7=SPS, 8=PPS, 6=SEI
NAL types (H.265): 19=IDR_W_RADL, 20=IDR_N_LP, 32=VPS, 33=SPS, 34=PPS
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Start code patterns
START_CODE_3 = b"\x00\x00\x01"
START_CODE_4 = b"\x00\x00\x00\x01"

# H.264 NAL unit types (nal_unit_type values)
H264_NAL_NON_IDR = 1
H264_NAL_SEI = 6
H264_NAL_SPS = 7
H264_NAL_PPS = 8
H264_NAL_IDR = 5

H264_NAL_TYPES = list(range(1, 13)) + list(range(19, 25))

# H.265 NAL unit types (first byte, upper 5 bits)
H265_NAL_TRAIL = 0
H265_NAL_STSA = 1
H265_NAL_RADL = 2
H265_NAL_RASL = 3
H265_NAL_VPS = 32
H265_NAL_SPS_HEVC = 33
H265_NAL_PPS_HEVC = 34
H265_NAL_AUD = 35
H265_NAL_EOS = 36
H265_NAL_EOB = 37
H265_NAL_IDR_W_RADL = 19
H265_NAL_IDR_N_LP = 20
H265_NAL_CRA = 21

# Preserve-critical NAL types (SPS, PPS, VPS)
CRITICAL_NAL_TYPES_H264 = {H264_NAL_SPS, H264_NAL_PPS}
CRITICAL_NAL_TYPES_H265 = {H265_NAL_VPS, H265_NAL_SPS_HEVC, H265_NAL_PPS_HEVC}

# Non-critical types safe for deletion
H264_DELETABLE = list(range(1, 7)) + list(range(9, 25))
H265_DELETABLE = list(range(0, 19)) + list(range(21, 32)) + list(range(35, 40))


@dataclass
class NalUnit:
    """A single NAL unit."""

    start_code: bytes
    header: bytes
    payload: bytes = field(default_factory=bytes)

    @property
    def unit_type(self) -> int:
        return self.header[0] & 0x1F

    @property
    def nal_ref_idc(self) -> int:
        return (self.header[0] >> 5) & 0x03


def parse_nal_units(data: bytes) -> list[NalUnit] | None:
    """Parse NAL units from an elementary bitstream.

    Scans for start codes (0x00000001 or 0x000001) and extracts
    complete NAL units. Returns None if no start code is found.
    """
    units: list[NalUnit] = []
    pos = 0
    n = len(data)

    # Find first start code
    while pos < n:
        if data[pos : pos + 4] == START_CODE_4:
            start_code = START_CODE_4
            break
        if data[pos : pos + 3] == START_CODE_3:
            start_code = START_CODE_3
            break
        pos += 1
    else:
        return None

    header_start = pos + len(start_code)
    if header_start >= n:
        return None

    prev_start_code = start_code

    pos = header_start + 1  # +1 for the header byte (minimal unit)

    while pos < n:
        # Scan for next start code
        next_code: bytes | None = None
        next_pos = pos

        while next_pos < n:
            if data[next_pos : next_pos + 4] == START_CODE_4:
                next_code = START_CODE_4
                break
            if data[next_pos : next_pos + 3] == START_CODE_3:
                next_code = START_CODE_3
                break
            next_pos += 1

        if next_code is None:
            # No more start codes — last unit runs to end of data
            payload = data[header_start:]
            units.append(
                NalUnit(start_code=prev_start_code, header=payload[:1], payload=payload[1:])
            )
            break

        # Unit is from unit_start to next_pos
        payload = data[header_start:next_pos]
        units.append(NalUnit(start_code=prev_start_code, header=payload[:1], payload=payload[1:]))

        # Advance to next unit
        prev_start_code = next_code
        header_start = next_pos + len(next_code)
        if header_start >= n:
            break
        pos = header_start + 1

    return units if units else None


def serialize_nal_units(units: list[NalUnit]) -> bytes:
    """Serialize NAL units back to bytes."""
    buf = bytearray()
    for unit in units:
        buf.extend(unit.start_code)
        buf.extend(unit.header)
        buf.extend(unit.payload)
    return bytes(buf)


class NalMutator:
    """Structure-aware NAL unit bitstream mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        self._rng = rng or random
        units = parse_nal_units(data)
        if units is None or not units:
            return self._generate_random_nal_stream(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 8)
        mutators = [
            self._mutate_nal_type,
            self._mutate_nal_ref_idc,
            self._mutate_sps,
            self._mutate_pps,
            self._mutate_slice,
            self._duplicate_nal,
            self._delete_nal,
            self._reorder_nal,
            self._generate_random_nal_stream,
        ]
        result = mutators[op](units, max_len)
        if isinstance(result, list):
            return serialize_nal_units(result)[:max_len]
        return result[:max_len]

    def _mutate_nal_type(self, units: list[NalUnit], max_len: int) -> list[NalUnit]:
        """Corrupt nal_unit_type field (lower 5 bits of header)."""
        target = self._rng.choice(units)
        if target.header:
            hdr = bytearray(target.header)
            weird_types = [0, 6, 10, 12, 15, 20, 24, 30]
            new_type = self._rng.choice(weird_types)
            hdr[0] = (hdr[0] & 0xE0) | (new_type & 0x1F)
            target.header = bytes(hdr)
        return units

    def _mutate_nal_ref_idc(self, units: list[NalUnit], max_len: int) -> list[NalUnit]:
        """Corrupt nal_ref_idc field (upper 2 bits of header)."""
        target = self._rng.choice(units)
        if target.header:
            hdr = bytearray(target.header)
            new_ref = self._rng.choice([0, 1, 2, 3])
            hdr[0] = (hdr[0] & 0x9F) | (new_ref << 5)
            target.header = bytes(hdr)
        return units

    def _mutate_sps(self, units: list[NalUnit], max_len: int) -> list[NalUnit]:
        """Apply byte corruption to SPS NAL units."""
        for unit in units:
            if unit.unit_type in (H264_NAL_SPS, H265_NAL_SPS_HEVC) and len(unit.payload) > 4:
                data = bytearray(unit.payload)
                for _ in range(self._rng.randint(1, min(4, len(data)))):
                    idx = self._rng.randint(0, len(data) - 1)
                    data[idx] ^= 1 << self._rng.randint(0, 7)
                unit.payload = bytes(data)
        return units

    def _mutate_pps(self, units: list[NalUnit], max_len: int) -> list[NalUnit]:
        """Apply byte corruption to PPS NAL units."""
        for unit in units:
            if unit.unit_type in (H264_NAL_PPS, H265_NAL_PPS_HEVC) and unit.payload:
                data = bytearray(unit.payload)
                for _ in range(self._rng.randint(1, min(4, len(data)))):
                    idx = self._rng.randint(0, len(data) - 1)
                    data[idx] ^= 1 << self._rng.randint(0, 7)
                unit.payload = bytes(data)
        return units

    def _mutate_slice(self, units: list[NalUnit], max_len: int) -> list[NalUnit]:
        """Apply byte corruption to slice NAL units."""
        for unit in units:
            if (
                unit.unit_type
                in (H264_NAL_NON_IDR, H264_NAL_IDR, H265_NAL_IDR_W_RADL, H265_NAL_IDR_N_LP)
                and unit.payload
            ):
                data = bytearray(unit.payload)
                for _ in range(self._rng.randint(1, min(4, len(data)))):
                    idx = self._rng.randint(0, len(data) - 1)
                    data[idx] ^= 1 << self._rng.randint(0, 7)
                unit.payload = bytes(data)
        return units

    def _duplicate_nal(self, units: list[NalUnit], max_len: int) -> list[NalUnit]:
        """Clone a random NAL unit after the original."""
        if units:
            idx = self._rng.randint(0, len(units) - 1)
            orig = units[idx]
            dup = NalUnit(
                start_code=orig.start_code, header=orig.header[:], payload=orig.payload[:]
            )
            units.insert(idx + 1, dup)
        return units

    def _delete_nal(self, units: list[NalUnit], max_len: int) -> list[NalUnit]:
        """Delete a non-critical NAL unit (keep SPS/PPS/VPS)."""
        deletable = [
            i
            for i, u in enumerate(units)
            if u.unit_type not in CRITICAL_NAL_TYPES_H264
            and u.unit_type not in CRITICAL_NAL_TYPES_H265
        ]
        if deletable:
            idx = self._rng.choice(deletable)
            units.pop(idx)
        return units

    def _reorder_nal(self, units: list[NalUnit], max_len: int) -> list[NalUnit]:
        """Swap two random NAL units."""
        if len(units) >= 2:
            i, j = self._rng.sample(list(range(len(units))), 2)
            units[i], units[j] = units[j], units[i]
        return units

    def _generate_random_nal_stream(self, _units=None, max_len: int = 65536, rng=None) -> bytes:
        """Generate a minimal random H.264 NAL stream from scratch."""
        # An int in the first slot is a max_len passed positionally. Without
        # this the cap lands in the vestigial placeholder and is dropped, and
        # the generator silently falls back to its own default -- the same
        # overload bmp/gzip/jpeg/zlib already handle and document.
        if isinstance(_units, int):
            max_len = _units
        self._rng = rng or self._rng
        units: list[NalUnit] = []

        # Minimal valid stream: SPS, PPS, IDR slice
        sps_payload = bytes(self._rng.randint(0, 255) for _ in range(self._rng.randint(8, 24)))
        units.append(NalUnit(start_code=START_CODE_4, header=bytes([0x67]), payload=sps_payload))

        pps_payload = bytes(self._rng.randint(0, 255) for _ in range(self._rng.randint(2, 8)))
        units.append(NalUnit(start_code=START_CODE_3, header=bytes([0x68]), payload=pps_payload))

        slice_payload = bytes(self._rng.randint(0, 255) for _ in range(self._rng.randint(16, 64)))
        units.append(NalUnit(start_code=START_CODE_4, header=bytes([0x65]), payload=slice_payload))

        result = serialize_nal_units(units)
        return result[:max_len]
