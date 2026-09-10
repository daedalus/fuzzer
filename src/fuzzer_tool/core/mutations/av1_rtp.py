"""Structure-aware AV1 over RTP mutations.

Parses AV1 OBU headers to identify obu_type, temporal_id, spatial_id,
and tile layout. Targets temporal_id/spatial_id corruption and OBU size overflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool


@dataclass
class Av1Obu:
    """An AV1 Open Bitstream Unit."""

    type: int
    temporal_id: int
    spatial_id: int
    size: int
    offset: int
    data: bytes


def parse_av1_obus(data: bytes) -> list[Av1Obu] | None:
    """Parse AV1 OBUs from data."""
    if len(data) < 2:
        return None

    obus = []
    pos = 0

    while pos + 2 <= len(data):
        header = data[pos]
        obu_type = (header >> 3) & 0x7
        temporal_id = data[pos + 1] & 0x7
        spatial_id = (data[pos + 1] >> 3) & 0x7

        # Handle OBU header size variants
        size = 1
        if header & 0x02:
            if pos + 3 > len(data):
                break
            size = data[pos + 2]
            if size & 0x80:
                # Two-byte size: the continuation byte is at pos + 3, which
                # the guard above does not cover -- it only proves pos + 2 is
                # readable. A 3-byte buffer whose last byte has 0x80 set
                # raised IndexError here.
                if pos + 4 > len(data):
                    break
                size = (size & 0x7F) << 8
                size |= data[pos + 3]
                pos += 3
            else:
                pos += 2
        else:
            pos += 2

        if pos + size > len(data):
            break

        obus.append(Av1Obu(obu_type, temporal_id, spatial_id, size, pos, data[pos : pos + size]))
        pos += size

    return obus if obus else None


class Av1RtpMutator:
    """Structure-aware AV1 over RTP mutator.

    Targets temporal_id/spatial_id corruption and OBU size overflow.
    """

    def __init__(self, seed=None):
        # One pool per mutator, built once. Callers that own a pool pass it
        # as ``rng=`` and it wins for that call; this is the standalone
        # default, never the stdlib module (Hard Rule 16).
        rng = RandPool(seed=seed)
        self._rng = rng

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        """Apply one AV1 over RTP-specific mutation."""
        self._rng = rng or self._rng
        obus = parse_av1_obus(data)
        if not obus:
            return self._generate_random_av1(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 4)
        mutators = [
            self._mutate_temporal_id,
            self._mutate_spatial_id,
            self._mutate_obu_type,
            self._mutate_obu_size,
            # The generator replaces the input rather than editing it, so it
            # takes neither the buffer nor the OBUs the other entries
            # do. Adapted here rather than given vestigial `_data`/`_obus`
            # parameters: that placeholder shape is exactly what f5435af had
            # to unpick across ten generators, where a positional `max_len`
            # landed in the placeholder and the generator silently fell back
            # to its own default.
            lambda _data, _obus, max_len: self._generate_random_av1(max_len=max_len, rng=self._rng),
        ]
        result = mutators[op](data, obus, max_len)
        return result[:max_len]

    def _mutate_temporal_id(self, data: bytes, obus: list[Av1Obu], max_len: int) -> bytes:
        """Mutate temporal_id field."""
        if not obus:
            return data
        raw = bytearray(data)
        obu = self._rng.choice(obus)
        header_pos = obu.offset - 2
        if header_pos >= 0 and header_pos + 2 <= len(raw):
            raw[header_pos + 1] = (raw[header_pos + 1] & 0xF8) | self._rng.randint(0, 7)
        return bytes(raw[:max_len])

    def _mutate_spatial_id(self, data: bytes, obus: list[Av1Obu], max_len: int) -> bytes:
        """Mutate spatial_id field."""
        if not obus:
            return data
        raw = bytearray(data)
        obu = self._rng.choice(obus)
        header_pos = obu.offset - 2
        if header_pos >= 0 and header_pos + 2 <= len(raw):
            raw[header_pos + 1] = (raw[header_pos + 1] & 0xC7) | (self._rng.randint(0, 7) << 3)
        return bytes(raw[:max_len])

    def _mutate_obu_type(self, data: bytes, obus: list[Av1Obu], max_len: int) -> bytes:
        """Mutate obu_type field."""
        if not obus:
            return data
        raw = bytearray(data)
        obu = self._rng.choice(obus)
        header_pos = obu.offset - 2
        if header_pos >= 0:
            raw[header_pos] = (raw[header_pos] & 0x87) | (self._rng.randint(0, 7) << 3)
        return bytes(raw[:max_len])

    def _mutate_obu_size(self, data: bytes, obus: list[Av1Obu], max_len: int) -> bytes:
        """Mutate OBU size to trigger overflow."""
        if not obus:
            return data
        raw = bytearray(data)
        obu = self._rng.choice(obus)
        size_pos = obu.offset - 1
        if size_pos >= 0:
            raw[size_pos] = 0xFF
        return bytes(raw[:max_len])

    def _generate_random_av1(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Generate minimal AV1 OBU with corrupt temporal_id/spatial_id."""
        self._rng = rng or self._rng
        r = self._rng

        result = bytearray()
        result += bytes([0x28 | (r.randint(0, 7) << 3)])  # OBU header
        result += bytes([0x00 | (r.randint(0, 7) << 3) | r.randint(0, 7)])  # temporal/spatial id
        result += bytes([0xFF])  # OBU size (overflow)
        result += b"\x00" * 64

        return bytes(result[:max_len])


__all__ = ["parse_av1_obus", "parse_av1", "Av1RtpMutator", "Av1Obu"]

# Alias used by the operator handler.
parse_av1 = parse_av1_obus
