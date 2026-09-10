"""Structure-aware TIFF mutations.

Parses TIFF structure (Byte-Order, header, IFD entries).
Targets IFD entry count overflow, offset wrapping, and tag manipulation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool


@dataclass
class TiffIfdEntry:
    """A TIFF IFD entry."""

    tag: int
    type_num: int
    count: int
    value_offset: int


@dataclass
class TiffHeader:
    """TIFF header structure."""

    byte_order: str  # "II" or "MM"
    magic: int
    offset: int
    num_entries: int
    ifd_entries: list[TiffIfdEntry]


def parse_tiff(data: bytes) -> TiffHeader | None:
    """Parse TIFF structure from data."""
    if len(data) < 8:
        return None

    byte_order = data[:2]
    if byte_order not in (b"II", b"MM"):
        return None

    magic = struct.unpack_from("<H", data, 2)[0]
    if magic != 0x002A:
        return None

    offset = struct.unpack_from("<I", data, 4)[0]
    num_entries = struct.unpack_from("<H", data, offset)[0]

    entries = []
    entry_size = 12
    ifd_pos = offset + 2

    for _i in range(min(num_entries, 100)):
        if ifd_pos + entry_size > len(data):
            break
        tag = struct.unpack_from("<H", data, ifd_pos)[0]
        type_num = struct.unpack_from("<H", data, ifd_pos + 2)[0]
        count = struct.unpack_from("<I", data, ifd_pos + 4)[0]
        value_offset = struct.unpack_from("<I", data, ifd_pos + 8)[0]
        entries.append(TiffIfdEntry(tag, type_num, count, value_offset))
        ifd_pos += entry_size

    return TiffHeader(byte_order.decode(), magic, offset, num_entries, entries)


class TiffMutator:
    """Structure-aware TIFF mutator.

    Targets IFD entry count overflow, offset wrapping, and tag manipulation.
    """

    def __init__(self, seed=None):
        # One pool per mutator, built once. Callers that own a pool pass it
        # as ``rng=`` and it wins for that call; this is the standalone
        # default, never the stdlib module (Hard Rule 16).
        rng = RandPool(seed=seed)
        self._rng = rng

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        """Apply one TIFF-specific mutation."""
        self._rng = rng or self._rng
        header = parse_tiff(data)
        if not header or header.num_entries == 0:
            return self._generate_random_tiff(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 4)
        mutators = [
            self._mutate_ifd_count,
            self._corrupt_offsets,
            self._mutate_tag,
            self._corrupt_byte_order,
            # The generator replaces the input rather than editing it, so it
            # takes neither the buffer nor the parsed header the other entries
            # do. Adapted here rather than given vestigial `_data`/`_header`
            # parameters: that placeholder shape is exactly what f5435af had
            # to unpick across ten generators, where a positional `max_len`
            # landed in the placeholder and the generator silently fell back
            # to its own default.
            lambda _data, _header, max_len: self._generate_random_tiff(
                max_len=max_len, rng=self._rng
            ),
        ]
        result = mutators[op](data, header, max_len)
        return result[:max_len]

    def _mutate_ifd_count(self, data: bytes, header: TiffHeader, max_len: int) -> bytes:
        """Corrupt IFD entry count to trigger overflow."""
        raw = bytearray(data)
        offset = header.offset
        if offset + 2 <= len(raw):
            struct.pack_into("<H", raw, offset, 0xFFFF)
        return bytes(raw[:max_len])

    def _corrupt_offsets(self, data: bytes, header: TiffHeader, max_len: int) -> bytes:
        """Corrupt IFD entry offsets."""
        raw = bytearray(data)
        entry_size = 12
        ifd_pos = header.offset + 2

        for i in range(min(4, header.num_entries)):
            pos = ifd_pos + i * entry_size
            if pos + 4 <= len(raw):
                struct.pack_into("<I", raw, pos + 8, 0xFFFFFFFF)
        return bytes(raw[:max_len])

    def _mutate_tag(self, data: bytes, header: TiffHeader, max_len: int) -> bytes:
        """Mutate IFD tag values."""
        raw = bytearray(data)
        entry_size = 12
        ifd_pos = header.offset + 2

        if header.num_entries > 0 and ifd_pos + entry_size <= len(raw):
            struct.pack_into("<H", raw, ifd_pos, self._rng.randint(0, 0xFFFF))
        return bytes(raw[:max_len])

    def _corrupt_byte_order(self, data: bytes, header: TiffHeader, max_len: int) -> bytes:
        """Corrupt byte-order marker to cause misalignment."""
        raw = bytearray(data)
        raw[0] = 0x4D  # "M" instead of "I"
        return bytes(raw[:max_len])

    def _generate_random_tiff(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Generate minimal TIFF with corrupt IFD."""
        self._rng = rng or self._rng
        result = bytearray()
        result += b"II"  # little-endian
        result += struct.pack("<H", 0x002A)  # magic
        result += struct.pack("<I", 8)  # IFD offset
        result += struct.pack("<H", 0xFFFF)  # IFD entry count (overflow)
        result += b"\x00" * 64

        return bytes(result[:max_len])


__all__ = ["parse_tiff", "TiffMutator", "TiffHeader", "TiffIfdEntry"]
