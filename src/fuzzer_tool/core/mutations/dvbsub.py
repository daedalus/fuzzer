"""Structure-aware DVBSub mutations.

Parses DVBSub subtitle codec data in WTV/EBML containers.
Targets CVE-2026-70628: signed integer overflow wraps guard.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool

# WTV/EBML markers
WTV_MAGIC = b"\x1a\x45\xdf\xa3"


@dataclass
class DvbsubBuffer:
    """A DVBSub buffer descriptor."""

    size: int
    offset: int
    data: bytes


def parse_dvbsub(data: bytes) -> list[DvbsubBuffer] | None:
    """Parse DVBSub buffer structures from data."""
    if len(data) < 8:
        return None

    buffers = []
    pos = 0

    while pos + 8 <= len(data):
        size = struct.unpack_from("<I", data, pos)[0]
        offset = struct.unpack_from("<I", data, pos + 4)[0]

        if size < 0x10000000 and offset < 0x10000000:
            buf_data = data[pos + 8 : pos + 8 + size] if pos + 8 + size <= len(data) else b""
            buffers.append(DvbsubBuffer(size, offset, buf_data))
            pos += 8 + size
        else:
            break

    return buffers if buffers else None


class DvbsubMutator:
    """Structure-aware DVBSub mutator.

    Targets CVE-2026-70628: buffer size/offset overflow.
    """

    def __init__(self, seed=None):
        # One pool per mutator, built once. Callers that own a pool pass it
        # as ``rng=`` and it wins for that call; this is the standalone
        # default, never the stdlib module (Hard Rule 16).
        rng = RandPool(seed=seed)
        self._rng = rng

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        """Apply one DVBSub-specific mutation."""
        self._rng = rng or self._rng
        buffers = parse_dvbsub(data)
        if not buffers:
            return self._generate_random_dvbsub(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 3)
        mutators = [
            self._mutate_buffer_size,
            self._mutate_buffer_offset,
            self._corrupt_buffer_data,
            # The generator replaces the input rather than editing it, so it
            # takes neither the buffer nor the parsed buffers the other entries
            # do. Adapted here rather than given vestigial `_data`/`_buffers`
            # parameters: that placeholder shape is exactly what f5435af had
            # to unpick across ten generators, where a positional `max_len`
            # landed in the placeholder and the generator silently fell back
            # to its own default.
            lambda _data, _buffers, max_len: self._generate_random_dvbsub(
                max_len=max_len, rng=self._rng
            ),
        ]
        result = mutators[op](data, buffers, max_len)
        return result[:max_len]

    def _mutate_buffer_size(self, data: bytes, buffers: list[DvbsubBuffer], max_len: int) -> bytes:
        """Corrupt buffer size to trigger signed overflow."""
        if not buffers:
            return data
        raw = bytearray(data)
        buf = self._rng.choice(buffers)
        pos = buf.offset
        if pos + 4 <= len(raw):
            struct.pack_into("<I", raw, pos, 0x80000000)
        return bytes(raw[:max_len])

    def _mutate_buffer_offset(
        self, data: bytes, buffers: list[DvbsubBuffer], max_len: int
    ) -> bytes:
        """Corrupt buffer offset to trigger signed overflow."""
        if not buffers:
            return data
        raw = bytearray(data)
        buf = self._rng.choice(buffers)
        pos = buf.offset + 4
        if pos + 4 <= len(raw):
            struct.pack_into("<I", raw, pos, 0xFFFFFFFF)
        return bytes(raw[:max_len])

    def _corrupt_buffer_data(self, data: bytes, buffers: list[DvbsubBuffer], max_len: int) -> bytes:
        """Corrupt buffer data to cause mismatches."""
        if not buffers:
            return data
        raw = bytearray(data)
        buf = self._rng.choice(buffers)
        pos = buf.offset + 8
        if pos < len(raw):
            corrupt_len = min(32, len(raw) - pos)
            raw[pos : pos + corrupt_len] = bytes(
                [self._rng.randint(0, 255) for _ in range(corrupt_len)]
            )
        return bytes(raw[:max_len])

    def _generate_random_dvbsub(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Generate minimal data with malicious DVBSub structure."""
        self._rng = rng or self._rng
        result = bytearray()
        result += struct.pack("<I", 0x10000000)  # size
        result += struct.pack("<I", 0xFFFFFFFF)  # offset (signed overflow)
        result += b"\x00" * 64

        return bytes(result[:max_len])


__all__ = ["parse_dvbsub", "DvbsubMutator", "DvbsubBuffer"]
