"""Structure-aware CFHD mutations.

Parses CFHD chunk structure (frame headers, block headers, coefficient blocks).
Targets coefficient block corruption and header tampering.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool


@dataclass
class CfhdFrameHeader:
    """CFHD frame header structure."""

    version: int
    frame_num: int
    slice_count: int
    block_count: int
    width: int
    height: int


def parse_cfhd_frame_header(data: bytes) -> CfhdFrameHeader | None:
    """Parse CFHD frame header from data."""
    if len(data) < 24:
        return None

    return CfhdFrameHeader(
        version=struct.unpack_from("<I", data, 0)[0],
        frame_num=struct.unpack_from("<I", data, 4)[0],
        slice_count=struct.unpack_from("<I", data, 8)[0],
        block_count=struct.unpack_from("<I", data, 12)[0],
        width=struct.unpack_from("<I", data, 16)[0],
        height=struct.unpack_from("<I", data, 20)[0],
    )


class CfhdMutator:
    """Structure-aware CFHD mutator.

    Targets coefficient block corruption and header tampering.
    """

    def __init__(self, seed=None):
        # One pool per mutator, built once. Callers that own a pool pass it
        # as ``rng=`` and it wins for that call; this is the standalone
        # default, never the stdlib module (Hard Rule 16).
        rng = RandPool(seed=seed)
        self._rng = rng
    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        """Apply one CFHD-specific mutation."""
        self._rng = rng or self._rng

        frame_header = parse_cfhd_frame_header(data)
        if not frame_header or frame_header.block_count == 0:
            return self._generate_random_cfhd(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 4)
        mutators = [
            self._mutate_frame_header,
            self._corrupt_coeff_blocks,
            self._add_zero_block,
            self._remove_slice_header,
            self._generate_random_cfhd,
        ]
        result = mutators[op](data, frame_header, max_len)
        return result[:max_len]

    def _mutate_frame_header(self, data: bytes, header: CfhdFrameHeader, max_len: int) -> bytes:
        """Mutate frame header fields."""
        raw = bytearray(data)
        op = self._rng.randint(0, 5)

        if op == 0 and header.version > 0:
            struct.pack_into("<I", raw, 0, header.version - 1)
        elif op == 1:
            struct.pack_into("<I", raw, 4, header.frame_num + 1)
        elif op == 2:
            struct.pack_into("<I", raw, 8, header.slice_count + 2)
        elif op == 3:
            struct.pack_into("<I", raw, 12, max(1, header.block_count - 2))
        elif op == 4 and header.width > 1:
            struct.pack_into("<I", raw, 16, header.width - 16)

        return bytes(raw[:max_len])

    def _corrupt_coeff_blocks(self, data: bytes, header: CfhdFrameHeader, max_len: int) -> bytes:
        """Corrupt coefficient blocks to break decoding."""
        raw = bytearray(data)
        coeff_offset = 24

        for _ in range(min(8, header.block_count)):
            if coeff_offset + 4 <= len(raw):
                corrupt_len = min(64, len(raw) - coeff_offset)
                raw[coeff_offset : coeff_offset + corrupt_len] = bytes(
                    [self._rng.randint(0, 255) for _ in range(corrupt_len)]
                )
            coeff_offset += 64

        return bytes(raw[:max_len])

    def _add_zero_block(self, data: bytes, header: CfhdFrameHeader, max_len: int) -> bytes:
        """Add a zero block header to frame."""
        raw = bytearray(data)
        if len(raw) < 24 + 32:  # frame header + one block
            return self._generate_random_cfhd(max_len=max_len, rng=self._rng)

        frame_header_size = 24
        block_start = frame_header_size + header.block_count * 32
        if block_start + 32 <= len(raw):
            raw[block_start : block_start + 32] = bytes(32)

        return bytes(raw[:max_len])

    def _remove_slice_header(self, data: bytes, header: CfhdFrameHeader, max_len: int) -> bytes:
        """Remove a slice header by shifting data."""
        raw = bytearray(data)
        slice_header_size = 8
        slice_start = 24

        if slice_start + slice_header_size <= len(raw):
            del raw[slice_start : slice_start + slice_header_size]

        return bytes(raw[:max_len])

    def _generate_random_cfhd(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Generate minimal CFHD frame with corrupt header."""
        self._rng = rng or self._rng

        result = bytearray()
        result += struct.pack("<I", 0x100)  # version
        result += struct.pack("<I", 0x10000000)  # frame_num (signed overflow)
        result += struct.pack("<I", 0xFFFFFFFF)  # slice_count (signed overflow)
        result += struct.pack("<I", 0xFFFFFFFF)  # block_count (signed overflow)
        result += struct.pack("<I", 0x10)  # width
        result += struct.pack("<I", 0x10)  # height

        return bytes(result[:max_len])


__all__ = ["parse_cfhd_frame_header", "parse_cfhd", "CfhdMutator", "CfhdFrameHeader"]

# Alias used by the operator handler.
parse_cfhd = parse_cfhd_frame_header
