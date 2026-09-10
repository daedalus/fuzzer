"""Structure-aware RASC mutations.

Parses RASC INIT/DLTA chunk structure in AVI containers.
Targets chunk size/offset corruption and sequence header tampering.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool


@dataclass
class RascChunk:
    """A RASC chunk descriptor."""

    chunk_type: int  # INIT=0, DLTA=1
    size: int
    offset: int
    seq_num: int
    data: bytes


def parse_rasc(data: bytes) -> list[RascChunk] | None:
    """Parse RASC chunk structure from data."""
    if len(data) < 8:
        return None

    chunks = []
    pos = 0

    while pos + 8 <= len(data):
        chunk_type = data[pos]
        size = struct.unpack_from("<I", data, pos + 1)[0]
        offset = struct.unpack_from("<I", data, pos + 5)[0]

        if chunk_type > 1:
            break

        seq_num = struct.unpack_from("<I", data, pos + 8)[0] if pos + 12 <= len(data) else 0
        chunk_data = data[pos + 12 : pos + 12 + size] if pos + 12 + size <= len(data) else b""
        chunks.append(RascChunk(chunk_type, size, offset, seq_num, chunk_data))
        pos += 12 + size

    return chunks if chunks else None


class RascMutator:
    """Structure-aware RASC mutator.

    Targets chunk size/offset corruption and sequence header tampering.
    """

    def __init__(self, seed=None):
        # One pool per mutator, built once. Callers that own a pool pass it
        # as ``rng=`` and it wins for that call; this is the standalone
        # default, never the stdlib module (Hard Rule 16).
        rng = RandPool(seed=seed)
        self._rng = rng

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        """Apply one RASC-specific mutation."""
        self._rng = rng or self._rng

        chunks = parse_rasc(data)
        if not chunks:
            return self._generate_random_rasc(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 3)
        mutators = [
            self._mutate_chunk_size,
            self._mutate_chunk_offset,
            self._mutate_seq_num,
            # The generator replaces the input rather than editing it, so it
            # takes neither the buffer nor the parsed chunks the other entries
            # do. Adapted here rather than given vestigial `_data`/`_chunks`
            # parameters: that placeholder shape is exactly what f5435af had
            # to unpick across ten generators, where a positional `max_len`
            # landed in the placeholder and the generator silently fell back
            # to its own default.
            lambda _data, _chunks, max_len: self._generate_random_rasc(
                max_len=max_len, rng=self._rng
            ),
        ]
        result = mutators[op](data, chunks, max_len)
        return result[:max_len]

    def _mutate_chunk_size(self, data: bytes, chunks: list[RascChunk], max_len: int) -> bytes:
        """Corrupt chunk size to trigger signed overflow."""
        raw = bytearray(data)
        chunk = self._rng.choice(chunks)
        pos = chunk.offset + 1 if chunk.offset > 0 else 1
        if pos + 4 <= len(raw):
            struct.pack_into("<I", raw, pos, 0xFFFFFFFF)
        return bytes(raw[:max_len])

    def _mutate_chunk_offset(self, data: bytes, chunks: list[RascChunk], max_len: int) -> bytes:
        """Corrupt chunk offset to cause out-of-bounds reads."""
        raw = bytearray(data)
        chunk = self._rng.choice(chunks)
        pos = chunk.offset + 5 if chunk.offset > 0 else 5
        if pos + 4 <= len(raw):
            struct.pack_into("<I", raw, pos, 0xFFFFFFFF)
        return bytes(raw[:max_len])

    def _mutate_seq_num(self, data: bytes, chunks: list[RascChunk], max_len: int) -> bytes:
        """Mutate sequence number to break ordering."""
        raw = bytearray(data)
        chunk = self._rng.choice(chunks)
        pos = chunk.offset + 8 if chunk.offset > 0 else 8
        if pos + 4 <= len(raw):
            struct.pack_into("<I", raw, pos, chunk.seq_num + 1)
        return bytes(raw[:max_len])

    def _generate_random_rasc(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Generate minimal RASC structure with corrupt chunk."""
        self._rng = rng or self._rng

        result = bytearray()
        result += bytes([0x00])  # chunk_type (INIT)
        result += struct.pack("<I", 0xFFFFFFFF)  # size (signed overflow)
        result += struct.pack("<I", 0xFFFFFFFF)  # offset (signed overflow)
        result += struct.pack("<I", 0x10000000)  # seq_num (signed overflow)
        result += b"\x00" * 64

        return bytes(result[:max_len])


__all__ = ["parse_rasc", "RascMutator", "RascChunk"]
