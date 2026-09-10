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


# Record layout, as `_generate_random_rasc` writes it: chunk_type(1) at 0,
# size(4) at 1, offset(4) at 5, seq_num(4) at 9, payload from 13. The
# offsets are named because the parser previously used 8 and 12 for the
# last two, which is one byte short at both: seq_num re-read the top byte
# of offset (a generated 0x10000000 parsed back as 255) and the stride
# dropped a byte per record.
_RASC_SIZE_OFF = 1
_RASC_OFFSET_OFF = 5
_RASC_SEQ_OFF = 9
_RASC_HEADER_LEN = 13
# Everything up to and including `offset` is read unconditionally.
_RASC_MIN_RECORD = _RASC_OFFSET_OFF + 4


def parse_rasc(data: bytes) -> list[RascChunk] | None:
    """Parse RASC chunk structure from data."""
    if len(data) < _RASC_MIN_RECORD:
        return None

    chunks = []
    pos = 0

    while pos + _RASC_MIN_RECORD <= len(data):
        chunk_type = data[pos]
        size = struct.unpack_from("<I", data, pos + _RASC_SIZE_OFF)[0]
        offset = struct.unpack_from("<I", data, pos + _RASC_OFFSET_OFF)[0]

        if chunk_type > 1:
            break

        have_seq = pos + _RASC_SEQ_OFF + 4 <= len(data)
        seq_num = struct.unpack_from("<I", data, pos + _RASC_SEQ_OFF)[0] if have_seq else 0
        end = pos + _RASC_HEADER_LEN + size
        chunk_data = data[pos + _RASC_HEADER_LEN : end] if end <= len(data) else b""
        chunks.append(RascChunk(chunk_type, size, offset, seq_num, chunk_data))
        pos += _RASC_HEADER_LEN + size

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
            struct.pack_into("<I", raw, pos, (chunk.seq_num + 1) & 0xFFFFFFFF)
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
