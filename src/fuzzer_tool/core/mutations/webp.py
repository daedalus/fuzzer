"""Structure-aware WebP mutator.

WebP structure (RIFF container):
  [RIFF][u32le size][WEBP][chunks...]
  Chunk: [fourcc: 4 bytes][u32le size][payload][even-pad]

Payload layouts (field views for targeted mutation):
  VP8  — [3B tag][u32le frame size][9D 01 2A start code][u16le w-1][u16le h-1]
  VP8L — [2F][4 bytes: 14-bit w-1 | 14-bit h-1 | alpha | version]
  VP8X — [u8 flags][u24le w-1][u24le h-1]
  ANMF — [u24le x][u24le y][u24le w][u24le h][u24le dur][u8 flags]
         followed by embedded chunks (children)

Serialization writes the stored size_orig in each chunk header verbatim
(declared-size-vs-payload mismatch is a deliberate corruption probe) and
recomputes only the outer RIFF size field.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.generic import _swap_pair

import random
import struct
from dataclasses import dataclass, field

# Chunk fourcc values that can be swapped in
CHUNK_TYPES = [b"VP8 ", b"VP8L", b"VP8X", b"ANIM", b"ANMF", b"ALPH", b"EXIF", b"ICCP", b"XMP "]
WEIRD_FOURCCS = [b"\x00\x00\x00\x00", b"\xff\xff\xff\xff", b"    ", b"xxxx", b"????"]

# Interesting declared chunk sizes
SIZE_VALUES = [0, 1, 8, 12, 16, 0xFFFFFFFF]

# Interesting VP8X flag values
VP8X_FLAG_VALUES = [0x00, 0x02, 0x04, 0x08, 0x10, 0xFF]

# Interesting ANMF duration values (in milliseconds)
DURATION_VALUES = [0, 1, 10, 100, 1000, 0xFFFFFF]


@dataclass
class Chunk:
    """A single RIFF chunk inside a WebP file."""

    fourcc: bytes
    size_orig: int  # original declared size
    payload: bytes
    children: list[Chunk] = field(default_factory=list)


def _parse_children(data: bytes, start: int) -> tuple[list[Chunk], bool]:
    """Parse chunk children starting at *start*.

    Returns (chunks, ok). ok is False if any chunk is truncated.
    """
    chunks: list[Chunk] = []
    pos = start
    while pos + 8 <= len(data):
        fourcc = data[pos : pos + 4]
        csize = struct.unpack_from("<I", data, pos + 4)[0]
        if pos + 8 + csize > len(data):
            return chunks, False
        payload = data[pos + 8 : pos + 8 + csize]
        children: list[Chunk] = []
        if fourcc == b"ANMF" and len(payload) >= 16:
            sub, ok = _parse_children(payload, 16)
            if ok:
                children = sub
        chunks.append(Chunk(fourcc, csize, payload, children))
        pos += 8 + csize + (csize & 1)
    return chunks, pos == len(data)


def parse_webp(data: bytes) -> list[Chunk] | None:
    """Parse a WebP file into a list of chunks.

    Returns None unless the data starts with RIFF, the RIFF size field
    matches the buffer length, and the WEBP tag is present. Trailing
    garbage after the last chunk also yields None.
    """
    if len(data) < 12:
        return None
    if data[:4] != b"RIFF":
        return None
    if struct.unpack_from("<I", data, 4)[0] != len(data) - 8:
        return None
    if data[8:12] != b"WEBP":
        return None

    chunks, ok = _parse_children(data, 12)
    if not ok or not chunks:
        return None
    return chunks


def serialize_webp(chunks: list[Chunk]) -> bytes:
    """Serialize chunks back to bytes.

    The outer RIFF size is recomputed; each chunk header writes its
    stored size_orig verbatim (may mismatch the payload on purpose).
    """
    body = bytearray()
    for c in chunks:
        body.extend(c.fourcc)
        body.extend(struct.pack("<I", c.size_orig))
        body.extend(c.payload)
        if len(c.payload) & 1:
            body.append(0)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + bytes(body)


def _find_chunks(chunks: list[Chunk], *fourccs: bytes) -> list[Chunk]:
    found: list[Chunk] = []
    for c in chunks:
        if c.fourcc in fourccs:
            found.append(c)
        found.extend(_find_chunks(c.children, *fourccs))
    return found


def _mutate_u24(payload: bytearray, off: int, value: int) -> None:
    payload[off : off + 3] = struct.pack("<I", value & 0xFFFFFF)[:3]


class WebpMutator:
    """Structure-aware WebP mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        self._rng = rng or random
        chunks = parse_webp(data)
        if chunks is None:
            return self._generate_random_webp(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 10)
        mutators = [
            self._mutate_chunk_type,
            self._mutate_chunk_size,
            self._mutate_riff_size,
            self._mutate_vp8_header,
            self._mutate_vp8l_header,
            self._mutate_vp8x,
            self._mutate_anmf_field,
            self._swap_chunks,
            self._duplicate_chunk,
            self._delete_chunk,
            self._truncate_chunk,
        ]
        result = mutators[op](chunks, max_len)
        if isinstance(result, list):
            return serialize_webp(result)[:max_len]
        return result[:max_len]

    def _mutate_chunk_type(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        target = self._rng.choice(chunks)
        current = target.fourcc
        options = [t for t in CHUNK_TYPES if t != current] + WEIRD_FOURCCS
        target.fourcc = self._rng.choice(options)
        return chunks

    def _mutate_chunk_size(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        target = self._rng.choice(chunks)
        target.size_orig = self._rng.choice(
            SIZE_VALUES + [self._rng.randint(0, max_len), len(target.payload)]
        )
        return chunks

    def _mutate_riff_size(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        """Rewrite the outer RIFF size field (deliberately inconsistent)."""
        raw = bytearray(serialize_webp(chunks))
        new_size = self._rng.choice(
            [0, 4, 8, max_len, 0xFFFFFFFF, self._rng.randint(0, max(1, max_len))]
        )
        struct.pack_into("<I", raw, 4, new_size & 0xFFFFFFFF)
        return bytes(raw)

    def _mutate_vp8_header(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        vp8 = _find_chunks(chunks, b"VP8 ")
        if not vp8:
            return chunks
        target = self._rng.choice(vp8)
        payload = bytearray(target.payload)
        if len(payload) < 14:
            return chunks
        # [3B tag][u32le frame size][9D 01 2A][u16le w-1][u16le h-1]
        if self._rng.random() < 0.5:
            payload[3:7] = struct.pack(
                "<I",
                self._rng.choice(
                    [0, 1, len(payload), 0xFFFFFFFF, self._rng.randint(0, 0xFFFFFFFF)]
                ),
            )
        else:
            off = self._rng.choice([10, 12])
            payload[off : off + 2] = struct.pack(
                "<H", self._rng.choice([0, 1, 2, 0xFFFF, self._rng.randint(0, 0xFFFF)])
            )
        target.payload = bytes(payload)
        return chunks

    def _mutate_vp8l_header(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        vp8l = _find_chunks(chunks, b"VP8L")
        if not vp8l:
            return chunks
        target = self._rng.choice(vp8l)
        payload = bytearray(target.payload)
        if len(payload) < 5:
            return chunks
        if self._rng.random() < 0.5:
            # Toggle the alpha-is-used bit (bit 28)
            bits = struct.unpack_from("<I", payload, 1)[0]
            bits ^= 1 << 28
            struct.pack_into("<I", payload, 1, bits)
        else:
            bits = struct.unpack_from("<I", payload, 1)[0]
            w1 = bits & 0x3FFF
            h1 = (bits >> 14) & 0x3FFF
            if self._rng.random() < 0.5:
                w1 = self._rng.choice([0, 1, 2, 0x3FFF, self._rng.randint(0, 0x3FFF)])
            else:
                h1 = self._rng.choice([0, 1, 2, 0x3FFF, self._rng.randint(0, 0x3FFF)])
            bits = w1 | (h1 << 14) | (bits & 0xC0000000)
            struct.pack_into("<I", payload, 1, bits)
        target.payload = bytes(payload)
        return chunks

    def _mutate_vp8x(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        vp8x = _find_chunks(chunks, b"VP8X")
        if not vp8x:
            return chunks
        target = self._rng.choice(vp8x)
        payload = bytearray(target.payload)
        if len(payload) < 10:
            return chunks
        if self._rng.random() < 0.5:
            payload[0] = self._rng.choice(VP8X_FLAG_VALUES + [self._rng.randint(0, 0xFF)])
        else:
            off = self._rng.choice([1, 4])
            _mutate_u24(
                payload, off, self._rng.choice([0, 1, 2, 0xFFFFFF, self._rng.randint(0, 0xFFFFFF)])
            )
        target.payload = bytes(payload)
        return chunks

    def _mutate_anmf_field(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        anmf = _find_chunks(chunks, b"ANMF")
        if not anmf:
            return chunks
        target = self._rng.choice(anmf)
        payload = bytearray(target.payload)
        if len(payload) < 16:
            return chunks
        field_off = self._rng.choice([0, 3, 6, 9, 12])
        if field_off == 12:
            payload[12:15] = struct.pack("<I", self._rng.choice(DURATION_VALUES))[:3]
        elif field_off == 15:
            payload[15] = self._rng.randint(0, 0xFF)
        else:
            _mutate_u24(
                payload,
                field_off,
                self._rng.choice([0, 1, 2, 0xFFFFFF, self._rng.randint(0, 0xFFFFFF)]),
            )
        target.payload = bytes(payload)
        return chunks

    def _swap_chunks(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        if (pair := _swap_pair(len(chunks), self._rng)) is not None:
            i, j = pair
            chunks[i], chunks[j] = chunks[j], chunks[i]
        return chunks

    def _duplicate_chunk(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        if chunks:
            idx = self._rng.randint(0, len(chunks) - 1)
            orig = chunks[idx]
            dup = Chunk(
                fourcc=orig.fourcc,
                size_orig=orig.size_orig,
                payload=orig.payload[:],
                children=list(orig.children),
            )
            chunks.insert(idx + 1, dup)
        return chunks

    def _delete_chunk(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        if len(chunks) > 1:
            chunks.pop(self._rng.randint(0, len(chunks) - 1))
        return chunks

    def _truncate_chunk(self, chunks: list[Chunk], max_len: int) -> list[Chunk]:
        if chunks:
            target = self._rng.choice(chunks)
            if len(target.payload) > 4:
                target.payload = target.payload[
                    : self._rng.randint(1, max(2, len(target.payload) // 2))
                ]
        return chunks

    def _generate_random_webp(self, _chunks=None, max_len: int = 4096, rng=None) -> bytes:
        """Generate a minimal random WebP file."""
        # An int in the first slot is a max_len passed positionally. Without
        # this the cap lands in the vestigial placeholder and is dropped, and
        # the generator silently falls back to its own default -- the same
        # overload bmp/gzip/jpeg/zlib already handle and document.
        if isinstance(_chunks, int):
            max_len = _chunks
        self._rng = rng or self._rng

        # VP8X chunk: flags + canvas dimensions (w-1, h-1)
        vp8x_payload = (
            bytes([0x00]) + struct.pack("<I", 0x000000)[:3] + struct.pack("<I", 0x000000)[:3]
        )
        vp8x = Chunk(b"VP8X", len(vp8x_payload), vp8x_payload)

        # Minimal VP8L chunk: 2F + 4 bytes (w-1=0, h-1=0, alpha=1, version=0)
        vp8l_payload = b"\x2f" + struct.pack("<I", 1 << 28)
        vp8l = Chunk(b"VP8L", len(vp8l_payload), vp8l_payload)

        chunks = [vp8x, vp8l]
        if self._rng.random() < 0.3:
            anmf_payload = (
                struct.pack("<I", 0)[:3]
                + struct.pack("<I", 0)[:3]
                + struct.pack("<I", 1)[:3]
                + struct.pack("<I", 1)[:3]
                + struct.pack("<I", 100)[:3]
                + bytes([0x00])
            )
            anmf_payload += serialize_webp([vp8l])
            anmf = Chunk(b"ANMF", len(anmf_payload), anmf_payload)
            chunks.append(anmf)

        return serialize_webp(chunks)[:max_len]
