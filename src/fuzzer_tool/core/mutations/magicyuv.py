"""Structure-aware MagicYUV mutations.

Parses AVI stream headers (BMH5/BMH6 chunks) to identify slice_height
fields. Targets the CVE-2026-8461 condition: last slice's sheight > height
causes dst pointer to go past buf_end, writing into AVBuffer.free.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass

# AVI chunk types for video streams
AVI_VideoChunks = [b"vprp", b"strh", b"strf", b"data"]

# Known AVI FourCCs for video streams (MagicYUV: BMH5/BMH6)
AVI_FOURCC = {
    b"bmh5": b"BMH5",  # MagicYUV codec
    b"bmh6": b"BMH6",  # MagicYUV codec
    b"RGB": b"RGB ",  # 8bpp uncompressed
    b"RLE": b"RLE ",  # Run-length encoded
}


@dataclass
class AviChunk:
    """A single AVI chunk: FourCC(4) + chunk_size(4 LE) + data."""

    fourcc: bytes
    chunk_size: int
    data: bytes

    def to_bytes(self) -> bytes:
        return self.fourcc + self.chunk_size.to_bytes(4, "little") + self.data


def parse_avi_chunks(data: bytes) -> tuple[bytes, list[AviChunk]] | None:
    """Parse AVI chunks. Returns (main_header, chunks) or None.

    Recognizes AVI format (ckid = "RIFF" + form_type = "AVI ") then
    walks LIST/ckid pairs until "movi" is seen. MagicYUV chunks appear as
    "vprp" (video properties) with slice_height in bytes 28-31 (relative to
    chunk start). This implements the pattern of existing riff.py but
    isolates MagicYUV-specific slice_height parsing.
    """
    if len(data) < 12 or data[:4] != b"RIFF":
        return None
    form_type = data[8:12]
    if form_type != b"AVI ":
        return None

    pos = 12
    n = len(data)
    chunks: list[AviChunk] = []
    while pos + 8 <= n:
        fourcc = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        body_start = pos + 8
        body_end = min(body_start + chunk_size, n)
        body = data[body_start:body_end]
        chunks.append(AviChunk(fourcc, chunk_size, body))
        pos = body_end + ((body_end - body_start) % 2)  # pad to even

        if fourcc == b"movi":
            break  # stop at movi (last list in AVI)

    return (b"AVI ", chunks) if chunks else None


def serialize_avi_chunks(form_type: bytes, chunks: list[AviChunk]) -> bytes:
    """Serialize AVI chunks back to bytes."""
    body = form_type + b"".join(c.to_bytes() for c in chunks)
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def _find_chunk(chunks: list[AviChunk], target_fourcc: bytes) -> AviChunk | None:
    return next((c for c in chunks if c.fourcc == target_fourcc), None)


def _parse_magicyuv_stream_header(data: bytes) -> int | None:
    """Extract slice_height from a vprp chunk.

    MagicYUV stream header layout (BMH5/BMH6 chunk type):
    Bytes 0-3: chunk_type (always "vprp")
    Bytes 4-7: chunk_size (little-endian)
    Bytes 8-11: magic_yuv_header_flags (1 bit per field)
    Bytes 12-15: magic_yuv_image_width (LE)
    Bytes 16-19: magic_yuv_image_height (LE)
    Bytes 20-23: magic_yuv_stride (LE)  <- used as stride per slice
    Bytes 24-27: magic_yuv_slice_height (LE)  <- per-slice slice_height
    Bytes 28-31: magic_yuv_frame_count (LE)
    Bytes 32-...: slice data (magic_yuv_slice_height * magic_yuv_stride * magic_yuv_frame_count)

    We need to clamp height and stride to valid sizes before calculating
    dst overflow.
    """
    if len(data) < 32:
        return None
    width = struct.unpack_from("<I", data, 12)[0]
    height = struct.unpack_from("<I", data, 16)[0]
    stride = struct.unpack_from("<I", data, 20)[0]
    slice_height = struct.unpack_from("<I", data, 24)[0]
    return slice_height


class MagicYUVMutator:
    """Structure-aware MagicYUV mutator.

    Targets slice_height field in AVI stream header chunks to trigger
    CVE-2026-8461 (dst pointer goes past buffer when slice_height does not
    evenly divide image height).

    Also corrupts width/height/stride to create edge-case arithmetic.
    """

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        """Apply one MagicYUV-specific mutation."""
        rng = rng or random
        parsed = parse_avi_chunks(data)
        if not parsed:
            return self._generate_random_magicyuv(max_len, rng=rng)

        form_type, chunks = parsed
        op = rng.randint(0, 5)
        mutators = [
            self._mutate_slice_height,
            self._mutate_width,
            self._mutate_height,
            self._mutate_stride,
            self._delete_chunk,
            self._generate_random_magicyuv,
        ]
        return mutators[op](form_type, chunks, max_len)[:max_len]

    def _mutate_slice_height(self, form_type: bytes, chunks: list[AviChunk], max_len: int) -> bytes:
        """Corrupt slice_height to create uneven remainders.

        For 640x480 frames with stride=2560, slice_height = 31 causes a
        31*2560 = 79360 bytes of Y (first slice) plus 2560*17 = 43520 bytes
        of U/V (remaining slices) total 122880 bytes, but chroma subsampling
        means dst buffer may be sized for (height * stride) where height = 480,
        stride = 2560 -> 1228800 bytes, and slice_height = 31 gives remainder
        480 % 31 = 4 → out-of-bounds write at dst + height*stride = 1228800 + 4*2560 = 1239680
        """
        target = _find_chunk(chunks, b"vprp")
        if not target or len(target.data) < 28:
            return serialize_avi_chunks(form_type, chunks)[:max_len]

        slice_height = _parse_magicyuv_stream_header(target.data)
        if slice_height is None:
            return serialize_avi_chunks(form_type, chunks)[:max_len]

        # Create slice_height that doesn't evenly divide height
        # height from chunk header if available, otherwise guess from slice data
        if len(target.data) >= 16:
            height = struct.unpack_from("<I", target.data, 16)[0]
        else:
            height = 480  # default common height

        # Choose problematic remainders (e.g., 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 31)
        problematic = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 31, 63, 127, 255]
        new_slice_height = rng.choice(problematic)

        # Write back
        modified = bytearray(target.data)
        struct.pack_into("<I", modified, 24, new_slice_height)
        target.data = bytes(modified)

        return serialize_avi_chunks(form_type, chunks)[:max_len]

    def _mutate_width(self, form_type: bytes, chunks: list[AviChunk], max_len: int) -> bytes:
        """Corrupt image width to create slice_height overflow."""
        target = _find_chunk(chunks, b"vprp")
        if not target or len(target.data) < 12:
            return serialize_avi_chunks(form_type, chunks)[:max_len]

        width = struct.unpack_from("<I", target.data, 12)[0]
        # Corrupt width to create problematic width/height combos
        new_width = rng.choice([0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095])

        modified = bytearray(target.data)
        struct.pack_into("<I", modified, 12, new_width)
        target.data = bytes(modified)

        return serialize_avi_chunks(form_type, chunks)[:max_len]

    def _mutate_height(self, form_type: bytes, chunks: list[AviChunk], max_len: int) -> bytes:
        """Corrupt image height to mismatch slice_height."""
        target = _find_chunk(chunks, b"vprp")
        if not target or len(target.data) < 16:
            return serialize_avi_chunks(form_type, chunks)[:max_len]

        height = struct.unpack_from("<I", target.data, 16)[0]
        # Corrupt height to create remainder when divided by slice_height
        if height > 100:
            new_height = height + rng.choice([1, 2, 3, 5, 7, 9, 11, 13, 15])
        else:
            new_height = rng.choice([0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 511])

        modified = bytearray(target.data)
        struct.pack_into("<I", modified, 16, new_height)
        target.data = bytes(modified)

        return serialize_avi_chunks(form_type, chunks)[:max_len]

    def _mutate_stride(self, form_type: bytes, chunks: list[AviChunk], max_len: int) -> bytes:
        """Corrupt stride to create slice_height * height overflow."""
        target = _find_chunk(chunks, b"vprp")
        if not target or len(target.data) < 20:
            return serialize_avi_chunks(form_type, chunks)[:max_len]

        stride = struct.unpack_from("<I", target.data, 20)[0]
        # Create problematic stride (too small for chroma subsampling)
        new_stride = rng.choice([0, 1, 2, 3, 7, 15, 31, 63, 127, 255])

        modified = bytearray(target.data)
        struct.pack_into("<I", modified, 20, new_stride)
        target.data = bytes(modified)

        return serialize_avi_chunks(form_type, chunks)[:max_len]

    def _delete_chunk(self, form_type: bytes, chunks: list[AviChunk], max_len: int) -> bytes:
        """Delete a random chunk (except critical ones)."""
        if len(chunks) <= 2:
            return serialize_avi_chunks(form_type, chunks)[:max_len]

        deletable = [c for c in chunks if c.fourcc not in (b"LIST", b"movi")]
        if not deletable:
            return serialize_avi_chunks(form_type, chunks)[:max_len]

        target = rng.choice(deletable)
        chunks.remove(target)

        return serialize_avi_chunks(form_type, chunks)[:max_len]

    def _generate_random_magicyuv(self, max_len: int = 4096, rng=None) -> bytes:
        """Generate a minimal valid MagicYUV AVI stream."""
        rng = rng or random
        # Common MagicYUV resolution
        width = 640
        height = 480
        stride = 2560
        # Choose slice_height that doesn't evenly divide height to trigger bug
        problematic = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 31, 63, 127, 255]
        slice_height = rng.choice(problematic)
        frame_count = 1

        # Build vprp chunk header
        header = bytearray(32)
        header[0:4] = b"vprp"
        header[4:8] = (32).to_bytes(4, "little")
        header[12:16] = width.to_bytes(4, "little")
        header[16:20] = height.to_bytes(4, "little")
        header[20:24] = stride.to_bytes(4, "little")
        header[24:28] = slice_height.to_bytes(4, "little")
        header[28:32] = frame_count.to_bytes(4, "little")

        vprp_chunk = AviChunk(b"vprp", 32, bytes(header))

        # Build minimal AVI structure: RIFF + "AVI " + LIST + movi
        # For simplicity, we'll create a single vprp chunk plus movi container
        movi_chunk = AviChunk(b"movi", 12, b"00dc00db")  # one frame key+delta
        chunks = [vprp_chunk, movi_chunk]

        return serialize_avi_chunks(b"AVI ", chunks)[:max_len]
