"""Structure-aware gzip mutations.

Parses gzip header fields, deflate stream, and trailer.
Applies targeted corruption to header flags, compression parameters,
deflate stream data, and trailer fields. Falls back to random
gzip generation when input is not valid gzip.
"""

from __future__ import annotations

import random
import struct
import zlib
from dataclasses import dataclass


GZIP_MAGIC = b"\x1f\x8b"
DEFLATE = 8

# Header flag bits
FTEXT = 1 << 0
FHCRC = 1 << 1
FEXTRA = 1 << 2
FNAME = 1 << 3
FCOMMENT = 1 << 4


@dataclass
class GzipInfo:
    """Parsed gzip header information."""

    method: int
    flags: int
    mtime: int
    xfl: int
    os: int
    # Optional fields
    extra: bytes = b""
    filename: bytes = b""
    comment: bytes = b""
    hcrc: int = 0
    # Data
    compressed_data: bytes = b""
    original_crc: int = 0
    original_size: int = 0
    # Raw header bytes (first 10)
    header: bytearray = None

    def __post_init__(self):
        if self.header is None:
            self.header = bytearray(10)


def parse_gzip(data: bytes) -> GzipInfo | None:
    """Parse gzip file header and trailer.

    Returns None if data doesn't start with gzip magic bytes.
    """
    if len(data) < 18 or data[:2] != GZIP_MAGIC:
        return None

    method = data[2]
    flags = data[3]
    mtime = struct.unpack("<I", data[4:8])[0]
    xfl = data[8]
    os = data[9]

    pos = 10

    # Extra field
    extra = b""
    if flags & FEXTRA:
        if pos + 2 > len(data):
            return None
        xlen = struct.unpack("<H", data[pos : pos + 2])[0]
        pos += 2
        if pos + xlen > len(data):
            return None
        extra = data[pos : pos + xlen]
        pos += xlen

    # Filename (null-terminated)
    filename = b""
    if flags & FNAME:
        end = data.find(b"\x00", pos)
        if end == -1:
            return None
        filename = data[pos:end]
        pos = end + 1

    # Comment (null-terminated)
    comment = b""
    if flags & FCOMMENT:
        end = data.find(b"\x00", pos)
        if end == -1:
            return None
        comment = data[pos:end]
        pos = end + 1

    # Header CRC16
    hcrc = 0
    if flags & FHCRC:
        if pos + 2 > len(data):
            return None
        hcrc = struct.unpack("<H", data[pos : pos + 2])[0]
        pos += 2

    # Compressed data: everything up to the last 8 bytes (trailer)
    if len(data) - pos < 8:
        return None

    compressed_data = data[pos : len(data) - 8]
    original_crc = struct.unpack("<I", data[len(data) - 8 : len(data) - 4])[0]
    original_size = struct.unpack("<I", data[len(data) - 4 :])[0]

    return GzipInfo(
        method=method,
        flags=flags,
        mtime=mtime,
        xfl=xfl,
        os=os,
        extra=extra,
        filename=filename,
        comment=comment,
        hcrc=hcrc,
        compressed_data=compressed_data,
        original_crc=original_crc,
        original_size=original_size,
        header=bytearray(data[:10]),
    )


def serialize_gzip(info: GzipInfo) -> bytes:
    """Serialize GzipInfo back to gzip bytes."""
    buf = bytearray(info.header)

    if info.flags & FEXTRA:
        buf.extend(struct.pack("<H", len(info.extra)))
        buf.extend(info.extra)
    if info.flags & FNAME:
        buf.extend(info.filename)
        buf.append(0)
    if info.flags & FCOMMENT:
        buf.extend(info.comment)
        buf.append(0)
    if info.flags & FHCRC:
        buf.extend(struct.pack("<H", info.hcrc))

    buf.extend(info.compressed_data)
    buf.extend(struct.pack("<I", info.original_crc))
    buf.extend(struct.pack("<I", info.original_size))
    return bytes(buf)


def _corrupt_field(data: bytearray, offset: int, size: int) -> None:
    """Apply random corruption to a field."""
    if size == 1:
        data[offset] = random.randint(0, 255)
    elif size == 2:
        val = struct.unpack("<H", data[offset : offset + 2])[0]
        method = random.randint(0, 3)
        if method == 0:
            val ^= 1 << random.randint(0, 15)
        elif method == 1:
            val = random.choice([0, 1, 0x7FFF, 0xFFFF])
        elif method == 2:
            val = max(0, val + random.choice([-2, -1, 1, 2]))
        else:
            val = random.randint(0, 0xFFFF)
        struct.pack_into("<H", data, offset, val)
    elif size == 4:
        val = struct.unpack("<I", data[offset : offset + 4])[0]
        method = random.randint(0, 3)
        if method == 0:
            val ^= 1 << random.randint(0, 31)
        elif method == 1:
            val = random.choice([0, 1, 0x7FFFFFFF, 0xFFFFFFFF])
        elif method == 2:
            val = max(0, val + random.choice([-2, -1, 1, 2, 256]))
        else:
            val = random.randint(0, 0xFFFFFFFF)
        struct.pack_into("<I", data, offset, val)


class GzipMutator:
    """Structure-aware gzip mutator.

    Dispatches one of 12 mutation operations per call, targeting
    specific gzip structures for maximum code-path diversity.
    """

    def mutate(self, data: bytes, max_len: int = 4096) -> bytes:
        """Apply one structure-aware gzip mutation."""
        info = parse_gzip(data)
        if info is None:
            return self._generate_random_gzip(max_len)

        op = random.randint(0, 11)
        mutators = [
            self._mutate_flags,
            self._mutate_method,
            self._mutate_xfl_os,
            self._mutate_deflate_stream,
            self._mutate_deflate_block,
            self._corrupt_trailer,
            self._swap_trailer_fields,
            self._truncate_gzip,
            self._inject_junk_before_deflate,
            self._mutate_extra_field,
            self._generate_random_gzip,
            self._mutate_deflate_strategy,
        ]
        result = mutators[op](info, max_len)
        if isinstance(result, GzipInfo):
            return serialize_gzip(result)[:max_len]
        return result[:max_len]

    def _mutate_flags(self, info: GzipInfo, max_len: int) -> GzipInfo:
        """Corrupt the flags byte."""
        info.header[3] = random.randint(0, 15)
        return info

    def _mutate_method(self, info: GzipInfo, max_len: int) -> GzipInfo:
        """Corrupt the compression method."""
        info.header[2] = random.choice([8, 0, 1, 2, 255])
        return info

    def _mutate_xfl_os(self, info: GzipInfo, max_len: int) -> GzipInfo:
        """Corrupt XFL or OS fields."""
        if random.random() < 0.5:
            info.header[8] = random.randint(0, 255)  # XFL
        else:
            info.header[9] = random.randint(0, 255)  # OS
        return info

    def _mutate_deflate_stream(self, info: GzipInfo, max_len: int) -> GzipInfo:
        """Flip random bytes in the deflate stream."""
        if info.compressed_data:
            data = bytearray(info.compressed_data)
            for _ in range(random.randint(1, min(8, len(data)))):
                pos = random.randint(0, len(data) - 1)
                data[pos] ^= 1 << random.randint(0, 7)
            info.compressed_data = bytes(data)
        return info

    def _mutate_deflate_block(self, info: GzipInfo, max_len: int) -> GzipInfo:
        """Replace a chunk of the deflate stream with random data."""
        if info.compressed_data and len(info.compressed_data) > 4:
            data = bytearray(info.compressed_data)
            chunk_start = random.randint(0, len(data) - 2)
            chunk_len = random.randint(1, min(16, len(data) - chunk_start))
            for i in range(chunk_start, chunk_start + chunk_len):
                data[i] = random.randint(0, 255)
            info.compressed_data = bytes(data)
        return info

    def _corrupt_trailer(self, info: GzipInfo, max_len: int) -> GzipInfo:
        """Corrupt the CRC32 or original size in the trailer."""
        if random.random() < 0.5:
            info.original_crc = random.randint(0, 0xFFFFFFFF)
        else:
            info.original_size = random.randint(0, 0xFFFFFFFF)
        return info

    def _swap_trailer_fields(self, info: GzipInfo, max_len: int) -> GzipInfo:
        """Swap CRC and size fields in the trailer."""
        info.original_crc, info.original_size = info.original_size, info.original_crc
        return info

    def _truncate_gzip(self, info: GzipInfo, max_len: int) -> bytes:
        """Remove the trailer to produce a truncated gzip."""
        return bytes(info.header) + info.compressed_data

    def _inject_junk_before_deflate(self, info: GzipInfo, max_len: int) -> GzipInfo:
        """Inject random bytes between header and deflate stream."""
        junk = bytes(random.randint(0, 255) for _ in range(random.randint(1, 32)))
        info.compressed_data = junk + info.compressed_data
        return info

    def _mutate_extra_field(self, info: GzipInfo, max_len: int) -> GzipInfo:
        """Corrupt or inject an extra field."""
        if info.flags & FEXTRA and info.extra:
            data = bytearray(info.extra)
            if data:
                idx = random.randint(0, len(data) - 1)
                data[idx] = random.randint(0, 255)
                info.extra = bytes(data)
        else:
            # Inject extra field
            extra_data = bytes(random.randint(0, 255) for _ in range(random.randint(2, 16)))
            info.flags |= FEXTRA
            info.extra = extra_data
        return info

    def _mutate_deflate_strategy(self, info: GzipInfo, max_len: int) -> GzipInfo:
        """Corrupt the deflate stream's block type bits (first 3 bits)."""
        if info.compressed_data:
            data = bytearray(info.compressed_data)
            # Flip block-type bits in the first byte
            data[0] ^= random.randint(1, 7)
            info.compressed_data = bytes(data)
        return info

    def _generate_random_gzip(self, info_or_max=None, max_len: int = 4096) -> bytes:
        """Generate a minimal random gzip from scratch."""
        if isinstance(info_or_max, GzipInfo):
            max_len = max_len
        elif isinstance(info_or_max, int):
            max_len = info_or_max

        # Random uncompressed data
        payload_len = random.randint(1, min(128, max_len - 20))
        payload = bytes(random.randint(0, 255) for _ in range(payload_len))

        # Compress with deflate
        compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        compressed = compressor.compress(payload) + compressor.flush()

        # Build header
        header = bytearray(10)
        header[0:2] = GZIP_MAGIC
        header[2] = DEFLATE
        header[3] = 0  # flags
        struct.pack_into("<I", header, 4, 0)  # mtime
        header[8] = 0  # XFL
        header[9] = 255  # OS = unknown

        # Trailer
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        size = len(payload) & 0xFFFFFFFF

        return bytes(header) + compressed + struct.pack("<I", crc) + struct.pack("<I", size)
