"""Structure-aware zlib (RFC 1950) mutations.

Parses zlib header (CMF/FLG), compressed DEFLATE data, and Adler-32
trailer. Applies targeted corruption to header flags, compression
parameters, deflate stream data, and trailer. Falls back to random
zlib generation when input is not valid zlib.

Note: This is the ZLIB format (RFC 1950), not GZIP (RFC 1952).
ZLIB: 2-byte header + DEFLATE + 4-byte Adler-32 trailer
GZIP: 10-byte header + optional fields + DEFLATE + CRC32 + size
"""

from __future__ import annotations

import random
import struct
import zlib
from dataclasses import dataclass


ZLIB_MIN_SIZE = 6  # 2 header + 0 data + 4 trailer


@dataclass
class ZlibInfo:
    """Parsed zlib stream information."""

    cmf: int  # Compression Method and Flags
    flg: int  # Flags
    # Parsed from CMF
    cm: int  # Compression method (8 = deflate)
    cinfo: int  # Log2(window size) - 8
    # Parsed from FLG
    fdict: int  # 1 = preset dictionary follows
    flevel: int  # Compression level hint
    # Data
    compressed_data: bytes = b""
    # Trailer
    adler32: int = 0
    # Raw bytes
    header: bytearray = None

    def __post_init__(self):
        if self.header is None:
            self.header = bytearray(2)


def _adler32(data: bytes) -> int:
    """Compute Adler-32 checksum (RFC 1950)."""
    return zlib.adler32(data) & 0xFFFFFFFF


def parse_zlib(data: bytes) -> ZlibInfo | None:
    """Parse zlib stream header and trailer.

    Returns None if data doesn't match zlib format.
    ZLIB header: CMF byte (CM=8, CINFO<=7) + FLG byte with valid checksum.
    """
    if len(data) < ZLIB_MIN_SIZE:
        return None

    cmf = data[0]
    flg = data[1]

    # Validate: CM=8 (deflate), CINFO<=7
    cm = cmf & 0x0F
    cinfo = (cmf >> 4) & 0x0F
    if cm != 8 or cinfo > 7:
        return None

    # Validate FLG: (CMF * 256 + FLG) must be multiple of 31
    if (cmf * 256 + flg) % 31 != 0:
        return None

    fdict = (flg >> 5) & 1
    flevel = (flg >> 6) & 3

    pos = 2

    # Optional preset dictionary
    if fdict:
        if pos + 4 > len(data):
            return None
        pos += 4  # skip DICTID (4 bytes)

    # Compressed data: everything up to last 4 bytes (Adler-32 trailer)
    if len(data) - pos < 4:
        return None

    compressed_data = data[pos : len(data) - 4]
    adler32 = struct.unpack(">I", data[len(data) - 4 :])[0]

    return ZlibInfo(
        cmf=cmf,
        flg=flg,
        cm=cm,
        cinfo=cinfo,
        fdict=fdict,
        flevel=flevel,
        compressed_data=compressed_data,
        adler32=adler32,
        header=bytearray(data[:2]),
    )


def serialize_zlib(info: ZlibInfo) -> bytes:
    """Serialize ZlibInfo back to zlib bytes, recomputing FLG checksum."""
    # Recompute FLG so (CMF * 256 + FLG) % 31 == 0
    cmf = info.cmf
    flg = info.flg & 0xE0  # preserve FLEVEL and FDICT bits
    # Set FCHECK so checksum is valid
    flg = (flg + (31 - (cmf * 256 + flg) % 31)) % 31

    buf = bytearray([cmf, flg])

    if info.fdict:
        buf.extend(b"\x00\x00\x00\x00")  # placeholder DICTID

    buf.extend(info.compressed_data)

    # Recompute Adler-32 over the uncompressed data
    # (We don't have the original uncompressed data here, so we
    # keep the stored adler32. The caller should set it correctly
    # if they want a valid stream.)
    buf.extend(struct.pack(">I", info.adler32))

    return bytes(buf)


def _corrupt_field(data: bytearray, offset: int, size: int) -> None:
    """Apply random corruption to a field."""
    if size == 1:
        data[offset] = random.randint(0, 255)
    elif size == 2:
        val = struct.unpack(">H", data[offset : offset + 2])[0]
        method = random.randint(0, 3)
        if method == 0:
            val ^= 1 << random.randint(0, 15)
        elif method == 1:
            val = random.choice([0, 1, 0x7FFF, 0xFFFF])
        elif method == 2:
            val = max(0, val + random.choice([-2, -1, 1, 2]))
        else:
            val = random.randint(0, 0xFFFF)
        struct.pack_into(">H", data, offset, val)
    elif size == 4:
        val = struct.unpack(">I", data[offset : offset + 4])[0]
        method = random.randint(0, 3)
        if method == 0:
            val ^= 1 << random.randint(0, 31)
        elif method == 1:
            val = random.choice([0, 1, 0x7FFFFFFF, 0xFFFFFFFF])
        elif method == 2:
            val = max(0, val + random.choice([-2, -1, 1, 2, 256]))
        else:
            val = random.randint(0, 0xFFFFFFFF)
        struct.pack_into(">I", data, offset, val)


class ZlibMutator:
    """Structure-aware zlib mutator.

    Dispatches one of 10 mutation operations per call, targeting
    specific zlib structures for maximum code-path diversity.
    """

    def mutate(self, data: bytes, max_len: int = 4096) -> bytes:
        """Apply one structure-aware zlib mutation."""
        info = parse_zlib(data)
        if info is None:
            return self._generate_random_zlib(max_len)

        op = random.randint(0, 9)
        mutators = [
            self._mutate_cmf,
            self._mutate_flevel,
            self._mutate_deflate_stream,
            self._mutate_deflate_block,
            self._corrupt_trailer,
            self._swap_header_nibbles,
            self._truncate_zlib,
            self._inject_junk_before_deflate,
            self._mutate_window_size,
            self._generate_random_zlib,
        ]
        result = mutators[op](info, max_len)
        if isinstance(result, ZlibInfo):
            return serialize_zlib(result)[:max_len]
        return result[:max_len]

    def _mutate_cmf(self, info: ZlibInfo, max_len: int) -> ZlibInfo:
        """Corrupt the CMF byte (CM and CINFO)."""
        method = random.randint(0, 3)
        if method == 0:
            # Valid CM=8, corrupt CINFO
            info.cmf = 8 | (random.randint(0, 15) << 4)
        elif method == 1:
            # Corrupt CM (not deflate)
            info.cmf = random.randint(0, 255)
        elif method == 2:
            # Extreme CINFO (large window)
            info.cmf = 8 | (7 << 4)  # CINFO=7 → 32K window
        else:
            info.cmf = random.randint(0, 255)
        return info

    def _mutate_flevel(self, info: ZlibInfo, max_len: int) -> ZlibInfo:
        """Corrupt the compression level hint in FLG."""
        info.flevel = random.randint(0, 3)
        return info

    def _mutate_deflate_stream(self, info: ZlibInfo, max_len: int) -> ZlibInfo:
        """Flip random bytes in the deflate stream."""
        if info.compressed_data:
            data = bytearray(info.compressed_data)
            for _ in range(random.randint(1, min(8, len(data)))):
                pos = random.randint(0, len(data) - 1)
                data[pos] ^= 1 << random.randint(0, 7)
            info.compressed_data = bytes(data)
        return info

    def _mutate_deflate_block(self, info: ZlibInfo, max_len: int) -> ZlibInfo:
        """Replace a chunk of the deflate stream with random data."""
        if info.compressed_data and len(info.compressed_data) > 4:
            data = bytearray(info.compressed_data)
            chunk_start = random.randint(0, len(data) - 2)
            chunk_len = random.randint(1, min(16, len(data) - chunk_start))
            for i in range(chunk_start, chunk_start + chunk_len):
                data[i] = random.randint(0, 255)
            info.compressed_data = bytes(data)
        return info

    def _corrupt_trailer(self, info: ZlibInfo, max_len: int) -> ZlibInfo:
        """Corrupt the Adler-32 trailer."""
        info.adler32 = random.randint(0, 0xFFFFFFFF)
        return info

    def _swap_header_nibbles(self, info: ZlibInfo, max_len: int) -> ZlibInfo:
        """Swap CM and CINFO nibbles in the CMF byte."""
        cm = info.cmf & 0x0F
        cinfo = (info.cmf >> 4) & 0x0F
        info.cmf = (cm << 4) | cinfo
        return info

    def _truncate_zlib(self, info: ZlibInfo, max_len: int) -> bytes:
        """Remove the trailer to produce a truncated zlib stream."""
        return bytes(info.header) + info.compressed_data

    def _inject_junk_before_deflate(self, info: ZlibInfo, max_len: int) -> ZlibInfo:
        """Inject random bytes between header and deflate stream."""
        junk = bytes(random.randint(0, 255) for _ in range(random.randint(1, 32)))
        info.compressed_data = junk + info.compressed_data
        return info

    def _mutate_window_size(self, info: ZlibInfo, max_len: int) -> ZlibInfo:
        """Change the window size (CINFO field)."""
        info.cinfo = random.randint(0, 7)
        return info

    def _generate_random_zlib(self, info_or_max=None, max_len: int = 4096) -> bytes:
        """Generate a minimal random zlib stream from scratch."""
        if isinstance(info_or_max, int):
            max_len = info_or_max

        # Random uncompressed data
        payload_len = random.randint(1, min(128, max_len - 10))
        payload = bytes(random.randint(0, 255) for _ in range(payload_len))

        # Compress with deflate (wbits=15 for zlib format)
        compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS)
        compressed = compressor.compress(payload) + compressor.flush()

        # Build CMF: CM=8 (deflate), CINFO=7 (32K window)
        cinfo = 7
        cmf = 8 | (cinfo << 4)

        # Build FLG: FLEVEL=3 (best compression), FDICT=0
        flevel = 3
        fdict = 0
        flg_base = (flevel << 6) | (fdict << 5)
        # Compute FCHECK so (CMF * 256 + FLG) % 31 == 0
        flg = (flg_base + (31 - (cmf * 256 + flg_base) % 31)) % 31

        header = bytes([cmf, flg])
        adler = _adler32(payload)

        return header + compressed + struct.pack(">I", adler)
