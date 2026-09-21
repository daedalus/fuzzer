"""Minimal *valid* cold-start seeds for the formats the target profiler can name.

``SeedPicker._format_aware_seed`` used to build these inline. Every one of them
was rejected by a real decoder (measured with Pillow / ``zlib`` / ``gzip``):

- png   — IHDR declared 10 data bytes instead of 13, and there was no IDAT
- jpeg  — SOI/APP0/EOI only: no DQT, SOF, DHT or SOS
- gif   — declared a 256-entry colour table and then ended
- bmp   — DIB header one field short, pixel row absent
- zlib  — a ``78 9c`` header prepended to a stream that already had one
- gzip  — a zlib-wrapped stream (header + Adler-32) where raw deflate is required

A mutation-based fuzzer needs its starting corpus to reach the code *behind*
header validation, so these are now built to decode cleanly. Everything here is
stdlib-only and deterministic; ``tests/test_minimal_seeds.py`` checks each one
against the format's own parser (and Pillow where installed).
"""

from __future__ import annotations

import struct
import zlib

__all__ = [
    "minimal_bmp",
    "minimal_gif",
    "minimal_gzip",
    "minimal_jpeg",
    "minimal_png",
    "minimal_zlib",
    "MINIMAL_SEEDS",
]


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def minimal_png() -> bytes:
    """1x1 8-bit RGB PNG: IHDR (13 bytes), one IDAT, IEND."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")  # filter byte + one RGB pixel
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


# 1x1 grayscale baseline JPEG, 331 bytes (JFIF, DQT, SOF0, the four standard
# DHT tables, SOS, EOI). The tables are what make it decodable; a JPEG without
# them is rejected by every decoder before entropy decoding starts.
_JPEG_HEX = (
    "ffd8ffe000104a46494600010100000100010000ffdb004300100b0c0e0c0a100e0d0e12"
    "11101318281a181616183123251d283a333d3c3933383740485c4e404457453738506d51"
    "575f626768673e4d71797064785c656763ffc0000b080001000101011100ffc4001f0000"
    "010501010101010100000000000000000102030405060708090a0bffc400b51000020103"
    "03020403050504040000017d01020300041105122131410613516107227114328191a108"
    "2342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445"
    "464748494a535455565758595a636465666768696a737475767778797a83848586878889"
    "8a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9"
    "cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda00080101"
    "00003f002bffd9"
)


def minimal_jpeg() -> bytes:
    return bytes.fromhex(_JPEG_HEX)


def minimal_gif() -> bytes:
    """1x1 GIF89a with a 2-entry global colour table and one LZW image."""
    return (
        b"GIF89a"
        + struct.pack("<HH", 1, 1)
        + b"\x80\x00\x00"  # GCT present, 2 entries; background 0; aspect 0
        + b"\x00\x00\x00\xff\xff\xff"  # colour table
        + b"\x2c" + struct.pack("<HHHH", 0, 0, 1, 1) + b"\x00"  # image descriptor
        + b"\x02\x02\x44\x01\x00"  # LZW min code size 2, one 2-byte sub-block
        + b"\x3b"  # trailer
    )


def minimal_bmp() -> bytes:
    """1x1 24-bit BMP: 14-byte file header, 40-byte BITMAPINFOHEADER, one padded row."""
    row = b"\x00\x00\x00\x00"  # BGR pixel + 1 byte of row padding to a multiple of 4
    dib = struct.pack("<IiiHHIIiiII", 40, 1, 1, 1, 24, 0, len(row), 0, 0, 0, 0)
    header = b"BM" + struct.pack("<IHHI", 14 + len(dib) + len(row), 0, 0, 14 + len(dib))
    return header + dib + row


def minimal_zlib() -> bytes:
    return zlib.compress(b"\x00")


def minimal_gzip() -> bytes:
    """gzip member: 10-byte header, *raw* deflate, CRC-32, ISIZE."""
    payload = b"\x00"
    comp = zlib.compressobj(wbits=-15)
    raw = comp.compress(payload) + comp.flush()
    return (
        b"\x1f\x8b\x08\x00" + b"\x00\x00\x00\x00" + b"\x00\xff"
        + raw
        + struct.pack("<II", zlib.crc32(payload), len(payload))
    )


MINIMAL_SEEDS = {
    "png": minimal_png,
    "jpeg": minimal_jpeg,
    "gif": minimal_gif,
    "bmp": minimal_bmp,
    "zlib": minimal_zlib,
    "gzip": minimal_gzip,
}
