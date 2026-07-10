#!/usr/bin/env python3
"""Generate malformed PNG files targeting libpng error paths and crash conditions.

Covers:
- Signature/chunk structure corruption
- IHDR field violations (dimensions, color type, bit depth, CRC)
- IDAT decompression failures (zlib corruption, truncation, bombs)
- Filter byte out-of-range
- Ancillary chunk malformation (PLTE, tRNS, iCCP, tEXt)
- Edge cases (zero dimensions, interlace, overflow)
"""

import struct
import zlib
from pathlib import Path

OUT = Path("/tmp/malformed_png")
OUT.mkdir(exist_ok=True)

PNG_SIG = b"\x89PNG\r\n\x1a\n"
cases = {}


def chunk(ctype: bytes, data: bytes) -> bytes:
    """PNG chunk: length(4) + type(4) + data + crc(4)."""
    c = ctype + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)


def ihdr(w=2, h=2, bd=8, ct=2):
    return chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bd, ct, 0, 0, 0))


def make_png(*chunks):
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def idat_rows(w, h, color_type=2, bit_depth=8, rows=None):
    """Generate IDAT chunk(s) with valid filtered rows."""
    bpp = (3 if color_type == 2 else 4) * bit_depth // 8
    row_len = 1 + w * bpp
    raw = b""
    for y in range(h):
        filt = 0
        if rows and y < len(rows):
            raw += bytes([filt]) + rows[y]
        else:
            raw += bytes([filt] + [128] * (row_len - 1))
    return chunk(b"IDAT", zlib.compress(raw))


# ══════════════════════════════════════════════════════════════════════
# Signature / header corruption
# ══════════════════════════════════════════════════════════════════════

cases["truncated_header"] = PNG_SIG[:4]

cases["bad_signature"] = b"\x89PNG\r\n\x1a\x00" + b"\x00" * 20

# ══════════════════════════════════════════════════════════════════════
# IHDR violations
# ══════════════════════════════════════════════════════════════════════

cases["zero_length_ihdr"] = PNG_SIG + chunk(b"IHDR", b"")

cases["truncated_ihdr"] = PNG_SIG + chunk(b"IHDR", b"\x00" * 10)

cases["huge_width"] = PNG_SIG + ihdr(w=0xFFFFFFFF, h=1)

cases["huge_height"] = PNG_SIG + ihdr(w=1, h=0xFFFFFFFF)

cases["bad_color_type"] = PNG_SIG + ihdr(ct=99)

cases["bad_bit_depth"] = PNG_SIG + ihdr(bd=128)

cases["zero_width"] = make_png(
    ihdr(w=0),
    chunk(b"IDAT", zlib.compress(b"\x00" * 10)),
    chunk(b"IEND", b""),
)

cases["interlaced_zero_height"] = PNG_SIG + chunk(
    b"IHDR", struct.pack(">IIBBBBB", 2, 0, 8, 2, 0, 0, 1)
)

cases["overflow_rowbytes"] = PNG_SIG + ihdr(w=0x80000000, h=1, bd=16)

# ── Bad CRC on IHDR ──
_ihdr_data = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
cases["bad_crc_ihdr"] = (
    PNG_SIG
    + struct.pack(">I", len(_ihdr_data))
    + b"IHDR" + _ihdr_data
    + struct.pack(">I", 0xDEADBEEF)
)

# ══════════════════════════════════════════════════════════════════════
# IDAT / zlib decompression
# ══════════════════════════════════════════════════════════════════════

cases["garbage_idat"] = make_png(ihdr(), chunk(b"IDAT", b"\x00" * 50), chunk(b"IEND", b""))

# Truncated IDAT — valid start, cut off mid-stream
_raw_row = bytes([0] + [0x80] * (1 + 2 * 3))
cases["truncated_idat"] = make_png(ihdr(), chunk(b"IDAT", zlib.compress(_raw_row)[:8]))

cases["no_idat"] = make_png(ihdr(), chunk(b"IEND", b""))

cases["empty_then_valid_idat"] = make_png(
    ihdr(),
    chunk(b"IDAT", b""),
    chunk(b"IDAT", zlib.compress(b"\x00" * 20)),
    chunk(b"IEND", b""),
)

cases["split_idat"] = make_png(
    ihdr(),
    *[
        chunk(b"IDAT", d)
        for d in (lambda c: [c[: len(c) // 2], c[len(c) // 2 :]])(
            zlib.compress(b"\x00" + b"\x80" * 100)
        )
    ],
    chunk(b"IEND", b""),
)

cases["duplicate_idat"] = make_png(ihdr(), idat_rows(2, 2), idat_rows(2, 2), chunk(b"IEND", b""))

# ── Zlib corruption ──

cases["bad_zlib_header"] = make_png(
    ihdr(), chunk(b"IDAT", b"\x00\x00" + b"\x00" * 50), chunk(b"IEND", b"")
)

cases["corrupt_zlib_stream"] = make_png(
    ihdr(),
    chunk(b"IDAT", b"\x78\x01" + b"\xff" * 50),
    chunk(b"IEND", b""),
)

# Stored decompression bomb — no-compression zlib block expands to 16KB
_stored = b"\x00" * 16384
_block = b"\x01"  # BFINAL=1, BTYPE=00 (stored)
_block += struct.pack("<H", len(_stored))
_block += struct.pack("<H", len(_stored) ^ 0xFFFF)
_block += _stored
_zlib_bomb = bytes([0x78, 30]) + _block  # FLG=30 so (CMF*256+FLG)%31==0
cases["stored_decompression_bomb"] = make_png(ihdr(w=4096, h=4), chunk(b"IDAT", _zlib_bomb))

cases["decompression_bomb"] = make_png(
    ihdr(w=16384, h=16384),
    chunk(b"IDAT", zlib.compress(b"\x00" + b"\x00" * (16384 * 4))),
)

# Non-final stored blocks — inflate never terminates
_inf = b"\x78\x01"
for _ in range(100):
    _inf += b"\x00" + b"\x00\x00" + b"\xff\xff"
cases["infinite_idat"] = make_png(ihdr(), chunk(b"IDAT", _inf), chunk(b"IEND", b""))

# Chunk declares more data than present
cases["chunk_overread"] = (
    PNG_SIG
    + struct.pack(">I", 100)
    + b"IHDR"
    + struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)[:4]
)

cases["chunk_length_mismatch"] = (
    PNG_SIG + ihdr() + b"IDAT" + struct.pack(">I", 4) + b"\x00" * 200
)

# ══════════════════════════════════════════════════════════════════════
# Filter bytes
# ══════════════════════════════════════════════════════════════════════

cases["bad_filter"] = make_png(
    ihdr(),
    idat_rows(2, 2, rows=[bytes([255] + [128] * 5) for _ in range(2)]),
    chunk(b"IEND", b""),
)

cases["filter_6"] = make_png(
    ihdr(w=4, h=4),
    chunk(b"IDAT", zlib.compress(bytes([6] + [0] * 15) * 4)),
    chunk(b"IEND", b""),
)

cases["filter_255"] = make_png(
    ihdr(w=4, h=4),
    chunk(b"IDAT", zlib.compress(bytes([255] + [0] * 15) * 4)),
    chunk(b"IEND", b""),
)

cases["filter_row0_reference"] = make_png(
    ihdr(w=8, h=8, bd=1, ct=0),
    chunk(b"IDAT", zlib.compress(bytes([3] + [0xFF] * 2) * 8)),
    chunk(b"IEND", b""),
)

# ══════════════════════════════════════════════════════════════════════
# Ancillary chunk malformation
# ══════════════════════════════════════════════════════════════════════

cases["text_before_ihdr"] = make_png(
    chunk(b"tEXt", b"key\x00value"),
    ihdr(),
    idat_rows(2, 2),
    chunk(b"IEND", b""),
)

cases["bad_plte"] = make_png(
    chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 3, 0, 0, 0)),
    chunk(b"PLTE", b"\xff" * 4),
    idat_rows(2, 2, color_type=3),
    chunk(b"IEND", b""),
)

cases["tiny_plte_8bit"] = make_png(
    chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 3, 0, 0, 0)),
    chunk(b"PLTE", b"\xff\x00\x00"),
    chunk(b"IDAT", zlib.compress(b"\x00" + b"\x00" * 16) * 16),
    chunk(b"IEND", b""),
)

cases["huge_plte_1bit"] = make_png(
    chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 1, 3, 0, 0, 0)),
    chunk(b"PLTE", b"\xff\x00\x00" * 256),
    chunk(b"IDAT", zlib.compress(b"\x00" + b"\x00" * 2) * 16),
    chunk(b"IEND", b""),
)

cases["bad_trns"] = make_png(
    ihdr(),
    chunk(b"tRNS", b"\xff" * 100),
    chunk(b"IDAT", zlib.compress(b"\x00" + b"\x80" * 8) * 2),
    chunk(b"IEND", b""),
)

cases["bad_iccp"] = make_png(
    ihdr(),
    chunk(b"iCCP", b"Profile\x00\x00" + b"\x78\x01" + b"\xff" * 30),
    chunk(b"IDAT", zlib.compress(b"\x00" + b"\x80" * 8) * 2),
    chunk(b"IEND", b""),
)

cases["huge_text_chunk"] = make_png(
    ihdr(),
    chunk(b"tEXt", b"K" + b"\x00" + b"V" * 1048576),
    chunk(b"IDAT", zlib.compress(b"\x00" * 20)),
    chunk(b"IEND", b""),
)

cases["unknown_chunk_mid"] = make_png(
    ihdr(),
    chunk(b"IDAT", zlib.compress(b"\x00" * 20)),
    chunk(b"ABCD", b"\x00" * 100),
    chunk(b"IEND", b""),
)

# ══════════════════════════════════════════════════════════════════════
# Edge cases / stress
# ══════════════════════════════════════════════════════════════════════

cases["extra_after_iend"] = make_png(ihdr(), idat_rows(2, 2), chunk(b"IEND", b"")) + b"\xff" * 64

cases["large_rowbytes_16bit"] = make_png(
    chunk(b"IHDR", struct.pack(">IIBBBBB", 10000, 1, 16, 6, 0, 0, 0)),
    chunk(b"IDAT", zlib.compress(b"\x00" + b"\x00" * 80000)),
    chunk(b"IEND", b""),
)

cases["tiny_interlaced"] = make_png(
    chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 1)),
    chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80")),
    chunk(b"IEND", b""),
)

# ══════════════════════════════════════════════════════════════════════
# Write output
# ══════════════════════════════════════════════════════════════════════

for name, data in cases.items():
    (OUT / f"{name}.png").write_bytes(data)

print(f"Wrote {len(cases)} malformed PNGs to {OUT}")
for name in sorted(cases):
    print(f"  {name}.png ({len(cases[name])} bytes)")
