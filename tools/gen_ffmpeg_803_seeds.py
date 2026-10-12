#!/usr/bin/env python3
"""Potential crash seeds aimed at FFmpeg 8.0.3.

Derived from the 8.1.x Changelog (the 8.0.3->8.1.3 diff as bug titles): every
seed reaches a function 8.0.3 ships unguarded and 8.1.3 hardened, so any crash
it produces is 8.0.3-only. Per-hunk analysis lives next to the seeds in
``DIFF_DEEPDIVE.md`` / ``DIFF_DEEPDIVE_9.0.2.md``.

Usage:
    python tools/gen_ffmpeg_803_seeds.py [--out DIR]

Output is written to <out>/seeds/ (the layout load_corpus reads); --out
defaults to ~/fuzzing/ffmpeg_803_potential_crashes. The container baselines
are minted separately by tools/mint_ffmpeg_803_baselines.sh.
"""

import argparse
import os
import struct

DEFAULT_OUT = os.path.expanduser("~/fuzzing/ffmpeg_803_potential_crashes")


def le32(v):
    return struct.pack("<I", v)


def le16(v):
    return struct.pack("<H", v)


def be32(v):
    return struct.pack(">I", v)


def be16(v):
    return struct.pack(">H", v)


def write_seeds(out):
    """Write the crafted 8.0.3 crash seeds into ``<out>/seeds/``."""

    def w(name, data):
        with open(os.path.join(out, name), "wb") as f:
            f.write(data)
        print(f"{name}: {len(data)}B")

    # ---------------- 8.0.3 unguarded imports ----------------
    # genh: type1 THP, interleave_size=1. 8.1.3 rejects != 2.
    g = bytearray(b"GENH") + le32(2) + le32(1) + le32(44100) + le32(0) + le32(1000)
    g += le32(12) + le32(0x800) + le32(0) + le32(0x100) + le32(0x200)
    g += le32(1) + le32(0) + le32(0) + le32(0)
    g += b"\x00" * (0x800 - len(g)) + b"\x41" * 64
    w("genh_thp_interleave1.genh", bytes(g))

    # binka: header ok, one packet whose declared size exceeds the data present.
    b = bytearray(b"1FCB\x01") + b"\x02" + le16(44100) + le32(1000)
    b += b"\x00" * 8 + le16(0) + b"\x00\x00" + b"\x00\x00" + le16(200) + b"\x42" * 4
    w("binka_trunc_packet.bin", bytes(b))

    # sierravmd: VMD_HEADER_SIZE 0x330; probe wants buf[0:2]==0x32e, 8-bit palette.
    VMD = 0x330
    s = bytearray(VMD)
    struct.pack_into("<H", s, 0, VMD - 2)
    struct.pack_into("<H", s, 12, 320)
    struct.pack_into("<H", s, 14, 240)
    struct.pack_into("<H", s, 804, 11025)
    w("vmd_min.vmd", bytes(s) + b"\x00" * 64)

    # rsd: probe "RSD2".."RSD6", sane channels/rate.
    r = bytearray(b"RSD2") + b"\x00" * 4 + le32(2) + b"\x00" * 4 + le32(44100)
    w("rsd_min.rsd", bytes(r) + b"\x00" * 64)

    # codec2: magic + major 0 / minor 8.
    w("codec2_min.c2", bytes([0xC0, 0xDE, 0xC2, 0x00, 0x08]) + b"\x00" * 64)

    # vividas: magic.
    w("vividas_min.viv", b"vividas03" + b"\x00" * 128)

    # vpk: the confirmed 8.0.3 SIGFPE reproducer. Inlined so the generator no longer
    # depends on the fuzzer output tree (the original crash file was retired).
    VPK_DIV0 = bytes.fromhex("204b5056565000f804003b03613956323636303850")
    w("vpk_div0.vpk", VPK_DIV0)

    # ---------------- extra crafted magic seeds ----------------
    # w64: riff/wave GUIDs, fmt chunk with an overflowing size.
    GRIFF = b"riff" + bytes([0x2E, 0x91, 0xCF, 0x11, 0xA5, 0xD6, 0x28, 0xDB, 0x04, 0xC1, 0, 0])
    GWAVE = b"wave" + bytes(
        [0xF3, 0xAC, 0xD3, 0x11, 0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A]
    )
    GFMT = b"fmt " + bytes([0xF3, 0xAC, 0xD3, 0x11, 0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A])
    x = bytearray(GRIFF) + struct.pack("<Q", 0xFFFFFFFF)
    x += GWAVE + GFMT + struct.pack("<Q", 0xFFFFFFFFFFFFFFFF) + b"\x00" * 32
    w("w64_overflow.w64", bytes(x))

    # caf: caff + version 1 + desc chunk (32) then a bogus chunk.
    c = b"caff" + be16(1) + be16(0) + b"desc" + struct.pack(">Q", 32)
    c += struct.pack(">I", 0) * 4 + struct.pack(">d", 44100.0)
    c += b"data" + struct.pack(">Q", 0xFFFFFFFF)
    w("caf_min.caf", c)

    # sox: header_size (buf+4) and channels (buf+24) must be non-zero.
    sx = bytearray(b".SoX") + le32(8) + b"\x00" * 8
    sx += struct.pack("<d", 44100.0) + le32(2) + le32(0) + b"\x00" * 64
    w("sox_min.sox", bytes(sx))

    # dhav: magic + variant byte.
    w("dhav_min.dav", b"DHAV\xf1" + b"\x00" * 256)

    # mcc: FCC probe string.
    w("mcc_min.mcc", b"File Format=MacCaption_MCC V1.0\r\n\x00" * 4)

    # ty: chunked TiVo PES fileid (CHUNK_SIZE = 128*1024).
    CHUNK = 128 * 1024
    t = bytearray()
    for _ in range(2):
        t += be32(0xF5467ABD) + be32(0x02) + be32(CHUNK) + b"\x00" * (CHUNK - 12)
    w("ty_series1.ty", bytes(t))

    # spdif: IEC 61937 sync words (AAC burst), boundary-sized to the probe bug.
    SYNC = b"\x72\xf8\x1f\x4e" + b"\x00\x00"  # Pa/Pb placeholder, Pc=0 (AAC?)
    w("spdif_sync.bin", (SYNC + b"\x00" * 8) + (SYNC + b"\x00" * 8) + b"\x00" * 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="pack root; seeds are written to <out>/seeds/ (default: %(default)s)",
    )
    args = ap.parse_args()

    out = os.path.join(args.out, "seeds")
    os.makedirs(out, exist_ok=True)
    write_seeds(out)
    print("seeds dir:", out)


if __name__ == "__main__":
    main()
