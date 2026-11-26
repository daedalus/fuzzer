#!/usr/bin/env python3
"""Generate crash/coverage seeds for the FFmpeg and/or GStreamer fuzz targets.

Two seed families are produced under ``<out>/seeds/``:

* ``ff_*`` -- the FFmpeg 9.0.2 crash seeds, reused verbatim from
  ``tools/gen_ffmpeg_91_seeds`` (imported, not copied, so the two packs never
  drift). Those seeds and their rationale are documented there.
* ``gs_*`` -- GStreamer demuxer seeds for ``targets/gstreamer_read.c``.

GStreamer routing contract (see ``targets/gstreamer_read.c``)
------------------------------------------------------------
The target type-finds the input, then pushes it into the matching demuxer:

    EBML magic  (1A 45 DF A3)      -> matroskademux
    RIFF....WAVE                   -> wavparse
    RIFF....AVI\x20                -> avidemux
    FLV\x01                        -> flvdemux
    <box>ftyp.. at offset 4        -> qtdemux
    (no typefind match)            -> kFallback[byte0 % 5]

A seed that misses its format's magic falls through to the ``byte0 % 5``
fallback and is parsed by the *wrong* demuxer under a synthesised caps, so its
crafted sub-structure never runs. Every ``gs_`` seed below therefore carries
valid container magic for its intended demuxer; the crafted malformation sits in
a sub-structure the demuxer's own ``chain``/parse code reaches. ``tests/
test_regression_ffmpeg_gst_seeds.py`` re-derives the route from the bytes and
asserts each seed reaches the intended demuxer.

Usage
-----
    python tools/gen_ffmpeg_gst_seeds.py --target ffmpeg    --out /tmp/seeds
    python tools/gen_ffmpeg_gst_seeds.py --target gstreamer --out /tmp/seeds
    python tools/gen_ffmpeg_gst_seeds.py --target both      --out /tmp/seeds
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
from pathlib import Path

# Make the `tools` namespace package importable when run as a script (cwd-
# independent), matching how the test suite imports it under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# FFmpeg seeds and byte helpers come from the 9.1 pack -- single source of truth.
from tools.gen_ffmpeg_91_seeds import SEEDS as FFMPEG_SEEDS  # noqa: E402
from tools.gen_ffmpeg_91_seeds import be16, be32, be64, le32  # noqa: E402

DEFAULT_OUT = os.path.expanduser("~/fuzzing/crash_seeds")

# AMF double whose bit pattern is a quiet NaN (FLV onMetaData duration).
AMF_NAN = 0x7FF8000000000000
FLV_PAD_TO = 128

# --- Matroska / EBML element ids (spec-level, shared with FFmpeg) -----------
EBML_MAGIC = b"\x1a\x45\xdf\xa3"
ID_SEGMENT = b"\x18\x53\x80\x67"
ID_INFO = b"\x15\x49\xa9\x66"
ID_TIMECODESCALE = b"\x2a\xd7\xb1"
ID_DURATION = b"\x44\x89"
ID_TRACKS = b"\x16\x54\xae\x6b"
ID_TRACKENTRY = b"\xae"
ID_TRACKNUMBER = b"\xd7"
ID_TRACKUID = b"\x73\xc5"
ID_TRACKTYPE = b"\x83"
ID_CODECID = b"\x86"
ID_CODECPRIVATE = b"\x63\xa2"
ID_VIDEO = b"\xe0"
ID_PIXELWIDTH = b"\xb0"
ID_PIXELHEIGHT = b"\xba"
ID_CLUSTER = b"\x1f\x43\xb6\x75"

TRACKTYPE_VIDEO = 1
TRACKTYPE_AUDIO = 2
MK_TIMECODESCALE = 1_000_000  # nanoseconds per tick (1 ms)


def ebml_len(n: int) -> bytes:
    """Encode ``n`` as the shortest EBML size vint (1..4 bytes here).

    EBML marks the vint width with the position of the first set bit:
    ``1xxxxxxx`` (7-bit), ``01xxxxxx xxxxxxxx`` (14-bit), etc.
    """
    if n < 0x7F:
        return bytes([0x80 | n])
    if n < 0x3FFF:
        return bytes([0x40 | (n >> 8), n & 0xFF])
    if n < 0x1FFFFF:
        return bytes([0x20 | (n >> 16), (n >> 8) & 0xFF, n & 0xFF])

    return bytes([0x10 | (n >> 24), (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF])


def ebml_elem(elem_id: bytes, payload: bytes) -> bytes:
    """One EBML element: id + size-vint + payload."""
    return elem_id + ebml_len(len(payload)) + payload


def ebml_uint(elem_id: bytes, value: int, width: int) -> bytes:
    """Fixed-width unsigned EBML integer element."""
    return ebml_elem(elem_id, value.to_bytes(width, "big"))


def _ebml_header(doctype: bytes = b"matroska") -> bytes:
    """Minimal EBML header a Matroska typefinder + demuxer accept."""
    body = (
        b"\x42\x86\x81\x01"  # EBMLVersion = 1
        + b"\x42\xf7\x81\x01"  # EBMLReadVersion = 1
        + b"\x42\xf2\x81\x04"  # EBMLMaxIDLength = 4
        + b"\x42\xf3\x81\x08"  # EBMLMaxSizeLength = 8
        + b"\x42\x82"
        + ebml_len(len(doctype))
        + doctype  # DocType
        + b"\x42\x87\x81\x02"  # DocTypeVersion = 2
        + b"\x42\x85\x81\x02"  # DocTypeReadVersion = 2
    )
    return EBML_MAGIC + ebml_len(len(body)) + body


# ---------------------------------------------------------------------------
# GStreamer matroskademux seeds
# ---------------------------------------------------------------------------


def gs_mkv_duration_inf():
    """Info.Duration is +inf -> matroskademux duration/segment float math.

    gst_matroska_demux_parse_info reads Duration as a gdouble and scales it by
    TimecodeScale into the segment stop; a non-finite value reaches that math
    unchecked on the pre-hardening path.
    """
    dur = ebml_elem(ID_DURATION, struct.pack(">d", float("inf")))
    info = ebml_uint(ID_TIMECODESCALE, MK_TIMECODESCALE, 3) + dur
    segment = ebml_elem(ID_INFO, info) + ID_CLUSTER  # bare Cluster id ends parse

    return _ebml_header() + ebml_elem(ID_SEGMENT, segment)


def gs_mkv_video_dims():
    """Huge PixelWidth/PixelHeight -> video track size computation overflow.

    The dimensions feed buffer/stride sizing in the video stream setup; 0xFFFFFFFF
    on each axis makes the product overflow 32 bits.
    """
    video = ebml_uint(ID_PIXELWIDTH, 0xFFFFFFFF, 4) + ebml_uint(ID_PIXELHEIGHT, 0xFFFFFFFF, 4)
    entry = (
        ebml_uint(ID_TRACKNUMBER, 1, 1)
        + ebml_uint(ID_TRACKUID, 1, 1)
        + ebml_uint(ID_TRACKTYPE, TRACKTYPE_VIDEO, 1)
        + ebml_elem(ID_CODECID, b"V_UNCOMPRESSED")
        + ebml_elem(ID_VIDEO, video)
    )
    tracks = ebml_elem(ID_TRACKS, ebml_elem(ID_TRACKENTRY, entry))
    segment = tracks + ID_CLUSTER

    return _ebml_header() + ebml_elem(ID_SEGMENT, segment)


def gs_mkv_codecprivate():
    """CodecPrivate size vint outruns the bytes present -> truncation path.

    The declared length (4 KiB) exceeds the 4 bytes that follow, so the demuxer
    reads a CodecPrivate span longer than the available input.
    """
    present = b"\x01\x02\x03\x04"
    cp = ID_CODECPRIVATE + ebml_len(0x1000) + present  # claims 4096, ships 4
    entry = (
        ebml_uint(ID_TRACKNUMBER, 1, 1)
        + ebml_uint(ID_TRACKUID, 1, 1)
        + ebml_uint(ID_TRACKTYPE, TRACKTYPE_AUDIO, 1)
        + ebml_elem(ID_CODECID, b"A_VORBIS")
        + cp
    )
    tracks = ebml_elem(ID_TRACKS, ebml_elem(ID_TRACKENTRY, entry))

    return _ebml_header() + ebml_elem(ID_SEGMENT, tracks)


# ---------------------------------------------------------------------------
# GStreamer qtdemux seed
# ---------------------------------------------------------------------------


def _mp4_box(box_type: bytes, payload: bytes) -> bytes:
    """One MP4/QuickTime box: 32-bit size + 4-char type + payload."""
    return be32(8 + len(payload)) + box_type + payload


def gs_quicktime_mvhd():
    """ftyp at offset 4 (typefind) + moov/mvhd with a huge v1 duration.

    qtdemux parses the Movie Header; a version-1 mvhd carries 64-bit times and a
    0x7FFF... duration drives the timescale/duration math.
    """
    ftyp = _mp4_box(b"ftyp", b"isom" + be32(0) + b"isom")

    mvhd_body = (
        bytes([1])
        + b"\x00\x00\x00"  # version 1, flags
        + be64(0)
        + be64(0)  # creation / modification time
        + be32(1000)  # timescale
        + be64(0x7FFFFFFFFFFFFFFF)  # duration
    )
    mvhd = _mp4_box(b"mvhd", mvhd_body)
    moov = _mp4_box(b"moov", mvhd)

    return ftyp + moov


# ---------------------------------------------------------------------------
# GStreamer wavparse seed
# ---------------------------------------------------------------------------


def gs_wav_missing_fmt():
    """RIFF/WAVE with a data chunk but no fmt -> wavparse error path.

    wavparse expects fmt before data; a lone data chunk with an oversized
    declared size sends it down the missing-format branch.
    """
    data_chunk = b"data" + le32(0xFFFFFFFF)
    payload = b"WAVE" + data_chunk

    return b"RIFF" + le32(len(payload)) + payload


# ---------------------------------------------------------------------------
# GStreamer avidemux seed
# ---------------------------------------------------------------------------


def _riff_list(list_type: bytes, payload: bytes) -> bytes:
    """A RIFF LIST chunk: 'LIST' + size + list-type + payload."""
    return b"LIST" + le32(4 + len(payload)) + list_type + payload


def gs_avi_bad_avih():
    """RIFF/AVI container with a malformed avih -> gst_avi_demux_parse_avih.

    dwMicroSecPerFrame = 0 and dwStreams = 0xFFFFFFFF drive the header parse
    with degenerate frame timing and an implausible stream count.
    """
    avih_body = (
        le32(0)  # dwMicroSecPerFrame = 0
        + le32(0)  # dwMaxBytesPerSec
        + le32(0)  # dwPaddingGranularity
        + le32(0)  # dwFlags
        + le32(0)  # dwTotalFrames
        + le32(0)  # dwInitialFrames
        + le32(0xFFFFFFFF)  # dwStreams
        + le32(0)  # dwSuggestedBufferSize
        + le32(0)  # dwWidth
        + le32(0)  # dwHeight
        + le32(0) * 4  # dwReserved[4]
    )
    avih = b"avih" + le32(len(avih_body)) + avih_body
    hdrl = _riff_list(b"hdrl", avih)
    movi = _riff_list(b"movi", b"")
    payload = b"AVI " + hdrl + movi

    return b"RIFF" + le32(len(payload)) + payload


# ---------------------------------------------------------------------------
# GStreamer flvdemux seed
# ---------------------------------------------------------------------------


def gs_flv_nan_duration():
    """FLV magic + NaN AMF onMetaData duration -> flvdemux metadata parse."""
    body = b"\x02" + be16(len("onMetaData")) + b"onMetaData"
    body += b"\x08" + be32(0)
    body += be16(len("duration")) + b"duration"
    body += b"\x00" + be64(AMF_NAN)
    body += b"\x00\x00\x09"

    tag = bytes([18]) + be32(len(body))[1:] + b"\x00" * 4 + b"\x00" * 3
    out = b"FLV" + bytes([1, 0]) + be32(9) + be32(0) + tag + body

    return out.ljust(FLV_PAD_TO, b"\x00")


# ---------------------------------------------------------------------------
# Seed registry
# ---------------------------------------------------------------------------

GSTREAMER_SEEDS = {
    "matroska_duration_inf.mkv": gs_mkv_duration_inf,
    "matroska_video_dims.mkv": gs_mkv_video_dims,
    "matroska_codecprivate.mkv": gs_mkv_codecprivate,
    "quicktime_mvhd.mp4": gs_quicktime_mvhd,
    "wav_missing_fmt.wav": gs_wav_missing_fmt,
    "avi_bad_avih.avi": gs_avi_bad_avih,
    "flv_nan_duration.flv": gs_flv_nan_duration,
}


def write_seeds(seeds_dir, seeds_dict, prefix=""):
    """Atomically publish ``seeds_dict`` into ``seeds_dir`` with ``prefix``.

    Each seed is staged in the *parent* directory (outside the scanned
    ``seeds/`` tree) via an exclusive, random temp name, then renamed into
    place, so load_corpus never sees a partial file and a planted temp-name
    symlink cannot redirect the write.
    """
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
        default=DEFAULT_OUT,
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
