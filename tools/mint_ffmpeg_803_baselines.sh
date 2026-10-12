#!/bin/bash
# Mint the container mutation baselines for the FFmpeg 8.0.3 seed pack.
#
# These are short real streams, one per container the 8.1.3 parser fixes touch
# (mov trun/sgpd/keys/iref, mpegts PES, matroska, avidec LIST/INFO, ogg, flv,
# mxf, w64, apng, and the image probes). They are mutation starting points, not
# crash reproducers.
#
# Usage:
#   tools/mint_ffmpeg_803_baselines.sh [--out DIR]
#
# Output: <out>/seeds/baseline.<ext>; --out defaults to
# ~/fuzzing/ffmpeg_803_potential_crashes. Crafted seeds are generated separately
# by tools/gen_ffmpeg_803_seeds.py.
#
# Requirements: ffmpeg with libx264, libvorbis, libmp3lame on PATH.

set -u

OUT="${HOME}/fuzzing/ffmpeg_803_potential_crashes"
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

OUT="${OUT}/seeds"
mkdir -p "$OUT"
FF=(ffmpeg -y -hide_banner -loglevel error)
V="testsrc2=size=64x64:rate=5:duration=1"
A="sine=frequency=440:duration=1"
fail=0

mint() {  # mint <name> <ffmpeg args...>
    local name="$1"; shift
    if "${FF[@]}" "$@" "$OUT/$name"; then
        printf '  ok  %-18s %8d B\n' "$name" "$(stat -c%s "$OUT/$name")"
    else
        printf 'FAIL  %s\n' "$name" >&2
        fail=$((fail + 1))
    fi
}

mint baseline.mp4  -f lavfi -i "$V" -f lavfi -i "$A" -c:v libx264 -pix_fmt yuv420p -c:a aac  -shortest
mint baseline.ts   -f lavfi -i "$V" -f lavfi -i "$A" -c:v mpeg2video -c:a mp2 -f mpegts
mint baseline.mkv  -f lavfi -i "$V" -f lavfi -i "$A" -c:v libx264 -pix_fmt yuv420p -c:a libvorbis -shortest -f matroska
mint baseline.avi  -f lavfi -i "$V" -f lavfi -i "$A" -c:v mpeg4 -c:a libmp3lame -shortest -f avi
mint baseline.ogg  -f lavfi -i "$A" -c:a libvorbis -f ogg
mint baseline.flv  -f lavfi -i "$V" -f lavfi -i "$A" -c:v flv -c:a libmp3lame -shortest -f flv
mint baseline.mxf  -f lavfi -i "testsrc2=size=64x64:rate=25:duration=1" -f lavfi -i "$A" -c:v mpeg2video -c:a pcm_s16le -ar 48000 -f mxf
mint baseline.w64  -f lavfi -i "$A" -c:a pcm_s16le -f w64
mint baseline.apng -f lavfi -i "testsrc2=size=32x32:rate=2:duration=1" -c:v apng -f apng
mint baseline.tiff -f lavfi -i "testsrc2=size=32x32" -frames:v 1
mint baseline.ppm  -f lavfi -i "testsrc2=size=32x32" -frames:v 1
mint baseline.bmp  -f lavfi -i "testsrc2=size=32x32" -frames:v 1 -pix_fmt bgr24

exit "$fail"
