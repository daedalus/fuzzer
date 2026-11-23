#!/bin/bash
# Build vendored GStreamer (core + base + good) with clang, static, no-ASAN
# (default) or ASAN.  Output: $FUZZ_BUILD_ROOT/gstreamer-<ver>-<opts>/
#
# Usage: tools/build_gstreamer.sh [--asan] [--version=1.28.9] [-j N]
# Requires: meson, ninja, clang, bison, flex, libglib2.0-dev
set -e
: "${FUZZ_VENDOR_ROOT:=$HOME/fuzzing/vendoring}"
: "${FUZZ_BUILD_ROOT:=$HOME/fuzzing/builds}"
GST_VERSION="${GST_VERSION:-1.28.9}"
OPTS=noasan; JOBS=$(nproc)
while [ $# -gt 0 ]; do
    case "$1" in
        --asan) OPTS=asan ;;
        --version=*) GST_VERSION="${1#*=}" ;;
        -j) shift; JOBS="$1" ;;
    esac; shift
done
SRC="$FUZZ_VENDOR_ROOT/gstreamer-$GST_VERSION"
OUT="$FUZZ_BUILD_ROOT/gstreamer-$GST_VERSION-$OPTS"
[ -d "$SRC" ] || { echo "ERROR: run tools/vendor_gstreamer.sh first"; exit 1; }
SAN=(); [ "$OPTS" = asan ] && SAN=(-Db_sanitize=address)
export CC=clang CXX=clang++
if [ ! -f "$OUT/build.ninja" ]; then
    meson setup "$OUT" "$SRC" --wrap-mode=nofallback \
        --default-library=static --buildtype=debugoptimized "${SAN[@]}" \
        -Dauto_features=disabled -Dgpl=disabled -Dtests=disabled \
        -Dexamples=disabled -Ddoc=disabled -Dintrospection=disabled \
        -Dgtk_doc=disabled -Dnls=disabled -Dgst-examples=disabled \
        -Dgstreamer:tools=disabled -Dgstreamer:benchmarks=disabled \
        -Dbase=enabled -Dgood=enabled \
        -Dgst-plugins-base:app=enabled -Dgst-plugins-base:typefind=enabled \
        -Dgst-plugins-base:playback=enabled -Dgst-plugins-base:audioconvert=enabled \
        -Dgst-plugins-base:videoconvertscale=enabled \
        -Dgst-plugins-good:isomp4=enabled -Dgst-plugins-good:matroska=enabled \
        -Dgst-plugins-good:wavparse=enabled -Dgst-plugins-good:avi=enabled \
        -Dgst-plugins-good:flv=enabled
fi
ninja -C "$OUT" -j "$JOBS"
echo "=== GStreamer $GST_VERSION built ($OPTS): $OUT ==="
