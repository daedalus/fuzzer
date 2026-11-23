#!/bin/bash
# Build vendored GStreamer (core + base + good) with clang, static, no-ASAN
# (default) or ASAN.  Output: $FUZZ_BUILD_ROOT/gstreamer-<ver>-<opts>/
#
# Usage: tools/build_gstreamer.sh [--asan] [--cov] [--target] [--version=1.28.9] [-j N]
#   --cov     instrument the libs (trace-pc-guard,trace-cmp); build dir suffix -cov
#   --target  also build $FUZZ_BUILD_ROOT/gstreamer/gstreamer_read_<ver>_<opts>.so
# Requires: meson, ninja, clang, bison, flex, libglib2.0-dev
set -e
: "${FUZZ_VENDOR_ROOT:=$HOME/fuzzing/vendoring}"
: "${FUZZ_BUILD_ROOT:=$HOME/fuzzing/builds}"
GST_VERSION="${GST_VERSION:-1.28.9}"
OPTS=noasan; JOBS=$(nproc); COV=0; TARGET=0
while [ $# -gt 0 ]; do
    case "$1" in
        --asan) OPTS=asan ;;
        --cov) COV=1 ;;
        --target) TARGET=1 ;;
        --version=*) GST_VERSION="${1#*=}" ;;
        -j) shift; JOBS="$1" ;;
    esac; shift
done
SRC="$FUZZ_VENDOR_ROOT/gstreamer-$GST_VERSION"
[ "$COV" = 1 ] && OPTS="$OPTS-cov"
OUT="$FUZZ_BUILD_ROOT/gstreamer-$GST_VERSION-$OPTS"
[ -d "$SRC" ] || { echo "ERROR: run tools/vendor_gstreamer.sh first"; exit 1; }
SAN=(); [ "$OPTS" = asan ] && SAN=(-Db_sanitize=address)
export CC=clang CXX=clang++
[ "$COV" = 1 ] && export CFLAGS="-fsanitize-coverage=trace-pc-guard,trace-cmp -fno-omit-frame-pointer"
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
if [ "$TARGET" = 1 ]; then
    REPO="$(cd "$(dirname "$0")/.." && pwd)"
    SO="$FUZZ_BUILD_ROOT/gstreamer/gstreamer_read_${GST_VERSION}_${OPTS}.so"
    mkdir -p "$(dirname "$SO")"
    LIBS=$(find "$OUT" -name 'libgst*.a' | sort)
    CFL=$(pkg-config --cflags gstreamer-1.0 2>/dev/null || true)
    clang -O2 -g -fPIC -shared -fno-omit-frame-pointer -D__AFL_CMPLOG=1 -D__AFL_PRIVATE_HEAP=1 -Wl,-Bsymbolic -Wl,-z,nodelete \
        -include "$REPO/src/fuzzer_tool/adapters/afl_shim.c" \
        -I"$SRC/subprojects/gstreamer" -I"$SRC/subprojects/gstreamer/libs" -I"$OUT/subprojects/gstreamer" \
        $(pkg-config --cflags glib-2.0 gobject-2.0) \
        -o "$SO" "$REPO/targets/gstreamer_read.c" \
        -Wl,--start-group $LIBS -Wl,--end-group \
        $(pkg-config --libs gobject-2.0 gmodule-2.0 gio-2.0 glib-2.0) -lm -ldl -lpthread \
        -Wl,--export-dynamic
    echo "target: $SO"
fi
echo "=== GStreamer $GST_VERSION built ($OPTS): $OUT ==="
