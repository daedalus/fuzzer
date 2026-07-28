#!/bin/bash
# Download and build FFmpeg as vendored static libraries with ASAN + trace-pc-guard.
# Produces .a archives for linking into fuzz targets.
#
# Usage:
#   tools/vendor_ffmpeg.sh                # Build with clang + ASAN + trace-cmp
#   tools/vendor_ffmpeg.sh --asan         # Same as default
#   tools/vendor_ffmpeg.sh --fast         # Build without ASAN (gcc, no sanitizer)
#
# Requirements: clang (for ASAN/trace-cmp), make, pkg-config (optional)
# First run downloads ~200MB and builds in ~10 min on 8-core.

set -e

VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor"
FFMPEG_VERSION="${FFMPEG_VERSION:-7.1.3}"
FFMPEG_DIR="$VENDOR_DIR/ffmpeg"
DOWNLOAD_URL="https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
TARBALL="/tmp/ffmpeg-${FFMPEG_VERSION}.tar.xz"

# Parse flags
BUILD_FAST=0
for arg in "$@"; do
    [ "$arg" = "--fast" ] && BUILD_FAST=1
done

# Compiler selection
CC="clang"
if [ "$BUILD_FAST" -eq 1 ]; then
    CC="gcc"
fi
if ! command -v "$CC" &>/dev/null; then
    echo "ERROR: $CC not found. Install clang: sudo apt install clang"
    exit 1
fi

# Flags
SCOV_FLAGS=""
ASAN_FLAGS=""
if [ "$BUILD_FAST" -eq 0 ]; then
    SCOV_FLAGS="-fsanitize-coverage=trace-cmp,trace-pc-guard"
    ASAN_FLAGS="-fsanitize=address"
fi
CFLAGS="-O2 -g -fPIC $ASAN_FLAGS $SCOV_FLAGS"

mkdir -p "$VENDOR_DIR"

# ── Step 1: Download and extract ─────────────────────────────────
if [ ! -d "$FFMPEG_DIR" ]; then
    echo "[1/4] Downloading FFmpeg $FFMPEG_VERSION..."
    if [ ! -f "$TARBALL" ]; then
        curl -L -o "$TARBALL" "$DOWNLOAD_URL"
    fi
    echo "  Extracting to $FFMPEG_DIR..."
    tar -xf "$TARBALL" -C "$VENDOR_DIR"
    mv "$VENDOR_DIR/ffmpeg-${FFMPEG_VERSION}" "$FFMPEG_DIR"
else
    echo "[1/4] FFmpeg source already at $FFMPEG_DIR"
fi

# ── Step 2: Configure ────────────────────────────────────────────
echo "[2/4] Configuring FFmpeg..."
(cd "$FFMPEG_DIR" && \
    CC="$CC" CFLAGS="$CFLAGS" \
    ./configure \
        --enable-static \
        --disable-shared \
        --enable-pic \
        --enable-demuxers \
        --enable-decoders \
        --enable-parsers \
        --enable-bsfs \
        --disable-encoders \
        --disable-muxers \
        --disable-filters \
        --disable-protocols \
        --disable-network \
        --disable-autodetect \
        --disable-doc \
        --disable-htmlpages \
        --disable-manpages \
        --disable-podpages \
        --disable-txtpages \
        --disable-programs \
        --disable-debug \
        --disable-x86asm \
        --extra-cflags="$CFLAGS" \
    2>&1 | tail -5) || {
    echo "ERROR: configure failed"
    exit 1
}

# ── Step 3: Build ────────────────────────────────────────────────
NPROC=$(nproc 2>/dev/null || echo 4)
echo "[3/4] Building FFmpeg ($NPROC cores)..."
(cd "$FFMPEG_DIR" && make -j"$NPROC" -s 2>&1 | tail -5) || {
    echo "ERROR: make failed"
    exit 1
}

# ── Step 4: Verify ───────────────────────────────────────────────
echo "[4/4] Verifying..."
MISSING=0
for lib in libavformat libavcodec libavutil libswresample; do
    a="$FFMPEG_DIR/$lib/$lib.a"
    if [ -f "$a" ]; then
        size=$(du -h "$a" | cut -f1)
        echo "  OK: $a ($size)"
    else
        echo "  MISSING: $a"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some libraries failed to build"
    exit 1
fi

# Count trace-cmp callbacks if ASAN build
if [ "$BUILD_FAST" -eq 0 ]; then
    for lib in libavformat libavcodec; do
        a="$FFMPEG_DIR/$lib/$lib.a"
        tc=$(nm "$a" 2>/dev/null | grep -c 'U.*trace_cmp' || echo 0)
        echo "  $lib: $tc trace-cmp callers"
    done
fi

echo "=== FFmpeg vendored successfully ==="
echo "Libraries: $FFMPEG_DIR/{libavformat,libavcodec,libavutil,libswresample}/*.a"
echo "Headers:   $FFMPEG_DIR/{libavformat,libavcodec,libavutil,libswresample}/"
