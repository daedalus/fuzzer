#!/bin/bash
# Download and build FFmpeg as vendored static libraries for fuzzing.
#
# Build modes (pick one):
#   tools/vendor_ffmpeg.sh              # default: --nosan (clang, coverage-only, no ASAN)
#   tools/vendor_ffmpeg.sh --nosan      # clang + trace-pc-guard,trace-cmp, NO sanitizer
#                                       #   -> vendor/ffmpeg      (linked by _nosan .so targets)
#   tools/vendor_ffmpeg.sh --asan       # clang + ASAN + trace-pc-guard,trace-cmp
#                                       #   -> vendor/ffmpeg_asan (linked by _asan targets)
#   tools/vendor_ffmpeg.sh --fast       # gcc, no instrumentation (fastest build, no coverage)
#                                       #   -> vendor/ffmpeg_fast
#
# Component set:
#   default            = full (all demuxers/decoders/parsers/bsfs) — matches upstream fuzzing
#   --minimal          = small audio-focused set (mov/matroska/wav/flac/mp3/ogg + aac/flac/mp3/
#                        vorbis/pcm). Builds in a few minutes even on 1 core. Good for CI / laptops.
#   FFMPEG_COMPONENTS  = space-separated configure flags to override the component set entirely.
#
# Source (tried in order; first reachable wins). Override with FFMPEG_SRC=<url-or-gitref>.
#   1. https://ffmpeg.org/releases/ffmpeg-<ver>.tar.xz        (upstream release tarball)
#   2. https://codeload.github.com/FFmpeg/FFmpeg/tar.gz/refs/tags/n<ver>   (GitHub tarball mirror)
#   3. git clone --depth 1 --branch n<ver> https://github.com/FFmpeg/FFmpeg   (git fallback)
# The GitHub mirrors matter in locked-down/CI networks where ffmpeg.org egress is blocked.
#
# Requirements: clang (for --nosan/--asan), make, curl or git, tar/xz.
# A full instrumented build downloads ~15MB (tarball) and takes ~10 min on 8 cores.

set -e

VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor"
FFMPEG_VERSION="${FFMPEG_VERSION:-7.1.3}"

# ── Parse flags ──────────────────────────────────────────────────
MODE="nosan"          # nosan | asan | fast
MINIMAL=0
for arg in "$@"; do
    case "$arg" in
        --nosan) MODE="nosan" ;;
        --asan)  MODE="asan" ;;
        --fast)  MODE="fast" ;;
        --minimal) MINIMAL=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# ── Per-mode compiler + flags + output dir ───────────────────────
case "$MODE" in
    nosan)
        CC="clang"
        SAN_FLAGS=""
        SCOV_FLAGS="-fsanitize-coverage=trace-cmp,trace-pc-guard"
        FFMPEG_DIR="${FFMPEG_DIR:-$VENDOR_DIR/ffmpeg}"
        ;;
    asan)
        CC="clang"
        SAN_FLAGS="-fsanitize=address"
        SCOV_FLAGS="-fsanitize-coverage=trace-cmp,trace-pc-guard"
        FFMPEG_DIR="${FFMPEG_DIR:-$VENDOR_DIR/ffmpeg_asan}"
        ;;
    fast)
        CC="gcc"
        SAN_FLAGS=""
        SCOV_FLAGS=""
        FFMPEG_DIR="${FFMPEG_DIR:-$VENDOR_DIR/ffmpeg_fast}"
        ;;
esac

if ! command -v "$CC" &>/dev/null; then
    echo "ERROR: $CC not found. Install clang: sudo apt install clang" >&2
    exit 1
fi

# -fno-omit-frame-pointer: afl_shim.c's caller-context edge hashing walks
# __builtin_return_address(1) into this library's frame (see FRAME_POINTER in
# tools/build_targets.sh); without it the walk silently returns the wrong caller.
CFLAGS="-O2 -g -fPIC -fno-omit-frame-pointer $SAN_FLAGS $SCOV_FLAGS"

# For a coverage-only (no-ASAN) build, -fsanitize-coverage=trace-pc-guard makes
# clang emit calls to __sanitizer_cov_trace_pc_guard{,_init} + __sanitizer_cov_
# trace_cmp*. With ASAN those live in the ASAN runtime, so configure's link-tests
# resolve them; without a sanitizer runtime they are undefined and *every*
# configure compile-and-link probe fails ("clang is unable to create an
# executable file"). The real callbacks come from afl_shim.c at harness link
# time, and `make` only archives .o into .a (no linking), so we satisfy configure
# with a throwaway no-op stub passed via --extra-ldflags. It never enters the .a.
STUB_LDFLAGS=""
if [ -n "$SCOV_FLAGS" ] && [ -z "$SAN_FLAGS" ]; then
    STUB_SRC="$(mktemp /tmp/sancov_stub.XXXXXX.c)"
    STUB_OBJ="${STUB_SRC%.c}.o"
    cat > "$STUB_SRC" <<'STUB'
#include <stdint.h>
void __sanitizer_cov_trace_pc_guard(uint32_t *g){(void)g;}
void __sanitizer_cov_trace_pc_guard_init(uint32_t *s,uint32_t *e){(void)s;(void)e;}
void __sanitizer_cov_trace_cmp1(uint8_t a,uint8_t b){(void)a;(void)b;}
void __sanitizer_cov_trace_cmp2(uint16_t a,uint16_t b){(void)a;(void)b;}
void __sanitizer_cov_trace_cmp4(uint32_t a,uint32_t b){(void)a;(void)b;}
void __sanitizer_cov_trace_cmp8(uint64_t a,uint64_t b){(void)a;(void)b;}
void __sanitizer_cov_trace_const_cmp1(uint8_t a,uint8_t b){(void)a;(void)b;}
void __sanitizer_cov_trace_const_cmp2(uint16_t a,uint16_t b){(void)a;(void)b;}
void __sanitizer_cov_trace_const_cmp4(uint32_t a,uint32_t b){(void)a;(void)b;}
void __sanitizer_cov_trace_const_cmp8(uint64_t a,uint64_t b){(void)a;(void)b;}
void __sanitizer_cov_trace_switch(uint64_t v,uint64_t *c){(void)v;(void)c;}
STUB
    "$CC" -O1 -c "$STUB_SRC" -o "$STUB_OBJ"
    STUB_LDFLAGS="$STUB_OBJ"
fi

# ── Component selection ──────────────────────────────────────────
if [ -n "$FFMPEG_COMPONENTS" ]; then
    COMPONENTS="$FFMPEG_COMPONENTS"
elif [ "$MINIMAL" -eq 1 ]; then
    COMPONENTS="--disable-everything \
        --enable-demuxer=mov,matroska,wav,aiff,flac,mp3,ogg \
        --enable-decoder=pcm_s16le,pcm_s16be,pcm_u8,flac,mp3,aac,vorbis \
        --enable-parser=aac,flac,mpegaudio,vorbis \
        --enable-protocol=file"
else
    COMPONENTS="--enable-demuxers --enable-decoders --enable-parsers --enable-bsfs \
        --disable-encoders --disable-muxers --disable-filters --disable-protocols \
        --enable-protocol=file --disable-network --disable-autodetect"
fi

mkdir -p "$VENDOR_DIR"

# ── Step 1: Acquire source (multi-source with fallbacks) ─────────
fetch_source() {
    [ -d "$FFMPEG_DIR" ] && [ -f "$FFMPEG_DIR/configure" ] && { echo "[1/4] Source present at $FFMPEG_DIR"; return 0; }
    local tmp="$VENDOR_DIR/.ffsrc"; rm -rf "$tmp"; mkdir -p "$tmp"
    local tarball="$tmp/ffmpeg.tar"

    # explicit override
    if [ -n "$FFMPEG_SRC" ]; then
        echo "[1/4] Fetching from FFMPEG_SRC=$FFMPEG_SRC"
        if [[ "$FFMPEG_SRC" == *.git || "$FFMPEG_SRC" == git://* ]]; then
            git clone --depth 1 "$FFMPEG_SRC" "$FFMPEG_DIR" && return 0
        else
            curl -fL -o "$tarball" "$FFMPEG_SRC" && _extract "$tarball" && return 0
        fi
        echo "ERROR: FFMPEG_SRC fetch failed" >&2; return 1
    fi

    # 1. upstream release tarball
    echo "[1/4] Trying ffmpeg.org release tarball..."
    if curl -fL --connect-timeout 15 -o "$tarball" \
         "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" 2>/dev/null; then
        _extract "$tarball" && return 0
    fi
    # 2. GitHub codeload tarball (works where ffmpeg.org is blocked)
    echo "    ffmpeg.org unreachable — trying GitHub codeload tarball..."
    if curl -fL --connect-timeout 15 -o "$tarball" \
         "https://codeload.github.com/FFmpeg/FFmpeg/tar.gz/refs/tags/n${FFMPEG_VERSION}" 2>/dev/null; then
        _extract "$tarball" && return 0
    fi
    # 3. git clone fallback
    echo "    codeload failed — trying git clone of tag n${FFMPEG_VERSION}..."
    if git clone --depth 1 --branch "n${FFMPEG_VERSION}" \
         https://github.com/FFmpeg/FFmpeg "$FFMPEG_DIR" 2>/dev/null; then
        return 0
    fi
    echo "ERROR: all FFmpeg source mirrors failed (ffmpeg.org, codeload, github git)." >&2
    echo "       Set FFMPEG_SRC=<tarball-url|git-url> to a reachable mirror." >&2
    return 1
}
_extract() {
    local tarball="$1"
    echo "  Extracting to $FFMPEG_DIR..."
    tar -xf "$tarball" -C "$VENDOR_DIR/.ffsrc"
    local top; top="$(find "$VENDOR_DIR/.ffsrc" -maxdepth 1 -type d -name 'FFmpeg*' -o -maxdepth 1 -type d -name 'ffmpeg*' | head -1)"
    [ -z "$top" ] && { echo "ERROR: unexpected archive layout" >&2; return 1; }
    rm -rf "$FFMPEG_DIR"; mv "$top" "$FFMPEG_DIR"
}
fetch_source

# ── Step 2: Configure ────────────────────────────────────────────
echo "[2/4] Configuring FFmpeg ($MODE${MINIMAL:+, minimal})..."
(cd "$FFMPEG_DIR" && \
    CC="$CC" CFLAGS="$CFLAGS" \
    ./configure \
        --cc="$CC" \
        --enable-static \
        --disable-shared \
        --enable-pic \
        $COMPONENTS \
        --disable-doc \
        --disable-htmlpages \
        --disable-manpages \
        --disable-podpages \
        --disable-txtpages \
        --disable-programs \
        --disable-debug \
        --disable-x86asm \
        --extra-cflags="$CFLAGS" \
        --extra-ldflags="$SAN_FLAGS $SCOV_FLAGS $STUB_LDFLAGS" \
    2>&1 | tail -5) || {
    echo "ERROR: configure failed (see $FFMPEG_DIR/ffbuild/config.log)"
    exit 1
}

# ── Step 3: Build ────────────────────────────────────────────────
NPROC=$(nproc 2>/dev/null || echo 4)
echo "[3/4] Building FFmpeg ($NPROC cores)..."
(cd "$FFMPEG_DIR" && make -j"$NPROC" -s \
    libavutil/libavutil.a libavcodec/libavcodec.a \
    libavformat/libavformat.a libswresample/libswresample.a 2>&1 | tail -5) || {
    echo "ERROR: make failed"
    exit 1
}

# ── Step 4: Verify ───────────────────────────────────────────────
echo "[4/4] Verifying..."
MISSING=0
for lib in libavformat libavcodec libavutil libswresample; do
    a="$FFMPEG_DIR/$lib/$lib.a"
    if [ -f "$a" ]; then
        echo "  OK: $a ($(du -h "$a" | cut -f1))"
    else
        echo "  MISSING: $a"; MISSING=1
    fi
done
[ "$MISSING" -eq 1 ] && { echo "ERROR: Some libraries failed to build"; exit 1; }

if [ -n "$SCOV_FLAGS" ]; then
    for lib in libavformat libavcodec; do
        a="$FFMPEG_DIR/$lib/$lib.a"
        tc=$(nm "$a" 2>/dev/null | grep -c 'U.*trace_cmp' || true)
        echo "  $lib: $tc trace-cmp call sites"
    done
fi

[ -n "$STUB_LDFLAGS" ] && rm -f "$STUB_SRC" "$STUB_OBJ"

echo "=== FFmpeg vendored successfully ($MODE) ==="
echo "Libraries: $FFMPEG_DIR/{libavformat,libavcodec,libavutil,libswresample}/*.a"
echo "Next:      tools/build_ffmpeg_ready.sh   (links the ready-to-fuzz harness)"
