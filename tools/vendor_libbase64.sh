#!/bin/bash
# Download libbase64 (aklomp/base64) source for fuzz target building.
# Extracts to $FUZZ_VENDOR_ROOT/libbase64/ — no configure step: the codecs
# are plain C files. What upstream's Makefile generates is lib/config.h,
# which selects which codecs get compiled in; tools/build_targets.sh writes
# that per build suffix so this tree stays pristine (see
# compile_libbase64_objects()).
#
# Usage:
#   tools/vendor_libbase64.sh                   # Download & verify
#   tools/vendor_libbase64.sh --fast            # Skip the object build-check
#   tools/vendor_libbase64.sh --version=0.5.3   # Specific version (default: 0.5.3)
#
# Requirements: curl, tar, clang (or gcc)
#
# After vendoring, build the libbase64 fuzz target with:
#   tools/build_targets.sh
#
# The library objects are compiled WITHOUT the AFL shim and linked into the
# target wrapper, which is the only translation unit that gets
# `-include afl_shim.c` (Hard Rule 9). Each SIMD codec is its own TU with its
# own -m flag, so the objects are built one per codec.

set -e

# VENDOR_ROOT: see vendor_ffmpeg.sh for the layout rationale.
#   default: ~/fuzzing/vendoring/   (canonical home)
#   override: FUZZ_VENDOR_ROOT=...
#   legacy:   --in-tree-vendor restores the in-tree `vendor/` location.
: "${FUZZ_VENDOR_ROOT:=$HOME/fuzzing/vendoring}"
VENDOR_DIR="${IN_TREE_VENDOR:+$(cd "$(dirname "$0")/.." && pwd)/vendor}"
VENDOR_DIR="${VENDOR_ROOT:-$VENDOR_DIR}"
VENDOR_DIR="${VENDOR_DIR:-$FUZZ_VENDOR_ROOT}"
mkdir -p "$VENDOR_DIR"
for arg in "$@"; do
    [ "$arg" = "--in-tree-vendor" ] && VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor"
done

LIBBASE64_VERSION="${LIBBASE64_VERSION:-0.5.3}"
LIBBASE64_DIR="$VENDOR_DIR/libbase64"
DOWNLOAD_URL="https://github.com/aklomp/base64/archive/refs/tags/v${LIBBASE64_VERSION}.tar.gz"
TARBALL="/tmp/libbase64-${LIBBASE64_VERSION}.tar.gz"
BUILD_CHECK=1

# Parse flags
for arg in "$@"; do
    [ "$arg" = "--fast" ] && BUILD_CHECK=0
    case "$arg" in
        --version=*)
            LIBBASE64_VERSION="${arg#*=}"
            DOWNLOAD_URL="https://github.com/aklomp/base64/archive/refs/tags/v${LIBBASE64_VERSION}.tar.gz"
            TARBALL="/tmp/libbase64-${LIBBASE64_VERSION}.tar.gz"
            ;;
    esac
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
ok() { echo -e "  ${GREEN}OK${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

mkdir -p "$VENDOR_DIR"

# ── Step 1: Download and extract ─────────────────────────────────
if [ ! -d "$LIBBASE64_DIR" ]; then
    echo "[1/3] Downloading libbase64 $LIBBASE64_VERSION..."
    if [ ! -f "$TARBALL" ]; then
        echo "  Fetching $DOWNLOAD_URL..."
        curl -fSL -o "$TARBALL" "$DOWNLOAD_URL" || {
            echo "ERROR: Download failed. Try a different version with --version=N"
            exit 1
        }
    fi
    echo "  Extracting..."
    tar -xf "$TARBALL" -C "$VENDOR_DIR"
    mv "$VENDOR_DIR/base64-${LIBBASE64_VERSION}" "$LIBBASE64_DIR"
    ok "Source extracted to $LIBBASE64_DIR"
else
    echo "[1/3] libbase64 source already at $LIBBASE64_DIR"
fi

# Verify source structure
if [ ! -f "$LIBBASE64_DIR/lib/lib.c" ] || [ ! -f "$LIBBASE64_DIR/lib/codec_choose.c" ]; then
    echo "ERROR: Invalid libbase64 source tree at $LIBBASE64_DIR"
    exit 1
fi

# ── Step 2: Build-check the library objects ──────────────────────
# libbase64 has no configure. The only thing worth checking is that the
# core sources plus one SIMD codec compile here; the per-codec -m flags are
# probed by build_targets.sh, which is what decides config.h.
if [ "$BUILD_CHECK" -eq 1 ]; then
    echo "[2/3] Build-checking libbase64 objects..."
    cc="clang"
    command -v clang &>/dev/null || {
        warn "clang not found, falling back to gcc (shallower edge coverage — see README)"
        cc="gcc"
    }
    tmpdir="$(mktemp -d)"
    printf '#define HAVE_AVX2 1\n' > "$tmpdir/config.h"
    rc=0
    for src in lib/lib lib/codec_choose lib/tables/tables lib/arch/generic/codec; do
        $cc -O2 -g -fPIC -fno-omit-frame-pointer -I"$tmpdir" -I"$LIBBASE64_DIR/lib" \
            -I"$LIBBASE64_DIR/include" -c "$LIBBASE64_DIR/${src}.c" \
            -o "$tmpdir/$(basename "$src").o" 2>/dev/null || rc=$?
    done
    $cc -O2 -g -fPIC -mavx2 -I"$tmpdir" -I"$LIBBASE64_DIR/lib" -I"$LIBBASE64_DIR/include" \
        -c "$LIBBASE64_DIR/lib/arch/avx2/codec.c" -o "$tmpdir/codec.o" 2>/dev/null || rc=$?
    rm -rf "$tmpdir"
    if [ $rc -eq 0 ]; then
        ok "libbase64 core + AVX2 objects compile ($cc)"
    else
        warn "libbase64 objects failed to compile with $cc"
    fi
else
    echo "[2/3] Skipping build-check (--fast)"
fi

# ── Step 3: Verify ──────────────────────────────────────────────
echo "[3/3] Verifying..."
MISSING=0
for f in include/libbase64.h lib/lib.c lib/codec_choose.c lib/codecs.h lib/env.h \
         lib/tables/tables.c lib/tables/tables.h lib/arch/generic/codec.c \
         lib/arch/ssse3/codec.c lib/arch/sse41/codec.c lib/arch/sse42/codec.c \
         lib/arch/avx/codec.c lib/arch/avx2/codec.c lib/arch/avx512/codec.c \
         LICENSE; do
    if [ -f "$LIBBASE64_DIR/$f" ]; then
        ok "$f"
    else
        warn "MISSING: $LIBBASE64_DIR/$f"
        MISSING=1
    fi
done

# A config.h here means someone ran upstream's make in the vendored tree.
# lib/*.c do `#include "config.h"`, which resolves next to the source before
# any -I, so a stale one silently overrides the generated per-suffix one and
# drops codecs from the build without a symptom.
if [ -f "$LIBBASE64_DIR/lib/config.h" ]; then
    warn "stale $LIBBASE64_DIR/lib/config.h: remove it (rm) — it overrides the generated one"
fi

if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some source files missing"
    exit 1
fi

echo "=== libbase64 vendored successfully ==="
echo "Source: $LIBBASE64_DIR/"
echo ""
echo "Build the fuzz target with:"
echo "  tools/build_targets.sh"
