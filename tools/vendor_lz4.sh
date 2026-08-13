#!/bin/bash
# Download LZ4 source for fuzz target building.
# Extracts to vendor/lz4/ — no configure step needed (LZ4 is a plain
# source drop: lib/*.c compiles standalone with no generated headers).
#
# Usage:
#   tools/vendor_lz4.sh                 # Download & verify
#   tools/vendor_lz4.sh --fast          # Skip the object build-check
#   tools/vendor_lz4.sh --version=1.10.0  # Specific version (default: 1.10.0)
#
# Requirements: curl, tar, clang (or gcc)
#
# After vendoring, build the lz4 fuzz target with:
#   tools/build_targets.sh
#
# The library objects are compiled WITHOUT the AFL shim and linked into the
# target wrapper, which is the only translation unit that gets
# `-include afl_shim.c` (Hard Rule 8). Passing -include to the library
# sources too would emit __afl_map_shm/__afl_area/__afl_guarded_call into
# every object and the final link fails with multiple-definition errors.

set -e

VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor"
LZ4_VERSION="${LZ4_VERSION:-1.10.0}"
LZ4_DIR="$VENDOR_DIR/lz4"
DOWNLOAD_URL="https://github.com/lz4/lz4/archive/refs/tags/v${LZ4_VERSION}.tar.gz"
TARBALL="/tmp/lz4-${LZ4_VERSION}.tar.gz"
BUILD_CHECK=1

# Parse flags
for arg in "$@"; do
    [ "$arg" = "--fast" ] && BUILD_CHECK=0
    case "$arg" in
        --version=*)
            LZ4_VERSION="${arg#*=}"
            DOWNLOAD_URL="https://github.com/lz4/lz4/archive/refs/tags/v${LZ4_VERSION}.tar.gz"
            TARBALL="/tmp/lz4-${LZ4_VERSION}.tar.gz"
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
if [ ! -d "$LZ4_DIR" ]; then
    echo "[1/3] Downloading LZ4 $LZ4_VERSION..."
    if [ ! -f "$TARBALL" ]; then
        echo "  Fetching $DOWNLOAD_URL..."
        curl -fSL -o "$TARBALL" "$DOWNLOAD_URL" || {
            echo "ERROR: Download failed. Try a different version with --version=N"
            exit 1
        }
    fi
    echo "  Extracting..."
    tar -xf "$TARBALL" -C "$VENDOR_DIR"
    mv "$VENDOR_DIR/lz4-${LZ4_VERSION}" "$LZ4_DIR"
    ok "Source extracted to $LZ4_DIR"
else
    echo "[1/3] LZ4 source already at $LZ4_DIR"
fi

# Verify source structure
if [ ! -f "$LZ4_DIR/lib/lz4.c" ] || [ ! -f "$LZ4_DIR/lib/lz4frame.c" ]; then
    echo "ERROR: Invalid LZ4 source tree at $LZ4_DIR"
    exit 1
fi

# ── Step 2: Build-check the library objects ──────────────────────
# LZ4 has no configure; the only thing worth checking is that the four
# sources the target links actually compile in this toolchain.
if [ "$BUILD_CHECK" -eq 1 ]; then
    echo "[2/3] Build-checking LZ4 objects..."
    cc="clang"
    command -v clang &>/dev/null || {
        warn "clang not found, falling back to gcc (shallower edge coverage — see README)"
        cc="gcc"
    }
    tmpdir="$(mktemp -d)"
    rc=0
    for src in lz4 lz4frame lz4hc xxhash; do
        $cc -O2 -g -fPIC -fno-omit-frame-pointer -I"$LZ4_DIR/lib" \
            -c "$LZ4_DIR/lib/${src}.c" -o "$tmpdir/${src}.o" 2>/dev/null || rc=$?
    done
    rm -rf "$tmpdir"
    if [ $rc -eq 0 ]; then
        ok "lz4 objects compile ($cc)"
    else
        warn "lz4 objects failed to compile with $cc"
    fi
else
    echo "[2/3] Skipping build-check (--fast)"
fi

# ── Step 3: Verify ──────────────────────────────────────────────
echo "[3/3] Verifying..."
MISSING=0
for f in lib/lz4.c lib/lz4.h lib/lz4frame.c lib/lz4frame.h lib/lz4frame_static.h \
         lib/lz4hc.c lib/lz4hc.h lib/xxhash.c lib/xxhash.h lib/LICENSE; do
    if [ -f "$LZ4_DIR/$f" ]; then
        ok "$f"
    else
        warn "MISSING: $LZ4_DIR/$f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some source files missing"
    exit 1
fi

echo "=== LZ4 vendored successfully ==="
echo "Source: $LZ4_DIR/"
echo ""
echo "Build the fuzz target with:"
echo "  tools/build_targets.sh"
