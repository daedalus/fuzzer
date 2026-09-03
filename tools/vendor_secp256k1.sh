#!/bin/bash
# Download libsecp256k1 source for fuzz target building.
# Extracts to vendor/secp256k1/ — no configure step needed: the precomputed
# ECMULT tables are committed (src/precomputed_ecmult{,_gen}.c) and the
# wide-multiplication implementation is auto-detected in src/util.h from
# __SIZEOF_INT128__, so src/secp256k1.c compiles standalone with just
# -I src -I include.
#
# Usage:
#   tools/vendor_secp256k1.sh                 # Download & verify
#   tools/vendor_secp256k1.sh --fast          # Skip the object build-check
#   tools/vendor_secp256k1.sh --version=0.8.0  # Specific version (default: 0.8.0)
#
# Requirements: curl, tar, clang (or gcc)
#
# After vendoring, build the secp256k1 fuzz target with:
#   tools/build_targets.sh
#
# The library objects are compiled WITHOUT the AFL shim and linked into the
# target wrapper, which is the only translation unit that gets
# `-include afl_shim.c` (Hard Rule 8). Passing -include to the library
# sources too would emit __afl_map_shm/__afl_area/__afl_guarded_call into
# every object and the final link fails with multiple-definition errors.
# Modules (ecdh, recovery, extrakeys, schnorrsig, musig, ellswift,
# silentpayments) are #ifdef-gated header includes inside secp256k1.c, so
# the target enables them by adding -DENABLE_MODULE_<NAME> at link time.

set -e

# VENDOR_ROOT: see vendor_ffmpeg.sh for the layout rationale.
: "${FUZZ_VENDOR_ROOT:=$HOME/fuzzing/vendoring}"
VENDOR_DIR="${IN_TREE_VENDOR:+$(cd "$(dirname "$0")/.." && pwd)/vendor}"
VENDOR_DIR="${VENDOR_ROOT:-$VENDOR_DIR}"
VENDOR_DIR="${VENDOR_DIR:-$FUZZ_VENDOR_ROOT}"
mkdir -p "$VENDOR_DIR"
for arg in "$@"; do
    [ "$arg" = "--in-tree-vendor" ] && VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor"
done

SECP256K1_VERSION="${SECP256K1_VERSION:-0.8.0}"
SECP256K1_DIR="$VENDOR_DIR/secp256k1"
DOWNLOAD_URL="https://github.com/bitcoin-core/secp256k1/archive/refs/tags/v${SECP256K1_VERSION}.tar.gz"
TARBALL="/tmp/secp256k1-${SECP256K1_VERSION}.tar.gz"
BUILD_CHECK=1

# Parse flags
for arg in "$@"; do
    [ "$arg" = "--fast" ] && BUILD_CHECK=0
    case "$arg" in
        --version=*)
            SECP256K1_VERSION="${arg#*=}"
            DOWNLOAD_URL="https://github.com/bitcoin-core/secp256k1/archive/refs/tags/v${SECP256K1_VERSION}.tar.gz"
            TARBALL="/tmp/secp256k1-${SECP256K1_VERSION}.tar.gz"
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
if [ ! -d "$SECP256K1_DIR" ]; then
    echo "[1/3] Downloading libsecp256k1 $SECP256K1_VERSION..."
    if [ ! -f "$TARBALL" ]; then
        echo "  Fetching $DOWNLOAD_URL..."
        curl -fSL -o "$TARBALL" "$DOWNLOAD_URL" || {
            echo "ERROR: Download failed. Try a different version with --version=N"
            exit 1
        }
    fi
    echo "  Extracting..."
    tar -xf "$TARBALL" -C "$VENDOR_DIR"
    mv "$VENDOR_DIR/secp256k1-${SECP256K1_VERSION}" "$SECP256K1_DIR"
    ok "Source extracted to $SECP256K1_DIR"
else
    echo "[1/3] libsecp256k1 source already at $SECP256K1_DIR"
fi

# Verify source structure
if [ ! -f "$SECP256K1_DIR/src/secp256k1.c" ] || [ ! -f "$SECP256K1_DIR/include/secp256k1.h" ]; then
    echo "ERROR: Invalid libsecp256k1 source tree at $SECP256K1_DIR"
    exit 1
fi

# ── Step 2: Build-check the library objects ──────────────────────
if [ "$BUILD_CHECK" -eq 1 ]; then
    echo "[2/3] Build-checking libsecp256k1 objects..."
    cc="clang"
    command -v clang &>/dev/null || {
        warn "clang not found, falling back to gcc (shallower edge coverage — see README)"
        cc="gcc"
    }
    tmpdir="$(mktemp -d)"
    rc=0
    # Core library plus the committed precomputed ECMULT tables.
    for src in secp256k1 precomputed_ecmult precomputed_ecmult_gen; do
        $cc -O2 -g -fPIC -fno-omit-frame-pointer -I"$SECP256K1_DIR/src" -I"$SECP256K1_DIR/include" \
            -c "$SECP256K1_DIR/src/${src}.c" -o "$tmpdir/${src}.o" 2>/dev/null || rc=$?
    done
    # Full-module pass: enable every module whose sources exist in this tree.
    module_flags=""
    for m in ecdh recovery extrakeys schnorrsig musig ellswift silentpayments; do
        [ -d "$SECP256K1_DIR/src/modules/$m" ] && \
            module_flags="$module_flags -DENABLE_MODULE_$(echo "$m" | tr '[:lower:]' '[:upper:]')"
    done
    if [ -n "$module_flags" ]; then
        $cc -O2 -g -fPIC -fno-omit-frame-pointer -I"$SECP256K1_DIR/src" -I"$SECP256K1_DIR/include" \
            $module_flags -c "$SECP256K1_DIR/src/secp256k1.c" \
            -o "$tmpdir/secp256k1_modules.o" 2>/dev/null || rc=$?
    fi
    rm -rf "$tmpdir"
    if [ $rc -eq 0 ]; then
        ok "libsecp256k1 objects compile ($cc${module_flags:+ with modules})"
    else
        warn "libsecp256k1 objects failed to compile with $cc"
    fi
else
    echo "[2/3] Skipping build-check (--fast)"
fi

# ── Step 3: Verify ──────────────────────────────────────────────
echo "[3/3] Verifying..."
MISSING=0
for f in src/secp256k1.c src/precomputed_ecmult.c src/precomputed_ecmult_gen.c \
         include/secp256k1.h include/secp256k1_preallocated.h COPYING; do
    if [ -f "$SECP256K1_DIR/$f" ]; then
        ok "$f"
    else
        warn "MISSING: $SECP256K1_DIR/$f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some source files missing"
    exit 1
fi

echo "=== libsecp256k1 vendored successfully ==="
echo "Source: $SECP256K1_DIR/"
echo ""
echo "Build the fuzz target with:"
echo "  tools/build_targets.sh"
