#!/bin/bash
# Vendor brainflayer + its pinned libsecp256k1 for the differential fuzz
# targets ec_pubkey_diff and hash_simd_diff.
#
# Extracts to vendor/brainflayer/, with the pinned libsecp256k1 (the commit
# brainflayer's own submodule points at) under vendor/brainflayer/secp256k1/.
# brainflayer builds that library with autotools (./autogen.sh && ./configure),
# but we need neither libtool nor a configure run: ec_pubkey_fast.c pulls the
# library's field/group/scalar/ecmult_gen implementations in as headers, and
# the reference secp256k1.c compiles from a hand-written libsecp256k1-config.h
# this script drops in. That config selects the 10x26 field, which is what
# ec_pubkey_fast.c hard-codes an #include of — the two translation units must
# agree on the field representation.
#
# Usage:
#   tools/vendor_brainflayer.sh                 # Clone, configure, verify
#   tools/vendor_brainflayer.sh --fast          # Skip the object build-check
#
# Override the pins via the environment:
#   BRAINFLAYER_REPO   (default: https://github.com/daedalus/brainflayer.git)
#   BRAINFLAYER_REF    (default: the commit validated with these targets)
#   SECP256K1_CLASSIC_REPO / SECP256K1_CLASSIC_REF
#
# Requirements: git, clang (or gcc), OpenSSL headers (libssl-dev). No libtool,
# no autoconf, no GMP.
#
# After vendoring, build the targets with:
#   tools/build_targets.sh
#
# The library sources (ec_pubkey_fast.c, mmapf.c, sha256_pub.c,
# ripemd160_256.c, the reference secp256k1.c) are compiled WITHOUT the AFL
# shim and linked into the target wrapper, which is the only translation unit
# that gets `-include afl_shim.c` (Hard Rule 8). Passing -include to the
# library sources too would emit __afl_map_shm/__afl_area/__afl_guarded_call
# into every object and the final link fails with multiple-definition errors.

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

# Pins. brainflayer @ daedalus fork, which pins libsecp256k1 @ acb7f97 — the
# commit ec_pubkey_fast.c was written against (it has the scalar module and
# the modern context API the reference side calls).
BRAINFLAYER_REPO="${BRAINFLAYER_REPO:-https://github.com/daedalus/brainflayer.git}"
BRAINFLAYER_REF="${BRAINFLAYER_REF:-05dec877cf0e7fd8a0656740cbf5354021bc00d4}"
SECP256K1_CLASSIC_REPO="${SECP256K1_CLASSIC_REPO:-https://github.com/bitcoin-core/secp256k1.git}"
SECP256K1_CLASSIC_REF="${SECP256K1_CLASSIC_REF:-acb7f97eb82dfbbdb797354e1550b910055b4422}"

BRAINFLAYER_DIR="$VENDOR_DIR/brainflayer"
SECP256K1_DIR="$BRAINFLAYER_DIR/secp256k1"
BUILD_CHECK=1
for arg in "$@"; do
    [ "$arg" = "--fast" ] && BUILD_CHECK=0
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
ok() { echo -e "  ${GREEN}OK${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

# Fetch exactly one commit into a fresh checkout (no full history, works for
# a bare commit SHA the way --branch cannot).
fetch_commit() {
    local repo="$1" ref="$2" dest="$3"
    rm -rf "$dest"
    mkdir -p "$dest"
    ( cd "$dest"
      git init -q
      git remote add origin "$repo"
      if git fetch -q --depth 1 origin "$ref" 2>/dev/null; then
          git checkout -q FETCH_HEAD
      else
          # Host refuses fetch-by-sha; fall back to a shallow history scan.
          git fetch -q --depth 200 origin
          git checkout -q "$ref"
      fi
      rm -rf .git )
}

# ── Step 1: brainflayer ──────────────────────────────────────────
if [ ! -f "$BRAINFLAYER_DIR/ec_pubkey_fast.c" ]; then
    echo "[1/4] Cloning brainflayer ($BRAINFLAYER_REF)..."
    fetch_commit "$BRAINFLAYER_REPO" "$BRAINFLAYER_REF" "$BRAINFLAYER_DIR"
    ok "brainflayer at $BRAINFLAYER_DIR"
else
    echo "[1/4] brainflayer already at $BRAINFLAYER_DIR"
fi

# ── Step 2: pinned libsecp256k1 ──────────────────────────────────
if [ ! -f "$SECP256K1_DIR/src/secp256k1.c" ]; then
    echo "[2/4] Cloning libsecp256k1 ($SECP256K1_CLASSIC_REF)..."
    fetch_commit "$SECP256K1_CLASSIC_REPO" "$SECP256K1_CLASSIC_REF" "$SECP256K1_DIR"
    ok "libsecp256k1 at $SECP256K1_DIR"
else
    echo "[2/4] libsecp256k1 already at $SECP256K1_DIR"
fi

# ── Step 3: build config ─────────────────────────────────────────
# Hand-written in place of ./configure. 10x26 field to match the #include in
# ec_pubkey_fast.c; 8x32 scalar (no __int128 needed); num=none so there is no
# GMP/OpenSSL bignum dependency; builtin inverses as num=none requires. The
# ECMULT_* sizes are the stock configure defaults.
echo "[3/4] Writing libsecp256k1-config.h..."
cat > "$SECP256K1_DIR/src/libsecp256k1-config.h" <<'EOF'
#ifndef LIBSECP256K1_CONFIG_H
#define LIBSECP256K1_CONFIG_H
#define USE_NUM_NONE 1
#define USE_FIELD_INV_BUILTIN 1
#define USE_SCALAR_INV_BUILTIN 1
#define USE_FIELD_10X26 1
#define USE_SCALAR_8X32 1
#define ECMULT_WINDOW_SIZE 15
#define ECMULT_GEN_PREC_BITS 4
#endif
EOF
ok "libsecp256k1-config.h (10x26 field, 8x32 scalar, num=none)"

# ── Step 4: build-check ──────────────────────────────────────────
if [ "$BUILD_CHECK" -eq 1 ]; then
    echo "[4/4] Build-checking objects..."
    cc="clang"
    command -v clang &>/dev/null || {
        warn "clang not found, falling back to gcc (shallower edge coverage — see README)"
        cc="gcc"
    }
    tmpdir="$(mktemp -d)"
    rc=0
    # ec_pubkey_diff side: fast path, mmap helper, reference library.
    $cc -O2 -g -fPIC -fno-omit-frame-pointer -Wno-unused-function \
        -I"$BRAINFLAYER_DIR" -c "$BRAINFLAYER_DIR/ec_pubkey_fast.c" \
        -o "$tmpdir/ec_pubkey_fast.o" 2>/dev/null || rc=$?
    $cc -O2 -g -fPIC -fno-omit-frame-pointer \
        -c "$BRAINFLAYER_DIR/mmapf.c" -o "$tmpdir/mmapf.o" 2>/dev/null || rc=$?
    $cc -O2 -g -fPIC -fno-omit-frame-pointer -DHAVE_CONFIG_H=1 \
        -I"$SECP256K1_DIR" -I"$SECP256K1_DIR/src" -I"$SECP256K1_DIR/include" \
        -c "$SECP256K1_DIR/src/secp256k1.c" -o "$tmpdir/secp256k1.o" 2>/dev/null || rc=$?
    # hash_simd_diff side: SIMD + scalar hashers (needs OpenSSL headers).
    $cc -O2 -g -fPIC -fno-omit-frame-pointer \
        -c "$BRAINFLAYER_DIR/sha256_pub.c" -o "$tmpdir/sha256_pub.o" 2>/dev/null || rc=$?
    $cc -O2 -g -fPIC -fno-omit-frame-pointer \
        -c "$BRAINFLAYER_DIR/ripemd160_256.c" -o "$tmpdir/ripemd160_256.o" 2>/dev/null || rc=$?
    rm -rf "$tmpdir"
    if [ $rc -eq 0 ]; then
        ok "all objects compile ($cc)"
    else
        warn "some objects failed to compile with $cc (is libssl-dev installed?)"
    fi
else
    echo "[4/4] Skipping build-check (--fast)"
fi

# ── Verify ───────────────────────────────────────────────────────
MISSING=0
for f in ec_pubkey_fast.c ec_pubkey_fast.h mmapf.c mmapf.h \
         sha256_pub.c sha256_pub.h sha256_simd.h \
         ripemd160_256.c ripemd160_256.h ripemd160_simd.h \
         secp256k1/src/secp256k1.c secp256k1/include/secp256k1.h \
         secp256k1/src/libsecp256k1-config.h; do
    if [ -f "$BRAINFLAYER_DIR/$f" ]; then
        ok "$f"
    else
        warn "MISSING: $BRAINFLAYER_DIR/$f"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some source files missing"
    exit 1
fi

echo "=== brainflayer vendored successfully ==="
echo "Source: $BRAINFLAYER_DIR/"
echo ""
echo "Build the fuzz targets with:"
echo "  tools/build_targets.sh"
