#!/bin/bash
# Download OpenSSL source for fuzz target building.
# Extracts to vendor/openssl/ — read-only here. tools/build_targets.sh
# configures and builds libcrypto.a out of tree under $FUZZ_BUILD_ROOT/openssl/.
#
# Usage:
#   tools/vendor_openssl.sh                    # Download & verify
#   tools/vendor_openssl.sh --fast             # Skip the Configure check
#   tools/vendor_openssl.sh --version=3.5.4    # Specific version (default: 4.1.0; 3.5.x is LTS)
#   tools/vendor_openssl.sh --version=master   # git HEAD -> vendor/openssl-master/ (AVX2 decoder)
#   tools/vendor_openssl.sh --version=master --commit=<sha>   # pin a master commit
#
# Requirements: curl, tar, perl, make, clang (or gcc); git for master
#
# After vendoring, build the openssl fuzz target with:
#   tools/build_targets.sh
#
# libcrypto is built WITHOUT the AFL shim and linked into the target
# wrapper, the only TU that gets `-include afl_shim.c` (Hard Rule 9).

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

OPENSSL_VERSION="${OPENSSL_VERSION:-4.1.0}"
OPENSSL_DIR="$VENDOR_DIR/openssl"
CONFIG_CHECK=1

# Parse flags
for arg in "$@"; do
    [ "$arg" = "--fast" ] && CONFIG_CHECK=0
    case "$arg" in
        --version=*) OPENSSL_VERSION="${arg#*=}" ;;
        --commit=*) OPENSSL_COMMIT="${arg#*=}" ;;
    esac
done

# master is a moving target with no tarball: shallow git clone into its own
# tree, so release-built targets (vendor/openssl/) never change under it.
# build_targets.sh stamps master builds with the commit (VERSION.dat stays
# x.y.0-dev across commits).
OPENSSL_GIT_URL="https://github.com/openssl/openssl.git"
if [ "$OPENSSL_VERSION" = "master" ]; then
    OPENSSL_DIR="$VENDOR_DIR/openssl-master"
fi
DOWNLOAD_URL="https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VERSION}/openssl-${OPENSSL_VERSION}.tar.gz"
TARBALL="/tmp/openssl-${OPENSSL_VERSION}.tar.gz"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
ok() { echo -e "  ${GREEN}OK${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

# ── Step 1 (master): clone, optionally pin a commit ──────────────
if [ "$OPENSSL_VERSION" = "master" ] && [ ! -d "$OPENSSL_DIR" ]; then
    echo "[1/3] Cloning OpenSSL master..."
    git clone -q --depth 1 "$OPENSSL_GIT_URL" "$OPENSSL_DIR"
    if [ -n "${OPENSSL_COMMIT:-}" ]; then
        git -C "$OPENSSL_DIR" fetch -q --depth 1 origin "$OPENSSL_COMMIT"
        git -C "$OPENSSL_DIR" checkout -q FETCH_HEAD
    fi
    ok "master $(git -C "$OPENSSL_DIR" rev-parse --short HEAD) at $OPENSSL_DIR"
elif [ "$OPENSSL_VERSION" = "master" ]; then
    echo "[1/3] OpenSSL master already at $OPENSSL_DIR ($(git -C "$OPENSSL_DIR" rev-parse --short HEAD 2>/dev/null)); move it aside to refresh"
fi

# ── Step 1: Download, verify checksum, extract ───────────────────
if [ "$OPENSSL_VERSION" = "master" ]; then
    :  # cloned above
elif [ ! -d "$OPENSSL_DIR" ]; then
    echo "[1/3] Downloading OpenSSL $OPENSSL_VERSION..."
    if [ ! -f "$TARBALL" ]; then
        echo "  Fetching $DOWNLOAD_URL..."
        curl -fSL -o "$TARBALL" "$DOWNLOAD_URL" || {
            echo "ERROR: Download failed. Try a different version with --version=N"
            exit 1
        }
    fi
    # Upstream publishes <tarball>.sha256 next to each release.
    expected="$(curl -fsSL "$DOWNLOAD_URL.sha256" | awk '{print $1}')" || expected=""
    if [ -n "$expected" ]; then
        actual="$(sha256sum "$TARBALL" | awk '{print $1}')"
        if [ "$expected" != "$actual" ]; then
            rm -f "$TARBALL"
            echo "ERROR: sha256 mismatch (expected $expected, got $actual)"
            exit 1
        fi
        ok "sha256 $actual"
    else
        warn "no upstream .sha256 fetched; tarball unverified"
    fi
    echo "  Extracting..."
    tar -xf "$TARBALL" -C "$VENDOR_DIR"
    mv "$VENDOR_DIR/openssl-${OPENSSL_VERSION}" "$OPENSSL_DIR"
    ok "Source extracted to $OPENSSL_DIR"
else
    echo "[1/3] OpenSSL source already at $OPENSSL_DIR"
fi

# A tree of another version is never reused silently: targets would build
# against code the caller did not ask for. Move it aside, never delete it.
have="$(awk -F= '$1=="MAJOR"||$1=="MINOR"||$1=="PATCH"{printf "%s%s", sep, $2; sep="."}' \
    "$OPENSSL_DIR/VERSION.dat" 2>/dev/null)"
if [ "$OPENSSL_VERSION" != "master" ] && [ "$have" != "$OPENSSL_VERSION" ]; then
    echo "ERROR: $OPENSSL_DIR holds OpenSSL ${have:-unknown}, wanted $OPENSSL_VERSION"
    echo "  mv $OPENSSL_DIR $OPENSSL_DIR-${have:-old}   # then re-run"
    exit 1
fi

# Verify source structure
if [ ! -f "$OPENSSL_DIR/Configure" ] || [ ! -f "$OPENSSL_DIR/crypto/x509/x_all.c" ]; then
    echo "ERROR: Invalid OpenSSL source tree at $OPENSSL_DIR"
    exit 1
fi

# ── Step 2: Configure check (out of tree, discarded) ─────────────
# The real build happens in build_targets.sh; this only proves Configure
# accepts the toolchain and option set it uses.
if [ "$CONFIG_CHECK" -eq 1 ]; then
    echo "[2/3] Configure-checking OpenSSL..."
    cc="clang"
    command -v clang &>/dev/null || {
        warn "clang not found, falling back to gcc (shallower edge coverage — see README)"
        cc="gcc"
    }
    tmpdir="$(mktemp -d)"
    # 4.1.0 out-of-tree + no-apps writes apps/include/ without creating it.
    mkdir -p "$tmpdir/apps/include"
    if (cd "$tmpdir" && CC="$cc" perl "$OPENSSL_DIR/Configure" no-shared no-asm \
            no-tests no-apps no-docs no-module no-dso no-threads >/dev/null 2>&1); then
        ok "Configure accepts $cc"
    else
        warn "Configure failed with $cc"
    fi
    rm -rf "$tmpdir"
else
    echo "[2/3] Skipping Configure check (--fast)"
fi

# ── Step 3: Verify ──────────────────────────────────────────────
echo "[3/3] Verifying..."
MISSING=0
for f in Configure LICENSE.txt include/openssl/x509.h.in include/openssl/pkcs7.h.in \
         crypto/asn1/tasn_dec.c crypto/x509/x_all.c; do
    if [ -f "$OPENSSL_DIR/$f" ]; then
        ok "$f"
    else
        warn "MISSING: $OPENSSL_DIR/$f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some source files missing"
    exit 1
fi

echo "=== OpenSSL vendored successfully ==="
echo "Source: $OPENSSL_DIR/"
echo ""
echo "Build the fuzz target with:"
echo "  tools/build_targets.sh"
