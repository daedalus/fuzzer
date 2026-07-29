#!/bin/bash
# Download and configure GNU Grep source for fuzz target building.
# Extracts to vendor/grep/ and runs configure to generate config.h.
#
# Usage:
#   tools/vendor_grep.sh              # Download & configure
#   tools/vendor_grep.sh --fast       # Skip config (source only)
#   tools/vendor_grep.sh --version N  # Specific version (default: 3.11)
#
# Requirements: curl, tar, gcc, make (for configure)
#
# After vendoring, build the grep fuzz target with:
#   tools/build_targets.sh
# or directly:
#   gcc -O2 -g -include src/fuzzer_tool/adapters/afl_shim.c \
#       -o targets/grep_read targets/grep_read.c

set -e

VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor"
GREP_VERSION="${GREP_VERSION:-3.11}"
GREP_DIR="$VENDOR_DIR/grep"
DOWNLOAD_URL="https://ftp.gnu.org/gnu/grep/grep-${GREP_VERSION}.tar.xz"
TARBALL="/tmp/grep-${GREP_VERSION}.tar.xz"
BUILD_CONFIGURE=1

# Parse flags
for arg in "$@"; do
    [ "$arg" = "--fast" ] && BUILD_CONFIGURE=0
    case "$arg" in
        --version=*) GREP_VERSION="${arg#*=}"; DOWNLOAD_URL="https://ftp.gnu.org/gnu/grep/grep-${GREP_VERSION}.tar.xz"; TARBALL="/tmp/grep-${GREP_VERSION}.tar.xz" ;;
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
if [ ! -d "$GREP_DIR" ]; then
    echo "[1/3] Downloading GNU Grep $GREP_VERSION..."
    if [ ! -f "$TARBALL" ]; then
        echo "  Fetching $DOWNLOAD_URL..."
        curl -fSL -o "$TARBALL" "$DOWNLOAD_URL" || {
            echo "ERROR: Download failed. Try a different version with --version=N"
            exit 1
        }
    fi
    echo "  Extracting..."
    tar -xf "$TARBALL" -C "$VENDOR_DIR"
    mv "$VENDOR_DIR/grep-${GREP_VERSION}" "$GREP_DIR"
    ok "Source extracted to $GREP_DIR"
else
    echo "[1/3] GNU Grep source already at $GREP_DIR"
fi

# Verify source structure
if [ ! -f "$GREP_DIR/src/grep.c" ] || [ ! -f "$GREP_DIR/configure" ]; then
    echo "ERROR: Invalid grep source tree at $GREP_DIR"
    exit 1
fi

# ── Step 2: Configure ────────────────────────────────────────────
if [ "$BUILD_CONFIGURE" -eq 1 ] && [ ! -f "$GREP_DIR/config.h" ]; then
    echo "[2/3] Configuring GNU Grep..."
    (cd "$GREP_DIR" && ./configure --quiet 2>&1 | tail -5) || {
        echo "ERROR: configure failed"
        echo "  Try ./configure manually in $GREP_DIR"
        exit 1
    }
    if [ -f "$GREP_DIR/config.h" ]; then
        ok "config.h generated"
    else
        warn "config.h not found after configure"
    fi
elif [ "$BUILD_CONFIGURE" -eq 0 ]; then
    echo "[2/3] Skipping configure (--fast)"
elif [ -f "$GREP_DIR/config.h" ]; then
    echo "[2/3] config.h already exists"
fi

# ── Step 3: Verify ──────────────────────────────────────────────
echo "[3/3] Verifying..."
MISSING=0
for f in src/grep.c src/kwset.c src/kwset.h src/kwsearch.c src/dfasearch.c \
         src/searchutils.c src/search.h src/grep.h config.h; do
    if [ -f "$GREP_DIR/$f" ]; then
        ok "$f"
    else
        warn "MISSING: $GREP_DIR/$f"
        MISSING=1
    fi
done

# Check PCRE2 support (for -P mode)
if [ -f "$GREP_DIR/src/pcresearch.c" ]; then
    if grep -q "pcre2" "$GREP_DIR/config.h" 2>/dev/null; then
        ok "PCRE2 support enabled in config.h"
    else
        warn "PCRE2 may not be enabled — grep -P mode may fail on some patterns"
    fi
fi

if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some source files missing"
    exit 1
fi

echo "=== GNU Grep vendored successfully ==="
echo "Source: $GREP_DIR/"
echo "Config: $GREP_DIR/config.h"
echo ""
echo "Build the fuzz target with:"
echo "  gcc -O2 -g -include src/fuzzer_tool/adapters/afl_shim.c \\"
echo "      -o targets/grep_read targets/grep_read.c"
