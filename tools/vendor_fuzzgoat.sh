#!/bin/bash
# Download and configure Fuzzgoat source for fuzz target building.
# Extracts to vendor/fuzzgoat/.
#
# Usage:
#   tools/vendor_fuzzgoat.sh              # Clone/download
#   tools/vendor_fuzzgoat.sh --fast       # Skip if already present
#
# Requirements: git
#
# After vendoring, build the fuzz target with:
#   tools/build_targets.sh
# or directly:
#   gcc -O2 -g -Ivendor/fuzzgoat -include src/fuzzer_tool/adapters/afl_shim.c \
#       -o targets/fuzzgoat_read targets/fuzzgoat_read.c \
#       vendor/fuzzgoat/fuzzgoat.c -lm

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

FUZZGOAT_DIR="$VENDOR_DIR/fuzzgoat"
REPO_URL="https://github.com/fuzzstati0n/fuzzgoat.git"

# Parse flags
SKIP_CLONE=0
for arg in "$@"; do
    [ "$arg" = "--fast" ] && SKIP_CLONE=1
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
ok() { echo -e "  ${GREEN}OK${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

mkdir -p "$VENDOR_DIR"

# ── Step 1: Clone repository ──────────────────────────────────────
if [ "$SKIP_CLONE" -eq 1 ] && [ -d "$FUZZGOAT_DIR" ]; then
    echo "[1/2] Fuzzgoat source already at $FUZZGOAT_DIR"
else
    echo "[1/2] Cloning Fuzzgoat..."
    if [ -d "$FUZZGOAT_DIR" ]; then
        rm -rf "$FUZZGOAT_DIR"
    fi
    git clone --depth 1 "$REPO_URL" "$FUZZGOAT_DIR" 2>&1 | tail -5
    ok "Source cloned to $FUZZGOAT_DIR"
fi

# Verify source structure
if [ ! -f "$FUZZGOAT_DIR/fuzzgoat.c" ] || [ ! -f "$FUZZGOAT_DIR/fuzzgoat.h" ]; then
    echo "ERROR: Invalid fuzzgoat source tree at $FUZZGOAT_DIR"
    exit 1
fi

# ── Step 2: Verify ────────────────────────────────────────────────
echo "[2/2] Verifying..."
for f in fuzzgoat.c fuzzgoat.h main.c; do
    if [ -f "$FUZZGOAT_DIR/$f" ]; then
        ok "$f"
    else
        warn "MISSING: $FUZZGOAT_DIR/$f"
    fi
done

echo "=== Fuzzgoat vendored successfully ==="
echo "Source: $FUZZGOAT_DIR/"
echo ""
echo "Build the fuzz target with:"
echo "  tools/build_targets.sh"
