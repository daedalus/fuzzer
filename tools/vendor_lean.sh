#!/bin/bash
# Download the Lean 4 toolchain for tools/tla_extract.py --backend lean.
# Lands at $FUZZ_VENDOR_ROOT/lean/<version>/ (bin/lean, bin/lake, bin/leanc).
#
# Usage:
#   tools/vendor_lean.sh                     # Download & verify
#   tools/vendor_lean.sh --version=4.36.0    # Specific release (default: 4.36.0)
#
# Requirements: curl, unzip

set -e

# VENDOR_ROOT: see vendor_ffmpeg.sh for the layout rationale.
#   default: ~/fuzzing/vendoring/   (canonical home)
#   override: FUZZ_VENDOR_ROOT=...
#   legacy:   --in-tree-vendor restores the in-tree `vendor/` location.
: "${FUZZ_VENDOR_ROOT:=$HOME/fuzzing/vendoring}"
VENDOR_DIR="${IN_TREE_VENDOR:+$(cd "$(dirname "$0")/.." && pwd)/vendor}"
VENDOR_DIR="${VENDOR_ROOT:-$VENDOR_DIR}"
VENDOR_DIR="${VENDOR_DIR:-$FUZZ_VENDOR_ROOT}"
for arg in "$@"; do
    [ "$arg" = "--in-tree-vendor" ] && VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor"
done

LEAN_VERSION="${LEAN_VERSION:-4.36.0}"
for arg in "$@"; do
    case "$arg" in
        --version=*) LEAN_VERSION="${arg#*=}" ;;
    esac
done

case "$(uname -m)" in
    aarch64|arm64) ARCH_SUFFIX="linux_aarch64" ;;
    *) ARCH_SUFFIX="linux" ;;
esac

LEAN_DIR="$VENDOR_DIR/lean"
DEST="$LEAN_DIR/$LEAN_VERSION"
ARCHIVE="lean-${LEAN_VERSION}-${ARCH_SUFFIX}"
# The .zip asset: unzip is everywhere, zstd is not.
DOWNLOAD_URL="https://releases.lean-lang.org/lean4/v${LEAN_VERSION}/${ARCHIVE}.zip"

GREEN='\033[0;32m'
NC='\033[0m'
ok() { echo -e "  ${GREEN}OK${NC}: $1"; }

mkdir -p "$LEAN_DIR"

# ── Step 1: Download ─────────────────────────────────────────────
if [ ! -x "$DEST/bin/lean" ]; then
    echo "[1/2] Downloading Lean $LEAN_VERSION..."
    tmp="$(mktemp -d "$LEAN_DIR/lean.XXXXXX")"
    curl -fSL -o "$tmp/lean.zip" "$DOWNLOAD_URL" || {
        rm -rf "$tmp"
        echo "ERROR: Download failed. Try a different version with --version=N"
        exit 1
    }
    unzip -q "$tmp/lean.zip" -d "$tmp"
    rm -rf "$DEST"
    mv "$tmp/$ARCHIVE" "$DEST"
    rm -rf "$tmp"
    ok "$DEST"
else
    echo "[1/2] Lean already at $DEST"
fi

# ── Step 2: Verify ──────────────────────────────────────────────
echo "[2/2] Verifying..."
"$DEST/bin/lean" --version | grep -q "$LEAN_VERSION" || {
    echo "ERROR: $DEST/bin/lean does not report version $LEAN_VERSION"
    exit 1
}
ok "$("$DEST/bin/lean" --version)"

echo "=== Lean vendored successfully ==="
echo "Check the forkserver model with:"
echo "  python tools/tla_extract.py forkserver --check --backend lean"
