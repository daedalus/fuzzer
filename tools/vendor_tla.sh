#!/bin/bash
# Download tla2tools.jar (TLC model checker) for tools/tla_extract.py --check.
# Lands at $FUZZ_VENDOR_ROOT/tla/tla2tools.jar; needs java on PATH to run.
#
# Usage:
#   tools/vendor_tla.sh                    # Download & verify
#   tools/vendor_tla.sh --version=1.8.0    # Specific release (default: 1.8.0)
#
# Requirements: curl, java

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

TLA_VERSION="${TLA_VERSION:-1.8.0}"
for arg in "$@"; do
    case "$arg" in
        --version=*) TLA_VERSION="${arg#*=}" ;;
    esac
done

TLA_DIR="$VENDOR_DIR/tla"
JAR="$TLA_DIR/tla2tools.jar"
DOWNLOAD_URL="https://github.com/tlaplus/tlaplus/releases/download/v${TLA_VERSION}/tla2tools.jar"

GREEN='\033[0;32m'
NC='\033[0m'
ok() { echo -e "  ${GREEN}OK${NC}: $1"; }

mkdir -p "$TLA_DIR"

# ── Step 1: Download ─────────────────────────────────────────────
if [ ! -f "$JAR" ]; then
    echo "[1/2] Downloading tla2tools.jar $TLA_VERSION..."
    tmp="$(mktemp "$TLA_DIR/tla2tools.XXXXXX")"
    curl -fSL -o "$tmp" "$DOWNLOAD_URL" || {
        rm -f "$tmp"
        echo "ERROR: Download failed. Try a different version with --version=N"
        exit 1
    }
    mv "$tmp" "$JAR"
    ok "$JAR"
else
    echo "[1/2] tla2tools.jar already at $JAR"
fi

# ── Step 2: Verify ──────────────────────────────────────────────
echo "[2/2] Verifying..."
command -v java &>/dev/null || { echo "ERROR: java not on PATH"; exit 1; }
# SANY (the parser TLC uses) on the hand-written I/O model: proves the jar runs and the model parses.
java -cp "$JAR" tla2sany.SANY "$(dirname "$0")/tla/Syscalls.tla" &>/dev/null || {
    echo "ERROR: $JAR cannot parse tools/tla/Syscalls.tla"
    exit 1
}
ok "SANY parses Syscalls.tla"

echo "=== TLC vendored successfully ==="
echo "Check the forkserver model with:"
echo "  python tools/tla_extract.py forkserver --check"
