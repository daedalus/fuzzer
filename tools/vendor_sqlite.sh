#!/bin/bash
# Download the SQLite amalgamation for fuzz target building.
# Extracts to vendor/sqlite/ — no configure step needed: the amalgamation is
# a single pre-generated translation unit (sqlite3.c + sqlite3.h), which is
# exactly why it is the distribution SQLite recommends for embedding.
#
# Usage:
#   tools/vendor_sqlite.sh                  # Download & verify
#   tools/vendor_sqlite.sh --fast           # Skip the compile-check
#   tools/vendor_sqlite.sh --version=3.53.4 # Specific version (default: 3.53.4)
#   tools/vendor_sqlite.sh --url=URL        # Explicit amalgamation zip URL
#
# Requirements: curl, unzip, clang (or gcc)
#
# After vendoring, build the sqlite fuzz target with:
#   tools/build_targets.sh
#
# The amalgamation is compiled WITHOUT the AFL shim and linked into the
# target wrapper, which is the only translation unit that gets
# `-include afl_shim.c` (Hard Rule 8). Passing -include to sqlite3.c too
# would emit __afl_map_shm/__afl_area/__afl_guarded_call into both objects
# and the final link fails with multiple-definition errors.
#
# On the download URL: sqlite.org files the amalgamation under the *release
# year*, and the release year is not derivable from the version number, so
# this walks the plausible years newest-first rather than hardcoding one and
# breaking on the next release. The filename encodes the version as
# 3XXYY00 (3.53.4 -> 3530400). Behind a proxy that blocks sqlite.org, pass
# --url= with a mirror; the verify step below is what actually decides
# whether the tree is usable.

set -e

VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor"
SQLITE_VERSION="${SQLITE_VERSION:-3.53.4}"
SQLITE_DIR="$VENDOR_DIR/sqlite"
DOWNLOAD_URL="${SQLITE_URL:-}"
BUILD_CHECK=1

# Parse flags
for arg in "$@"; do
    [ "$arg" = "--fast" ] && BUILD_CHECK=0
    case "$arg" in
        --version=*) SQLITE_VERSION="${arg#*=}" ;;
        --url=*) DOWNLOAD_URL="${arg#*=}" ;;
    esac
done

# 3.53.4 -> 3530400
version_number() {
    local major minor patch
    major="${1%%.*}"
    minor="$(echo "$1" | cut -d. -f2)"
    patch="$(echo "$1" | cut -d. -f3)"
    [ -n "$patch" ] || patch=0
    printf '%d%02d%02d00\n' "$major" "$minor" "$patch"
}

SQLITE_NUM="$(version_number "$SQLITE_VERSION")"
ZIPNAME="sqlite-amalgamation-${SQLITE_NUM}.zip"
ZIPFILE="/tmp/${ZIPNAME}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
ok() { echo -e "  ${GREEN}OK${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

command -v unzip &>/dev/null || {
    echo "ERROR: unzip is required (the amalgamation ships as a .zip)"
    exit 1
}

mkdir -p "$VENDOR_DIR"

# ── Step 1: Download and extract ─────────────────────────────────
if [ ! -d "$SQLITE_DIR" ]; then
    echo "[1/3] Downloading SQLite $SQLITE_VERSION ($ZIPNAME)..."
    if [ ! -f "$ZIPFILE" ]; then
        if [ -n "$DOWNLOAD_URL" ]; then
            echo "  Fetching $DOWNLOAD_URL..."
            curl -fSL --connect-timeout 20 -o "$ZIPFILE" "$DOWNLOAD_URL" || {
                echo "ERROR: Download failed from $DOWNLOAD_URL"
                exit 1
            }
        else
            for year in $(seq "$(date +%Y)" -1 2022); do
                url="https://sqlite.org/${year}/${ZIPNAME}"
                echo "  Trying $url..."
                if curl -fSL --connect-timeout 20 -o "$ZIPFILE" "$url" 2>/dev/null; then
                    DOWNLOAD_URL="$url"
                    break
                fi
            done
            [ -n "$DOWNLOAD_URL" ] || {
                echo "ERROR: Could not find $ZIPNAME on sqlite.org."
                echo "       Try another version with --version=3.X.Y, or pass"
                echo "       an explicit --url=<amalgamation zip>."
                rm -f "$ZIPFILE"
                exit 1
            }
        fi
    fi
    echo "  Extracting..."
    rm -rf "$VENDOR_DIR/sqlite-amalgamation-${SQLITE_NUM}"
    unzip -q -o "$ZIPFILE" -d "$VENDOR_DIR"
    mv "$VENDOR_DIR/sqlite-amalgamation-${SQLITE_NUM}" "$SQLITE_DIR"
    ok "Source extracted to $SQLITE_DIR"
else
    echo "[1/3] SQLite source already at $SQLITE_DIR"
fi

# Verify source structure
if [ ! -f "$SQLITE_DIR/sqlite3.c" ] || [ ! -f "$SQLITE_DIR/sqlite3.h" ]; then
    echo "ERROR: Invalid SQLite amalgamation at $SQLITE_DIR"
    exit 1
fi

# ── Step 2: Compile-check the amalgamation ───────────────────────
# The amalgamation has no configure; the only thing worth checking is that
# it parses in this toolchain with the defines the target builds it with.
# -fsyntax-only rather than a real -O2 object: this is a 9 MB translation
# unit, an optimized compile is ~a minute, and build_targets.sh compiles it
# for real moments later anyway.
if [ "$BUILD_CHECK" -eq 1 ]; then
    echo "[2/3] Compile-checking the amalgamation..."
    cc="clang"
    command -v clang &>/dev/null || {
        warn "clang not found, falling back to gcc (shallower edge coverage — see README)"
        cc="gcc"
    }
    if $cc -fsyntax-only -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION=1 \
        -I"$SQLITE_DIR" "$SQLITE_DIR/sqlite3.c" 2>/dev/null; then
        ok "sqlite3.c parses ($cc)"
    else
        warn "sqlite3.c failed to compile with $cc"
    fi
else
    echo "[2/3] Skipping compile-check (--fast)"
fi

# ── Step 3: Verify ──────────────────────────────────────────────
echo "[3/3] Verifying..."
MISSING=0
for f in sqlite3.c sqlite3.h; do
    if [ -f "$SQLITE_DIR/$f" ]; then
        ok "$f"
    else
        warn "MISSING: $SQLITE_DIR/$f"
        MISSING=1
    fi
done
# sqlite3ext.h is only needed by loadable extensions, which the target
# builds with SQLITE_OMIT_LOAD_EXTENSION — absence is not fatal.
[ -f "$SQLITE_DIR/sqlite3ext.h" ] && ok "sqlite3ext.h" || warn "sqlite3ext.h absent (not required)"

if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some source files missing"
    exit 1
fi

VER_STRING="$(grep -m1 '#define SQLITE_VERSION ' "$SQLITE_DIR/sqlite3.h" | awk '{print $3}')"
echo "=== SQLite vendored successfully ==="
echo "Source:  $SQLITE_DIR/"
echo "Version: ${VER_STRING:-unknown}"
echo ""
echo "Build the fuzz target with:"
echo "  tools/build_targets.sh"
