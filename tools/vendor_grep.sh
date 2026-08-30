#!/bin/bash
# Download, configure and build GNU Grep source for fuzz target building.
# Extracts to vendor/grep/, runs configure to generate config.h, and builds
# lib/libgreputils.a.
#
# Usage:
#   tools/vendor_grep.sh              # Download, configure & build
#   tools/vendor_grep.sh --fast       # Skip configure and build (source only)
#   tools/vendor_grep.sh --version N  # Specific version (default: 3.11)
#
# Requirements: curl, tar, gcc, make (for configure)
#
# The archive is built with -fPIC because targets/grep_read.so links it into
# a shared object; without it the link fails with "relocation R_X86_64_PC32
# ... can not be used when making a shared object". -fno-omit-frame-pointer
# matches the convention for every other vendored library here.
#
# The archive is deliberately NOT instrumented. tools/build_targets.sh
# recompiles the three files the harness actually exercises (lib/dfa.c,
# lib/localeinfo.c, src/kwset.c) with -fsanitize-coverage=trace-pc-guard and
# links those objects ahead of the archive, so the matcher engines are
# covered while the gnulib support code is not. Instrumenting all of gnulib
# would spend bitmap on code no pattern reaches.
#
# After vendoring, build the grep fuzz target with:
#   tools/build_targets.sh

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

# ── Step 2: Configure and build the support library ──────────────
GREP_CFLAGS="-O2 -g -fPIC -fno-omit-frame-pointer"

if [ "$BUILD_CONFIGURE" -eq 1 ]; then
    if [ ! -f "$GREP_DIR/config.h" ]; then
        echo "[2/4] Configuring GNU Grep..."
        (cd "$GREP_DIR" && ./configure --quiet CFLAGS="$GREP_CFLAGS" 2>&1 | tail -5) || {
            echo "ERROR: configure failed"
            echo "  Try ./configure manually in $GREP_DIR"
            exit 1
        }
        if [ -f "$GREP_DIR/config.h" ]; then
            ok "config.h generated"
        else
            warn "config.h not found after configure"
        fi
    else
        echo "[2/4] config.h already exists"
    fi

    # libgreputils.a supplies the gnulib support symbols that dfa.c and
    # kwset.c call into (xalloc, obstack, mbrtowc wrappers, ...). Built
    # here rather than in build_targets.sh because it is a one-off that
    # takes minutes, while build_targets.sh reruns often.
    echo "[3/4] Building lib/libgreputils.a (-fPIC)..."
    if (cd "$GREP_DIR" && make -C lib CFLAGS="$GREP_CFLAGS" >/dev/null 2>&1); then
        ok "libgreputils.a built"
    else
        warn "make -C lib failed — targets/grep_read will be skipped by build_targets.sh"
    fi
else
    echo "[2/4] Skipping configure (--fast)"
    echo "[3/4] Skipping library build (--fast)"
fi

# ── Step 4: Verify ──────────────────────────────────────────────
echo "[4/4] Verifying..."
MISSING=0
# lib/dfa.c, lib/localeinfo.c and src/kwset.c are the files build_targets.sh
# recompiles with instrumentation; lib/libgreputils.a supplies the rest.
for f in src/grep.c src/kwset.c src/kwset.h src/kwsearch.c src/dfasearch.c \
         src/searchutils.c src/search.h src/grep.h config.h \
         lib/dfa.c lib/dfa.h lib/localeinfo.c lib/localeinfo.h; do
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

if [ -f "$GREP_DIR/lib/libgreputils.a" ]; then
    ok "lib/libgreputils.a"
elif [ "$BUILD_CONFIGURE" -eq 1 ]; then
    warn "MISSING: $GREP_DIR/lib/libgreputils.a"
    MISSING=1
fi

if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some source files missing"
    exit 1
fi

echo "=== GNU Grep vendored successfully ==="
echo "Source: $GREP_DIR/"
echo "Config: $GREP_DIR/config.h"
echo "Library: $GREP_DIR/lib/libgreputils.a"
echo ""
echo "Build the fuzz target with:"
echo "  tools/build_targets.sh"
