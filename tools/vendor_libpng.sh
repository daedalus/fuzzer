#!/bin/bash
# Download libpng + zlib sources for fuzz target building.
# Extracts to vendor/libpng/ and vendor/zlib/ — the build script will
# compile them as static libraries and link them into png_read.so.
#
# Usage:
#   tools/vendor_libpng.sh                 # Download & verify
#   tools/vendor_libpng.sh --fast          # Skip build-check
#   tools/vendor_libpng.sh --zlib-version=1.3.1  # Specific zlib version (default: 1.3.1)
#   tools/vendor_libpng.sh --png-version=1.6.48   # Specific libpng version (default: 1.6.48)
#
# Requirements: curl, tar, clang (or gcc)
#
# After vendoring, build the png fuzz target with:
#   tools/build_targets.sh

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

ZLIB_VERSION="${ZLIB_VERSION:-1.3.1}"
LIBPNG_VERSION="${LIBPNG_VERSION:-1.6.48}"
ZLIB_DIR="$VENDOR_DIR/zlib"
LIBPNG_DIR="$VENDOR_DIR/libpng"
ZLIB_TARBALL="/tmp/zlib-${ZLIB_VERSION}.tar.gz"
LIBPNG_TARBALL="/tmp/libpng-${LIBPNG_VERSION}.tar.gz"
BUILD_CHECK=1

# Parse flags
for arg in "$@"; do
    [ "$arg" = "--fast" ] && BUILD_CHECK=0
    case "$arg" in
        --zlib-version=*)
            ZLIB_VERSION="${arg#*=}"
            ZLIB_TARBALL="/tmp/zlib-${ZLIB_VERSION}.tar.gz"
            ;;
        --png-version=*)
            LIBPNG_VERSION="${arg#*=}"
            LIBPNG_TARBALL="/tmp/libpng-${LIBPNG_VERSION}.tar.gz"
            ;;
    esac
done

ZLIB_URL="https://github.com/madler/zlib/archive/refs/tags/v${ZLIB_VERSION}.tar.gz"
LIBPNG_URL="https://download.sourceforge.net/libpng/libpng16/${LIBPNG_VERSION}/libpng-${LIBPNG_VERSION}.tar.gz"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
ok() { echo -e "  ${GREEN}OK${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

mkdir -p "$VENDOR_DIR"

# ── Step 1: Download and extract ─────────────────────────────────
echo "[1/3] Vendoring zlib ${ZLIB_VERSION} and libpng ${LIBPNG_VERSION}..."

# zlib
if [ ! -d "$ZLIB_DIR" ]; then
    echo "  Fetching zlib..."
    if [ ! -f "$ZLIB_TARBALL" ]; then
        curl -fSL -o "$ZLIB_TARBALL" "$ZLIB_URL" || {
            echo "ERROR: zlib download failed. Try a different version with --zlib-version=N"
            exit 1
        }
    fi
    echo "  Extracting zlib..."
    tar -xf "$ZLIB_TARBALL" -C "$VENDOR_DIR"
    mv "$VENDOR_DIR/zlib-${ZLIB_VERSION}" "$ZLIB_DIR"
    ok "zlib extracted to $ZLIB_DIR"
else
    echo "  zlib already at $ZLIB_DIR"
fi

# libpng
if [ ! -d "$LIBPNG_DIR" ]; then
    echo "  Fetching libpng..."
    if [ ! -f "$LIBPNG_TARBALL" ]; then
        curl -fSL -o "$LIBPNG_TARBALL" "$LIBPNG_URL" || {
            echo "ERROR: libpng download failed. Try a different version with --png-version=N"
            exit 1
        }
    fi
    echo "  Extracting libpng..."
    tar -xf "$LIBPNG_TARBALL" -C "$VENDOR_DIR"
    # SourceForge tarballs extract as libpng-<version>
    extracted="$VENDOR_DIR/libpng-${LIBPNG_VERSION}"
    if [ ! -d "$extracted" ]; then
        # Fallback: auto-detect from tarball contents
        extracted="$VENDOR_DIR/$(tar -tzf "$LIBPNG_TARBALL" | head -1 | cut -f1 -d"/")"
    fi
    mv "$extracted" "$LIBPNG_DIR"
    ok "libpng extracted to $LIBPNG_DIR"
else
    echo "  libpng already at $LIBPNG_DIR"
fi

# Verify source structure
if [ ! -f "$ZLIB_DIR/zlib.h" ] || [ ! -f "$ZLIB_DIR/inflate.c" ]; then
    echo "ERROR: Invalid zlib source tree at $ZLIB_DIR"
    exit 1
fi
if [ ! -f "$LIBPNG_DIR/png.h" ] || [ ! -f "$LIBPNG_DIR/pngget.c" ]; then
    echo "ERROR: Invalid libpng source tree at $LIBPNG_DIR"
    exit 1
fi

# ── Step 2: Build-check the vendored libraries ────────────────────
# Compile zlib and libpng with coverage/comparison instrumentation baked
# into the static archives. The target wrapper provides the callback
# definitions via afl_shim.c / cmplog_shim.c, so the libraries only need
# to emit the __sanitizer_cov_trace_* calls.
if [ "$BUILD_CHECK" -eq 1 ]; then
    echo "[2/3] Build-checking vendored libraries..."
    cc="clang"
    command -v clang &>/dev/null || {
        warn "clang not found, falling back to gcc"
        cc="gcc"
    }
    tmpdir="$(mktemp -d)"

    # Coverage + cmplog-friendly flags for the static archives.
    # -fsanitize-coverage=trace-pc-guard: edge coverage
    # -fno-builtin-*: keep comparison calls at the PLT so cmplog can intercept
    #   them when the cmplog shim is linked (e.g., .so targets).
    # -fno-omit-frame-pointer: afl_shim.c's caller-context edge hashing
    # (-D__AFL_CTX_SENSITIVE=1) walks __builtin_return_address(1) into THIS
    # library's frame, so the flag is required here, not only in the TU that
    # includes the shim. Without it the walk silently returns the wrong
    # caller (measured) instead of crashing. See FRAME_POINTER in
    # tools/build_targets.sh.
    VENDOR_CFLAGS="-O2 -g -fPIC -fno-omit-frame-pointer -fsanitize-coverage=trace-pc-guard"
    VENDOR_CFLAGS="$VENDOR_CFLAGS -fno-builtin-memcmp -fno-builtin-bcmp -fno-builtin-strcmp"
    VENDOR_CFLAGS="$VENDOR_CFLAGS -fno-builtin-strncmp -fno-builtin-strcasecmp"
    VENDOR_CFLAGS="$VENDOR_CFLAGS -fno-builtin-strncasecmp -fno-builtin-memchr -fno-builtin-strstr"
    VENDOR_CFLAGS="$VENDOR_CFLAGS -fno-builtin-memmem -fno-builtin-strcasestr"

    # zlib static
    (cd "$ZLIB_DIR" && CC=$cc CFLAGS="$VENDOR_CFLAGS" \
        ./configure --static >/dev/null 2>&1 && make -j"$(nproc)" -s >/dev/null 2>&1) || \
        { warn "zlib static build failed"; rm -rf "$tmpdir"; exit 1; }
    ok "zlib static library built ($cc)"

    # libpng static against vendored zlib
    (cd "$LIBPNG_DIR" && CC=$cc \
        CPPFLAGS="-I../zlib" \
        CFLAGS="$VENDOR_CFLAGS -I../zlib" \
        LDFLAGS="-L../zlib" \
        ./configure --with-pkgconfig=no --enable-shared=no >/dev/null 2>&1 && \
        make -j"$(nproc)" -s >/dev/null 2>&1) || \
        { warn "libpng static build failed"; rm -rf "$tmpdir"; exit 1; }
    ok "libpng static library built ($cc)"

    rm -rf "$tmpdir"
else
    echo "[2/3] Skipping build-check (--fast)"
fi

# ── Step 3: Verify ────────────────────────────────────────────────
echo "[3/3] Verifying..."
MISSING=0
for f in zlib.h inflate.c deflate.c crc32.c zconf.h; do
    [ -f "$ZLIB_DIR/$f" ] && ok "zlib/$f" || { warn "MISSING: zlib/$f"; MISSING=1; }
done
for f in png.h pngget.c pngset.c pngread.c pngrtran.c pngrutil.c pngwio.c pngwutil.c png.c; do
    [ -f "$LIBPNG_DIR/$f" ] && ok "libpng/$f" || { warn "MISSING: libpng/$f"; MISSING=1; }
done

# Check that the static archives were produced
[ -f "$ZLIB_DIR/libz.a" ] && ok "zlib/libz.a" || { warn "MISSING: zlib/libz.a"; MISSING=1; }
[ -f "$LIBPNG_DIR/.libs/libpng16.a" ] && ok "libpng/.libs/libpng16.a" || { warn "MISSING: libpng/.libs/libpng16.a"; MISSING=1; }

if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some source files or build artifacts missing"
    exit 1
fi

echo "=== libpng + zlib vendored successfully ==="
echo "Sources: $ZLIB_DIR/ and $LIBPNG_DIR/"
echo ""
echo "Build the png fuzz target with:"
echo "  tools/build_targets.sh"
