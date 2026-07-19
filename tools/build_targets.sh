#!/bin/bash
# Build all fuzz targets with AFL edge coverage.
# Compiles both ASAN and no-ASAN variants.
#
# Usage:
#   tools/build_targets.sh          # Build all targets
#   tools/build_targets.sh --asan   # ASAN only
#   tools/build_targets.sh --fast   # No-ASAN only

set -e

FGREP="/home/dclavijo/my_code/fgrep"
SHIM="src/fuzzer_tool/adapters/afl_shim.c"
TARGETS="targets"
OPTS="${@:---all}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok() { echo -e "  ${GREEN}OK${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

# ── Compile fgrep library objects ──────────────────────────────────
compile_fgrep_objects() {
    local suffix="$1" flags="$2"
    echo "Compiling fgrep objects${suffix:+ ($suffix)}..."
    for src in regex_engine simd cpu; do
        gcc $flags -fPIC -O2 -g -I"$FGREP/include" -I"$FGREP/src" \
            -c "$FGREP/src/${src}.c" -o "/tmp/${src}${suffix}.o"
    done
    for src in output search bmh_simd io fileutil; do
        gcc $flags -fPIC -O2 -g -mavx2 -I"$FGREP/include" -I"$FGREP/src" \
            -c "$FGREP/src/${src}.c" -o "/tmp/${src}${suffix}.o"
    done
    ok "fgrep objects${suffix:+ ($suffix)}"
}

# ── Build a target ────────────────────────────────────────────────
build_target() {
    local src="$1" out="$2" libs="$3" extra_flags="$4"
    if [ ! -f "$src" ]; then
        warn "Source not found: $src"
        return 1
    fi
    gcc $extra_flags -O2 -g -include "$SHIM" \
        -o "$out" "$src" $libs 2>/dev/null
    ok "$(basename "$out")"
}

# ── Build a .so target ───────────────────────────────────────────
build_so_target() {
    local src="$1" out="$2" libs="$3" extra_flags="$4"
    if [ ! -f "$src" ]; then
        warn "Source not found: $src"
        return 1
    fi
    gcc $extra_flags -O2 -g -shared -fPIC -include "$SHIM" \
        -o "$out" "$src" $libs 2>/dev/null
    ok "$(basename "$out")"
}

# ── Build fgrep targets ──────────────────────────────────────────
build_fgrep_targets() {
    local suffix="$1" flags="$2" label="$3"
    echo "Building fgrep targets ($label)..."
    local FGREP_INC="-I$FGREP/include -I$FGREP/src"
    local FGREP_LIBS="/tmp/regex_engine${suffix}.o /tmp/simd${suffix}.o /tmp/cpu${suffix}.o"
    local FGREP_LIBS_FULL="$FGREP_LIBS /tmp/output${suffix}.o /tmp/search${suffix}.o /tmp/bmh_simd${suffix}.o /tmp/io${suffix}.o /tmp/fileutil${suffix}.o -lpthread"

    local out_suffix=""
    [ "$suffix" = "_nosan" ] && out_suffix="_nosan"
    build_target "$TARGETS/fuzz_regex_compile.c" "$TARGETS/fuzz_regex_compile${out_suffix}" "$FGREP_INC $FGREP_LIBS" "$flags"
    build_target "$TARGETS/fuzz_pattern_match.c" "$TARGETS/fuzz_pattern_match${out_suffix}" "$FGREP_INC $FGREP_LIBS" "$flags"
    build_target "$TARGETS/fuzz_search_pipeline.c" "$TARGETS/fuzz_search_pipeline${out_suffix}" "$FGREP_INC $FGREP_LIBS_FULL" "$flags"
}

# ── Build fgrep .so targets ─────────────────────────────────────
build_fgrep_so_targets() {
    local suffix="$1" flags="$2" label="$3"
    echo "Building fgrep .so targets ($label)..."
    local FGREP_INC="-I$FGREP/include -I$FGREP/src"
    local FGREP_LIBS="/tmp/regex_engine${suffix}.o /tmp/simd${suffix}.o /tmp/cpu${suffix}.o"
    local FGREP_LIBS_FULL="$FGREP_LIBS /tmp/output${suffix}.o /tmp/search${suffix}.o /tmp/bmh_simd${suffix}.o /tmp/io${suffix}.o /tmp/fileutil${suffix}.o -lpthread"

    local out_suffix=""
    [ "$suffix" = "_nosan" ] && out_suffix="_nosan"
    build_so_target "$TARGETS/fuzz_regex_compile.c" "$TARGETS/fuzz_regex_compile${out_suffix}.so" "$FGREP_INC $FGREP_LIBS" "$flags"
    build_so_target "$TARGETS/fuzz_pattern_match.c" "$TARGETS/fuzz_pattern_match${out_suffix}.so" "$FGREP_INC $FGREP_LIBS" "$flags"
    build_so_target "$TARGETS/fuzz_search_pipeline.c" "$TARGETS/fuzz_search_pipeline${out_suffix}.so" "$FGREP_INC $FGREP_LIBS_FULL" "$flags"
}

# ── Build simple targets ─────────────────────────────────────────
build_simple_targets() {
    local suffix="$1" flags="$2" label="$3"
    echo "Building simple targets ($label)..."
    local out_suffix=""
    [ "$suffix" = "_nosan" ] && out_suffix="_nosan"
    build_target "$TARGETS/asan_target.c" "$TARGETS/asan_target${out_suffix}" "" "$flags"
    build_target "$TARGETS/test_target.c" "$TARGETS/test_target${out_suffix}" "" "$flags"
    build_target "$TARGETS/proto_target.c" "$TARGETS/proto_target${out_suffix}" "" "$flags"
    build_target "$TARGETS/png_read.c" "$TARGETS/png_read${out_suffix}" "-lpng -lz" "$flags"
    build_target "$TARGETS/zlib_read.c" "$TARGETS/zlib_read${out_suffix}" "-lz" "$flags"
    build_target "$TARGETS/gzip_read.c" "$TARGETS/gzip_read${out_suffix}" "-lz" "$flags"
    build_target "$TARGETS/jpeg_read.c" "$TARGETS/jpeg_read${out_suffix}" "-ljpeg" "$flags"
}

# ── Build simple .so targets ────────────────────────────────────
build_simple_so_targets() {
    local suffix="$1" flags="$2" label="$3"
    echo "Building simple .so targets ($label)..."
    local out_suffix=""
    [ "$suffix" = "_nosan" ] && out_suffix="_nosan"
    build_so_target "$TARGETS/asan_target.c" "$TARGETS/asan_target${out_suffix}.so" "" "$flags"
    build_so_target "$TARGETS/test_target.c" "$TARGETS/test_target${out_suffix}.so" "" "$flags"
    build_so_target "$TARGETS/proto_target.c" "$TARGETS/proto_target${out_suffix}.so" "" "$flags"
    build_so_target "$TARGETS/png_read.c" "$TARGETS/png_read${out_suffix}.so" "-lpng -lz" "$flags"
    build_so_target "$TARGETS/zlib_read.c" "$TARGETS/zlib_read${out_suffix}.so" "-lz" "$flags"
    build_so_target "$TARGETS/gzip_read.c" "$TARGETS/gzip_read${out_suffix}.so" "-lz" "$flags"
    build_so_target "$TARGETS/jpeg_read.c" "$TARGETS/jpeg_read${out_suffix}.so" "-ljpeg" "$flags"
}

# ── Verify AFL symbols ───────────────────────────────────────────
verify_afl() {
    echo "Verifying AFL symbols..."
    local count=0
    for f in "$TARGETS"/fuzz_* "$TARGETS"/asan_target "$TARGETS"/asan_target_nosan "$TARGETS"/asan_target.so "$TARGETS"/asan_target_nosan.so \
             "$TARGETS"/png_read "$TARGETS"/png_read_nosan "$TARGETS"/png_read.so "$TARGETS"/png_read_nosan.so \
             "$TARGETS"/zlib_read "$TARGETS"/zlib_read_nosan "$TARGETS"/zlib_read.so "$TARGETS"/zlib_read_nosan.so \
             "$TARGETS"/gzip_read "$TARGETS"/gzip_read_nosan "$TARGETS"/gzip_read.so "$TARGETS"/gzip_read_nosan.so \
             "$TARGETS"/jpeg_read "$TARGETS"/jpeg_read_nosan "$TARGETS"/jpeg_read.so "$TARGETS"/jpeg_read_nosan.so \
             "$TARGETS"/test_target "$TARGETS"/test_target_nosan "$TARGETS"/test_target.so "$TARGETS"/test_target_nosan.so \
             "$TARGETS"/proto_target "$TARGETS"/proto_target_nosan "$TARGETS"/proto_target.so "$TARGETS"/proto_target_nosan.so; do
        [ -f "$f" ] || continue
        [[ "$f" == *.c ]] && continue
        local n=$(nm "$f" 2>/dev/null | grep -c __afl || true)
        if [ "$n" -gt 0 ]; then
            count=$((count + 1))
        else
            warn "$(basename "$f"): no AFL symbols"
        fi
    done
    ok "$count targets with AFL symbols"
}

# ── Verify fuzz_shm_run in .so targets ──────────────────────────
verify_shm_run() {
    echo "Verifying fuzz_shm_run in .so targets..."
    local ok_count=0
    local fail_count=0
    for f in "$TARGETS"/*.so; do
        [ -f "$f" ] || continue
        if nm "$f" 2>/dev/null | grep -q "fuzz_shm_run"; then
            ok_count=$((ok_count + 1))
        else
            warn "$(basename "$f"): missing fuzz_shm_run"
            fail_count=$((fail_count + 1))
        fi
    done
    ok "$ok_count .so targets with fuzz_shm_run"
    [ "$fail_count" -gt 0 ] && warn "$fail_count .so targets missing fuzz_shm_run"
}

# ── Main ──────────────────────────────────────────────────────────
echo "=== Building fuzz targets ==="

case "$OPTS" in
    --asan)
        compile_fgrep_objects "_asan" "-fsanitize=address"
        build_fgrep_targets "_asan" "-fsanitize=address" "ASAN"
        build_simple_targets "_asan" "-fsanitize=address" "ASAN"
        build_fgrep_so_targets "_asan" "-fsanitize=address" "ASAN"
        build_simple_so_targets "_asan" "-fsanitize=address" "ASAN"
        ;;
    --fast|--nosan)
        compile_fgrep_objects "_nosan" ""
        build_fgrep_targets "_nosan" "" "No-ASAN"
        build_simple_targets "_nosan" "" "No-ASAN"
        build_fgrep_so_targets "_nosan" "" "No-ASAN"
        build_simple_so_targets "_nosan" "" "No-ASAN"
        ;;
    *)
        compile_fgrep_objects "_asan" "-fsanitize=address"
        compile_fgrep_objects "_nosan" ""
        build_fgrep_targets "_asan" "-fsanitize=address" "ASAN"
        build_fgrep_targets "_nosan" "" "No-ASAN"
        build_simple_targets "_asan" "-fsanitize=address" "ASAN"
        build_simple_targets "_nosan" "" "No-ASAN"
        build_fgrep_so_targets "_asan" "-fsanitize=address" "ASAN"
        build_fgrep_so_targets "_nosan" "" "No-ASAN"
        build_simple_so_targets "_asan" "-fsanitize=address" "ASAN"
        build_simple_so_targets "_nosan" "" "No-ASAN"
        ;;
esac

verify_afl
verify_shm_run
echo "=== Done ==="
