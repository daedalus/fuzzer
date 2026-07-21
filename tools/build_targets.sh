#!/bin/bash
# Build all fuzz targets with AFL edge coverage.
# Compiles both ASAN and no-ASAN variants.
#
# Usage:
#   tools/build_targets.sh                # Build all targets
#   tools/build_targets.sh --asan         # ASAN only
#   tools/build_targets.sh --fast         # No-ASAN only
#   tools/build_targets.sh --cmplog       # Include cmplog in .so targets (build-time linking)
#   tools/build_targets.sh --asan --cmplog  # ASAN + cmplog in .so targets

set -e

FGREP="${FGREP_DIR:-/home/dclavijo/my_code/fgrep}"
TAILSLAYER="${TAILSLAYER_DIR:-/home/dclavijo/code/tailslayer}"
SHIM="src/fuzzer_tool/adapters/afl_shim.c"
CMPLOG_SHIM="src/fuzzer_tool/adapters/cmplog_shim.c"
TARGETS="targets"
OPTS="${@:---all}"
HAS_FGREP=0
[ -d "$FGREP/src" ] && HAS_FGREP=1
WITH_CMPLOG=0

# Parse --cmplog flag (can appear anywhere)
for arg in "$@"; do
    [ "$arg" = "--cmplog" ] && WITH_CMPLOG=1
done

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
    local rc=0
    gcc $extra_flags -O2 -g -include "$SHIM" \
        -o "$out" "$src" $libs 2>/dev/null || rc=$?
    if [ $rc -eq 0 ]; then
        ok "$(basename "$out")"
    else
        warn "failed: $(basename "$out")"
    fi
}

# ── Build a .so target ───────────────────────────────────────────
build_so_target() {
    local src="$1" out="$2" libs="$3" extra_flags="$4"
    local cmplog_obj="" cmplog_libs=""
    if [ ! -f "$src" ]; then
        warn "Source not found: $src"
        return 1
    fi
    if [ "$WITH_CMPLOG" -eq 1 ] && [ -f "$CMPLOG_SHIM" ]; then
        # Compile cmplog shim separately (without -include afl_shim.c
        # to avoid duplicate symbol definitions from -include hitting
        # both compilation units).
        local cmplog_obj_path="/tmp/fuzz_cmplog_$$.o"
        gcc $extra_flags -O2 -g -fPIC -c "$CMPLOG_SHIM" -o "$cmplog_obj_path" 2>/dev/null
        if [ -f "$cmplog_obj_path" ]; then
            cmplog_obj="$cmplog_obj_path"
            cmplog_libs="-ldl"
        fi
    fi
    local rc=0
    gcc $extra_flags -O2 -g -shared -fPIC -include "$SHIM" \
        -o "$out" "$src" $cmplog_obj $libs $cmplog_libs 2>/dev/null || rc=$?
    # Clean up temp cmplog object
    [ -n "$cmplog_obj" ] && rm -f "$cmplog_obj"
    if [ $rc -eq 0 ]; then
        ok "$(basename "$out")"
    else
        warn "failed: $(basename "$out")"
    fi
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
    build_so_target "$TARGETS/nop_target.c" "$TARGETS/nop_target${out_suffix}.so" "" "$flags"
}

# ── Build standalone .so targets with external deps ─────────────
build_standalone_so_targets() {
    local suffix="$1" flags="$2" label="$3"
    local out_suffix=""
    [ "$suffix" = "_nosan" ] && out_suffix="_nosan"

    # tailslayer — C++ target (g++), header-only library
    if [ -f "$TARGETS/tailslayer_read.cpp" ] && [ -d "$TAILSLAYER/include" ]; then
        local cxx=g++
        if command -v g++ &>/dev/null; then
            local src="$TARGETS/tailslayer_read.cpp"
            local out="$TARGETS/tailslayer_read${out_suffix}.so"
            local inc="-I$TAILSLAYER/include"
            local cmplog_obj="" cmplog_libs=""
            if [ "$WITH_CMPLOG" -eq 1 ] && [ -f "$CMPLOG_SHIM" ]; then
                # Compile cmplog shim with gcc (C, not C++) to avoid
                # C++ const-correctness conflict on memchr signature.
                local co="/tmp/fuzz_cmplog_tailslayer$$.o"
                gcc $flags -O2 -g -fPIC -c "$CMPLOG_SHIM" -o "$co" 2>/dev/null && cmplog_obj="$co" && cmplog_libs="-ldl"
            fi
            # Use g++ for link, but compile the .cpp with -include afl_shim only.
            # cmplog_shim.o is already a compiled C object (no -include needed).
            $cxx $flags -O2 -g -shared -fPIC -include "$SHIM" $inc \
                -o "$out" "$src" $cmplog_obj $cmplog_libs 2>/dev/null && ok "tailslayer_read${out_suffix}.so" || warn "failed: tailslayer_read${out_suffix}.so"
            [ -n "$cmplog_obj" ] && rm -f "$cmplog_obj"
        fi
    elif [ -f "$TARGETS/tailslayer_read.cpp" ] && [ ! -d "$TAILSLAYER/include" ]; then
        warn "tailslayer_read${out_suffix}.so: tailslayer headers not found at $TAILSLAYER/include, skipping"
    fi

    # lz4_read — needs LZ4 precompiled objects + include path
    local LZ4_DIR="${LZ4_DIR:-/home/dclavijo/code/lz4/lib}"
    local LZ4_OBJS="/tmp/lz4$suffix.o /tmp/lz4frame$suffix.o /tmp/lz4hc$suffix.o /tmp/xxhash$suffix.o"
    local LZ4_INC="-I$LZ4_DIR -DXXH_NAMESPACE=LZ4_"
    local all_exist=true
    for obj in $LZ4_OBJS; do [ -f "$obj" ] || all_exist=false; done
    if $all_exist && [ -f "$TARGETS/lz4_read.c" ]; then
        build_so_target "$TARGETS/lz4_read.c" "$TARGETS/lz4_read${out_suffix}.so" "$LZ4_OBJS -Wl,--export-dynamic -lpthread" "$flags $LZ4_INC"
    else
        warn "lz4_read${out_suffix}.so: LZ4 objects not found, skipping (build LZ4 lib first)"
    fi
}

# ── Verify AFL symbols ───────────────────────────────────────────
verify_afl() {
    echo "Verifying AFL symbols..."
    local count=0 fail_count=0
    for f in "$TARGETS"/fuzz_* "$TARGETS"/asan_target "$TARGETS"/asan_target_nosan "$TARGETS"/asan_target.so "$TARGETS"/asan_target_nosan.so \
             "$TARGETS"/png_read "$TARGETS"/png_read_nosan "$TARGETS"/png_read.so "$TARGETS"/png_read_nosan.so \
             "$TARGETS"/zlib_read "$TARGETS"/zlib_read_nosan "$TARGETS"/zlib_read.so "$TARGETS"/zlib_read_nosan.so \
             "$TARGETS"/gzip_read "$TARGETS"/gzip_read_nosan "$TARGETS"/gzip_read.so "$TARGETS"/gzip_read_nosan.so \
             "$TARGETS"/jpeg_read "$TARGETS"/jpeg_read_nosan "$TARGETS"/jpeg_read.so "$TARGETS"/jpeg_read_nosan.so \
             "$TARGETS"/test_target "$TARGETS"/test_target_nosan "$TARGETS"/test_target.so "$TARGETS"/test_target_nosan.so \
             "$TARGETS"/proto_target "$TARGETS"/proto_target_nosan "$TARGETS"/proto_target.so "$TARGETS"/proto_target_nosan.so \
             "$TARGETS"/nop_target "$TARGETS"/nop_target_nosan "$TARGETS"/nop_target.so "$TARGETS"/nop_target_nosan.so \
             "$TARGETS"/tailslayer_read "$TARGETS"/tailslayer_read.so \
             "$TARGETS"/lz4_read "$TARGETS"/lz4_read_nosan "$TARGETS"/lz4_read.so "$TARGETS"/lz4_read_nosan.so; do
        [ -f "$f" ] || continue
        [[ "$f" == *.c ]] && continue
        local n=$(nm "$f" 2>/dev/null | grep -c __afl || true)
        if [ "$n" -gt 0 ]; then
            count=$((count + 1))
        else
            warn "$(basename "$f"): no AFL symbols"
            fail_count=$((fail_count + 1))
        fi
    done
    ok "$count targets with AFL symbols"
    if [ "$fail_count" -gt 0 ]; then
        warn "$fail_count targets without AFL symbols"
    fi
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
    if [ "$fail_count" -gt 0 ]; then
        warn "$fail_count .so targets missing fuzz_shm_run"
    fi
}

# ── Verify cmplog symbols in .so targets ───────────────────────
verify_cmplog() {
    [ "$WITH_CMPLOG" -eq 0 ] && return 0
    echo "Verifying cmplog symbols in .so targets..."
    local ok_count=0
    local fail_count=0
    for f in "$TARGETS"/*.so; do
        [ -f "$f" ] || continue
        if nm "$f" 2>/dev/null | grep -q "__cmplog_reset"; then
            ok_count=$((ok_count + 1))
        else
            warn "$(basename "$f"): missing __cmplog_reset"
            fail_count=$((fail_count + 1))
        fi
    done
    ok "$ok_count .so targets with cmplog"
    if [ "$fail_count" -gt 0 ]; then
        warn "$fail_count .so targets missing cmplog"
    fi
}

# ── Main ──────────────────────────────────────────────────────────
echo "=== Building fuzz targets ==="
[ "$WITH_CMPLOG" -eq 1 ] && echo "[*] Cmplog: build-time linking enabled for .so targets"

if [ "$HAS_FGREP" -eq 0 ]; then
    warn "fgrep directory not found at $FGREP — skipping fgrep targets"
fi

case "$OPTS" in
    --asan)
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_asan" "-fsanitize=address"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_asan" "-fsanitize=address" "ASAN"
        build_simple_targets "_asan" "-fsanitize=address" "ASAN"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_asan" "-fsanitize=address" "ASAN"
        build_simple_so_targets "_asan" "-fsanitize=address" "ASAN"
        build_standalone_so_targets "_asan" "-fsanitize=address" "ASAN"
        ;;
    --fast|--nosan)
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_nosan" ""
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_nosan" "" "No-ASAN"
        build_simple_targets "_nosan" "" "No-ASAN"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_nosan" "" "No-ASAN"
        build_simple_so_targets "_nosan" "" "No-ASAN"
        build_standalone_so_targets "_nosan" "" "No-ASAN"
        ;;
    *)
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_asan" "-fsanitize=address"
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_nosan" ""
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_asan" "-fsanitize=address" "ASAN"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_nosan" "" "No-ASAN"
        build_simple_targets "_asan" "-fsanitize=address" "ASAN"
        build_simple_targets "_nosan" "" "No-ASAN"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_asan" "-fsanitize=address" "ASAN"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_nosan" "" "No-ASAN"
        build_simple_so_targets "_asan" "-fsanitize=address" "ASAN"
        build_simple_so_targets "_nosan" "" "No-ASAN"
        build_standalone_so_targets "_asan" "-fsanitize=address" "ASAN"
        build_standalone_so_targets "_nosan" "" "No-ASAN"
        ;;
esac

verify_afl
verify_shm_run
verify_cmplog
echo "=== Done ==="
