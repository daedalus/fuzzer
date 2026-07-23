#!/bin/bash
# Build all fuzz targets with AFL edge coverage.
# Compiles both ASAN and no-ASAN variants.
#
# Usage:
#   tools/build_targets.sh                            # Build all targets
#   tools/build_targets.sh --asan                     # ASAN only
#   tools/build_targets.sh --fast                     # No-ASAN only
#   tools/build_targets.sh --cmplog                   # Include cmplog in .so targets (build-time linking)
#   tools/build_targets.sh --asan --cmplog            # ASAN + cmplog in .so targets
#   tools/build_targets.sh --clang-scov               # Clang + compiler-inserted edge coverage (sancov)
#   tools/build_targets.sh --tracecmp                 # Clang + compiler-IR comparison tracing
#   tools/build_targets.sh --vendor-tracecmp          # Vendored libpng+zlib + trace-cmp targets
#   tools/build_targets.sh --vendor-tracecmp --asan   # Same with ASAN

set -e

FGREP="${FGREP_DIR:-/home/dclavijo/my_code/fgrep}"
TAILSLAYER="${TAILSLAYER_DIR:-/home/dclavijo/code/tailslayer}"
SHIM="src/fuzzer_tool/adapters/afl_shim.c"
CMPLOG_SHIM="src/fuzzer_tool/adapters/cmplog_shim.c"
TARGETS="targets"
VENDOR="vendor"
OPTS="${@:---all}"
HAS_FGREP=0
[ -d "$FGREP/src" ] && HAS_FGREP=1
WITH_CMPLOG=0
WITH_TRACECMP=0
WITH_VENDOR_TRACECMP=0
WITH_CLANG_SCOV=0
USE_CLANG=0

# Parse flags (can appear anywhere)
for arg in "$@"; do
    [ "$arg" = "--cmplog" ] && WITH_CMPLOG=1
    # --tracecmp implies --cmplog (the unified shim covers both layers)
    [ "$arg" = "--tracecmp" ] && WITH_CMPLOG=1 && WITH_TRACECMP=1
    [ "$arg" = "--vendor-tracecmp" ] && WITH_VENDOR_TRACECMP=1
    [ "$arg" = "--clang-scov" ] && WITH_CLANG_SCOV=1
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok() { echo -e "  ${GREEN}OK${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

# ── Compile fgrep library objects ──────────────────────────────────
compile_fgrep_objects() {
    local suffix="$1" flags="$2" cc="${3:-gcc}" extra_cflags="${4:-}"
    echo "Compiling fgrep objects${suffix:+ ($suffix)}..."
    for src in regex_engine simd cpu; do
        $cc $flags -fPIC -O2 -g $extra_cflags -I"$FGREP/include" -I"$FGREP/src" \
            -c "$FGREP/src/${src}.c" -o "/tmp/${src}${suffix}.o"
    done
    for src in output search bmh_simd io fileutil; do
        $cc $flags -fPIC -O2 -g -mavx2 $extra_cflags -I"$FGREP/include" -I"$FGREP/src" \
            -c "$FGREP/src/${src}.c" -o "/tmp/${src}${suffix}.o"
    done
    ok "fgrep objects${suffix:+ ($suffix)}"
}

# ── Build a target ────────────────────────────────────────────────
build_target() {
    local src="$1" out="$2" libs="$3" extra_flags="$4" cc="${5:-gcc}" extra_cflags="${6:-}"
    if [ ! -f "$src" ]; then
        warn "Source not found: $src"
        return 1
    fi
    local rc=0
    $cc $extra_flags -O2 -g $extra_cflags -include "$SHIM" \
        -o "$out" "$src" $libs 2>/dev/null || rc=$?
    if [ $rc -eq 0 ]; then
        ok "$(basename "$out")"
    else
        warn "failed: $(basename "$out")"
    fi
}

# ── Build a .so target ───────────────────────────────────────────
build_so_target() {
    local src="$1" out="$2" libs="$3" extra_flags="$4" cc="${5:-gcc}" extra_cflags="${6:-}"
    local cmplog_obj="" cmplog_libs=""
    if [ ! -f "$src" ]; then
        warn "Source not found: $src"
        return 1
    fi
    if [ "$WITH_CMPLOG" -eq 1 ] && [ -f "$CMPLOG_SHIM" ]; then
        local cmplog_obj_path="/tmp/fuzz_cmplog_$$.o"
        # Compile cmplog shim with clang (NOT gcc) — clang generates correct
        # __sanitizer_cov_trace_cmp* implementations. Do NOT add -fsanitize-coverage
        # to the shim — it PROVIDES the callbacks, not calls them.
        local CMPLOG_CC="clang"
        command -v clang &>/dev/null || CMPLOG_CC="$cc"
        $CMPLOG_CC -O2 -g -fPIC -c "$CMPLOG_SHIM" -o "$cmplog_obj_path" 2>/dev/null
        if [ -f "$cmplog_obj_path" ]; then
            cmplog_obj="$cmplog_obj_path"
            cmplog_libs="-ldl"
        fi
    fi
    local rc=0
    # Do NOT add -fsanitize-coverage=trace-cmp to the target's own code —
    # the target wrapper (png_read.c etc.) has almost no comparisons; all
    # the interesting comparisons are in the vendored libraries which are
    # compiled separately with trace-cmp. Adding it here just adds overhead
    # for zero benefit.
    # -Bsymbolic: prevents ASAN's LD_PRELOAD from overriding the trace-cmp
    # callbacks with no-ops (ASAN ships weak stubs that shadow our shim).
    local bsymbolic_flag=""
    local target_cc="$cc"
    if [ "$WITH_CMPLOG" -eq 1 ]; then
        bsymbolic_flag="-Wl,-Bsymbolic"
        if command -v clang &>/dev/null; then
            target_cc="clang"
        fi
    fi
    $target_cc $extra_flags -O2 -g $extra_cflags -shared -fPIC $bsymbolic_flag -include "$SHIM" \
        -o "$out" "$src" $cmplog_obj $libs $cmplog_libs 2>/dev/null || rc=$?
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
                # Do NOT add trace-cmp — the shim provides the callbacks.
                local co="/tmp/fuzz_cmplog_tailslayer$$.o"
                gcc $flags -O2 -g -fPIC -c "$CMPLOG_SHIM" -o "$co" 2>/dev/null && cmplog_obj="$co" && cmplog_libs="-ldl"
            fi
            # Use g++ for link, but compile the .cpp with -include afl_shim only.
            # cmplog_shim.o is already a compiled C object (no -include needed).
            local bsym_flag=""
            [ "$WITH_CMPLOG" -eq 1 ] && bsym_flag="-Wl,-Bsymbolic"
            $cxx $flags -O2 -g -shared -fPIC $bsym_flag -include "$SHIM" $inc \
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

# ── Compile vendored libraries with sancov instrumentation ───────
compile_vendored_libs() {
    local cc="$1" scov_flag="$2" suffix="$3"
    echo "Compiling vendored libraries ($cc ${scov_flag:-no-sancov})..."

    # zlib
    if [ -d "$VENDOR/zlib" ]; then
        (cd "$VENDOR/zlib" && CC=$cc CFLAGS="-O2 -g -fPIC ${scov_flag}" \
            ./configure --static 2>/dev/null && make -j$(nproc) 2>/dev/null) && \
            ok "zlib (vendored)" || warn "zlib (vendored) failed"
    else
        warn "vendored zlib not found at $VENDOR/zlib"
    fi

    # libpng (depends on zlib)
    if [ -d "$VENDOR/libpng" ] && [ -d "$VENDOR/zlib" ]; then
        (cd "$VENDOR/libpng" && CC=$cc \
            CFLAGS="-O2 -g -fPIC ${scov_flag} -I../../zlib" \
            LDFLAGS="-L../../zlib" \
            ./configure --with-pkgconfig=no 2>/dev/null && make -j$(nproc) 2>/dev/null) && \
            ok "libpng (vendored)" || warn "libpng (vendored) failed"
    else
        warn "vendored libpng not found or zlib missing"
    fi

    # libjpeg-turbo
    if [ -d "$VENDOR/libjpeg-turbo" ]; then
        (cd "$VENDOR/libjpeg-turbo" && \
            cmake -DCMAKE_C_COMPILER=$cc \
                  -DCMAKE_C_FLAGS="-O2 -g -fPIC ${scov_flag}" \
                  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                  -G "Unix Makefiles" . 2>/dev/null && \
            make -j$(nproc) 2>/dev/null) && \
            ok "libjpeg-turbo (vendored)" || warn "libjpeg-turbo (vendored) failed"
    else
        warn "vendored libjpeg-turbo not found"
    fi
}

# ── Build .so targets with vendored libraries ────────────────────
build_vendored_so_targets() {
    local suffix="$1" flags="$2" label="$3" cc="${4:-gcc}" extra_cflags="${5:-}"
    echo "Building vendored .so targets ($label)..."
    local out_suffix=""
    [ "$suffix" = "_nosan" ] && out_suffix="_nosan"

    local ZLIB_OBJS=""
    local ZLIB_INC=""
    local PNG_OBJS=""
    local PNG_INC=""
    local JPEG_OBJS=""
    local JPEG_INC=""

    if [ -d "$VENDOR/zlib" ]; then
        ZLIB_OBJS=$(ls "$VENDOR/zlib"/*.o 2>/dev/null | tr '\n' ' ')
        ZLIB_INC="-I$VENDOR/zlib"
    fi
    if [ -d "$VENDOR/libpng" ]; then
        PNG_OBJS=$(ls "$VENDOR/libpng"/*.o 2>/dev/null | tr '\n' ' ')
        PNG_INC="-I$VENDOR/libpng -I$VENDOR/libpng/scripts"
    fi
    if [ -d "$VENDOR/libjpeg-turbo" ]; then
        JPEG_OBJS=$(ls "$VENDOR/libjpeg-turbo"/*.o 2>/dev/null | tr '\n' ' ')
        JPEG_INC="-I$VENDOR/libjpeg-turbo"
    fi

    # png_read.so — vendored libpng + zlib
    if [ -n "$PNG_OBJS" ] && [ -n "$ZLIB_OBJS" ]; then
        build_so_target "$TARGETS/png_read.c" "$TARGETS/png_read${out_suffix}_scov.so" \
            "$PNG_OBJS $ZLIB_OBJS -lm -lpthread" "$flags" "$cc" "$extra_cflags $PNG_INC $ZLIB_INC"
    else
        warn "png_read${out_suffix}_scov.so: vendored objects missing, skipping"
    fi

    # zlib_read.so — vendored zlib
    if [ -n "$ZLIB_OBJS" ]; then
        build_so_target "$TARGETS/zlib_read.c" "$TARGETS/zlib_read${out_suffix}_scov.so" \
            "$ZLIB_OBJS -lm" "$flags" "$cc" "$extra_cflags $ZLIB_INC"
    else
        warn "zlib_read${out_suffix}_scov.so: vendored zlib objects missing, skipping"
    fi

    # gzip_read.so — vendored zlib
    if [ -n "$ZLIB_OBJS" ]; then
        build_so_target "$TARGETS/gzip_read.c" "$TARGETS/gzip_read${out_suffix}_scov.so" \
            "$ZLIB_OBJS -lm" "$flags" "$cc" "$extra_cflags $ZLIB_INC"
    else
        warn "gzip_read${out_suffix}_scov.so: vendored zlib objects missing, skipping"
    fi

    # jpeg_read.so — vendored libjpeg-turbo
    if [ -n "$JPEG_OBJS" ]; then
        build_so_target "$TARGETS/jpeg_read.c" "$TARGETS/jpeg_read${out_suffix}_scov.so" \
            "$JPEG_OBJS -lm -lpthread" "$flags" "$cc" "$extra_cflags $JPEG_INC"
    else
        warn "jpeg_read${out_suffix}_scov.so: vendored libjpeg-turbo objects missing, skipping"
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

# ── Vendored trace-cmp: rebuild libpng+zlib with trace-cmp, then link targets ─
VENDOR_ZLIB_DIR="$VENDOR/zlib"
VENDOR_LIBPNG_DIR="$VENDOR/libpng"

build_vendored_tracecmp_targets() {
    [ "$WITH_VENDOR_TRACECMP" -eq 0 ] && return 0

    local CC="clang"
    if ! command -v clang &>/dev/null; then
        warn "clang not found — --vendor-tracecmp requires clang"
        return 1
    fi
    local TRACE_FLAGS="-fsanitize-coverage=trace-cmp,trace-pc-guard"
    local ASAN_FLAGS=""
    for arg in "$@"; do
        [ "$arg" = "--asan" ] && ASAN_FLAGS="-fsanitize=address"
    done

    echo "Building vendored trace-cmp targets ($CC)..."
    local VENDOR_OK=0

    # ── zlib ────────────────────────────────────────────────────────
    if [ -f "$VENDOR_ZLIB_DIR/configure" ]; then
        echo "  [1/3] Compiling vendor/zlib with trace-cmp..."
        (cd "$VENDOR_ZLIB_DIR" && \
            CC=clang CFLAGS="-O2 -g -fPIC $TRACE_FLAGS" \
            ./configure --static 2>/dev/null && \
            make -j$(nproc) -s 2>/dev/null) && \
            ok "vendor/zlib (trace-cmp)" || warn "vendor/zlib build failed"
    else
        warn "vendor/zlib not found — skipping"
        VENDOR_OK=1
    fi

    # ── libpng ──────────────────────────────────────────────────────
    if [ -f "$VENDOR_LIBPNG_DIR/configure" ]; then
        echo "  [2/3] Compiling vendor/libpng with trace-cmp..."
        (cd "$VENDOR_LIBPNG_DIR" && \
            CC=clang CFLAGS="-O2 -g -fPIC $TRACE_FLAGS -I../zlib" \
            LDFLAGS="-L../zlib" \
            ./configure --enable-shared=no --quiet 2>/dev/null && \
            make -j$(nproc) -s 2>/dev/null) && \
            ok "vendor/libpng (trace-cmp)" || warn "vendor/libpng build failed"
    else
        warn "vendor/libpng not found — skipping"
        VENDOR_OK=1
    fi

    # Verify vendor .a files exist
    local ZLIB_A="$VENDOR_ZLIB_DIR/libz.a"
    local LIBPNG_A="$VENDOR_LIBPNG_DIR/.libs/libpng16.a"
    if [ ! -f "$ZLIB_A" ] || [ ! -f "$LIBPNG_A" ]; then
        warn "Vendor .a files missing (zlib: $(test -f "$ZLIB_A" && echo ok || echo missing), libpng: $(test -f "$LIBPNG_A" && echo ok || echo missing))"
        return 1
    fi

    # Verify trace-cmp callbacks in vendor objects
    local ZLIB_TC=$(nm "$ZLIB_A" 2>/dev/null | grep -c 'U.*trace_cmp' || echo 0)
    local LIBPNG_TC=$(nm "$LIBPNG_A" 2>/dev/null | grep -c 'U.*trace_cmp' || echo 0)
    echo "  Vendor trace-cmp callbacks: zlib=${ZLIB_TC}, libpng=${LIBPNG_TC}"

    # ── Build .so targets ───────────────────────────────────────────
    echo "  [3/3] Linking targets against vendored trace-cmp libs..."

    local LIBS="-lm"
    local VENDOR_LIBS="$LIBPNG_A $ZLIB_A $LIBS"
    local VENDOR_INC="-I$VENDOR_LIBPNG_DIR -I$VENDOR_ZLIB_DIR"
    local OUT_SUFFIX="_tracecmp"
    local ALL_FLAGS="$TRACE_FLAGS $ASAN_FLAGS"

    # png_read
    if [ -f "$TARGETS/png_read.c" ]; then
        $CC -O2 -g $ALL_FLAGS -shared -fPIC -include "$SHIM" \
            -o "$TARGETS/png_read${OUT_SUFFIX}.so" \
            "$TARGETS/png_read.c" $VENDOR_LIBS $VENDOR_INC 2>/dev/null && \
            ok "png_read${OUT_SUFFIX}.so" || warn "failed: png_read${OUT_SUFFIX}.so"
    fi

    # zlib_read
    if [ -f "$TARGETS/zlib_read.c" ]; then
        $CC -O2 -g $ALL_FLAGS -shared -fPIC -include "$SHIM" \
            -o "$TARGETS/zlib_read${OUT_SUFFIX}.so" \
            "$TARGETS/zlib_read.c" "$ZLIB_A" $LIBS 2>/dev/null && \
            ok "zlib_read${OUT_SUFFIX}.so" || warn "failed: zlib_read${OUT_SUFFIX}.so"
    fi

    # gzip_read
    if [ -f "$TARGETS/gzip_read.c" ]; then
        $CC -O2 -g $ALL_FLAGS -shared -fPIC -include "$SHIM" \
            -o "$TARGETS/gzip_read${OUT_SUFFIX}.so" \
            "$TARGETS/gzip_read.c" "$ZLIB_A" $LIBS 2>/dev/null && \
            ok "gzip_read${OUT_SUFFIX}.so" || warn "failed: gzip_read${OUT_SUFFIX}.so"
    fi

    # jpeg_read (needs system libjpeg — no vendored jpeg yet)
    if [ -f "$TARGETS/jpeg_read.c" ]; then
        $CC -O2 -g $ALL_FLAGS -shared -fPIC -include "$SHIM" \
            -o "$TARGETS/jpeg_read${OUT_SUFFIX}.so" \
            "$TARGETS/jpeg_read.c" -ljpeg $LIBS 2>/dev/null && \
            ok "jpeg_read${OUT_SUFFIX}.so" || warn "failed: jpeg_read${OUT_SUFFIX}.so"
    fi

    # Verify trace-cmp symbols are UNDEFINED (U) in output .so files
    echo "  Verifying trace-cmp callbacks in output targets..."
    for f in "$TARGETS"/*"${OUT_SUFFIX}.so" "$TARGETS/png_read_asan_tracecmp.so"; do
        [ -f "$f" ] || continue
        local tc_count=$(nm "$f" 2>/dev/null | grep -c 'trace_cmp' || echo 0)
        if [ "$tc_count" -gt 0 ]; then
            ok "$(basename "$f"): $tc_count trace-cmp callbacks"
        else
            warn "$(basename "$f"): no trace-cmp callbacks found"
        fi
    done

    # Also build ASAN variant with tracecmp compiled in (two-step:
    # compile tracecmp_shim.c separately, then link together).
    # The two-step build uses hidden visibility in tracecmp_shim.c,
    # preventing ASAN's LD_PRELOAD from overriding the callbacks.
    # Only builds when --asan is passed.
    local HAS_ASAN=0
    for _arg in "$@"; do [ "$_arg" = "--asan" ] && HAS_ASAN=1; done
    if [ "$HAS_ASAN" -eq 1 ] && [ -f "$TARGETS/png_read.c" ] && [ -f "src/fuzzer_tool/adapters/tracecmp_shim.c" ]; then
        local TC_SHIM_OBJ="/tmp/tracecmp_shim_asan_$$.o"
        # Link ASAN runtime statically via libasan.a to avoid LD_PRELOAD
        local ASAN_LIB="/usr/lib/gcc/x86_64-linux-gnu/14/libasan.a"
        if [ ! -f "$ASAN_LIB" ]; then
            ASAN_LIB=$(gcc -print-file-name=libasan.a 2>/dev/null)
        fi
        $CC -O2 -g -fsanitize=address -fvisibility=hidden -fPIC -c \
            "src/fuzzer_tool/adapters/tracecmp_shim.c" \
            -o "$TC_SHIM_OBJ" 2>/dev/null
        if [ -f "$ASAN_LIB" ] && [ -f "$TC_SHIM_OBJ" ]; then
            $CC -O2 -g \
                -fsanitize=address \
                -fsanitize-coverage=trace-cmp,trace-pc-guard \
                -shared -fPIC \
                -include "$SHIM" \
                -o "$TARGETS/png_read_asan_tracecmp.so" \
                "$TARGETS/png_read.c" "$TC_SHIM_OBJ" \
                $VENDOR_LIBS $VENDOR_INC \
                -Wl,--whole-archive "$ASAN_LIB" -Wl,--no-whole-archive \
                2>/dev/null && \
            rm -f "$TC_SHIM_OBJ" && \
            ok "png_read_asan_tracecmp.so (ASAN + tracecmp, no LD_PRELOAD needed)" || \
            warn "failed: png_read_asan_tracecmp.so"
        else
            [ -f "$TC_SHIM_OBJ" ] && rm -f "$TC_SHIM_OBJ"
            warn "libasan.a not found at $ASAN_LIB — skipping ASAN variant"
        fi
    fi

    # Verify trace-cmp callbacks (non-ASAN targets have U symbols)
    echo "  Verifying trace-cmp callbacks in output targets..."
    for src in png_read zlib_read gzip_read; do
        local src_file="$TARGETS/$src.c"
        local out_file="$TARGETS/${src}${OUT_SUFFIX}"
        [ -f "$src_file" ] || continue
        # Pick the right libs per target
        local tgt_libs="$LIBS"
        case "$src" in
            png_read) tgt_libs="$LIBPNG_A $ZLIB_A $LIBS" ;;
            zlib_read|gzip_read) tgt_libs="$ZLIB_A $LIBS" ;;
        esac
        $CC -O2 -g $ALL_FLAGS -include "$SHIM" \
            -o "$out_file" "$src_file" $tgt_libs $VENDOR_INC 2>/dev/null && \
            ok "$(basename "$out_file")" || warn "failed: $(basename "$out_file")"
    done

    echo "  Done — target suffix: ${OUT_SUFFIX}"
}

# ── Build trace-cmp targets (Clang -fsanitize-coverage=trace-cmp) ─
build_tracecmp_targets() {
    [ "$WITH_TRACECMP" -eq 0 ] && return 0

    local CC="gcc"
    if [ "$USE_CLANG" -eq 1 ]; then
        if command -v clang &>/dev/null; then
            CC="clang"
        else
            warn "clang not found — trace-cmp targets require clang"
            return 1
        fi
    elif command -v clang &>/dev/null; then
        CC="clang"
    else
        warn "clang not found — trace-cmp targets require clang"
        return 1
    fi

    echo "Building trace-cmp targets ($CC)..."
    local TRACE_FLAGS="-fsanitize-coverage=trace-cmp,trace-pc-guard"

    # tracecmp_target: exercises compiler-inlined comparisons
    local rc=0
    $CC -O2 -g $TRACE_FLAGS -include "$SHIM" \
        -o "$TARGETS/tracecmp_target" "$TARGETS/tracecmp_target.c" 2>/dev/null || rc=$?
    if [ $rc -eq 0 ]; then
        ok "tracecmp_target (trace-cmp)"
    else
        warn "failed: tracecmp_target (trace-cmp)"
    fi

    # tracecmp_target.so: same with shared library
    rc=0
    $CC -O2 -g $TRACE_FLAGS -shared -fPIC -include "$SHIM" \
        -o "$TARGETS/tracecmp_target.so" "$TARGETS/tracecmp_target.c" 2>/dev/null || rc=$?
    if [ $rc -eq 0 ]; then
        ok "tracecmp_target.so (trace-cmp)"
    else
        warn "failed: tracecmp_target.so (trace-cmp)"
    fi

    # Verify trace-cmp symbols in built targets
    for f in "$TARGETS/tracecmp_target" "$TARGETS/tracecmp_target.so"; do
        if [ -f "$f" ] && nm "$f" 2>/dev/null | grep -q "trace_cmp"; then
            ok "$(basename "$f"): trace-cmp callbacks present"
        fi
    done
}

# ── Main ──────────────────────────────────────────────────────────
echo "=== Building fuzz targets ==="
[ "$WITH_CMPLOG" -eq 1 ] && echo "[*] Cmplog: build-time linking enabled for .so targets"
[ "$WITH_TRACECMP" -eq 1 ] && echo "[*] Trace-cmp: compiler-IR comparison tracing enabled (requires clang)"
[ "$WITH_CLANG_SCOV" -eq 1 ] && echo "[*] Clang-scov: compiler-inserted edge coverage enabled (requires clang)"
[ "$WITH_VENDOR_TRACECMP" -eq 1 ] && echo "[*] Vendor-tracecmp: rebuild vendor libs + targets with trace-cmp (requires clang)"

if [ "$HAS_FGREP" -eq 0 ]; then
    warn "fgrep directory not found at $FGREP — skipping fgrep targets"
fi

# Dispatch by flags — multiple flags can be combined
HAS_ASAN_ARG=0
for _a in "$@"; do [ "$_a" = "--asan" ] && HAS_ASAN_ARG=1; done

BUILD_ASAN=0; BUILD_NOSAN=0
case "$OPTS" in
    --asan) BUILD_ASAN=1 ;;
    --fast|--nosan) BUILD_NOSAN=1 ;;
    --clang-scov) BUILD_ASAN=1; BUILD_NOSAN=1 ;;
    --vendor-tracecmp) [ "$HAS_ASAN_ARG" -eq 1 ] && BUILD_ASAN=1 || BUILD_NOSAN=1 ;;
    *) BUILD_ASAN=1; BUILD_NOSAN=1 ;;
esac

if [ "$BUILD_ASAN" -eq 1 ]; then
    # ASAN .so targets always include cmplog (linked shim, not LD_PRELOAD).
    # This ensures --cmplog works out of the box without extra flags.
    WITH_CMPLOG=1
    [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_asan" "-fsanitize=address"
    [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_asan" "-fsanitize=address" "ASAN"
    # Compile fgrep objects with clang + trace-pc-guard for .so targets
    # auto-instrumentation via afl_shim.c's __sanitizer_cov_trace_pc_guard
    if command -v clang &>/dev/null; then
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_asan_tcg" "-fsanitize=address" "clang" "-fsanitize-coverage=trace-pc-guard"
    else
        warn "clang not found — .so targets will lack auto edge coverage (manual __afl_map_edge only)"
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_asan_tcg" "-fsanitize=address"
    fi
    build_simple_targets "_asan" "-fsanitize=address" "ASAN"
    [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_asan_tcg" "-fsanitize=address" "ASAN"
    build_simple_so_targets "_asan" "-fsanitize=address" "ASAN"
    build_standalone_so_targets "_asan" "-fsanitize=address" "ASAN"
fi
if [ "$BUILD_NOSAN" -eq 1 ]; then
    [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_nosan" ""
    [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_nosan" "" "No-ASAN"
    if command -v clang &>/dev/null; then
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_nosan_tcg" "" "clang" "-fsanitize-coverage=trace-pc-guard"
    else
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_nosan_tcg" ""
    fi
    build_simple_targets "_nosan" "" "No-ASAN"
    [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_nosan_tcg" "" "No-ASAN"
    build_simple_so_targets "_nosan" "" "No-ASAN"
    build_standalone_so_targets "_nosan" "" "No-ASAN"
fi
if [ "$OPTS" = "--clang-scov" ]; then
    local SCOV_CC="clang"
    if ! command -v clang &>/dev/null; then
        warn "clang not found — --clang-scov requires clang"
    else
        local SCOV_FLAGS="-fsanitize-coverage=trace-pc-guard"
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_asan" "-fsanitize=address" "$SCOV_CC" "$SCOV_FLAGS"
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_nosan" "" "$SCOV_CC" "$SCOV_FLAGS"
        compile_vendored_libs "$SCOV_CC" "$SCOV_FLAGS" "_asan"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_asan" "-fsanitize=address" "Clang-scov"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_nosan" "" "Clang-scov"
        build_simple_targets "_asan" "-fsanitize=address" "Clang-scov"
        build_simple_targets "_nosan" "" "Clang-scov"
        build_vendored_so_targets "_asan" "-fsanitize=address" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        build_vendored_so_targets "_nosan" "" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_asan" "-fsanitize=address" "Clang-scov"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_nosan" "" "Clang-scov"
        build_standalone_so_targets "_asan" "-fsanitize=address" "Clang-scov"
        build_standalone_so_targets "_nosan" "" "Clang-scov"
    fi
fi
if [ "$WITH_VENDOR_TRACECMP" -eq 1 ]; then
    build_vendored_tracecmp_targets "$@"
fi

verify_afl
verify_shm_run
verify_cmplog
build_tracecmp_targets
echo "=== Done ==="
