#!/bin/bash
# Build all fuzz targets with AFL edge coverage.
# Compiles both ASAN and no-ASAN variants.
#
# Usage:
#   tools/build_targets.sh                            # Build all targets (default: ASAN + cmplog)
#   tools/build_targets.sh --fast                     # Build no-ASAN only
#   tools/build_targets.sh --cmplog                   # Include cmplog in .so targets (default: on; explicit for clarity)
#   tools/build_targets.sh --asan --cmplog            # Same as default
#   tools/build_targets.sh --clang-scov               # Clang + compiler-inserted edge coverage (sancov)
#   tools/build_targets.sh --tracecmp                 # Clang + compiler-IR comparison tracing
#   tools/build_targets.sh --vendor-tracecmp          # Vendored libpng+zlib + trace-cmp targets
#   tools/build_targets.sh --vendor-tracecmp --asan   # Same with ASAN
#   tools/build_targets.sh --distance                 # AFLGo distance .so targets (_dist / _dist_asan)
#   tools/build_targets.sh --ngram                    # png_read k=2/k=3 n-gram flavors (_ng2 / _ng3)
#
# Some targets need vendored sources fetched first (all are optional — the
# build warns and skips when the tree is absent):
#   tools/vendor_lz4.sh      -> vendor/lz4      (lz4_read / lz4_read.so)
#   tools/vendor_grep.sh     -> vendor/grep
#   tools/vendor_ffmpeg.sh   -> vendor/ffmpeg
#   tools/vendor_secp256k1.sh -> vendor/secp256k1 (secp256k1_read.so)
#   tools/vendor_sqlite.sh   -> vendor/sqlite    (sqlite_read.so)

set -e

# `set -e` plus a long list of optional, best-effort build steps is a silent
# failure waiting to happen: any command returning non-zero kills the script
# mid-run, the last thing printed is whatever succeeded before it, and piping
# the run through `tee`/`tail` discards the exit code entirely.  That is
# exactly how `compile_fuzzgoat_object`'s `return 1` on an absent optional
# vendor tree came to abort every build on a machine without vendor/fuzzgoat
# -- after grep_read and before ALL of the .so targets, perf_shim and the
# entire verify pass, with a green OK as the final line.  Report aborts loudly
# and locate them.  `set -E` propagates the trap into functions and subshells.
set -E
trap 'rc=$?; printf "\n\033[0;31mFAIL\033[0m: build aborted (exit %d) at %s:%d\n       command: %s\n" \
      "$rc" "${BASH_SOURCE[0]}" "$LINENO" "$BASH_COMMAND" >&2' ERR

if [ -z "${TMPDIR:-}" ]; then
    mkdir -p /home/dclavijo/tmp
    export TMPDIR=/home/dclavijo/tmp
fi

FGREP="${FGREP_DIR:-vendor/fgrep}"
TAILSLAYER="${TAILSLAYER_DIR:-/home/dclavijo/code/tailslayer}"
SHIM="src/fuzzer_tool/adapters/afl_shim.c"
# Comparison logging lives in $SHIM behind -D__AFL_CMPLOG=1 (it used to be a
# separate cmplog_shim.c compiled to its own object and linked in). $CMPLOG_CFLAGS
# is what turns it on; $CMPLOG_LIBS is the -ldl its dlsym(RTLD_NEXT) layer needs.
CMPLOG_CFLAGS="-D__AFL_CMPLOG=1"
CMPLOG_LIBS="-ldl"
PERF_SHIM="src/fuzzer_tool/adapters/perf_shim.c"
TARGETS="targets"
VENDOR="vendor"
LZ4="${LZ4_DIR:-vendor/lz4}"
SECP256K1="${SECP256K1_DIR:-vendor/secp256k1}"
SQLITE="${SQLITE_DIR:-vendor/sqlite}"
# Shared by compile_sqlite_objects and the sqlite_read.so link: the wrapper
# includes sqlite3.h, and a header parsed under different SQLITE_* defines
# than the library it links against is the classic silent-ABI-mismatch bug
# (sqlite3_int64 widths, omitted APIs), so both sides use this one list.
SQLITE_DEFINES="-DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION=1 \
-DSQLITE_ENABLE_DESERIALIZE -DSQLITE_ENABLE_FTS4 -DSQLITE_ENABLE_FTS5 \
-DSQLITE_ENABLE_RTREE -DSQLITE_ENABLE_GEOPOLY -DSQLITE_ENABLE_DBSTAT_VTAB \
-DSQLITE_ENABLE_STMTVTAB"
TARGETS_MD5=".target.md5"
OPTS="${@:---all}"
HAS_FGREP=0
[ -d "$FGREP/src" ] && HAS_FGREP=1
HAS_FUZZGOAT=0
[ -f "$VENDOR/fuzzgoat/fuzzgoat.c" ] && HAS_FUZZGOAT=1
WITH_CMPLOG=1  # default: cmplog linked into .so targets
WITH_TRACECMP=1  # default: compiler-IR comparison tracing (needs clang)

# ── Keeping comparison constants visible at -O2 ──────────────────────
#
# At -O2 clang folds memcmp/strcmp against compile-time constants into
# inline integer compares, so nothing reaches the LD_PRELOAD shim.
# Measured on targets/cmplog_exercise.c, seed "AAAAAAAAAAAAAAAA", counting
# how many of its 10 magic constants ("CMPl", "OG!", "fuzz", ...) reach the
# cmplog pair pool:
#
#   -O2                                          0/10   (5 operands)
#   -O2 -fsanitize-coverage=trace-cmp, preloaded 0/10   (5 operands)
#   -O2 -fsanitize-coverage=trace-cmp, linked    0/10  (12 operands)
#   -O0                                          9/10  (21 operands)
#   -O2 $NOBUILTIN_CMP                          10/10  (24 operands)
#   -O2 $NOBUILTIN_CMP + trace-cmp, linked      10/10  (36 operands)
#
# trace-cmp alone does NOT recover them, whatever the optimization level:
# SanitizerCoverage instruments IR `icmp`, and clang's ExpandMemCmp is a
# CodeGen pass that runs *after* it. So the compare trace-cmp sees is
# `memcmp_result == 0` -- it logs the literal pair (0, 1), and only later
# does the memcmp become `cmpl $0x6C504D43,(%rbx)`. On the -O2 trace-cmp
# build 11 of 20 logged pairs were that degenerate (0, 1).
#
# -fno-builtin-<fn> is what actually works: it keeps the call at the PLT so
# the shim intercepts it, while leaving every other -O2 optimization on.
# The two are complementary, not alternatives -- trace-cmp still catches
# genuine inline integer compares and switch dispatch, which the libc layer
# cannot see at all. Together they beat -O0 on operand count by 71%.
NOBUILTIN_CMP="-fno-builtin-memcmp -fno-builtin-bcmp -fno-builtin-strcmp"
NOBUILTIN_CMP="$NOBUILTIN_CMP -fno-builtin-strncmp -fno-builtin-strcasecmp"
NOBUILTIN_CMP="$NOBUILTIN_CMP -fno-builtin-strncasecmp -fno-builtin-memchr"
NOBUILTIN_CMP="$NOBUILTIN_CMP -fno-builtin-strstr -fno-builtin-memmem"
# strcasestr is intercepted by the cmplog layer but was missing here, so any
# target clang chose to fold it in was invisible to the libc layer. This list
# and the interceptor list in afl_shim.c must be kept in step.
NOBUILTIN_CMP="$NOBUILTIN_CMP -fno-builtin-strcasestr"
NOBUILTIN_CMP="$NOBUILTIN_CMP -fno-builtin-wmemcmp -fno-builtin-wcscmp"
NOBUILTIN_CMP="$NOBUILTIN_CMP -fno-builtin-wcsncmp -fno-builtin-wcscasecmp"
NOBUILTIN_CMP="$NOBUILTIN_CMP -fno-builtin-strpbrk -fno-builtin-strspn"
NOBUILTIN_CMP="$NOBUILTIN_CMP -fno-builtin-strcspn -fno-builtin-memrchr"

# ── Frame pointers: required by caller-context edge hashing ──────────────
#
# afl_shim.c's __afl_get_caller_ctx() (default-on via __AFL_CTX_SENSITIVE=1)
# computes edge_id = hash(__builtin_return_address(1)) ^ prev_loc ^ cur_loc.
# return_address(1) walks one frame past trace_pc_guard's own frame to reach
# the call site of whoever called the function the edge lives in -- and that
# function is in the LIBRARY, not in the shim's translation unit. So the
# frame-pointer requirement lands on every vendored archive we link, not
# just on the TU that does `-include afl_shim.c`.
#
# Every vendored library here was built at -O2 without this flag while
# build_simple_so_targets linked them into _nosan targets with CTX enabled.
# The comment there asserted the archives were "rebuilt under the same
# contract"; only the --vendor-tracecmp path actually was.
#
# The failure mode is not reliably a crash. Measured on the two-TU minimal
# case (library TU at -O2, shim TU with frame pointers):
#
#   library -fno-omit-frame-pointer -> return_address(1) = 0x55555555506d
#                                      (correct: inside the real caller)
#   library -fomit-frame-pointer    -> return_address(1) = 0x7ffff7c2a1ca
#                                      (wrong: a libc address, one frame skipped)
#
# Neither run crashed. A silently wrong context is worse than the SEGV in
# docs/learnings/2026-08-11-rebuild-shim-ctx-segv.md, because it yields
# stable-but-meaningless edge IDs: distinct call chains collapse together
# and unrelated ones separate, so coverage counts rise while meaning falls.
# The SEGV only appears when the bogus rbp happens to be unmapped.
#
# Applied unconditionally rather than gated on CTX: the cost is one register
# at -O2, the archives are shared between CTX and non-CTX targets, and a
# stale archive built without it is invisible at link time. It also keeps
# stack traces walkable for the crash reports (see build_c_targets).
FRAME_POINTER="-fno-omit-frame-pointer"

WITH_VENDOR_TRACECMP=0
WITH_CLANG_SCOV=0
WITH_DISTANCE=0
WITH_NGRAM=0
WITH_MSAN=0
WITH_TSAN=0
WITH_FFMPEG_SANCOV=1  # auto-rebuild vendored FFmpeg with coverage if needed
USE_CLANG=0

# Parse flags (can appear anywhere)
for arg in "$@"; do
    [ "$arg" = "--cmplog" ] && WITH_CMPLOG=1
    # --tracecmp implies --cmplog (the unified shim covers both layers)
    [ "$arg" = "--tracecmp" ] && WITH_CMPLOG=1 && WITH_TRACECMP=1
    [ "$arg" = "--no-tracecmp" ] && WITH_TRACECMP=0
    [ "$arg" = "--vendor-tracecmp" ] && WITH_VENDOR_TRACECMP=1
    [ "$arg" = "--clang-scov" ] && WITH_CLANG_SCOV=1
    [ "$arg" = "--ffmpeg-sancov" ] && WITH_FFMPEG_SANCOV=1
    [ "$arg" = "--distance" ] && WITH_DISTANCE=1
    [ "$arg" = "--ngram" ] && WITH_NGRAM=1
    [ "$arg" = "--msan" ] && WITH_MSAN=1
    [ "$arg" = "--tsan" ] && WITH_TSAN=1
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok() { echo -e "  ${GREEN}OK${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

# ── Target checksum tracking (.target.md5, gitignored) ──────────────
# Records MD5 of every built binary and verifies them at the end.
# Escape a path for use in a POSIX basic/extended regex. Unescaped dots in
# a filename ("png_read.so") would otherwise match any character.
_md5_re() { printf '%s' "$1" | sed 's/[][\\.^$*+?(){}|/]/\\&/g'; }

# Snapshot the previous checksums, then start a fresh record.
#
# The truncation is what makes "removed" detectable at all: while the record
# was only appended to or replaced in place, it stayed a permanent superset
# of the baseline, so verify_target_md5's removed branch could never fire.
# Kept as a function rather than inline in Main so the invariant is testable
# without running a build.
snapshot_target_md5() {
    if [ -f "$TARGETS_MD5" ]; then
        cp -f "$TARGETS_MD5" "${TARGETS_MD5}.prev"
    fi
    : > "$TARGETS_MD5"
}

record_target_md5() {
    local file="$1"
    [ -f "$file" ] || return 0
    local md5 rel rel_re
    md5=$(md5sum "$file" | awk '{print $1}')
    rel="${file#./}"
    rel_re=$(_md5_re "$rel")
    # Drop any earlier entry for this path, so a target built twice in one
    # run (different flags, same output name) records its final state.
    if [ -f "$TARGETS_MD5" ]; then
        sed -i "\|^[0-9a-f]\+  ${rel_re}\$|d" "$TARGETS_MD5"
    fi
    echo "${md5}  ${rel}" >> "$TARGETS_MD5"
}

verify_target_md5() {
    local baseline="${TARGETS_MD5}.prev"
    if [ ! -f "$TARGETS_MD5" ]; then
        rm -f "$baseline"
        return 0
    fi
    echo "Verifying target checksums..."
    local new_count=0 changed_count=0 unchanged_count=0
    local removed_count=0 stale_count=0 kept_count=0
    local current_md5 rel rel_re stored_md5 prev_md5 line

    # Pass 1: every target built this run, against the previous build.
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        stored_md5=$(echo "$line" | awk '{print $1}')
        rel=$(echo "$line" | awk '{print $2}')
        [ -f "$rel" ] || continue
        rel_re=$(_md5_re "$rel")
        current_md5=$(md5sum "$rel" | awk '{print $1}')

        # The recorded checksum must still match the file. This is the only
        # check that is actually a *verification* rather than a diff against
        # the last build: it catches a binary replaced after it was built.
        if [ "$current_md5" != "$stored_md5" ]; then
            stale_count=$((stale_count + 1))
            warn "$rel: on-disk checksum does not match the recorded one"
        fi

        if [ -f "$baseline" ] && grep -qE "^[0-9a-f]+  ${rel_re}\$" "$baseline"; then
            prev_md5=$(grep -E "^[0-9a-f]+  ${rel_re}\$" "$baseline" | head -n1 | awk '{print $1}')
            if [ "$current_md5" = "$prev_md5" ]; then
                unchanged_count=$((unchanged_count + 1))
            else
                changed_count=$((changed_count + 1))
                warn "$rel: checksum changed"
            fi
        else
            new_count=$((new_count + 1))
            ok "$rel: new binary"
        fi
    done < "$TARGETS_MD5"

    # Pass 2: baseline entries this run did not rebuild.
    #
    # This is only meaningful because the record is truncated at the start
    # of a build (see Main). Before that it was append-or-replace and never
    # cleared, so the record was always a superset of the baseline and the
    # "removed" branch below could not fire at all.
    #
    # Not rebuilt is not the same as gone. A partial build (--asan alone)
    # legitimately skips most targets, so an entry whose binary still exists
    # is carried forward rather than dropped; only a vanished binary is
    # worth a warning.
    if [ -f "$baseline" ]; then
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            rel=$(echo "$line" | awk '{print $2}')
            rel_re=$(_md5_re "$rel")
            grep -qE "^[0-9a-f]+  ${rel_re}\$" "$TARGETS_MD5" 2>/dev/null && continue
            if [ -f "$rel" ]; then
                kept_count=$((kept_count + 1))
                echo "$line" >> "$TARGETS_MD5"
            else
                removed_count=$((removed_count + 1))
                warn "$rel: was built previously, binary is gone"
            fi
        done < "$baseline"
        rm -f "$baseline"
    fi

    ok "$unchanged_count targets unchanged"
    [ "$new_count" -gt 0 ] && ok "$new_count new targets"
    [ "$kept_count" -gt 0 ] && ok "$kept_count not rebuilt this run (carried forward)"
    [ "$changed_count" -gt 0 ] && warn "$changed_count targets changed"
    [ "$removed_count" -gt 0 ] && warn "$removed_count targets removed from build"
    [ "$stale_count" -gt 0 ] && warn "$stale_count targets modified after build"
    echo ""
    return 0
}

# ── Vendored libpng / zlib selection ────────────────────────────────
# Sets PNG_LIBS, ZLIB_LIBS, GZIP_LIBS, PNG_INC, ZLIB_INC in the caller's
# scope; callers declare them `local` first.
#
# One function rather than a copy per build mode. There were three, and
# within a day of being written they had already drifted: two omitted -lm
# on ZLIB_LIBS, two omitted -lpthread on PNG_LIBS, all three left GZIP_LIBS
# on the system library while handing gzip_read vendored headers, and only
# one said out loud which library it had picked.
#
# The link lines are the superset of what the three used. The extra
# -lm/-lpthread are no-ops where unneeded (glibc 2.34+ folds libpthread
# into libc), which is the cheaper error than a missing symbol in one build
# mode only.
select_png_zlib_libs() {
    local vendor_zlib="$VENDOR/zlib/libz.a"
    local vendor_png="$VENDOR/libpng/.libs/libpng16.a"

    PNG_LIBS="-lpng -lz"
    ZLIB_LIBS="-lz"
    GZIP_LIBS="-lz"
    PNG_INC=""
    ZLIB_INC=""

    if [ -f "$vendor_zlib" ]; then
        # gzip_read and zlib_read share ZLIB_INC, so they must share the
        # library: inflateInit2 bakes the header's ZLIB_VERSION into a
        # runtime check against whatever libz is linked, and a mismatch is
        # Z_VERSION_ERROR on every input with no build or runtime error.
        ZLIB_LIBS="$vendor_zlib -lm"
        GZIP_LIBS="$vendor_zlib -lm"
        ZLIB_INC="-I$VENDOR/zlib"
    fi

    if [ -f "$vendor_png" ] && [ -f "$vendor_zlib" ]; then
        PNG_LIBS="$vendor_png $vendor_zlib -lm -lpthread"
        PNG_INC="-I$VENDOR/libpng -I$VENDOR/libpng/scripts"
    fi

    # Always announce the choice. Whether vendor/ is populated changes what
    # a benchmark number means, and tools/eval_set.py pins targets/png_read
    # as a locked cell, so this must not be silent.
    local png_src="system" zlib_src="system"
    [ -f "$vendor_png" ] && [ -f "$vendor_zlib" ] && png_src="vendored"
    [ -f "$vendor_zlib" ] && zlib_src="vendored"
    echo "  libpng: $png_src, zlib: $zlib_src"
}

# ── Default compiler: prefer clang, fall back to gcc ────────────────
#
# clang is the default because it is the only compiler that can produce
# full automatic edge coverage here. Measured on targets/png_read.c:
#
#   clang -fsanitize-coverage=trace-pc-guard  -> 192 call sites, 127 bitmap slots
#   gcc   (manual __afl_map_edge only)        ->   0 call sites,  43 bitmap slots
#
# gcc's -fsanitize-coverage= accepts only trace-pc and trace-cmp, not the
# trace-pc-guard variant the AFL shim's edge callbacks are built on. The one
# gcc-compatible callback the shim implements, __sanitizer_cov_trace_pc(), is
# compiled only under __AFL_DISTANCE_MODE and depends on the AFLGo distance
# SHM — without it, gcc builds link but crash at runtime. So gcc targets fall
# back to the hand-placed __afl_map_edge() calls in the target wrappers, which
# see the wrapper's own branching but not the library internals underneath.
#
# gcc still builds every target correctly and is a fine fallback; it just
# yields shallower coverage. See README "Feature Compatibility Matrix".
_pick_cc() {
    if command -v clang &>/dev/null; then
        echo "clang"
    else
        # >&2 matters: warn() writes to stdout, and this function's whole
        # stdout is captured by DEFAULT_CC="$(_pick_cc)". Without the
        # redirect, DEFAULT_CC on a clang-less box becomes the warning text
        # *and* "gcc" — a two-line value that is not a command, so every
        # `$cc ...` compile fails with "command not found". Each compile
        # helper redirects stderr to /dev/null, so the only symptom is a
        # string of "objects failed" warnings and silently missing targets.
        warn "clang not found, falling back to gcc (shallower edge coverage — see README)" >&2
        echo "gcc"
    fi
}
DEFAULT_CC="$(_pick_cc)"

# ── Compile perf_shim.so ─────────────────────────────────────────
compile_perf_shim() {
    local cc="${1:-$DEFAULT_CC}"
    local out="src/fuzzer_tool/adapters/perf_shim.so"
    if [ ! -f "$PERF_SHIM" ]; then
        warn "perf_shim.c not found, skipping"
        return 1
    fi
    local rc=0
    $cc -O2 -g -shared -fPIC -o "$out" "$PERF_SHIM" 2>/dev/null || rc=$?
    if [ $rc -eq 0 ]; then
        ok "perf_shim.so"
    else
        warn "failed: perf_shim.so"
    fi
}

# ── Compile fgrep library objects ──────────────────────────────────
compile_fgrep_objects() {
    local suffix="$1" flags="$2" cc="${3:-$DEFAULT_CC}" extra_cflags="${4:-}"
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

# ── Compile lz4 library objects ────────────────────────────────────
# Vendored via tools/vendor_lz4.sh (extracts to vendor/lz4/).
#
# Compiled WITHOUT `-include $SHIM`: the shim goes only into the target
# wrapper (Hard Rule 8). `-include` applies to every .c on a command line,
# so compiling these alongside lz4_read.c would emit __afl_map_shm /
# __afl_area / __afl_guarded_call into all five objects and fail the link
# with multiple-definition errors.
# -DXXH_NAMESPACE=LZ4_ matches how upstream LZ4 builds its bundled xxhash;
# without it the xxhash symbols collide if anything else in the link pulls
# in its own copy.
compile_lz4_objects() {
    local suffix="$1" flags="$2" cc="${3:-$DEFAULT_CC}" extra_cflags="${4:-}"
    [ -d "$LZ4/lib" ] || return 1
    echo "Compiling lz4 objects${suffix:+ ($suffix)}..."
    local rc=0
    for src in lz4 lz4frame lz4hc xxhash; do
        $cc $flags -fPIC -O2 -g $extra_cflags -DXXH_NAMESPACE=LZ4_ -I"$LZ4/lib" \
            -c "$LZ4/lib/${src}.c" -o "/tmp/${src}${suffix}.o" 2>/dev/null || rc=$?
    done
    if [ $rc -eq 0 ]; then
        ok "lz4 objects${suffix:+ ($suffix)}"
    else
        warn "lz4 objects${suffix:+ ($suffix)} failed"
        return 1
    fi
}

# ── Compile secp256k1 library objects ─────────────────────────────
# Vendored via tools/vendor_secp256k1.sh (extracts to vendor/secp256k1/).
# libsecp256k1 is a plain source drop: the precomputed ECMULT tables are
# committed (src/precomputed_ecmult{,_gen}.c) and the wide-mul implementation
# auto-detects in src/util.h, so no configure step is needed. Modules (ecdh,
# recovery, extrakeys, schnorrsig, musig, ellswift, silentpayments) are
# #ifdef-gated header includes inside secp256k1.c — every module whose
# sources exist in the tree is enabled, giving the target the full API
# surface.
# Same shim discipline as the lz4 objects: compiled WITHOUT `-include $SHIM`
# (Hard Rule 8) — that flag applies to every .c on a command line, so
# compiling these alongside secp256k1_read.c would emit __afl_map_shm /
# __afl_area / __afl_guarded_call into all three objects and fail the link
# with multiple-definition errors.
compile_secp256k1_objects() {
    local suffix="$1" flags="$2" cc="${3:-$DEFAULT_CC}" extra_cflags="${4:-}"
    [ -d "$SECP256K1/src" ] || return 1
    echo "Compiling secp256k1 objects${suffix:+ ($suffix)}..."
    local rc=0
    local module_flags=""
    for m in ecdh recovery extrakeys schnorrsig musig ellswift silentpayments; do
        [ -d "$SECP256K1/src/modules/$m" ] && \
            module_flags="$module_flags -DENABLE_MODULE_$(echo "$m" | tr '[:lower:]' '[:upper:]')"
    done
    # Coverage instrumentation for the library objects: trace-pc-guard inserts
    # __sanitizer_cov_trace_pc_guard calls into every basic block without
    # emitting __afl_map_shm / __afl_area (those stay in the wrapper only via
    # -include $SHIM). This gives real library-level coverage without the
    # multiple-definition errors that -include $SHIM would cause.
    local cov_flag="-fsanitize-coverage=trace-pc-guard"
    for src in secp256k1 precomputed_ecmult precomputed_ecmult_gen; do
        $cc $flags $cov_flag -fPIC -O2 -g $extra_cflags $module_flags \
            -I"$SECP256K1/src" -I"$SECP256K1/include" \
            -c "$SECP256K1/src/${src}.c" -o "/tmp/${src}${suffix}.o" 2>/dev/null || rc=$?
    done
    if [ $rc -eq 0 ]; then
        ok "secp256k1 objects${suffix:+ ($suffix)}"
    else
        warn "secp256k1 objects${suffix:+ ($suffix)} failed"
        return 1
    fi
}

# ── Compile the sqlite amalgamation object ─────────────────────────
# Vendored via tools/vendor_sqlite.sh (extracts to vendor/sqlite/).
#
# One 9 MB translation unit, so this is the slowest single compile in the
# script (~1 min at -O2) — the object is cached in $TMPDIR across runs by
# suffix, same as the lz4/secp256k1 objects.
#
# Same shim discipline as those (Hard Rule 8): compiled WITHOUT
# `-include $SHIM`, since that flag applies to every .c on a command line
# and compiling sqlite3.c alongside sqlite_read.c would emit
# __afl_map_shm / __afl_area / __afl_guarded_call into both objects and
# fail the link with multiple-definition errors.
#
# Build defines follow SQLITE_ENABLE_* choices SQLite's own fuzzers use:
#   THREADSAFE=0            — the fuzzer drives one connection on one thread;
#                             mutex work is pure overhead per exec
#   OMIT_LOAD_EXTENSION     — no dlopen surface from a hostile file
#   ENABLE_DESERIALIZE      — sqlite3_deserialize(), the DB-image entry the
#                             target uses; a no-op on >= 3.36 (on by default
#                             unless SQLITE_OMIT_DESERIALIZE), kept for
#                             older vendored versions
#   FTS4/FTS5/RTREE/GEOPOLY/DBSTAT/STMTVTAB
#                           — virtual tables reachable from a corrupt schema
#                             or from SQL text; excluding them would hide the
#                             module code from the fuzzer entirely
# Coverage: trace-pc-guard instruments every basic block of the library
# without emitting the shim's symbols, the same trick compile_secp256k1_objects
# uses — without it the map only ever sees the wrapper's own landmarks.
compile_sqlite_objects() {
    local suffix="$1" flags="$2" cc="${3:-$DEFAULT_CC}" extra_cflags="${4:-}"
    [ -f "$SQLITE/sqlite3.c" ] || return 1
    echo "Compiling sqlite amalgamation${suffix:+ ($suffix)} (this takes a minute)..."
    local rc=0
    # trace-pc-guard is clang-only (see _pick_cc): gcc's -fsanitize-coverage=
    # takes trace-pc and trace-cmp and errors out on this one, which would
    # fail the whole compile and drop the target on a gcc-only box. Under gcc
    # the object is built without it and coverage falls back to the wrapper's
    # hand-placed __afl_map_edge() landmarks — shallower, but a working
    # target beats a skipped one.
    local cov_flag=""
    case "$cc" in
        *clang*) cov_flag="-fsanitize-coverage=trace-pc-guard" ;;
    esac
    $cc $flags $cov_flag -fPIC -O2 -g $extra_cflags $SQLITE_DEFINES -I"$SQLITE" \
        -c "$SQLITE/sqlite3.c" -o "/tmp/sqlite3${suffix}.o" 2>/dev/null || rc=$?
    if [ $rc -eq 0 ]; then
        ok "sqlite object${suffix:+ ($suffix)}"
    else
        warn "sqlite object${suffix:+ ($suffix)} failed"
        return 1
    fi
}

# ── Compile fuzzgoat library object ────────────────────────────────
# Fuzzgoat is vendored as source. To avoid multiple-definition errors with
# the AFL shim (which is injected via `-include $SHIM` into the wrapper TU
# only), compile fuzzgoat.c separately without the shim and link the object.
compile_fuzzgoat_object() {
    local flags="$1" cc="${2:-$DEFAULT_CC}" extra_cflags="${3:-}"
    # return 0, not 1: every other optional-vendor guard in this script
    # does the same, and under `set -e` a non-zero return from here aborts
    # the whole build.  Callers gate on $HAS_FUZZGOAT.
    [ -f "$VENDOR/fuzzgoat/fuzzgoat.c" ] || return 0
    $cc $flags -O2 -g $extra_cflags -I"$VENDOR/fuzzgoat" \
        -c "$VENDOR/fuzzgoat/fuzzgoat.c" -o /tmp/fuzzgoat.o 2>/dev/null
}

# ── Build a target ────────────────────────────────────────────────
build_target() {
    local src="$1" out="$2" libs="$3" extra_flags="$4" cc="${5:-$DEFAULT_CC}" extra_cflags="${6:-}"
    if [ ! -f "$src" ]; then
        warn "Source not found: $src"
        return 1
    fi
    local cmplog_cflags="" cmplog_libs=""
    # Vendored libraries (libpng/zlib/ffmpeg) are compiled with comparison
    # tracing, so their objects reference __sanitizer_cov_trace_const_cmp*;
    # an executable link needs a provider for those (a .so link tolerates
    # undefined symbols). Mirror build_so_target. Still excluded for
    # MSAN/TSAN: leaving the previous behaviour alone rather than assuming
    # the in-TU layer is safe there without measuring it.
    if [ "$WITH_CMPLOG" -eq 1 ] \
        && [[ "$extra_flags" != *-fsanitize=memory* ]] \
        && [[ "$extra_flags" != *-fsanitize=thread* ]]; then
        cmplog_cflags="$CMPLOG_CFLAGS"
        cmplog_libs="$CMPLOG_LIBS"
    fi
    local rc=0
    # FRAME_POINTER: ctx hashing is default-on in afl_shim.c; see the header.
    $cc $extra_flags -O2 -g $FRAME_POINTER $extra_cflags $cmplog_cflags -include "$SHIM" \
        -o "$out" "$src" $libs $cmplog_libs 2>/dev/null || rc=$?
    if [ $rc -eq 0 ]; then
        ok "$(basename "$out")"
        record_target_md5 "$out"
    else
        warn "failed: $(basename "$out")"
    fi
}

# ── Build a .so target ───────────────────────────────────────────
build_so_target() {
    local src="$1" out="$2" libs="$3" extra_flags="$4" cc="${5:-$DEFAULT_CC}" extra_cflags="${6:-}"
    local cmplog_cflags="" cmplog_libs=""
    if [ ! -f "$src" ]; then
        warn "Source not found: $src"
        return 1
    fi
    if [ "$WITH_CMPLOG" -eq 1 ]; then
        cmplog_cflags="$CMPLOG_CFLAGS"
        cmplog_libs="$CMPLOG_LIBS"
    fi
    # When ASAN is enabled, explicitly link libasan so it's resolved at load time.
    # clang -shared -fsanitize=address does NOT add NEEDED libasan.so.8 (unlike gcc),
    # leaving ASAN symbols unresolved and breaking ctypes.CDLL loading.
    if [[ "$extra_flags" == *-fsanitize=address* ]]; then
        libs="$libs -lasan"
    fi
    local rc=0
    # Do NOT add -fsanitize-coverage=trace-cmp to the target's own code —
    # the target wrapper (png_read.c etc.) has almost no comparisons; all
    # the interesting comparisons are in the vendored libraries which are
    # compiled separately with trace-cmp. Adding it here just adds overhead
    # for zero benefit.
    # -Bsymbolic: prevents ASAN's LD_PRELOAD from overriding the trace-cmp
    # callbacks with no-ops (ASAN ships weak stubs that shadow our shim).
    # The callbacks are also compiled hidden now, which is the stronger
    # guarantee; -Bsymbolic is kept because it costs nothing and still
    # covers the __afl_* symbols.
    local bsymbolic_flag=""
    local target_cc="$cc"
    if [ "$WITH_CMPLOG" -eq 1 ]; then
        bsymbolic_flag="-Wl,-Bsymbolic"
        if command -v clang &>/dev/null; then
            target_cc="clang"
        fi
    fi
    # FRAME_POINTER: ctx hashing is default-on in afl_shim.c; see the header.
    $target_cc $extra_flags -O2 -g $FRAME_POINTER $extra_cflags $cmplog_cflags -shared -fPIC $bsymbolic_flag -include "$SHIM" \
        -o "$out" "$src" $libs $cmplog_libs 2>/dev/null || rc=$?
    if [ $rc -eq 0 ]; then
        ok "$(basename "$out")"
        record_target_md5 "$out"
    else
        warn "failed: $(basename "$out")"
    fi
}

# ── Build fgrep targets ──────────────────────────────────────────
# The former fuzz_regex_compile / fuzz_pattern_match / fuzz_search_pipeline
# wrappers were consolidated into fgrep_read.c modes 4/5/6 (071a67f).
build_fgrep_targets() {
    local suffix="$1" flags="$2" label="$3"
    echo "Building fgrep targets ($label)..."
    local FGREP_INC="-I$FGREP/include -I$FGREP/src"

    local out_suffix=""
    [ "$suffix" = "_nosan" ] && out_suffix="_nosan"
    # fgrep_read includes fgrep .c files directly and needs -mavx2 for AVX2 intrinsics
    build_target "$TARGETS/fgrep_read.c" "$TARGETS/fgrep_read${out_suffix}" "$FGREP_INC -lpthread" "$flags -mavx2"
}

# ── Build fgrep .so targets ─────────────────────────────────────
build_fgrep_so_targets() {
    local suffix="$1" flags="$2" label="$3"
    echo "Building fgrep .so targets ($label)..."
    local FGREP_INC="-I$FGREP/include -I$FGREP/src"

    local out_suffix=""
    [[ "$suffix" == _asan* ]] && out_suffix="$suffix"
    # fgrep_read includes fgrep .c files directly — needs -mavx2 for AVX2 intrinsics
    build_so_target "$TARGETS/fgrep_read.c" "$TARGETS/fgrep_read${out_suffix}.so" "$FGREP_INC -lpthread" "$flags -mavx2"
}

# ── Build simple targets ─────────────────────────────────────────
build_simple_targets() {
    local suffix="$1" flags="$2" label="$3" cc="${4:-$DEFAULT_CC}" extra_cflags="${5:-}"
    echo "Building simple targets ($label)..."
    local out_suffix=""
    [ "$suffix" = "_nosan" ] && out_suffix="_nosan"

    local FFMPEG_LIBS="-lavformat -lavcodec -lavutil -lswresample -lm"
    local FFMPEG_INC="-I/usr/include/x86_64-linux-gnu"
    local VENDOR_FFMPEG_A="$VENDOR/ffmpeg/libavformat/libavformat.a"
    if [ -f "$VENDOR_FFMPEG_A" ]; then
        # Prefer the vendored tree (coverage-built; system ffmpeg headers are
        # not installed on every box), picking the ASAN-instrumented tree for
        # sanitizer passes.
        local ffmpeg_root="$VENDOR/ffmpeg"
        if [[ "$flags" == *-fsanitize=address* ]] && [ -f "$VENDOR/ffmpeg_asan/libavformat/libavformat.a" ]; then
            ffmpeg_root="$VENDOR/ffmpeg_asan"
        fi
        FFMPEG_LIBS="$ffmpeg_root/libavformat/libavformat.a $ffmpeg_root/libavcodec/libavcodec.a $ffmpeg_root/libavutil/libavutil.a $ffmpeg_root/libswresample/libswresample.a -lm -lz -llzma -lbz2 -lpthread -ldl"
        FFMPEG_INC="-I$ffmpeg_root"
    fi

    local PNG_LIBS ZLIB_LIBS GZIP_LIBS PNG_INC ZLIB_INC
    select_png_zlib_libs

    build_target "$TARGETS/asan_target.c" "$TARGETS/asan_target${out_suffix}" "" "$flags" "$cc" "$extra_cflags"
    build_target "$TARGETS/test_target.c" "$TARGETS/test_target${out_suffix}" "" "$flags" "$cc" "$extra_cflags"
    build_target "$TARGETS/proto_target.c" "$TARGETS/proto_target${out_suffix}" "" "$flags" "$cc" "$extra_cflags"
    build_target "$TARGETS/png_read.c" "$TARGETS/png_read${out_suffix}" "$PNG_LIBS" "$flags" "$cc" "$extra_cflags $PNG_INC"
    build_target "$TARGETS/zlib_read.c" "$TARGETS/zlib_read${out_suffix}" "$ZLIB_LIBS" "$flags" "$cc" "$extra_cflags $ZLIB_INC"
    build_target "$TARGETS/gzip_read.c" "$TARGETS/gzip_read${out_suffix}" "$GZIP_LIBS" "$flags" "$cc" "$extra_cflags $ZLIB_INC"
    build_target "$TARGETS/jpeg_read.c" "$TARGETS/jpeg_read${out_suffix}" "-ljpeg" "$flags" "$cc" "$extra_cflags"
    build_target "$TARGETS/ffmpeg_read.c" "$TARGETS/ffmpeg_read${out_suffix}" "$FFMPEG_LIBS" "$flags" "$DEFAULT_CC" "$FFMPEG_INC"
    build_target "$TARGETS/grep_read.c" "$TARGETS/grep_read${out_suffix}" "" "$flags"
    if [ "$HAS_FUZZGOAT" -eq 1 ]; then
        compile_fuzzgoat_object "$flags" "$cc" "$extra_cflags"
        build_target "$TARGETS/fuzzgoat_read.c" "$TARGETS/fuzzgoat_read${out_suffix}" "/tmp/fuzzgoat.o -lm" "$flags" "$cc" "-I$VENDOR/fuzzgoat"
    fi
}

# ── MSAN / TSAN standalone executables ─────────────────────────
# These deliberately build *executables only*, never .so targets:
#
#   MSAN reports uninitialized reads across the whole process, so every
#   piece of linked code must be MSAN-instrumented or it produces
#   false positives. A .so loaded via ctypes into an uninstrumented
#   CPython would report on Python's own allocations, making the mode
#   useless. Run through the subprocess/exec path instead.
#
#   TSAN likewise intercepts the whole runtime and does not compose with
#   being dlopen'd into an already-running interpreter.
#
# Both are clang-only (gcc's MSAN support does not exist; its TSAN is
# weaker) and both are opt-in via --msan / --tsan since they are slower
# than ASAN and only relevant to specific bug classes:
#   MSAN -> use-of-uninitialized-value (ASAN cannot see these at all)
#   TSAN -> data races (relevant for threaded targets)
#
# sanitizer.py already parses MemorySanitizer/ThreadSanitizer reports, so
# no runtime-side work is needed to consume these binaries.
build_sanitizer_targets() {
    local suffix="$1" flags="$2" label="$3"
    if ! command -v clang &>/dev/null; then
        warn "$label targets need clang — skipping"
        return 0
    fi
    echo "Building simple targets ($label)..."
    # -fno-omit-frame-pointer keeps stack traces usable in the reports that
    # sanitizer.py parses; -fPIE/-pie is required by MSAN.
    local common="$flags -fno-omit-frame-pointer -fPIE -pie"
    build_target "$TARGETS/asan_target.c" "$TARGETS/asan_target${suffix}" "" "$common" "clang"
    build_target "$TARGETS/test_target.c" "$TARGETS/test_target${suffix}" "" "$common" "clang"
    build_target "$TARGETS/proto_target.c" "$TARGETS/proto_target${suffix}" "" "$common" "clang"
    build_target "$TARGETS/grep_read.c" "$TARGETS/grep_read${suffix}" "" "$common" "clang"
    if [ "$HAS_FUZZGOAT" -eq 1 ]; then
        compile_fuzzgoat_object "$common" "clang" "-I$VENDOR/fuzzgoat"
        build_target "$TARGETS/fuzzgoat_read.c" "$TARGETS/fuzzgoat_read${suffix}" "/tmp/fuzzgoat.o -lm" "$common" "clang" "-I$VENDOR/fuzzgoat"
    fi
    # Targets linking uninstrumented system libraries (libpng/libz/libjpeg)
    # are intentionally omitted for MSAN: without an instrumented build of
    # those libraries every call into them reports a false uninitialized
    # read. Build them with --vendor-tracecmp-style vendored sources first
    # if you need MSAN coverage there.
    if [ "$label" != "MSAN" ]; then
        local PNG_LIBS ZLIB_LIBS GZIP_LIBS PNG_INC ZLIB_INC
        select_png_zlib_libs
        build_target "$TARGETS/png_read.c" "$TARGETS/png_read${suffix}" "$PNG_LIBS" "$common" "clang" "$PNG_INC"
        build_target "$TARGETS/zlib_read.c" "$TARGETS/zlib_read${suffix}" "$ZLIB_LIBS" "$common" "clang" "$ZLIB_INC"
        build_target "$TARGETS/gzip_read.c" "$TARGETS/gzip_read${suffix}" "$GZIP_LIBS" "$common" "clang" "$ZLIB_INC"
    fi
}

# ── Rebuild vendored FFmpeg with sancov coverage ───────────────
# Builds two variants:
#   suffix=""      →  vendor/ffmpeg/       (coverage only, no ASAN)
#   suffix="_asan" →  vendor/ffmpeg_asan/   (coverage + ASAN)
# FFmpeg's configure needs a stub for __sanitizer_cov_trace_pc_guard
# since those symbols are provided by cmplog_shim.o at final link.
build_vendored_ffmpeg_sancov() {
    local asan_suffix="${1:-}"  # "" or "_asan"
    [ "$WITH_FFMPEG_SANCOV" -eq 0 ] && return 0
    local SRC_DIR="$VENDOR/ffmpeg"
    local FFMPEG_DIR="$VENDOR/ffmpeg${asan_suffix}"
    [ -d "$SRC_DIR" ] || return 0
    # For the ASAN variant, copy source from the base if not already separate
    if [ "$asan_suffix" = "_asan" ] && [ ! -d "$FFMPEG_DIR" ]; then
        echo "  Copying FFmpeg source to vendor/ffmpeg_asan/..."
        cp -a "$SRC_DIR" "$FFMPEG_DIR"
    fi
    [ -d "$FFMPEG_DIR" ] || return 0
    # Check if FFmpeg libs already have desired instrumentation
    local has_cov=$(nm "$FFMPEG_DIR/libavformat/libavformat.a" 2>/dev/null | grep -c '__sanitizer_cov_trace_pc_guard' || true)
    [ "$has_cov" -gt 10 ] && return 0  # already instrumented

    local label="${asan_suffix:-" (nosan)"}"
    echo "Building vendored FFmpeg${label} with sancov coverage..."
    local cc="clang"
    if ! command -v clang &>/dev/null; then
        warn "clang not found — cannot rebuild FFmpeg with coverage"
        return 1
    fi
    # Create sancov stub for configure's link test
    local stub_dir="/tmp/ffcover_$$"
    mkdir -p "$stub_dir"
    cat > "$stub_dir/sancov_stub.c" << 'STUBEOF'
void __sanitizer_cov_trace_pc_guard_init(void *s, void *e) { (void)s; (void)e; }
void __sanitizer_cov_trace_pc_guard(void *g) { (void)g; }
void __sanitizer_cov_trace_cmp1(unsigned char a, unsigned char b) { (void)a; (void)b; }
void __sanitizer_cov_trace_cmp2(unsigned short a, unsigned short b) { (void)a; (void)b; }
void __sanitizer_cov_trace_cmp4(unsigned int a, unsigned int b) { (void)a; (void)b; }
void __sanitizer_cov_trace_cmp8(unsigned long long a, unsigned long long b) { (void)a; (void)b; }
void __sanitizer_cov_trace_const_cmp1(unsigned char a, unsigned char b) { (void)a; (void)b; }
void __sanitizer_cov_trace_const_cmp2(unsigned short a, unsigned short b) { (void)a; (void)b; }
void __sanitizer_cov_trace_const_cmp4(unsigned int a, unsigned int b) { (void)a; (void)b; }
void __sanitizer_cov_trace_const_cmp8(unsigned long long a, unsigned long long b) { (void)a; (void)b; }
void __sanitizer_cov_trace_switch(unsigned long long v, unsigned long long *r) { (void)v; (void)r; }
STUBEOF
    $cc -c -o "$stub_dir/sancov_stub.o" "$stub_dir/sancov_stub.c" 2>/dev/null
    ar rcs "$stub_dir/libsancov_stub.a" "$stub_dir/sancov_stub.o" 2>/dev/null
    local COV_FLAGS="-fsanitize-coverage=trace-pc-guard -fsanitize-coverage=trace-cmp"
    local LINK_FLAGS=""
    local EXTRA_LIBS="-lsancov_stub"
    if [ "$asan_suffix" = "_asan" ]; then
        COV_FLAGS="-fsanitize=address $COV_FLAGS"
        LINK_FLAGS="-fsanitize=address"
        EXTRA_LIBS="-lsancov_stub"
    fi
    (cd "$FFMPEG_DIR" && make clean >/dev/null 2>&1 || true)
    if (cd "$FFMPEG_DIR" && ./configure --cc="$cc" --extra-cflags="$COV_FLAGS" \
        --extra-ldflags="-L$stub_dir $LINK_FLAGS" --extra-libs="$EXTRA_LIBS" \
        --enable-static --disable-shared --disable-programs --disable-doc \
        --disable-encoders --disable-muxers --disable-devices --disable-filters \
        --disable-parsers --disable-bsfs --disable-postproc --disable-avdevice \
        --disable-pthreads --disable-network --disable-hwaccels --disable-cuvid \
        --disable-nvenc --disable-vaapi --disable-vdpau --disable-vulkan \
        >/dev/null 2>&1); then
        if (cd "$FFMPEG_DIR" && make -j$(nproc) -s >/dev/null 2>&1); then
            ok "vendored FFmpeg${label}"
        else
            warn "vendored FFmpeg${label} build failed"
        fi
    else
        warn "vendored FFmpeg${label} configure failed"
    fi
    rm -rf "$stub_dir"
}

# ── Build simple .so targets ────────────────────────────────────
build_simple_so_targets() {
    local suffix="$1" flags="$2" label="$3" cc="${4:-$DEFAULT_CC}" extra_cflags="${5:-}"
    echo "Building simple .so targets ($label)..."
    local out_suffix=""
    [[ "$suffix" == _asan* || "$suffix" == _ubsan* ]] && out_suffix="$suffix"

    # No-ASAN .so targets link the vendored libpng/zlib/ffmpeg archives and
    # enable call-stack-sensitive edge hashing. The context walk reaches into
    # those archives' frames at runtime, so they must carry frame pointers
    # too -- see FRAME_POINTER in the header of this file, which is applied
    # by compile_vendored_libs and by every tools/vendor_*.sh.
    #
    # The shim's ctx hashing and distance channel are default-on since the
    # afl_shim.c flip; the only thing the build must add is FRAME_POINTER on
    # every shim TU (build_so_target/build_target add it centrally), so the
    # context walk sees real callers instead of junk-or-zero frames.
    flags="$flags $FRAME_POINTER"

    # Prefer vendored static libraries when available. The vendored .a files
    # are compiled with -fsanitize-coverage=trace-pc-guard and
    # -fno-builtin-* so their comparisons remain visible to cmplog and
    # their edge coverage callbacks resolve against afl_shim.c / ASAN.
    local PNG_LIBS ZLIB_LIBS GZIP_LIBS PNG_INC ZLIB_INC
    select_png_zlib_libs

    local FFMPEG_LIBS="-lavformat -lavcodec -lavutil -lswresample -lm"
    local FFMPEG_INC="-I/usr/include/x86_64-linux-gnu"
    # Select vendored FFmpeg path based on suffix:
    #   _asan → vendor/ffmpeg_asan/ (ASAN + coverage)
    #   _ubsan / _nosan / "" → vendor/ffmpeg/ (coverage only)
    local ffmpeg_vendor_dir="$VENDOR/ffmpeg"
    [[ "$suffix" == _asan* ]] && ffmpeg_vendor_dir="$VENDOR/ffmpeg_asan"
    local VENDOR_FFMPEG_A="$ffmpeg_vendor_dir/libavformat/libavformat.a"
    if [ -f "$VENDOR_FFMPEG_A" ]; then
        FFMPEG_LIBS="$ffmpeg_vendor_dir/libavformat/libavformat.a $ffmpeg_vendor_dir/libavcodec/libavcodec.a $ffmpeg_vendor_dir/libavutil/libavutil.a $ffmpeg_vendor_dir/libswresample/libswresample.a -lm -lz -llzma -lbz2 -lpthread -ldl"
        FFMPEG_INC="-I$ffmpeg_vendor_dir"
        echo "  Using vendored FFmpeg static libraries ($ffmpeg_vendor_dir)"
    fi

    build_so_target "$TARGETS/asan_target.c" "$TARGETS/asan_target${out_suffix}.so" "" "$flags" "$cc" "$extra_cflags"
    build_so_target "$TARGETS/test_target.c" "$TARGETS/test_target${out_suffix}.so" "" "$flags" "$cc" "$extra_cflags"
    build_so_target "$TARGETS/proto_target.c" "$TARGETS/proto_target${out_suffix}.so" "" "$flags" "$cc" "$extra_cflags"
    build_so_target "$TARGETS/png_read.c" "$TARGETS/png_read${out_suffix}.so" "$PNG_LIBS" "$flags" "$cc" "$extra_cflags $PNG_INC"
    build_so_target "$TARGETS/zlib_read.c" "$TARGETS/zlib_read${out_suffix}.so" "$ZLIB_LIBS" "$flags" "$cc" "$extra_cflags $ZLIB_INC"
    build_so_target "$TARGETS/gzip_read.c" "$TARGETS/gzip_read${out_suffix}.so" "$GZIP_LIBS" "$flags" "$cc" "$extra_cflags $ZLIB_INC"
    build_so_target "$TARGETS/jpeg_read.c" "$TARGETS/jpeg_read${out_suffix}.so" "-ljpeg" "$flags" "$cc" "$extra_cflags"
    build_so_target "$TARGETS/nop_target.c" "$TARGETS/nop_target${out_suffix}.so" "" "$flags" "$cc" "$extra_cflags"
    build_so_target "$TARGETS/ffmpeg_read.c" "$TARGETS/ffmpeg_read${out_suffix}.so" "$FFMPEG_LIBS" "$flags" "$cc" "$extra_cflags $FFMPEG_INC"
    build_so_target "$TARGETS/grep_read.c" "$TARGETS/grep_read${out_suffix}.so" "" "$flags" "$cc" "$extra_cflags"
    if [ "$HAS_FUZZGOAT" -eq 1 ]; then
        compile_fuzzgoat_object "$flags" "$cc" "-I$VENDOR/fuzzgoat"
        build_so_target "$TARGETS/fuzzgoat_read.c" "$TARGETS/fuzzgoat_read${out_suffix}.so" "/tmp/fuzzgoat.o -lm" "$flags" "$cc" "-I$VENDOR/fuzzgoat"
    fi
}

# ── Build standalone .so targets with external deps ─────────────
build_standalone_so_targets() {
    local suffix="$1" flags="$2" label="$3" cc="${4:-$DEFAULT_CC}" extra_cflags="${5:-}"
    local out_suffix=""
    [[ "$suffix" == _asan* || "$suffix" == _ubsan* ]] && out_suffix="$suffix"

    # tailslayer — C++ target (g++), header-only library
    if [ -f "$TARGETS/tailslayer_read.cpp" ] && [ -d "$TAILSLAYER/include" ]; then
        local cxx=g++
        if command -v g++ &>/dev/null; then
            local src="$TARGETS/tailslayer_read.cpp"
            local out="$TARGETS/tailslayer_read${out_suffix}.so"
            local inc="-I$TAILSLAYER/include"
            local cmplog_cflags="" cmplog_libs=""
            # This one is a C++ TU. The cmplog layer's memchr/strstr/strcasestr
            # interceptors use the C signatures; C++ overloads the const-ness,
            # so keep it off here rather than fight the declaration mismatch.
            # (It was already effectively separate: the old build compiled the
            # shim as its own C object precisely to dodge this.)
            local bsym_flag=""
            [ "$WITH_CMPLOG" -eq 1 ] && bsym_flag="-Wl,-Bsymbolic"
            $cxx $flags -O2 -g $FRAME_POINTER -shared -fPIC $bsym_flag -include "$SHIM" $inc \
                -o "$out" "$src" $cmplog_cflags $cmplog_libs 2>/dev/null && ok "tailslayer_read${out_suffix}.so" || warn "failed: tailslayer_read${out_suffix}.so"
        fi
    elif [ -f "$TARGETS/tailslayer_read.cpp" ] && [ ! -d "$TAILSLAYER/include" ]; then
        warn "tailslayer_read${out_suffix}.so: tailslayer headers not found at $TAILSLAYER/include, skipping"
    fi

    # lz4_read — vendored LZ4 (tools/vendor_lz4.sh extracts to vendor/lz4).
    # The library objects are compiled here with the same sanitizer flags as
    # the wrapper, so ASAN variants instrument the decoder itself rather than
    # just the wrapper around it. They are built WITHOUT `-include $SHIM`:
    # that flag applies to every .c on a command line, so compiling the
    # vendored sources alongside lz4_read.c would emit __afl_map_shm /
    # __afl_area / __afl_guarded_call into all five objects and fail the
    # link with multiple-definition errors (Hard Rule 8).
    local LZ4_OBJS="/tmp/lz4$suffix.o /tmp/lz4frame$suffix.o /tmp/lz4hc$suffix.o /tmp/xxhash$suffix.o"
    local LZ4_INC="-I$LZ4/lib -DXXH_NAMESPACE=LZ4_"
    if [ ! -f "$TARGETS/lz4_read.c" ]; then
        :  # target source absent — nothing to build
    elif compile_lz4_objects "$suffix" "$flags" "$DEFAULT_CC"; then
        build_so_target "$TARGETS/lz4_read.c" "$TARGETS/lz4_read${out_suffix}.so" "$LZ4_OBJS -Wl,--export-dynamic -lpthread" "$flags $LZ4_INC"
    else
        warn "lz4_read${out_suffix}.so: vendor/lz4 not found, skipping (run tools/vendor_lz4.sh)"
    fi

    # secp256k1_read — vendored libsecp256k1 (tools/vendor_secp256k1.sh
    # extracts to vendor/secp256k1). Same object discipline as lz4_read
    # above: library objects are compiled separately, without -include $SHIM.
    # _nosan builds a distinct _nosan.so instead of overwriting the base .so.
    local SECP256K1_OBJS="/tmp/secp256k1${suffix}.o /tmp/precomputed_ecmult${suffix}.o /tmp/precomputed_ecmult_gen${suffix}.o"
    local SECP256K1_INC="-I$SECP256K1/src -I$SECP256K1/include"
    if [ ! -f "$TARGETS/secp256k1_read.c" ]; then
        :  # target source absent — nothing to build
    elif [ "$suffix" = "_nosan" ]; then
        # Always compile fresh _nosan objects with coverage, and emit a
        # distinct _nosan.so so the base .so from the "" pass is preserved.
        if compile_secp256k1_objects "$suffix" "$flags" "$DEFAULT_CC"; then
            build_so_target "$TARGETS/secp256k1_read.c" "$TARGETS/secp256k1_read_nosan.so" "$SECP256K1_OBJS -Wl,--export-dynamic" "$flags $SECP256K1_INC"
        else
            warn "secp256k1_read_nosan.so: compile failed, skipping"
        fi
    elif compile_secp256k1_objects "$suffix" "$flags" "$DEFAULT_CC"; then
        build_so_target "$TARGETS/secp256k1_read.c" "$TARGETS/secp256k1_read${out_suffix}.so" "$SECP256K1_OBJS -Wl,--export-dynamic" "$flags $SECP256K1_INC"
    else
        warn "secp256k1_read${out_suffix}.so: vendor/secp256k1 not found, skipping (run tools/vendor_secp256k1.sh)"
    fi

    # sqlite_read — vendored SQLite amalgamation (tools/vendor_sqlite.sh
    # extracts to vendor/sqlite). Same object discipline as lz4_read and
    # secp256k1_read above: sqlite3.c is compiled separately, without
    # -include $SHIM. _nosan builds a distinct _nosan.so instead of
    # overwriting the base .so.
    local SQLITE_OBJS="/tmp/sqlite3${suffix}.o"
    # -lm: sqlite's math functions; -lpthread even at THREADSAFE=0 because
    # the shim's own machinery links against it.
    local SQLITE_LIBS="$SQLITE_OBJS -lm -lpthread -Wl,--export-dynamic"
    local SQLITE_INC="-I$SQLITE $SQLITE_DEFINES"
    if [ ! -f "$TARGETS/sqlite_read.c" ]; then
        :  # target source absent — nothing to build
    elif [ ! -f "$SQLITE/sqlite3.c" ]; then
        warn "sqlite_read${out_suffix}.so: vendor/sqlite not found, skipping (run tools/vendor_sqlite.sh)"
    elif compile_sqlite_objects "$suffix" "$flags" "$DEFAULT_CC"; then
        if [ "$suffix" = "_nosan" ]; then
            build_so_target "$TARGETS/sqlite_read.c" "$TARGETS/sqlite_read_nosan.so" "$SQLITE_LIBS" "$flags $SQLITE_INC"
        else
            build_so_target "$TARGETS/sqlite_read.c" "$TARGETS/sqlite_read${out_suffix}.so" "$SQLITE_LIBS" "$flags $SQLITE_INC"
        fi
    else
        warn "sqlite_read${out_suffix}.so: amalgamation failed to compile, skipping"
    fi
}

# ── Compile vendored libraries with sancov instrumentation ───────
compile_vendored_libs() {
    local cc="$1" scov_flag="$2" suffix="$3"
    echo "Compiling vendored libraries ($cc ${scov_flag:-no-sancov})..."

    # Keep comparison calls visible to cmplog even at -O2.
    local no_builtin_cmp="$NOBUILTIN_CMP"

    # zlib
    if [ -d "$VENDOR/zlib" ]; then
        (cd "$VENDOR/zlib" && CC=$cc CFLAGS="-O2 -g -fPIC ${scov_flag} ${no_builtin_cmp} ${FRAME_POINTER}" \
            ./configure --static 2>/dev/null && make -j$(nproc) 2>/dev/null) && \
            ok "zlib (vendored)" || warn "zlib (vendored) failed"
    else
        warn "vendored zlib not found at $VENDOR/zlib"
    fi

    # libpng (depends on zlib)
    if [ -d "$VENDOR/libpng" ] && [ -d "$VENDOR/zlib" ]; then
        (cd "$VENDOR/libpng" && CC=$cc \
            CPPFLAGS="-I../zlib" \
            CFLAGS="-O2 -g -fPIC ${scov_flag} ${no_builtin_cmp} ${FRAME_POINTER} -I../zlib" \
            LDFLAGS="-L../zlib" \
            ./configure --with-pkgconfig=no 2>/dev/null && make -j$(nproc) 2>/dev/null) && \
            ok "libpng (vendored)" || warn "libpng (vendored) failed"
    else
        warn "vendored libpng not found or zlib missing"
    fi

    # libjpeg-turbo
    if [ -d "$VENDOR/libjpeg-turbo" ]; then
        (cd "$VENDOR/libjpeg-turbo" && \
            cmake -DCMAKE_C_COMPILER=$cc \
                  -DCMAKE_C_FLAGS="-O2 -g -fPIC ${scov_flag} ${no_builtin_cmp} ${FRAME_POINTER}" \
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
    local suffix="$1" flags="$2" label="$3" cc="${4:-$DEFAULT_CC}" extra_cflags="${5:-}"
    echo "Building vendored .so targets ($label)..."
    local out_suffix=""
    [[ "$suffix" == _asan* ]] && out_suffix="$suffix"

    local ZLIB_OBJS=""
    local ZLIB_INC=""
    local PNG_OBJS=""
    local PNG_INC=""
    local JPEG_OBJS=""
    local JPEG_INC=""

    # Link against the built static archives (libz.a / libpng16.a), not a
    # glob of loose *.o files. `make all` in these trees also builds test/
    # tool binaries (pngtest.c, zlib's test/example.c, minigzip.c, ...)
    # whose .o files sit in the same directory and each define their own
    # main() -- glob every *.o and the final link fails with "multiple
    # definition of main" against the fuzz target wrapper's own main().
    # The .a archive is exactly the library's object subset with none of
    # that, including the ones nested under mips/intel/powerpc/ that a
    # flat top-level glob would miss entirely anyway.
    if [ -d "$VENDOR/zlib" ]; then
        [ -f "$VENDOR/zlib/libz.a" ] && ZLIB_OBJS="$VENDOR/zlib/libz.a"
        ZLIB_INC="-I$VENDOR/zlib"
    fi
    if [ -d "$VENDOR/libpng" ]; then
        [ -f "$VENDOR/libpng/.libs/libpng16.a" ] && PNG_OBJS="$VENDOR/libpng/.libs/libpng16.a"
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
    for f in "$TARGETS"/fuzz_* "$TARGETS"/fgrep_read "$TARGETS"/fgrep_read_nosan \
             "$TARGETS"/fgrep_read.so "$TARGETS"/fgrep_read_nosan.so \
             "$TARGETS"/asan_target "$TARGETS"/asan_target_nosan "$TARGETS"/asan_target.so "$TARGETS"/asan_target_nosan.so \
             "$TARGETS"/png_read "$TARGETS"/png_read_nosan "$TARGETS"/png_read.so "$TARGETS"/png_read_nosan.so \
             "$TARGETS"/zlib_read "$TARGETS"/zlib_read_nosan "$TARGETS"/zlib_read.so "$TARGETS"/zlib_read_nosan.so \
             "$TARGETS"/gzip_read "$TARGETS"/gzip_read_nosan "$TARGETS"/gzip_read.so "$TARGETS"/gzip_read_nosan.so \
             "$TARGETS"/jpeg_read "$TARGETS"/jpeg_read_nosan "$TARGETS"/jpeg_read.so "$TARGETS"/jpeg_read_nosan.so \
             "$TARGETS"/ffmpeg_read "$TARGETS"/ffmpeg_read_nosan "$TARGETS"/ffmpeg_read.so "$TARGETS"/ffmpeg_read_nosan.so \
             "$TARGETS"/test_target "$TARGETS"/test_target_nosan "$TARGETS"/test_target.so "$TARGETS"/test_target_nosan.so \
             "$TARGETS"/proto_target "$TARGETS"/proto_target_nosan "$TARGETS"/proto_target.so "$TARGETS"/proto_target_nosan.so \
             "$TARGETS"/nop_target "$TARGETS"/nop_target_nosan "$TARGETS"/nop_target.so "$TARGETS"/nop_target_nosan.so \
             "$TARGETS"/tailslayer_read "$TARGETS"/tailslayer_read.so \
             "$TARGETS"/lz4_read "$TARGETS"/lz4_read_nosan "$TARGETS"/lz4_read.so "$TARGETS"/lz4_read_nosan.so \
             "$TARGETS"/secp256k1_read.so "$TARGETS"/secp256k1_read_nosan.so \
             "$TARGETS"/sqlite_read.so "$TARGETS"/sqlite_read_nosan.so \
             "$TARGETS"/grep_read "$TARGETS"/grep_read_nosan "$TARGETS"/grep_read.so "$TARGETS"/grep_read_nosan.so \
             "$TARGETS"/fuzzgoat_read "$TARGETS"/fuzzgoat_read_nosan "$TARGETS"/fuzzgoat_read.so "$TARGETS"/fuzzgoat_read_nosan.so; do
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

# ── Verify compiler-inserted edge coverage in .so targets ──────
#
# verify_afl only checks that the shim's __afl_* symbols are present, and they
# always are: the shim is -include'd into every target. It says nothing about
# whether the compiler inserted any CALLS to them. A .so carrying the shim but
# built without -fsanitize-coverage records zero edges, and the fuzzer prints
# "AFL instrumentation: detected" and then `shm: 0` for the whole campaign.
# Check the thing that actually produces edges.
verify_sancov() {
    [ "$WITH_CLANG_SCOV" -eq 0 ] && return 0
    echo "Verifying sancov instrumentation in .so targets..."
    local ok_count=0
    local fail_count=0
    for f in "$TARGETS"/*.so; do
        [ -f "$f" ] || continue
        nm "$f" 2>/dev/null | grep -q "fuzz_shm_run" || continue
        # Look for the __sancov_guards SECTION, not for __sanitizer_cov_*
        # symbols. The symbols are the shim's own callback definitions and are
        # present either way -- measured: an uninstrumented .so and an
        # instrumented one both report 12 of them to nm. Only the guard
        # section is emitted by -fsanitize-coverage=trace-pc-guard, and it is
        # the array the instrumented call sites index into.
        if readelf -S "$f" 2>/dev/null | grep -q "__sancov_guards"; then
            ok_count=$((ok_count + 1))
        else
            warn "$(basename "$f"): no __sancov_guards — in-process modes record ZERO edges"
            fail_count=$((fail_count + 1))
        fi
    done
    ok "$ok_count .so targets with compiler-inserted edge coverage"
    if [ "$fail_count" -gt 0 ]; then
        warn "$fail_count .so targets carry fuzz_shm_run but no edge instrumentation"
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

# ── Verify vendored trace-cmp caller/implementation resolution ──
# Checks that vendored libraries (zlib, libpng) have trace-cmp
# callers (U __sanitizer_cov_trace_cmp*), and that the final .so
# has those callers resolved (no U trace_cmp, only T definitions
# from cmplog_shim.o) with -Bsymbolic preventing ASAN override.
verify_vendor_tracecmp() {
    [ "$WITH_CMPLOG" -eq 0 ] && return 0
    # Only run when vendored .a files exist
    local ZLIB_A="$VENDOR/zlib/libz.a"
    local LIBPNG_A="$VENDOR/libpng/.libs/libpng16.a"
    [ -f "$ZLIB_A" ] || [ -f "$LIBPNG_A" ] || return 0

    echo "Verifying vendored trace-cmp resolution..."

    # ── Vendor static libs: count U trace_cmp callers ──────────────
    local vendor_zlib_callers=0 vendor_png_callers=0
    if [ -f "$ZLIB_A" ]; then
        vendor_zlib_callers=$(nm "$ZLIB_A" 2>/dev/null | grep -c 'U.*trace_cmp' || true)
        ok "vendor/zlib: $vendor_zlib_callers trace-cmp callers (U)"
    fi
    if [ -f "$LIBPNG_A" ]; then
        vendor_png_callers=$(nm "$LIBPNG_A" 2>/dev/null | grep -c 'U.*trace_cmp' || true)
        ok "vendor/libpng: $vendor_png_callers trace-cmp callers (U)"
    fi
    local total_vendor_callers=$((vendor_zlib_callers + vendor_png_callers))

    # ── Final .so: check resolution + -Bsymbolic ──────────────────
    local resolved_ok=0 resolved_fail=0 bsymbolic_ok=0 bsymbolic_fail=0
    local total_implementations=0 total_so_callers=0
    for f in "$TARGETS"/*.so; do
        [ -f "$f" ] || continue
        local impl_count=$(nm "$f" 2>/dev/null | grep -cE '[Tt].*trace_cmp' || true)
        local undef_count=$(nm "$f" 2>/dev/null | grep -c 'U.*trace_cmp' || true)
        total_implementations=$((total_implementations + impl_count))
        total_so_callers=$((total_so_callers + undef_count))

        if [ "$undef_count" -eq 0 ] && [ "$impl_count" -gt 0 ]; then
            resolved_ok=$((resolved_ok + 1))
        elif [ "$impl_count" -eq 0 ]; then
            warn "$(basename "$f"): no trace-cmp implementations (T)"
            resolved_fail=$((resolved_fail + 1))
        else
            warn "$(basename "$f"): $undef_count unresolved trace-cmp callers (U)"
            resolved_fail=$((resolved_fail + 1))
        fi

        if readelf -d "$f" 2>/dev/null | grep -q 'SYMBOLIC'; then
            bsymbolic_ok=$((bsymbolic_ok + 1))
        else
            warn "$(basename "$f"): missing -Bsymbolic (ASAN may override trace-cmp)"
            bsymbolic_fail=$((bsymbolic_fail + 1))
        fi
    done

    local total_tested=$((resolved_ok + resolved_fail))
    if [ "$total_vendor_callers" -gt 0 ]; then
        echo "  Vendor callers: $total_vendor_callers | .so implems: $total_implementations | .so unresolved: $total_so_callers"
    fi
    ok "$resolved_ok/$total_tested .so targets: trace-cmp fully resolved"
    if [ "$bsymbolic_fail" -eq 0 ]; then
        ok "$bsymbolic_ok/$total_tested .so targets: -Bsymbolic present"
    else
        warn "$bsymbolic_fail .so targets missing -Bsymbolic"
    fi
}

# ── Build AFLGo distance .so targets ──────────────────────────────
# Compiles the target wrapper with -fsanitize-coverage=trace-pc +
# -D__AFL_DISTANCE_MODE so the shim accumulates per-block distances
# into the SHM tail.  The cmplog shim is linked in (like the default
# .so builds) so `--cmplog` keeps the fuzzer in direct_lite mode
# instead of falling back to the persistent loader, where the tail is
# not streamed.  Builds ASAN and no-ASAN variants following the repo's
# suffix convention (no suffix = no-ASAN, _asan = ASAN).
# NOTE: vendored libraries keep their default trace-pc-guard
# instrumentation, so distance is measured on the wrapper's own blocks
# only; rebuild the vendor libs with trace-pc to extend distance into
# library code.
build_distance_so_targets() {
    [ "$WITH_DISTANCE" -eq 0 ] && return 0
    if ! command -v clang &>/dev/null; then
        warn "clang not found — --distance requires clang"
        return 1
    fi
    echo "Building distance .so targets (trace-pc + AFLGo channel + cmplog)..."
    for variant in "nosan:-O2" "asan:-O2 -fsanitize=address -lasan"; do
        local label="${variant%%:*}"
        local extra_flags="${variant#*:}"
        local out_suffix="_dist"
        [ "$label" = "asan" ] && out_suffix="_dist_asan"
        for spec in "png_read:-lpng -lz" "zlib_read:-lz" "gzip_read:-lz" "jpeg_read:-ljpeg" "test_target:" "proto_target:"; do
            local name="${spec%%:*}"
            local libs="${spec#*:}"
            [ -f "$TARGETS/$name.c" ] || continue
            local cmplog_cflags=""
            local cmplog_libs=""
            if [ "$WITH_CMPLOG" -eq 1 ]; then
                cmplog_cflags="$CMPLOG_CFLAGS"
                cmplog_libs="$CMPLOG_LIBS"
            fi
            clang $extra_flags -g $FRAME_POINTER -D__AFL_DISTANCE_MODE -fsanitize-coverage=trace-pc \
                $cmplog_cflags -shared -fPIC -Wl,-Bsymbolic -include "$SHIM" \
                -o "$TARGETS/${name}${out_suffix}.so" "$TARGETS/$name.c" \
                $libs $cmplog_libs 2>/dev/null
            if [ -f "$TARGETS/${name}${out_suffix}.so" ]; then
                ok "${name}${out_suffix}.so ($label)"
            else
                warn "failed: ${name}${out_suffix}.so ($label)"
            fi
        done
    done
}

# ── Build png_read n-gram flavors (_ng2 / _ng3) ───────────────────
# trace-pc wrapper + -D__AFL_NGRAM_K={2,3} + the AFLGo channel. Linked
# against vendored libpng+zlib REBUILT with trace-pc so library blocks
# emit __sanitizer_cov_trace_pc too — the K-Scheduler node table must
# cover library code or the horizon graph sees only the wrapper
# (docs/kscheduler_centrality_port.md §3 build-scope caveat).
# NOTE: rebuilds vendor/<lib> in place, clobbering artifacts of earlier
# vendor passes — same last-wins behaviour as --vendor-tracecmp. Run
# this pass last if you need both flavors' libs.
build_ngram_so_targets() {
    [ "$WITH_NGRAM" -eq 0 ] && return 0
    if ! command -v clang &>/dev/null; then
        warn "clang not found — --ngram requires clang"
        return 1
    fi
    echo "Building n-gram .so flavors (trace-pc + __AFL_NGRAM_K + AFLGo channel)..."

    local NG_VENDOR_LIBS="-lpng -lz -lm"
    local NG_VENDOR_INC=""
    if [ -f "$VENDOR_LIBPNG_DIR/configure" ] && [ -f "$VENDOR_ZLIB_DIR/configure" ]; then
        echo "  Rebuilding vendored libpng+zlib with trace-pc..."
        (cd "$VENDOR_ZLIB_DIR" && \
            make -s clean >/dev/null 2>&1; \
            CC=clang CFLAGS="-O2 -g -fPIC $FRAME_POINTER -fsanitize-coverage=trace-pc" \
            ./configure --static 2>/dev/null && \
            make -j$(nproc) -s 2>/dev/null) && \
            ok "vendor/zlib (trace-pc)" || warn "vendor/zlib build failed"
        (cd "$VENDOR_LIBPNG_DIR" && \
            make -s clean >/dev/null 2>&1; \
            CC=clang CFLAGS="-O2 -g -fPIC $FRAME_POINTER -fsanitize-coverage=trace-pc -I../zlib" \
            LDFLAGS="-L../zlib" \
            ./configure --enable-shared=no --quiet 2>/dev/null && \
            make -j$(nproc) -s 2>/dev/null) && \
            ok "vendor/libpng (trace-pc)" || warn "vendor/libpng build failed"
        local PNG_A="$VENDOR_LIBPNG_DIR/.libs/libpng16.a"
        local Z_A="$VENDOR_ZLIB_DIR/libz.a"
        if [ -f "$PNG_A" ] && [ -f "$Z_A" ]; then
            NG_VENDOR_LIBS="$PNG_A $Z_A -lm"
            NG_VENDOR_INC="-I$VENDOR_LIBPNG_DIR -I$VENDOR_ZLIB_DIR"
        else
            warn "vendor .a missing after rebuild — system libpng/zlib fallback gives wrapper-only node coverage"
        fi
    else
        warn "vendor sources missing (run tools/vendor_libpng.sh) — system libpng/zlib fallback gives wrapper-only node coverage"
    fi

    for k in 2 3; do
        local cmplog_cflags=""
        local cmplog_libs=""
        if [ "$WITH_CMPLOG" -eq 1 ]; then
            cmplog_cflags="$CMPLOG_CFLAGS"
            cmplog_libs="$CMPLOG_LIBS"
        fi
        clang -O2 -g $FRAME_POINTER -D__AFL_DISTANCE_MODE -D__AFL_NGRAM_K=$k \
            -fsanitize-coverage=trace-pc $cmplog_cflags \
            -shared -fPIC -Wl,-Bsymbolic -include "$SHIM" \
            -o "$TARGETS/png_read_ng${k}.so" "$TARGETS/png_read.c" \
            $NG_VENDOR_LIBS $NG_VENDOR_INC $cmplog_libs 2>/dev/null
        if [ -f "$TARGETS/png_read_ng${k}.so" ]; then
            ok "png_read_ng${k}.so"
        else
            warn "failed: png_read_ng${k}.so"
        fi
    done
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
    # Caller-context edge hashing is default-on inside afl_shim.c, which only
    # the target TUs include; the libs here never see the define. What the
    # whole linked chain DOES need is frame pointers: the context walk
    # dereferences the caller's saved-frame-pointer slot at runtime (see
    # afl_shim.c), so every TU under these targets carries FRAME_POINTER.
    local CTX_FLAGS="$FRAME_POINTER"
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
            CC=clang CFLAGS="-O2 -g -fPIC $CTX_FLAGS $TRACE_FLAGS" \
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
            CC=clang CFLAGS="-O2 -g -fPIC $CTX_FLAGS $TRACE_FLAGS -I../zlib" \
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
    local ALL_FLAGS="$CTX_FLAGS $TRACE_FLAGS $ASAN_FLAGS"

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
            $CC -O2 -g $FRAME_POINTER \
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

    # The callbacks must be COMPILED IN, not LD_PRELOADed.
    #
    # -fsanitize-coverage pulls in compiler-rt's sancov runtime, which ships
    # *weak no-op definitions* of __sanitizer_cov_trace_{,const_}cmp{1,2,4,8}.
    # The executable is searched before LD_PRELOAD libraries in the global
    # symbol lookup order, so those stubs win and every callback returns
    # immediately -- the preloaded shim is never reached. Measured: an -O2
    # trace-cmp build of cmplog_exercise.c logged 4 CMP lines under
    # LD_PRELOAD (all from memchr, i.e. the libc layer only) and 20 with the
    # shim linked. A strong definition in the same link beats the weak stub;
    # -D__AFL_CMPLOG=1 puts one directly in the target's own TU, and marks
    # the callbacks hidden so nothing can interpose them afterwards.

    # $NOBUILTIN_CMP keeps memcmp/strcmp at the PLT so the libc layer still
    # sees their operands; trace-cmp cannot recover those (see the note at
    # the top of this file). The two layers are complementary.
    local rc=0
    local spec src
    for spec in "tracecmp_target" "cmplog_exercise"; do
        src="$TARGETS/$spec.c"
        [ -f "$src" ] || { warn "source not found: $src"; continue; }

        rc=0
        $CC -O2 -g $FRAME_POINTER $TRACE_FLAGS $NOBUILTIN_CMP $CMPLOG_CFLAGS -include "$SHIM" \
            -o "$TARGETS/${spec}_tcg" "$src" $CMPLOG_LIBS 2>/dev/null || rc=$?
        if [ $rc -eq 0 ]; then
            ok "${spec}_tcg (trace-cmp + no-builtin)"
        else
            warn "failed: ${spec}_tcg (trace-cmp)"
        fi

        rc=0
        $CC -O2 -g $FRAME_POINTER $TRACE_FLAGS $NOBUILTIN_CMP $CMPLOG_CFLAGS -shared -fPIC -Wl,-Bsymbolic \
            -include "$SHIM" \
            -o "$TARGETS/${spec}_tcg.so" "$src" $CMPLOG_LIBS 2>/dev/null || rc=$?
        if [ $rc -eq 0 ]; then
            ok "${spec}_tcg.so (trace-cmp + no-builtin)"
        else
            warn "failed: ${spec}_tcg.so (trace-cmp)"
        fi
    done

    # Verify the callbacks are DEFINED (T) in the target, not left undefined
    # for a preload that cannot win the lookup, and not the runtime's weak
    # no-op stub (W).
    for f in "$TARGETS/tracecmp_target_tcg" "$TARGETS/cmplog_exercise_tcg"; do
        [ -f "$f" ] || continue
        # T or t. The callbacks are compiled with hidden visibility now, and
        # the static linker resolves a fully-linked hidden global to a LOCAL
        # symbol -- nm prints 't'. That is strictly stronger than the 'T'
        # this check was written to assert: local means nothing outside the
        # binary can interpose it. 'W' is still the failure (the sancov
        # runtime's weak no-op won and cmplog would see nothing).
        if nm "$f" 2>/dev/null | grep -qE "^[0-9a-f]+ [Tt] __sanitizer_cov_trace_const_cmp4$"; then
            ok "$(basename "$f"): trace-cmp callbacks compiled in (non-interposable)"
        else
            warn "$(basename "$f"): trace-cmp callbacks are weak no-ops — cmplog will see nothing"
        fi
        # $NOBUILTIN_CMP must have kept memcmp a real call. With the shim
        # linked in, the call binds directly to the shim's own definition
        # rather than going through the PLT -- so match either form. If it
        # matches neither, ExpandMemCmp folded the comparison and the
        # constants are gone.
        if objdump -d "$f" 2>/dev/null | grep -qE "call.*<memcmp(@plt)?>"; then
            ok "$(basename "$f"): memcmp still a call (constants reach the pool)"
        else
            warn "$(basename "$f"): memcmp folded away — check \$NOBUILTIN_CMP"
        fi
    done
}

# ── Print build feature matrix ──────────────────────────────────
# Always shows which features and target groups this invocation will build.
print_feature_matrix() {
    echo ""
    echo "Build feature matrix (flags: $OPTS):"
    printf '  %-20s %-12s %s\n' "Feature" "State" "Notes"
    printf '  %-20s %-12s %s\n' "-------" "-----" "-----"

    local state
    state=$([ "$WITH_CMPLOG" -eq 1 ] && echo "ON (default)" || echo "OFF")
    printf '  %-20s %-12s %s\n' "cmplog" "$state" "comparison tracing linked into .so targets"
    state=$([ "$WITH_TRACECMP" -eq 1 ] && echo "ON (default)" || echo "OFF")
    printf "  %-20s %-12s %s\n" "tracecmp" "$state" "compiler-IR tracing + no-builtin cmp (clang)"
    state=$([ "$WITH_CLANG_SCOV" -eq 1 ] && echo "ON" || echo "OFF")
    printf '  %-20s %-12s %s\n' "clang-scov" "$state" "compiler-inserted edge coverage (clang)"
    state=$([ "$WITH_VENDOR_TRACECMP" -eq 1 ] && echo "ON" || echo "OFF")
    printf '  %-20s %-12s %s\n' "vendor-tracecmp" "$state" "rebuild vendor libs + targets with trace-cmp (clang)"
    state=$([ "$WITH_DISTANCE" -eq 1 ] && echo "ON" || echo "OFF")
    printf '  %-20s %-12s %s\n' "distance" "$state" "AFLGo SHM-tail distance .so targets (clang)"
    state=$([ "$WITH_FFMPEG_SANCOV" -eq 1 ] && echo "ON (default)" || echo "OFF")
    printf '  %-20s %-12s %s\n' "ffmpeg-sancov" "$state" "auto-rebuild vendored FFmpeg with coverage"

    state=$([ "$BUILD_ASAN" -eq 1 ] && echo "ON" || echo "OFF")
    printf '  %-20s %-12s %s\n' "ASAN variants" "$state" "executables + .so targets"
    printf '  %-20s %-12s %s\n' "UBSAN variants" "$state" "built alongside ASAN (.so)"
    state=$([ "$WITH_MSAN" -eq 1 ] && echo "ON" || echo "OFF")
    printf '  %-20s %-12s %s\n' "MSAN variants" "$state" "--msan: uninit-value bugs (exe only, clang)"
    state=$([ "$WITH_TSAN" -eq 1 ] && echo "ON" || echo "OFF")
    printf '  %-20s %-12s %s\n' "TSAN variants" "$state" "--tsan: data races (exe only, clang)"
    state=$([ "$BUILD_NOSAN" -eq 1 ] && echo "ON" || echo "OFF")
    printf '  %-20s %-12s %s\n' "No-ASAN variants" "$state" "executables + .so targets"

    state=$([ "$HAS_FGREP" -eq 1 ] && echo "BUILD" || echo "SKIP")
    printf '  %-20s %-12s %s\n' "fgrep targets" "$state" "$([ "$HAS_FGREP" -eq 1 ] && echo "found at $FGREP" || echo "vendor/fgrep not found")"
    state=$([ -d "$TAILSLAYER/include" ] && echo "BUILD" || echo "SKIP")
    printf '  %-20s %-12s %s\n' "tailslayer_read" "$state" "$([ -d "$TAILSLAYER/include" ] && echo "found at $TAILSLAYER" || echo "headers not found at $TAILSLAYER/include")"
    state=$([ -d "$LZ4/lib" ] && echo "BUILD" || echo "SKIP")
    printf '  %-20s %-12s %s\n' "lz4_read" "$state" "$([ -d "$LZ4/lib" ] && echo "found at $LZ4" || echo "not vendored — run tools/vendor_lz4.sh")"
    state=$([ "$HAS_FUZZGOAT" -eq 1 ] && echo "BUILD" || echo "SKIP")
    printf '  %-20s %-12s %s\n' "fuzzgoat_read" "$state" "$([ "$HAS_FUZZGOAT" -eq 1 ] && echo "found at $VENDOR/fuzzgoat" || echo "not vendored — run tools/vendor_fuzzgoat.sh")"
    state=$([ -d "$SECP256K1/src" ] && echo "BUILD" || echo "SKIP")
    printf '  %-20s %-12s %s\n' "secp256k1_read" "$state" "$([ -d "$SECP256K1/src" ] && echo "found at $SECP256K1" || echo "not vendored — run tools/vendor_secp256k1.sh")"
    state=$([ -f "$SQLITE/sqlite3.c" ] && echo "BUILD" || echo "SKIP")
    printf '  %-20s %-12s %s\n' "sqlite_read" "$state" "$([ -f "$SQLITE/sqlite3.c" ] && echo "found at $SQLITE" || echo "not vendored — run tools/vendor_sqlite.sh")"
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────
echo "=== Building fuzz targets ==="

snapshot_target_md5

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
    --ngram) BUILD_NOSAN=1 ;;
    *) BUILD_ASAN=1; BUILD_NOSAN=1 ;;
esac

print_feature_matrix

if [ "$BUILD_ASAN" -eq 1 ]; then
    [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_asan" "-fsanitize=address"
    [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_asan" "-fsanitize=address" "ASAN"
    if command -v clang &>/dev/null; then
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_asan_tcg" "-fsanitize=address" "clang" "-fsanitize-coverage=trace-pc-guard"
    else
        warn "clang not found — .so targets will lack auto edge coverage (manual __afl_map_edge only)"
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_asan_tcg" "-fsanitize=address"
    fi
    build_simple_targets "_asan" "-fsanitize=address" "ASAN"
    [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_asan_tcg" "-fsanitize=address" "ASAN"
    build_vendored_ffmpeg_sancov "_asan"
    build_simple_so_targets "_asan" "-fsanitize=address" "ASAN"
    build_standalone_so_targets "_asan" "-fsanitize=address" "ASAN"
fi
# UBSAN targets: compiled with -fsanitize=undefined (runtime built in)
# Uses nosan (coverage-only) FFmpeg libs — UBSAN doesn't need ASAN instrumentation.
if [ "$BUILD_ASAN" -eq 1 ]; then
    echo "  Building UBSAN targets..."
    build_vendored_ffmpeg_sancov ""
    build_simple_so_targets "_ubsan" "-fsanitize=undefined" "UBSAN" "clang"
    build_standalone_so_targets "_ubsan" "-fsanitize=undefined" "UBSAN" "clang"
fi
# MSAN / TSAN: opt-in, executables only (see build_sanitizer_targets).
if [ "$WITH_MSAN" -eq 1 ]; then
    build_sanitizer_targets "_msan" "-fsanitize=memory -fsanitize-memory-track-origins=2" "MSAN"
fi
if [ "$WITH_TSAN" -eq 1 ]; then
    build_sanitizer_targets "_tsan" "-fsanitize=thread" "TSAN"
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
    build_vendored_ffmpeg_sancov ""
    build_simple_so_targets "_nosan" "" "No-ASAN"
    build_standalone_so_targets "_nosan" "" "No-ASAN"
fi
if [ "$WITH_CLANG_SCOV" -eq 1 ]; then
    SCOV_CC="clang"
    if ! command -v clang &>/dev/null; then
        warn "clang not found — --clang-scov requires clang"
    else
        SCOV_FLAGS="-fsanitize-coverage=trace-pc-guard"
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_asan" "-fsanitize=address" "$SCOV_CC" "$SCOV_FLAGS"
        [ "$HAS_FGREP" -eq 1 ] && compile_fgrep_objects "_nosan" "" "$SCOV_CC" "$SCOV_FLAGS"
        compile_vendored_libs "$SCOV_CC" "$SCOV_FLAGS" "_asan"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_asan" "-fsanitize=address" "Clang-scov"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_targets "_nosan" "" "Clang-scov"
        build_simple_targets "_asan" "-fsanitize=address" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        build_simple_targets "_nosan" "" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        # The .so targets need this pass just as much as the executables do,
        # and used to be left out of it entirely: build_simple_so_targets was
        # never called here, and build_standalone_so_targets took no
        # cc/extra_cflags to pass $SCOV_FLAGS through. Everything carrying
        # fuzz_shm_run -- i.e. everything the in-process and direct_lite modes
        # run -- therefore kept whatever the uninstrumented no-ASAN pass had
        # produced. `nm -D targets/test_target.so | grep -c sanitizer_cov`
        # returned 0, so every in-process campaign reported `shm: 0` and
        # `Edges discovered: 0` while still calling itself coverage-guided.
        # verify_sancov below now says so out loud.
        build_simple_so_targets "_asan" "-fsanitize=address" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        build_simple_so_targets "_nosan" "" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        build_vendored_so_targets "_asan" "-fsanitize=address" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        build_vendored_so_targets "_nosan" "" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_asan" "-fsanitize=address" "Clang-scov"
        [ "$HAS_FGREP" -eq 1 ] && build_fgrep_so_targets "_nosan" "" "Clang-scov"
        build_standalone_so_targets "_asan" "-fsanitize=address" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        build_standalone_so_targets "_nosan" "" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        # UBSAN .so targets are built in their own pass above and were the
        # last variant left uninstrumented -- verify_sancov flagged all nine
        # of them. They carry fuzz_shm_run like every other .so, so without
        # this they record zero edges in-process exactly as the _asan and
        # _nosan ones did. Gated on BUILD_ASAN to match the pass that creates
        # them; without that guard this would build UBSAN targets that the
        # rest of the run never asked for.
        if [ "$BUILD_ASAN" -eq 1 ]; then
            build_simple_so_targets "_ubsan" "-fsanitize=undefined" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
            build_standalone_so_targets "_ubsan" "-fsanitize=undefined" "Clang-scov" "$SCOV_CC" "$SCOV_FLAGS"
        fi
    fi
fi
if [ "$WITH_VENDOR_TRACECMP" -eq 1 ]; then
    build_vendored_tracecmp_targets "$@"
fi
if [ "$WITH_DISTANCE" -eq 1 ]; then
    build_distance_so_targets
fi
if [ "$WITH_NGRAM" -eq 1 ]; then
    build_ngram_so_targets
fi

# ── Compile perf_shim.so (utility library, not a fuzz target) ────
echo "Compiling utility libraries..."
compile_perf_shim

verify_afl
verify_shm_run
verify_sancov
verify_cmplog
verify_vendor_tracecmp
verify_target_md5
build_tracecmp_targets
echo "=== Done ==="
