#!/bin/bash
# Build the buggy Rust fuzz target (rust/buggy_target + targets/rust_target.c).
#
# Kept as its own script rather than folded into tools/build_targets.sh:
# that script's build_target()/build_so_target() already know how to link
# an arbitrary `libs` string against a C source with the shim -included,
# so a `cargo build` step ahead of them is all a Rust target actually
# needs -- this script is that step, plus the two build_*_target calls,
# factored out so it can be sourced or run standalone without adding a
# rustc/cargo dependency to the main build's otherwise C-toolchain-only
# path. See docs/handover/handover_rust_target_2026-09-15.md for the
# integration writeup, including why edge coverage from the Rust side
# itself is not available under the stable rustc this project builds
# with (no `-Z sanitizer-coverage-trace-pc-guard`).
#
# Compiler choice mirrors build_targets.sh's _pick_cc exactly, and for the
# same reason: gcc's -fsanitize-coverage= accepts trace-pc and trace-cmp
# but not trace-pc-guard, which is what afl_shim.c's edge callbacks are
# built on. An earlier version of this script hardcoded gcc and quietly
# shipped a wrapper with ZERO instrumented call sites -- not "shallower
# coverage", none at all, confirmed with objdump against both compilers
# (gcc: 0 calls to __sanitizer_cov_trace_pc_guard; clang with
# -fsanitize-coverage=trace-pc-guard: 84). The Rust internals stay opaque
# to the shim either way (see the handover doc), but the wrapper itself —
# the one piece of this target that CAN be instrumented on any toolchain
# available here — should actually be, and only clang can do it.
#
# Usage:
#   tools/build_rust_target.sh              # debug-friendly (no ASAN)
#   tools/build_rust_target.sh --asan       # ASAN build too
#
# Warns and skips (exit 0) if cargo is not installed, matching every
# vendor_*.sh script's convention for an optional, absent toolchain --
# this must never fail the rest of a `build_targets.sh` run just because
# Rust isn't set up on a given machine.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="$ROOT/rust/buggy_target"
WRAPPER_SRC="$ROOT/targets/rust_target.c"
SHIM="$ROOT/src/fuzzer_tool/adapters/afl_shim.c"
OUT_DIR="${FUZZ_BUILD_ROOT:-$ROOT/targets}"
FRAME_POINTER="-fno-omit-frame-pointer"  # matches build_targets.sh: ctx hashing needs it

WITH_ASAN=0
for arg in "$@"; do
    [ "$arg" = "--asan" ] && WITH_ASAN=1
done

warn() { printf '\033[0;33mWARN\033[0m: %s\n' "$*" >&2; }
ok()   { printf '\033[0;32mOK\033[0m: %s\n' "$1"; }
fail() { printf '\033[0;31mFAIL\033[0m: %s\n' "$1" >&2; exit 1; }

# Mirrors build_targets.sh's _pick_cc: prefer clang, warn and fall back to
# gcc otherwise. Unlike that function's general case, the fallback here
# isn't "shallower coverage" — it's zero coverage on the wrapper, since
# gcc has no trace-pc-guard at all (see the header comment above).
if command -v clang &>/dev/null; then
    CC="${CC:-clang}"
    COV_FLAG="-fsanitize-coverage=trace-pc-guard"
else
    warn "clang not found, falling back to gcc — the wrapper will carry" \
         "ZERO instrumented call sites (gcc has no -fsanitize-coverage=" \
         "trace-pc-guard); crash detection still works, coverage feedback" \
         "from targets/rust_target.c itself will not"
    CC="${CC:-gcc}"
    COV_FLAG=""
fi

if ! command -v cargo &>/dev/null; then
    warn "cargo not found — skipping the Rust target (install rustc+cargo to enable it)"
    exit 0
fi

if [ ! -f "$SHIM" ]; then
    fail "shim not found at $SHIM — run from a checkout with src/fuzzer_tool/adapters/afl_shim.c intact"
fi

# Optional: real edge coverage from INSIDE the Rust crate, not just the C
# wrapper. Needs nightly rustc's -Z sanitizer-coverage-trace-pc-guard,
# unavailable on stable. Auto-detects a toolchain fetched by
# tools/fetch_rust_nightly.sh (RUSTC_NIGHTLY env var takes precedence, so
# you can point at any nightly build you already trust instead). Verified
# empirically end to end for this crate: with it, rust_fuzz_entry alone
# picked up 12 real __sanitizer_cov_trace_pc_guard call sites (nm/objdump
# confirmed, not assumed) covering the branch structure of all four bugs
# — a real improvement over the wrapper-only 84 call sites clang alone
# gives, though the two numbers aren't directly comparable (the crate is
# small enough that release-mode inlining folds all four bug functions
# into rust_fuzz_entry, so 12 is "the whole crate's branches", not "one
# function's"). See docs/handover/handover_rust_target_2026-09-15.md for
# the full validation, the ASAN gap that's still open even with this
# toolchain (no rust-src component -> no -Z build-std), and — important —
# this toolchain's own provenance caveat before you rely on it for
# anything beyond a disposable local build.
DEFAULT_NIGHTLY="$HOME/.cache/rust-nightly-jolt/bin/rustc"
if [ -n "${RUSTC_NIGHTLY:-}" ]; then
    NIGHTLY_RUSTC="$RUSTC_NIGHTLY"
elif [ -x "$DEFAULT_NIGHTLY" ]; then
    NIGHTLY_RUSTC="$DEFAULT_NIGHTLY"
else
    NIGHTLY_RUSTC=""
fi

if [ -n "$NIGHTLY_RUSTC" ]; then
    if [ ! -x "$NIGHTLY_RUSTC" ]; then
        fail "RUSTC_NIGHTLY=$NIGHTLY_RUSTC is not executable"
    fi
    ok "using nightly rustc for real Rust-side edge coverage: $NIGHTLY_RUSTC"
    export RUSTC="$NIGHTLY_RUSTC"
    export RUSTFLAGS="-Cpasses=sancov-module -Cllvm-args=-sanitizer-coverage-level=3 -Cllvm-args=-sanitizer-coverage-trace-pc-guard"
else
    warn "no nightly rustc found (checked \$RUSTC_NIGHTLY and $DEFAULT_NIGHTLY)" \
         "— building with stable rustc; the crate itself will carry no" \
         "coverage instrumentation (run tools/fetch_rust_nightly.sh to enable" \
         "this — read that script's header first, it's an unofficial" \
         "third-party toolchain, opt in deliberately)"
fi

echo "Building rust/buggy_target (cargo release, staticlib)..."
( cd "$CRATE_DIR" && cargo build --release )

RLIB="$CRATE_DIR/target/release/libbuggy_rust_target.a"
[ -f "$RLIB" ] || fail "expected staticlib not found: $RLIB"

# Rust's staticlib pulls in libstd, which needs these even though the
# wrapper C source references none of them directly. Determined empirically
# (link failures name the missing symbol) rather than copied from a rustc
# --print flag, since `rustc --print native-static-libs` requires a crate
# that has already been built with that flag threaded through, which a
# plain `cargo build` does not do.
RUST_LIBS="-lpthread -ldl -lm -lrt -lutil -lgcc_s"

mkdir -p "$OUT_DIR"

build_variant() {
    local suffix="$1" extra_flags="$2" cc="$3"
    local exe_out="$OUT_DIR/rust_target${suffix}"
    local so_out="$OUT_DIR/rust_target${suffix}.so"
    local so_libs="$RUST_LIBS"
    [[ "$extra_flags" == *-fsanitize=address* ]] && so_libs="$so_libs -lasan"

    echo "  -> $exe_out"
    "$cc" $extra_flags $COV_FLAG -O2 -g $FRAME_POINTER -include "$SHIM" \
        -o "$exe_out" "$WRAPPER_SRC" "$RLIB" $RUST_LIBS
    ok "$(basename "$exe_out")"

    echo "  -> $so_out"
    "$cc" $extra_flags $COV_FLAG -O2 -g $FRAME_POINTER -shared -fPIC -include "$SHIM" \
        -o "$so_out" "$WRAPPER_SRC" "$RLIB" $so_libs
    ok "$(basename "$so_out")"
}

build_variant "" "" "$CC"

if [ "$WITH_ASAN" -eq 1 ]; then
    ASAN_CC="$CC"
    command -v clang &>/dev/null && ASAN_CC="clang"
    build_variant "_asan" "-fsanitize=address" "$ASAN_CC"
    warn "ASAN build linked, but note: the Rust static lib itself is NOT" \
         "ASAN-instrumented (stable rustc has no -Z sanitizer=address /" \
         "-Z build-std). ASAN here only covers targets/rust_target.c's" \
         "own few lines; the bugs inside rust/buggy_target that don't" \
         "reach unmapped memory on their own will not be caught by it." \
         "See the handover doc's 'What ASAN does and doesn't see' section."
fi

echo "Rust target build complete."
