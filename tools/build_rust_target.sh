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
    local suffix="$1" extra_flags="$2" cc="$3" rlib="$4" so_libs="$5"
    local exe_out="$OUT_DIR/rust_target${suffix}"
    local so_out="$OUT_DIR/rust_target${suffix}.so"

    echo "  -> $exe_out"
    "$cc" $extra_flags $COV_FLAG -O2 -g $FRAME_POINTER -include "$SHIM" \
        -o "$exe_out" "$WRAPPER_SRC" "$rlib" $RUST_LIBS
    ok "$(basename "$exe_out")"

    echo "  -> $so_out"
    "$cc" $extra_flags $COV_FLAG -O2 -g $FRAME_POINTER -shared -fPIC -include "$SHIM" \
        -o "$so_out" "$WRAPPER_SRC" "$rlib" $so_libs
    ok "$(basename "$so_out")"
}

build_variant "" "" "$CC" "$RLIB" "$RUST_LIBS"

if [ "$WITH_ASAN" -eq 1 ]; then
    if [ -n "$NIGHTLY_RUSTC" ]; then
        # Real ASAN instrumentation of the Rust code itself, not just the
        # wrapper — verified end to end (docs/handover/
        # handover_rust_target_2026-09-15.md): both O and W now produce a
        # genuine, fully symbolicated "AddressSanitizer: heap-buffer-
        # overflow" report pointing at the actual Rust source line, for
        # small overflows a plain build can't catch at all. Two things
        # were required beyond just adding -Z sanitizer=address, both
        # confirmed necessary by testing without them first:
        #   1. -Z build-std is NOT needed for this — ASAN's heap redzones
        #      come from intercepting malloc/free, which happens
        #      regardless of whether std itself is instrumented; build-std
        #      only matters for bugs inside std's own code. (This
        #      toolchain couldn't do -Z build-std anyway — see the
        #      handover doc on the missing rust-src component.)
        #   2. The C compiler linking the final binary needs an ASAN
        #      runtime that actually matches this rustc's bundled LLVM
        #      version, or you get "Your application is linked against
        #      incompatible ASan runtimes" (confirmed: system clang's
        #      default LLVM 18 runtime against this nightly's LLVM 20
        #      fails this way; clang-20 works). Auto-detected below by
        #      parsing `rustc --version --verbose`'s LLVM version and
        #      looking for a matching `clang-N`; install it yourself
        #      (`apt install clang-N`) if it's missing.
        LLVM_MAJOR="$("$NIGHTLY_RUSTC" --version --verbose | sed -n 's/^LLVM version: \([0-9]*\).*/\1/p')"
        ASAN_CC=""
        if [ -n "$LLVM_MAJOR" ] && command -v "clang-$LLVM_MAJOR" &>/dev/null; then
            ASAN_CC="clang-$LLVM_MAJOR"
        fi

        if [ -z "$ASAN_CC" ]; then
            warn "no clang-$LLVM_MAJOR found to match this nightly rustc's" \
                 "LLVM $LLVM_MAJOR — real Rust-side ASAN needs a matching" \
                 "runtime (confirmed: a mismatched major version fails" \
                 "with \"incompatible ASan runtimes\" at process start, not" \
                 "a build error). Install it (apt install clang-$LLVM_MAJOR)" \
                 "to enable this. Falling back to coverage-only ASAN (the" \
                 "wrapper only, same as without nightly)."
            ASAN_CC="$CC"
            build_variant "_asan" "-fsanitize=address" "$ASAN_CC" "$RLIB" "$RUST_LIBS -lasan"
        else
            ok "using $ASAN_CC (matches nightly's LLVM $LLVM_MAJOR) for real Rust-side ASAN"
            ASAN_TARGET_DIR="$(mktemp -d)"
            ( cd "$CRATE_DIR" && \
              RUSTC="$NIGHTLY_RUSTC" \
              RUSTFLAGS="-Z sanitizer=address -Cpasses=sancov-module -Cllvm-args=-sanitizer-coverage-level=3 -Cllvm-args=-sanitizer-coverage-trace-pc-guard" \
              cargo build --release --target-dir "$ASAN_TARGET_DIR" )
            ASAN_RLIB="$ASAN_TARGET_DIR/release/libbuggy_rust_target.a"
            [ -f "$ASAN_RLIB" ] || fail "expected ASAN staticlib not found: $ASAN_RLIB"
            # No explicit -lasan here: forcing one on top of what clang
            # auto-selects is exactly what produced the "incompatible
            # runtimes" error above when this was first tried.
            build_variant "_asan" "-fsanitize=address" "$ASAN_CC" "$ASAN_RLIB" "$RUST_LIBS"
            warn "in-process (.so, dlopen via ctypes) loading of this ASAN" \
                 "build was NOT verified to work — confirmed failing with" \
                 "an undefined-symbol error when loaded the naive way." \
                 "This project's own fuzzer.py already has an LD_PRELOAD-" \
                 "based mechanism for loading ASAN .so targets in-process" \
                 "(see _detect_asan and around it); whether it resolves" \
                 "this for a Rust-instrumented .so specifically was not" \
                 "tested in this pass. The executable variant (subprocess" \
                 "mode) was fully verified end to end and is the only" \
                 "path to trust for this ASAN build without further work."
        fi
    else
        warn "no nightly rustc found — ASAN build will not cover the Rust" \
             "code itself (stable rustc has no -Z sanitizer=address); only" \
             "targets/rust_target.c's own few lines get ASAN coverage." \
             "Run tools/fetch_rust_nightly.sh to enable real Rust-side ASAN."
        build_variant "_asan" "-fsanitize=address" "$CC" "$RLIB" "$RUST_LIBS -lasan"
    fi
fi

echo "Rust target build complete."
