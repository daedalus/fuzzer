# Handover: first Rust fuzz target (2026-09-15)

## What this is

`rust/buggy_target` (a Cargo staticlib) + `targets/rust_target.c` (a thin C
wrapper) + `tools/build_rust_target.sh` (standalone build script) +
`tests/test_regression_rust_target.py`. The first non-C/C++ target in the
tree, built deliberately buggy to answer one question: does the fuzzer's
crash detection work unmodified against a target whose actual bug lives in
Rust code, reached through a C ABI boundary? `afl_shim.c` was explicitly
out of scope for this pass and is untouched — confirmed with `git diff
--stat` before handing this off.

## How it's wired

`afl_shim.c` is a C source file, `-include`d into whichever translation
unit a build treats as "the target" (see `build_target()`/
`build_so_target()` in `tools/build_targets.sh`). rustc doesn't
participate in a C `-include`, so the crate is built separately
(`cargo build --release`, `crate-type = ["staticlib"]`) and
`targets/rust_target.c` is the actual `-include` target — it supplies the
usual `fuzz_test()`/`fuzz_shm_run()`/`main()` contract every other target
in `targets/` supplies, and its one call into
`rust_fuzz_entry()` (declared `extern "C"`, `#[no_mangle]`) is the whole
boundary. Link line:

```
clang -O2 -g -fno-omit-frame-pointer -fsanitize-coverage=trace-pc-guard \
    [-shared -fPIC] -include afl_shim.c \
    -o OUT targets/rust_target.c rust/buggy_target/target/release/libbuggy_rust_target.a \
    -lpthread -ldl -lm -lrt -lutil -lgcc_s
```

`tools/build_rust_target.sh` runs `cargo build --release` then this link
twice (executable + `.so`), following the warn-and-skip-if-toolchain-absent
convention every `vendor_*.sh` script already uses (`cargo` not found →
exit 0, not a build failure). Not folded into `tools/build_targets.sh`
itself, to keep that script's dependency surface C-toolchain-only; nothing
stops a future pass from adding a `--rust` flag there that just calls this
script, if that's wanted.

## The four bugs (`RUST` + selector byte, `targets/rust_target.c` → `rust/buggy_target/src/lib.rs`)

| selector | bug | crash without ASAN? |
|---|---|---|
| `S` | fixed-offset wild read (1 GiB past the buffer) | yes, reliably |
| `O` | read whose offset scales with an attacker byte (`n * 4 MiB`) | yes, for `n` past ~0 |
| `W` | same idea, a write | yes, for `n` past ~0 |
| `P` | safe-Rust panic via checked indexing (no `unsafe` on this path) | yes — `panic = "abort"` |

Verified through **both** real execution paths the fuzzer actually uses,
not just standalone:

- **In-process (`direct`/`direct_lite`)**: loaded the built `.so` via
  `ctypes.CDLL` and called `fuzz_shm_run` through `__afl_guarded_call`
  exactly as `adapters/inprocess.py` does. All four selectors return the
  expected negated signal (`RUSTP` → `-6`/SIGABRT, the other three →
  `-11`/SIGSEGV); a non-triggering input returns `0`. This is the
  regression suite's main coverage (`test_guarded_call_crash_detection`,
  `test_panic_reports_as_abort_not_segv`).
- **Subprocess mode**: plain `fork`+`exec` of the executable variant,
  crash read from `os.waitpid`'s wait status (`returncode < 0`) — the
  same mechanism `persistent_subprocess.py`'s fallback path and any
  one-shot execution rely on. Covered by
  `test_subprocess_mode_detects_crash_via_exit_status`.

## What does NOT work yet, and why (read before extending this)

**Compiler choice matters more here than it looks like it should — use
clang.** `afl_shim.c`'s edge map is populated by compiler-inserted calls
to `__sanitizer_cov_trace_pc_guard` (`-fsanitize-coverage=trace-pc-guard`).
That flag is clang-only: gcc's `-fsanitize-coverage=` accepts `trace-pc`
and `trace-cmp` but not the `trace-pc-guard` variant the shim's edge
callbacks are built on (same fact `tools/build_targets.sh`'s `_pick_cc`
is built around — see its header comment). An earlier version of this
target's build script and test suite built `targets/rust_target.c` with
gcc unconditionally. It linked, ran, and every crash-detection check
passed — gcc doesn't error on the missing flag, it just emits nothing —
so the wrapper silently carried **zero** instrumented call sites, not
"shallow" coverage, none. Confirmed both ways with `objdump -d | grep
__sanitizer_cov_trace_pc_guard`: 0 calls under gcc, 84 under clang with
the flag. Fixed: `tools/build_rust_target.sh` and the test fixtures now
mirror `_pick_cc` (prefer clang, warn-and-fall-back to gcc with an
explicit "zero coverage" warning rather than a silent one), and
`tests/test_regression_rust_target.py::test_wrapper_actually_has_instrumented_call_sites`
asserts real call sites exist rather than trusting a clean build — the
gap that let the gcc version go unnoticed in the first place.

**Update 2026-09-15 (later the same day): nightly rustc tried, edge
coverage from inside the crate now works.** The paragraph above described
this as unattempted because the official channel (`rustup.rs`/
`static.rust-lang.org`) is blocked by this sandbox's network policy
(`host_not_allowed`) and no official nightly binaries exist anywhere
else — confirmed by checking: `rust-lang/rust`'s own GitHub releases
carry only auto-generated source archives, never compiled binaries. The
user approved using an **unofficial third-party mirror** instead:
`a16z/rust`, a fork maintained for an unrelated project (their Jolt
zkVM) whose releases happen to include a full `x86_64-unknown-linux-gnu`
host nightly toolchain. `tools/fetch_rust_nightly.sh` fetches and pins it
by a self-computed SHA256 (there is no independently published one to
check against — read that script's header comment for exactly what that
does and doesn't mean for trust before running it or relying on it
beyond a disposable local build).

With it, `tools/build_rust_target.sh` sets
`RUSTC=<nightly rustc>` and
`RUSTFLAGS="-Cpasses=sancov-module -Cllvm-args=-sanitizer-coverage-level=3 -Cllvm-args=-sanitizer-coverage-trace-pc-guard"`
automatically (auto-detected via `$RUSTC_NIGHTLY` or the script's default
install path — nothing else in the build requires this to be present).
Verified empirically, not assumed: the resulting `.a`'s object code
references `__sanitizer_cov_trace_pc_guard`/`_init` (`nm -u`), and once
linked into the wrapper, `objdump -d -C` shows **12 real guard calls
inside `rust_fuzz_entry` itself** — release-mode inlining folds all four
bug functions into that one symbol, so 12 is the whole crate's branch
structure, not just one function's. Total call sites in the linked `.so`
went from 84 (wrapper-only, clang, stable rustc) to 95 (wrapper + crate,
clang, nightly rustc) — the arithmetic lines up. Crash detection was
re-verified against this build through the same two paths as before
(`__afl_guarded_call` and subprocess wait-status) and behaves identically.
`tests/test_regression_rust_target.py::test_nightly_build_instruments_inside_the_crate`
and `::test_nightly_build_still_detects_all_crashes` cover this, gated to
skip cleanly when no nightly toolchain is present (the default state —
this stays strictly opt-in given the toolchain's provenance).

**What's still not covered even with this toolchain: ASAN on the Rust
side.** This particular `a16z/rust` build does not bundle the `rust-src`
component (`$sysroot/lib/rustlib/src/rust` exists but is empty), and
`-Z build-std` — needed to recompile `core`/`std` with
`-Z sanitizer=address` — requires it. Getting that component would mean
going back to the still-blocked official channel. The ASAN section below
is otherwise unchanged.

**Even with clang, no edge coverage from *inside* the Rust code — using
only the default, stable-rustc build.** Those 84 call sites cover
`targets/rust_target.c` and the few lines of `afl_shim.c` it pulls in —
real, but it's the wrapper's own handful of lines, not `rust/buggy_target`.
Getting rustc to emit the same `__sanitizer_cov_trace_pc_guard` calls
needs `-Z sanitizer-coverage-trace-pc-guard`, which is nightly-only. The
`apt` package on this box is stable rustc 1.75 (`rustc --version`), no
`-Z` flags available. Consequence, **for a stable-only build without the
opt-in nightly toolchain above**: from the shim's point of view, the
entire Rust crate beyond the wrapper is one opaque call (the call to
`rust_fuzz_entry`) — the fuzzer's mutation engine gets no coverage
feedback distinguishing "reached `bug_oob_read`" from "reached
`bug_panic`" *inside* the crate. Crash detection is fully unaffected (it
does not depend on coverage instrumentation at all).

**What ASAN does and doesn't see.** Built an ASAN variant
(`tools/build_rust_target.sh --asan`) and confirmed empirically — not
assumed — that ASAN does not catch the `O`/`W` bugs at small overflow
sizes: `RUSTO\xc8` (200-byte overread) and `RUSTW\xc8` under
`-fsanitize=address` both exit `0`, no report. This is expected once
you know why: ASAN's redzone checks are inserted by the *compiler* at
each instrumented load/store; `rust/buggy_target`'s object code is not
compiled with `-fsanitize=address` (same nightly-only gap as coverage,
`-Z sanitizer=address` + `-Z build-std`), so its raw pointer reads/writes
never consult ASAN's shadow memory at all — regardless of whether the
memory they touch has a poisoned redzone. Only the C wrapper's few lines
are actually ASAN-covered. This is *why* `O`/`W` are designed to scale
their reach into clearly-unmapped memory (`n * 4 MiB`) rather than
imitating `heap_oob_target.c`'s small-overflow-into-a-redzone style: a
small overflow would have been a silent no-op bug here, invisible to both
the plain build and the ASAN build, which would have been a worse target
to hand off than an honest "no small-overflow demo without nightly rustc"
gap. If a future pass gets a nightly toolchain with the `rust-src` component
(this one doesn't — see the update above) plus
`-Z build-std -Z sanitizer=address` working for the crate, redesigning
`O`/`W` back to a small, ASAN-catchable overflow (matching the existing C
sanitizer targets' style) becomes worthwhile and is the natural next step.

**Standalone-executable crash signal can differ from the in-process one
for the same input.** Not a bug introduced here — reproduced against
`targets/test_target.c`'s existing `'S'` trigger (NULL function-pointer
call) built and run the exact same standalone way. `afl_shim.c`'s
`__afl_auto_init` constructor installs the SIGSEGV/SIGABRT/etc. handlers
unconditionally in every edge build; `__afl_crash_handler` always
`siglongjmp`s to `__afl_jmp_buf`. If nothing has called
`__afl_guarded_call` yet (true when you just run the compiled executable
directly, e.g. `printf 'RUSTP' | ./rust_target`), that jump buffer was
never `sigsetjmp`'d and the jump target is garbage — a SIGILL/SIGABRT from
deep in the target can come back out as a SIGSEGV instead, at some
unrelated point after the jump. Confirmed harmless for both real execution
modes: `__afl_guarded_call` always runs with its own jmp buf freshly set
up right before the entry function is called, and subprocess mode reads
the wait status regardless of which signal actually killed the process (a
crash is a crash either way `is_crash()` is concerned). Worth knowing if
anyone runs a target binary by hand while debugging and the signal number
looks wrong — it's the debugging convenience of running it standalone that
doesn't hold, not a shim bug in the modes that matter.

## Suggested next steps (not done here)

1. ~~Nightly rustc + `-Z sanitizer-coverage-trace-pc-guard` for real edge
   coverage from the Rust side~~ — **done**, see the 2026-09-15 update
   above (`tools/fetch_rust_nightly.sh` + auto-detection in
   `tools/build_rust_target.sh`). Remember it's an unofficial third-party
   toolchain, opt-in by design.
2. A toolchain with the `rust-src` component, so `-Z build-std
   -Z sanitizer=address` becomes possible — the `a16z/rust` mirror doesn't
   include it. Would need either a different mirror that does, or the
   still-blocked official channel. Once available: redesign `O`/`W` as
   small, redzone-catchable overflows matching the C sanitizer targets'
   style (currently they scale their reach into clearly-unmapped memory
   instead, specifically because nothing here could catch a small one).
3. cmplog for the Rust side would need the same nightly
   `-Z sanitizer-coverage-trace-cmp` support — not investigated at all in
   this pass, though the toolchain now in hand could plausibly support it;
   worth a quick check before assuming it needs more setup.
4. If it's ever worth wiring into `tools/build_targets.sh` proper (a
   `--rust` flag calling this script, or reusing `build_target`/
   `build_so_target` directly against the staticlib), the `RUST_LIBS`
   list in `tools/build_rust_target.sh` was determined empirically from
   link errors, not from `rustc --print native-static-libs` (which needs
   the crate already built with that flag threaded through) — worth
   double-checking against whatever rustc version ships wherever this
   runs next, since libstd's own native dependencies do shift across
   versions.
5. If this project ever wants to depend on the nightly toolchain more
   than opt-in-for-better-coverage, its provenance (unofficial, no
   published checksum, built for an unrelated project) is worth revisiting
   rather than just continuing to rely on the pin in
   `tools/fetch_rust_nightly.sh`.
