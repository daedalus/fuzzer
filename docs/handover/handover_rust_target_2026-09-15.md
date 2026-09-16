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

**Update 2026-09-16: O and W were redesigned.** They used to scale their
reach by a huge multiplier (`n * 4 MiB`) specifically so they'd crash
reliably even without ASAN — because at the time, nothing here could make
ASAN see the Rust side at all. That's no longer true (see the big update
below), so they're back to being real, small, realistic overflows —
`n` bytes past the allocation, no scaling — matching this project's other
sanitizer targets (`heap_oob_target.c`, `asan_target.c`). The table below
reflects the current design.

| selector | bug | crash without ASAN? |
|---|---|---|
| `S` | fixed-offset wild read (1 GiB past the buffer) | yes, reliably |
| `O` | read of `n` bytes past the allocation, `n` = an attacker byte | no, silently reads adjacent live memory (verified) |
| `W` | same idea, a write | no, silently writes adjacent live memory (verified) |
| `P` | safe-Rust panic via checked indexing (no `unsafe` on this path) | yes — `panic = "abort"` |

Verified through **both** real execution paths the fuzzer actually uses,
not just standalone:

- **In-process (`direct`/`direct_lite`)**: loaded the built `.so` via
  `ctypes.CDLL` and called `fuzz_shm_run` through `__afl_guarded_call`
  exactly as `adapters/inprocess.py` does. `S`/`P` return the expected
  negated signal (`RUSTP` → `-6`/SIGABRT, `RUSTS` → `-11`/SIGSEGV); `O`/`W`
  and a non-triggering input all return `0` (by design — see above). This
  is the regression suite's main coverage
  (`test_guarded_call_crash_detection`, `test_panic_reports_as_abort_not_segv`).
- **Subprocess mode**: plain `fork`+`exec` of the executable variant,
  crash read from `os.waitpid`'s wait status (`returncode < 0`) — the
  same mechanism `persistent_subprocess.py`'s fallback path and any
  one-shot execution rely on. Covered by
  `test_subprocess_mode_detects_crash_via_exit_status`.

## Update 2026-09-16: ASAN now genuinely catches O and W

Asked to "try fixing the ASAN crash" after the 2026-09-15 update below
documented it as a known gap. It's fixed for subprocess-mode execution,
verified end to end with a real, fully symbolicated
`AddressSanitizer: heap-buffer-overflow` report pointing at the actual
Rust source line — not assumed, not a raw crash mistaken for one. Getting
there required finding and fixing **three separate bugs**, each of which
would have silently defeated the others if left in place:

**1. `-Z build-std` turned out not to be needed at all.** The
2026-09-15 update below assumed ASAN parity needed `-Z build-std
-Z sanitizer=address` (rebuilding `core`/`std` from source), and that this
toolchain couldn't do it because it lacks the `rust-src` component. Tested
directly instead of continuing to assume: `-Z sanitizer=address` alone,
with no `-Z build-std`, compiles cleanly and produces real
`__asan_report_*` references in the crate's object code. This makes
sense in hindsight — ASAN's heap redzones come from intercepting
malloc/free, which happens regardless of whether the calling code's
instrumented, and instrumenting *this* crate's own loads/stores is all
that's needed to catch overflows *in this crate's own code*. `-Z build-std`
would only matter for a bug inside `std` itself. The missing `rust-src`
component is a real, still-open gap (see below), just not the one that
was blocking this.

**2. Mismatched ASAN runtime versions.** Linking the ASAN-instrumented
Rust object with the system's default `clang` (LLVM 18) failed at
process start with `Your application is linked against incompatible ASan
runtimes` — not a build error, a runtime one. The nightly rustc bundles
LLVM 20.1.7; `apt install clang-20` (LLVM 20.1.2 — close enough on the
major version to interoperate, confirmed) fixed it. Also confirmed:
forcing an explicit `-lasan` on the link line reintroduces the same
failure even with matching versions, by pulling in a second, differently-
sourced runtime on top of whatever clang already auto-selects for
`-fsanitize=address` — don't pass it manually. `tools/build_rust_target.sh`
now parses `rustc --version --verbose`'s `LLVM version:` line and looks
for a matching `clang-N`, warning and falling back to coverage-only ASAN
(the old, more limited behavior) if none is found.

**3. A real Rust UB bug in `bug_oob_write` itself, unrelated to the other
two.** Even with (1) and (2) both fixed, `W` produced no crash and no
ASAN report at all — just silently returned success. `bug_oob_write` took
`buf: &[u8]` and wrote through a `*mut u8` cast from `buf.as_ptr()`.
That's undefined behavior independent of the out-of-bounds access: a
`&[u8]` carries a `noalias`+readonly contract, so LLVM is entitled to
assume nothing writes through any pointer derived from one — and in the
real crate build (though not in a minimal isolated probe used to first
confirm write-side ASAN detection works at all), it exercised exactly
that entitlement and eliminated the entire write loop as a provably-dead
store. Fixed by changing `bug_oob_write` (and the `fuzz_me` dispatch that
calls it) to take the raw `*const u8`/`len` pair instead of a slice, so no
aliasing `&[u8]` is ever formed over memory the function is about to
mutate through a raw pointer. Worth remembering for any future bug added
here: deriving a mutable raw pointer from an existing shared reference and
writing through it is UB in Rust *regardless of whether the write is
in-bounds* — this has nothing to do with fuzzing or ASAN specifically, it
would have been silently wrong even in safe, bounds-checked-adjacent code.

**Also fixed along the way, in `targets/rust_target.c`:** `main()` used to
read into a fixed `char buf[256]` stack buffer regardless of actual input
length. ASAN's overflow check is a redzone around the *true* allocation
boundary, not around however much of it the caller says it's using — so
any Rust-side overread shorter than the buffer's remaining unused capacity
(most of them, for short inputs) stayed inside that allocation and never
reached a redzone at all, independent of whether the Rust side was
instrumented. `main()` now reads all of stdin into a heap buffer sized
exactly to what was actually read (`realloc` down to the exact length)
before calling `fuzz_test`, matching how the in-process ctypes path
already worked (a `ctypes` array is already exact-size).

**Confirmed working**, via `tests/test_regression_rust_target.py::test_asan_catches_small_overflows`:
both `O` and `W`, with a small `n` (10 in the test), produce a genuine
`AddressSanitizer: heap-buffer-overflow` report with a full, correctly
symbolicated stack trace through `main` → `fuzz_test` → `rust_fuzz_entry`
→ `fuzz_me` → the actual bug function, landing on the real `.rs` line
number. `W`'s report is via `__asan_memset` — LLVM vectorized the
byte-by-byte write loop into a `memset` call, and ASAN's interceptor
catches it there instead of via an inlined per-byte check; still a fully
attributed, correct report.

**Confirmed NOT working, still open:** loading this ASAN-instrumented
`.so` in-process via a naive `ctypes.CDLL` fails immediately with
`undefined symbol: __asan_option_detect_stack_use_after_return` — dlopen'ing
an ASAN-instrumented shared library into a host process that wasn't
itself built with ASAN is a known-fragile combination in general. This
project's own `services/fuzzer.py` already has a real mechanism for this
exact problem (`_detect_asan`, then `LD_PRELOAD`-ing the system's
`libasan.so.8` with `verify_asan_link_order=0` before `ctypes.CDLL` ever
runs) — but that mechanism is built around GCC's shared `libasan.so.8`,
and whether it's compatible with this nightly-rustc/clang-20 combination's
runtime specifically was **not tested** in this pass. Don't assume it
works without checking. The executable variant (subprocess mode) needed
none of that and is fully verified — trust that one, not the `.so` for
in-process use, without further work.

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

**Update 2026-09-16: ASAN on the Rust side is now real, for subprocess
mode.** The paragraph below was accurate as of the 2026-09-15 nightly-rustc
work, before ASAN was specifically debugged — see the "ASAN now genuinely
catches O and W" section near the top of this document for what changed,
what was fixed, and what's still open (in-process/dlopen loading of the
ASAN `.so`, not yet verified).

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

**What ASAN does and doesn't see — superseded, see the 2026-09-16 update
above.** (Originally: "ASAN does not catch the `O`/`W` bugs at small
overflow sizes... this is why they're designed to scale their reach into
clearly-unmapped memory instead." That turned out to be three separate,
independently fixable problems rather than one hard toolchain limit — the
update above walks through all three. `O`/`W` are back to small,
realistic overflow sizes as a result; see the bug table further up.)

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
2. ~~ASAN parity for the Rust side, redesign `O`/`W` as small,
   redzone-catchable overflows~~ — **done for subprocess mode**, see the
   2026-09-16 update near the top. `-Z build-std`/`rust-src` turned out
   to be unnecessary for this specific goal (it matters for bugs inside
   `std` itself, not for this crate's own code) — don't assume it's
   still needed before checking.
3. In-process (`.so`, `ctypes.CDLL`) loading of the ASAN build — **not
   done**, confirmed failing with an undefined-symbol error the naive
   way. This project's `services/fuzzer.py` has an existing
   `LD_PRELOAD`-based mechanism for ASAN `.so` targets in-process
   (`_detect_asan` and the code around it) built around GCC's system
   `libasan.so.8`; whether it's compatible with this nightly-rustc/
   clang-20 combination's runtime was not tested. Check that before
   assuming in-process ASAN fuzzing of this target works at all.
4. cmplog for the Rust side would need the same nightly
   `-Z sanitizer-coverage-trace-cmp` support — not investigated at all in
   this pass, though the toolchain now in hand could plausibly support it;
   worth a quick check before assuming it needs more setup.
5. If it's ever worth wiring into `tools/build_targets.sh` proper (a
   `--rust` flag calling this script, or reusing `build_target`/
   `build_so_target` directly against the staticlib), the `RUST_LIBS`
   list in `tools/build_rust_target.sh` was determined empirically from
   link errors, not from `rustc --print native-static-libs` (which needs
   the crate already built with that flag threaded through) — worth
   double-checking against whatever rustc version ships wherever this
   runs next, since libstd's own native dependencies do shift across
   versions.
6. If this project ever wants to depend on the nightly toolchain more
   than opt-in-for-better-coverage, its provenance (unofficial, no
   published checksum, built for an unrelated project) is worth revisiting
   rather than just continuing to rely on the pin in
   `tools/fetch_rust_nightly.sh`.
