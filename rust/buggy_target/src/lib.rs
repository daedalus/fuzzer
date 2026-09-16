//! Deliberately buggy fuzz target, in Rust, for exercising the fuzzer
//! against a non-C/C++ target for the first time.
//!
//! Mirrors targets/test_target.c's contract: a magic prefix selects which
//! bug fires, so the fuzzer's mutation engine has to *discover* each one
//! rather than the harness taking an if/else on len alone.
//!
//! Coverage instrumentation of this crate's own code is optional and needs
//! nightly rustc (`tools/fetch_rust_nightly.sh`, auto-detected by
//! `tools/build_rust_target.sh`) -- with the default stable toolchain,
//! afl_shim.c's edge map only sees the one call site in the C wrapper.
//! Crash detection does not depend on coverage instrumentation either way.
//! See docs/handover/handover_rust_target_2026-09-15.md for what's
//! verified for each toolchain combination.
//!
//! Four independent bugs behind the same "RUST" prefix, each needing its
//! own byte to be found by mutation -- deliberately not reachable by a
//! single short input, the same reason test_target.c gates on
//! buf[5] after matching "CRASH":
//!   RUST\x53...   ('S') wild out-of-bounds read, a fixed distance past
//!                 the allocation, far enough to fault even without ASAN.
//!   RUST\x4f<n>   ('O') out-of-bounds read of `n` bytes past the
//!                 allocation -- a real, small heap-buffer-overflow (like
//!                 heap_oob_target.c/asan_target.c elsewhere in this
//!                 project), not a fixed or scaled reach. Silent on a
//!                 plain build for most `n` (verified: it just reads
//!                 adjacent live memory); a genuine ASAN
//!                 heap-buffer-overflow report with real Rust-side
//!                 instrumentation (nightly toolchain + a matching-version
//!                 C compiler -- see the handover doc for exactly which
//!                 combination was verified to interoperate).
//!   RUST\x57<n>   ('W') same idea as 'O', but a write.
//!   RUST\x50      ('P') safe-Rust panic (checked indexing), which under
//!                 panic=abort surfaces as an abort trap rather than an
//!                 unwind, exercising the shim's abort()-interception
//!                 path from Rust code instead of C.

use std::os::raw::c_int;

/// C ABI entry point. Contract matches fuzz_test()/fuzz_shm_run() in every
/// existing C target: interpret `data[0..len]` as one input, return 0.
/// Actual crashes happen via signal (SIGSEGV/SIGABRT), not return value --
/// same convention the fuzzer already reads from C targets, so nothing on
/// the Python or shim side needs to know this target happens to be Rust.
///
/// # Safety
/// `data` must point to at least `len` readable bytes, or be null when
/// `len == 0`. The C wrapper (targets/rust_target.c) upholds this the same
/// way it upholds it for the C targets it was copied from.
#[no_mangle]
pub unsafe extern "C" fn rust_fuzz_entry(data: *const u8, len: usize) -> c_int {
    if data.is_null() || len == 0 {
        return 0;
    }
    fuzz_me(data, len);
    0
}

fn fuzz_me(data: *const u8, len: usize) {
    // Bounded by the caller-supplied len -- this slice itself is sound.
    // Every read-only bug below is reached *from* here by deliberately
    // stepping outside it. bug_oob_write is the one exception: it takes
    // the raw pointer instead (see its own doc comment for why forming a
    // `&[u8]` over memory about to be mutated through a raw pointer would
    // be its own, unrelated bug).
    let buf = unsafe { std::slice::from_raw_parts(data, len) };
    if buf.len() < 5 || &buf[0..4] != b"RUST" {
        return;
    }
    match buf[4] {
        b'S' => bug_wild_read(buf),
        b'O' => bug_oob_read(buf),
        b'W' => bug_oob_write(data, len),
        b'P' => bug_panic(buf),
        _ => {}
    }
}

/// Wild out-of-bounds read: walk far enough past the input buffer that the
/// read lands on an unmapped page essentially every time, independent of
/// ASAN. This is the Rust analogue of test_target.c's NULL-function-pointer
/// call -- a crash the plain (non-ASAN) build can also catch, so the
/// fuzzer's crash path gets exercised even on a fast/no-ASAN campaign.
fn bug_wild_read(buf: &[u8]) {
    // 1 GiB past the real allocation. Two calls to defeat a smart optimizer
    // treating the offset as always in-bounds and folding the whole
    // function to a no-op -- volatile_read-style through a raw pointer,
    // same intent as test_target.c's `volatile` crash_fn.
    const WILD_OFFSET: usize = 1 << 30;
    unsafe {
        let p = buf.as_ptr().add(WILD_OFFSET);
        let v = std::ptr::read_volatile(p);
        std::hint::black_box(v);
    }
}

/// Unchecked out-of-bounds *read* of a small, attacker-controlled number
/// of bytes (`buf[5]`) past the input — no scaling, no artificial reach.
/// This mirrors the project's other sanitizer targets (`heap_oob_target.c`,
/// `asan_target.c`): a real small heap-buffer-overflow, invisible on a
/// plain build (verified: it silently reads adjacent, live, mapped memory
/// rather than faulting, both when the underlying allocation is a heap
/// buffer sized exactly to the input — see `targets/rust_target.c`'s
/// `main()` — and via the in-process ctypes path, which allocates the
/// same way), and a genuine `AddressSanitizer: heap-buffer-overflow`
/// report — with a symbolicated stack trace into this file — once the
/// Rust code itself is ASAN-instrumented. That last part needed two
/// things this project's default toolchain doesn't have on its own:
/// nightly rustc's `-Z sanitizer=address` (no `-Z build-std` required —
/// confirmed unnecessary for catching overflows in *this* crate's own
/// code, since ASAN's heap redzones come from intercepting malloc/free
/// regardless of who calls them; `-Z build-std` only matters for bugs
/// *inside* std itself), and a C compiler whose bundled ASAN runtime
/// actually matches rustc's LLVM version closely enough to interoperate
/// (confirmed: system clang's default, LLVM 18, produces "incompatible
/// ASan runtimes" against this nightly's LLVM 20; `clang-20` works).
/// Full validation, including why the *previous* version of this bug
/// (scaled to a huge fixed reach) could never have shown any of this, in
/// docs/handover/handover_rust_target_2026-09-15.md.
fn bug_oob_read(buf: &[u8]) {
    if buf.len() < 6 {
        return;
    }
    let n = buf[5] as usize;
    unsafe {
        let p = buf.as_ptr().add(6);
        let mut acc: u32 = 0;
        for i in 0..n {
            acc = acc.wrapping_add(*p.add(i) as u32);
        }
        std::hint::black_box(acc);
    }
}

/// Same idea as bug_oob_read, but a write -- a distinct crash signature
/// (and, under ASAN, a distinct report: "heap-buffer-overflow ... WRITE
/// of size 1" instead of "READ of size 1") worth the fuzzer telling apart
/// from a read fault.
///
/// Takes the raw pointer/length, NOT a `&[u8]`, and this isn't
/// stylistic: an earlier version derived a `*mut u8` from a `&[u8]` and
/// wrote through that. That's undefined behavior in Rust independent of
/// bounds -- a `&[u8]` carries a noalias+readonly contract, so LLVM is
/// entitled to assume nothing writes through any pointer derived from it,
/// and in the actual crate build (not the minimal probe used to first
/// validate this) it exercised exactly that entitlement: the whole write
/// loop was optimized away as a provably-dead store, silently, no crash,
/// no ASAN report, nothing -- confirmed by comparing this function in
/// isolation (which reproduced it) against an otherwise-identical probe
/// that took `*mut u8` directly (which didn't). Fixed by never forming a
/// `&[u8]` over memory this function is about to mutate through a raw
/// pointer at all.
fn bug_oob_write(data: *const u8, len: usize) {
    if len < 6 {
        return;
    }
    // SAFETY: `data` is valid for `len` reads (rust_fuzz_entry's own
    // safety contract); reading a single byte at offset 5 is in-bounds.
    let n = unsafe { *data.add(5) } as usize;
    let p = unsafe { data.add(6) } as *mut u8;
    for i in 0..n {
        // SAFETY: deliberately NOT checked against `len` -- that missing
        // check is the bug under test.
        unsafe {
            *p.add(i) = 0x41;
        }
    }
}

/// Safe-Rust panic via checked indexing. No `unsafe` at all on this path --
/// included because a panic under panic=abort is a different signal
/// (illegal instruction / abort trap, depending on target and rustc
/// version) than either of the two segfaults above, and the fuzzer's crash
/// classification should not assume every crash from this binary is a
/// SIGSEGV.
fn bug_panic(buf: &[u8]) {
    let idx = buf.len() + 1; // guaranteed out of range
    let _ = buf[idx]; // panics: "index out of bounds"
}
