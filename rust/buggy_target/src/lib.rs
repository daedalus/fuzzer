//! Deliberately buggy fuzz target, in Rust, for exercising the fuzzer
//! against a non-C/C++ target for the first time.
//!
//! Mirrors targets/test_target.c's contract: a magic prefix selects which
//! bug fires, so the fuzzer's mutation engine has to *discover* each one
//! rather than the harness taking an if/else on len alone. Unlike
//! test_target.c, this file has no coverage instrumentation of its own
//! (see targets/rust_target.c and docs/handover/handover_rust_target_*.md
//! for why): the stable rustc shipped by the base image cannot emit
//! `-Z sanitizer-coverage-trace-pc-guard` calls, so afl_shim.c's edge map
//! only sees the one call site in the C wrapper. Crash detection (the
//! forkserver, the signal handler, ASAN reports) does not depend on that
//! and works exactly as it does for a C target.
//!
//! Four independent bugs behind the same "RUST" prefix, each needing its
//! own byte to be found by mutation -- deliberately not reachable by a
//! single short input, the same reason test_target.c gates on
//! buf[5] after matching "CRASH":
//!   RUST\x53...   ('S') wild out-of-bounds read, a fixed distance past
//!                 the allocation, far enough to fault even without ASAN.
//!   RUST\x4f<n>   ('O') out-of-bounds read whose distance past the
//!                 allocation scales with attacker-controlled byte `n` --
//!                 the fuzzer has to find a large enough n to reach
//!                 unmapped memory, rather than the offset being fixed.
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
    // Bounded by the caller-supplied len -- this slice itself is sound.
    // Every bug below is reached *from* here by deliberately stepping
    // outside it.
    let buf = std::slice::from_raw_parts(data, len);
    fuzz_me(buf);
    0
}

fn fuzz_me(buf: &[u8]) {
    if buf.len() < 5 || &buf[0..4] != b"RUST" {
        return;
    }
    match buf[4] {
        b'S' => bug_wild_read(buf),
        b'O' => bug_oob_read(buf),
        b'W' => bug_oob_write(buf),
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

/// Unchecked out-of-bounds *read*, distance scaled by an attacker-controlled
/// byte (buf[5]) so the fuzzer has to find a large enough value for the
/// overrun to actually reach unmapped memory -- same "search for the
/// magnitude" shape as bug_wild_read, just data-dependent instead of fixed.
///
/// This is NOT the small-overflow-into-a-redzone bug it looks like at first
/// (compare heap_oob_target.c/asan_target.c, which lean on ASAN redzones
/// for exactly that). It can't be, here: ASAN only checks shadow memory
/// from code the compiler instrumented, and rustc 1.75 (the stable
/// toolchain `apt` has on this box) has no `-Z sanitizer=address` /
/// `-Z build-std` to instrument the Rust side at all -- confirmed by
/// building this crate under `-fsanitize=address` in the C wrapper and
/// observing buf[5] up to 255 (the largest a single byte allows) pass
/// through silently, no ASAN report, no crash. A byte-sized overrun into a
/// libc redzone or glibc heap slack is real but invisible under this
/// toolchain either way. Scaling the reach by a few MiB per unit of `n`
/// sidesteps the whole question: large `n` now walks off the end of
/// mapped memory outright, which every build (ASAN or not) reports as a
/// plain SIGSEGV. Getting genuine ASAN parity for the Rust side is real
/// follow-up work, not a hidden gap -- see docs/handover/
/// handover_rust_target_2026-09-15.md.
fn bug_oob_read(buf: &[u8]) {
    if buf.len() < 6 {
        return;
    }
    const UNIT: usize = 4 * 1024 * 1024; // 4 MiB per step
    let n = buf[5] as usize;
    unsafe {
        let p = buf.as_ptr().add(6).add(n * UNIT);
        let v = std::ptr::read_volatile(p);
        std::hint::black_box(v);
    }
}

/// Same scaled-reach idea as bug_oob_read, but a write -- a distinct crash
/// signature worth the fuzzer telling apart from a read fault (writes to
/// a genuinely unmapped page still SIGSEGV, but a write landing just past
/// mapped memory on a read-only page, e.g. inside the same binary's
/// .rodata/.text if the offset happens to be small and negative-adjacent,
/// would fault differently than a read of the same address -- this
/// target's reach is large enough that it lands past everything mapped
/// either way, but the *kind* of fault the fuzzer's crash triage records
/// still differs by access type at the hardware level).
fn bug_oob_write(buf: &[u8]) {
    if buf.len() < 6 {
        return;
    }
    const UNIT: usize = 4 * 1024 * 1024; // 4 MiB per step
    let n = buf[5] as usize;
    unsafe {
        let p = buf.as_ptr().add(6).add(n * UNIT) as *mut u8;
        std::ptr::write_volatile(p, 0x41);
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
