/* Thin C wrapper around rust/buggy_target (see docs/handover/
 * handover_rust_target_2026-09-15.md for the full writeup).
 *
 * afl_shim.c is a C source file, -included into whichever translation
 * unit the build compiles as "the target" (see build_target()/
 * build_so_target() in tools/build_targets.sh and the shim's own top
 * comment). rustc does not participate in that -include, so the crate is
 * built separately as a staticlib and THIS file is the actual -include
 * target: it provides the same fuzz_test()/fuzz_shm_run()/main() contract
 * every other target in targets/ provides, and its one call into
 * rust_fuzz_entry() is the whole boundary.
 *
 * Coverage: with stable rustc (the apt default), edges from inside
 * rust_fuzz_entry are invisible to the shim -- this file's own handful of
 * lines are the only instrumented call sites. With the optional nightly
 * toolchain (tools/fetch_rust_nightly.sh, auto-detected by
 * tools/build_rust_target.sh), the crate itself gets real
 * __sanitizer_cov_trace_pc_guard calls too -- see the handover doc for
 * how that was verified.
 *
 * Crash detection works fully either way, regardless of coverage:
 * SIGSEGV/SIGABRT/illegal instruction from the Rust side unwinds through
 * this C frame exactly like a crash from C code, so __afl_guarded_call
 * (direct_lite/in-process) and the forkserver's waitpid() status
 * (subprocess/persistent) both see it and report it the same way.
 */
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>

/* Implemented in rust/buggy_target/src/lib.rs, linked as a static lib by
 * tools/build_rust_target.sh (standalone) or `build_target`/
 * `build_so_target --rust` in tools/build_targets.sh. */
extern int rust_fuzz_entry(const unsigned char *data, size_t len);

__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    return rust_fuzz_entry(buf, len);
}

/* Standard in-process entry point for fuzzer-tool .so mode (see
 * test_target.c, which this mirrors). */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_test(buf, size);
}

int main(void) {
    /* Read all of stdin into a HEAP buffer sized to exactly what was
     * read, not a fixed, over-provisioned scratch buffer. This matters
     * once the Rust side is ASAN-instrumented (see
     * docs/handover/handover_rust_target_2026-09-15.md): ASAN's
     * heap-buffer-overflow check is a redzone around the TRUE allocation
     * boundary, not around however much of it the caller says it's using
     * -- an earlier version of this file read into `char buf[256]` on the
     * stack regardless of actual input length, so any Rust-side overread
     * shorter than the remaining ~250 bytes of headroom silently stayed
     * inside that allocation and never reached ASAN's redzone at all,
     * even with fully working Rust-side ASAN instrumentation. Confirmed
     * empirically, not assumed: a small (a few bytes) overread against
     * that old fixed buffer produced no ASAN report, and the same overread
     * against an exact-size heap allocation (this version) does.
     */
    size_t cap = 4096, len = 0;
    unsigned char *data = malloc(cap);
    if (!data) return 0;
    for (;;) {
        if (len == cap) {
            size_t new_cap = cap * 2;
            unsigned char *grown = realloc(data, new_cap);
            if (!grown) { free(data); return 0; }
            data = grown;
            cap = new_cap;
        }
        ssize_t n = read(0, data + len, cap - len);
        if (n <= 0) break;
        len += (size_t)n;
    }
    if (len == 0) { free(data); return 0; }
    unsigned char *exact = realloc(data, len); /* shrink to the true size */
    if (exact) data = exact; /* a failed shrink just keeps the larger `cap`-sized block */
    int rc = fuzz_test(data, len);
    free(data);
    return rc;
}
