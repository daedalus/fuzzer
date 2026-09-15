/* Thin C wrapper around rust/buggy_target (see docs/handover/
 * handover_rust_target_2026-09-15.md for the full writeup).
 *
 * afl_shim.c is a C source file, -included into whichever translation
 * unit the build compiles as "the target" (see build_target()/
 * build_so_target() in tools/build_targets.sh and the shim's own top
 * comment). rustc does not participate in that -include, and the stable
 * toolchain available here cannot emit -Z sanitizer-coverage-trace-pc-guard
 * calls to make Rust code visible to the shim's edge map directly. So the
 * Rust crate is built separately as a staticlib, and THIS file is the
 * translation unit the shim gets -included into: it provides the same
 * fuzz_test()/fuzz_shm_run()/main() contract every other target in
 * targets/ provides, and its one call into Rust is the coverage boundary.
 *
 * That boundary is a real limitation, not a hidden one: edge coverage from
 * inside rust_fuzz_entry is invisible to the fuzzer, so the search cannot
 * distinguish "reached bug_oob_read" from "reached bug_panic" by coverage
 * feedback alone -- from the shim's point of view they are the same edge.
 * What still works fully is crash detection: SIGSEGV/SIGABRT/illegal
 * instruction from the Rust side unwinds through this C frame exactly like
 * a crash from C code, so __afl_guarded_call (direct_lite/in-process) and
 * the forkserver's waitpid() status (subprocess/persistent) both see it
 * and report it the same way. That is the property this target exists to
 * test first -- see the handover doc for what a next pass could add
 * (per-input random data via a length-prefixed corpus format to get any
 * coverage feedback at all without nightly rustc).
 */
#include <stddef.h>
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
    char buf[256];
    ssize_t n = read(0, buf, sizeof(buf) - 1);
    if (n <= 0) return 0;
    return fuzz_test((unsigned char *)buf, (size_t)n);
}
