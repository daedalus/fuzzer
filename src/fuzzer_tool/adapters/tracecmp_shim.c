/* tracecmp_shim.c — LD_PRELOAD shim for compiler-IR comparison tracing.

Implements Clang's -fsanitize-coverage=trace-cmp callbacks to intercept
comparisons at the IR level — after the compiler has already inlined/folded
small constant-length memcmp into integer compares. This catches comparisons
that symbol-based interposition (cmplog_shim.c) misses.

Intercepted callbacks:
  __sanitizer_cov_trace_cmp{1,2,4,8}
  __sanitizer_cov_trace_const_cmp{1,2,4,8}
  __sanitizer_cov_trace_switch

Both shims coexist: cmplog_shim intercepts libc calls, this shim intercepts
compiler-inlined comparisons. Both write to the same _CMPLOG_OUT file.

Protocol: each comparison writes a line to _CMPLOG_OUT:
  CMP <hex_operand1> <hex_operand2> <cmp_result> <len>\n

Usage:
  LD_PRELOAD=./tracecmp_shim.so _CMPLOG_OUT=/tmp/cmp.log ./target

Requires: target compiled with clang -fsanitize-coverage=trace-cmp,trace-pc-guard
*/

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Buffered writer ──────────────────────────────────────────────────
 * trace_cmp callbacks fire on EVERY comparison in the binary — per-call
 * fprintf is unacceptable. We buffer output and flush on full/exit/crash.
 */

#define BUFFER_SIZE (256 * 1024)  /* 256KB write buffer */
#define MAX_SWITCH_CASES 256

static char cmplog_buffer[BUFFER_SIZE];
static size_t cmplog_buf_pos = 0;
static FILE *cmplog_file = NULL;

static void flush_buffer(void) {
    if (cmplog_buf_pos == 0 || !cmplog_file) return;
    fwrite(cmplog_buffer, 1, cmplog_buf_pos, cmplog_file);
    cmplog_buf_pos = 0;
}

/* Append a CMP line to the buffer. n is the comparison width in bytes (1/2/4/8).
 * a and b are the operands, result is their signed comparison result. */
static inline void buffer_cmp(uint64_t a, uint64_t b, size_t n) {
    /* Worst case: "CMP " + 16 hex + " " + 16 hex + " " + "-9223372036854775808" + " " + "8" + "\n"
     * = ~72 chars. Leave generous margin. */
    if (cmplog_buf_pos + 80 > BUFFER_SIZE) {
        flush_buffer();
    }

    static const char hex[] = "0123456789abcdef";
    char *p = cmplog_buffer + cmplog_buf_pos;

    /* "CMP " */
    *p++ = 'C'; *p++ = 'M'; *p++ = 'P'; *p++ = ' ';

    /* Hex encode operand a (n bytes, little-endian) */
    for (size_t i = 0; i < n; i++) {
        uint8_t byte = (uint8_t)(a >> (i * 8));
        *p++ = hex[byte >> 4];
        *p++ = hex[byte & 0xf];
    }

    *p++ = ' ';

    /* Hex encode operand b (n bytes, little-endian) */
    for (size_t i = 0; i < n; i++) {
        uint8_t byte = (uint8_t)(b >> (i * 8));
        *p++ = hex[byte >> 4];
        *p++ = hex[byte & 0xf];
    }

    *p++ = ' ';

    /* Comparison result as signed decimal */
    int64_t result;
    if (a < b) result = -1;
    else if (a > b) result = 1;
    else result = 0;
    p += sprintf(p, "%ld %zu\n", (long)result, n);

    cmplog_buf_pos = (size_t)(p - cmplog_buffer);
}

/* ── Lifecycle ──────────────────────────────────────────────────────── */

static void flush_and_close(void) {
    flush_buffer();
    if (cmplog_file) {
        fclose(cmplog_file);
        cmplog_file = NULL;
    }
}

static void crash_handler(int sig) {
    flush_buffer();
    if (cmplog_file) fflush(cmplog_file);
    signal(sig, SIG_DFL);
    raise(sig);
}

static void install_crash_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
}

static int handlers_installed = 0;

static void __attribute__((constructor)) init_tracecmp(void) {
    const char *path = getenv("_CMPLOG_OUT");
    if (path && path[0]) {
        cmplog_file = fopen(path, "a");
    }
    if (!handlers_installed) {
        install_crash_handlers();
        handlers_installed = 1;
    }
}

static void __attribute__((destructor)) fini_tracecmp(void) {
    flush_and_close();
}

/* ── Typed comparison callbacks ───────────────────────────────────────
 * Clang inserts calls to these after each inlined comparison.
 * The _const variants have identical signatures — one operand is a
 * compile-time constant embedded in the instruction, but the callback
 * receives both as normal arguments.
 */

void __sanitizer_cov_trace_cmp1(uint8_t arg1, uint8_t arg2) {
    buffer_cmp(arg1, arg2, 1);
}

void __sanitizer_cov_trace_cmp2(uint16_t arg1, uint16_t arg2) {
    buffer_cmp(arg1, arg2, 2);
}

void __sanitizer_cov_trace_cmp4(uint32_t arg1, uint32_t arg2) {
    buffer_cmp(arg1, arg2, 4);
}

void __sanitizer_cov_trace_cmp8(uint64_t arg1, uint64_t arg2) {
    buffer_cmp(arg1, arg2, 8);
}

void __sanitizer_cov_trace_const_cmp1(uint8_t arg1, uint8_t arg2) {
    buffer_cmp(arg1, arg2, 1);
}

void __sanitizer_cov_trace_const_cmp2(uint16_t arg1, uint16_t arg2) {
    buffer_cmp(arg1, arg2, 2);
}

void __sanitizer_cov_trace_const_cmp4(uint32_t arg1, uint32_t arg2) {
    buffer_cmp(arg1, arg2, 4);
}

void __sanitizer_cov_trace_const_cmp8(uint64_t arg1, uint64_t arg2) {
    buffer_cmp(arg1, arg2, 8);
}

/* ── Switch statement tracing ─────────────────────────────────────────
 * Clang's -fsanitize-coverage=trace-switch passes the switch value and
 * a pointer to a uint64_t array where ref[-1] is the number of cases.
 * We log each case value against the switch value.
 */

void __sanitizer_cov_trace_switch(uint64_t val, uint64_t *ref) {
    if (!ref) return;

    /* ref[-1] holds the case count (Clang convention) */
    int64_t count = (int64_t)ref[-1];
    if (count <= 0 || count > MAX_SWITCH_CASES) return;

    for (int64_t i = 0; i < count; i++) {
        buffer_cmp(val, ref[i], 8);
    }
}

/* ── Public API for in-process mode ──────────────────────────────────
 * Called by the fuzzer via ctypes when trace-cmp is compiled into
 * the target .so. Named __tracecmp_reset to avoid conflict with
 * cmplog_shim.c's __cmplog_reset.
 */

__attribute__((visibility("default")))
void __tracecmp_reset(void) {
    if (cmplog_file) {
        flush_buffer();
        const char *path = getenv("_CMPLOG_OUT");
        if (path && path[0]) {
            fclose(cmplog_file);
            cmplog_file = fopen(path, "w");
        }
    }
}

__attribute__((visibility("default")))
const char *__tracecmp_get_path(void) {
    return getenv("_CMPLOG_OUT");
}

__attribute__((visibility("default")))
void __tracecmp_flush(void) {
    flush_buffer();
    if (cmplog_file) fflush(cmplog_file);
}
