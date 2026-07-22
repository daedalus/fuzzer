/* cmplog_shim.c — Unified LD_PRELOAD shim for comparison tracing.
 *
 * Two interception layers, one shared log file:
 *
 * 1. Symbol-based: intercepts libc comparison functions via dlsym(RTLD_NEXT)
 *    (memcmp, strcmp, strncmp, memchr, strcasecmp, strncasecmp, memmem,
 *    strstr, strcasestr) — catches explicit library calls at the PLT level.
 *
 * 2. Compiler-IR-based: implements Clang's -fsanitize-coverage=trace-cmp
 *    callbacks (__sanitizer_cov_trace_cmp{1,2,4,8}, trace_const_cmp*,
 *    trace_switch) — catches comparisons the compiler has inlined/folded
 *    into integer compares that symbol interposition cannot see.
 *
 * Both layers write to the same _CMPLOG_OUT file in the same CMP line format.
 *
 * Protocol:
 *   CMP <hex_operand1> <hex_operand2> <cmp_result> [<len>]\n
 *
 * Usage:
 *   LD_PRELOAD=./cmplog_shim.so _CMPLOG_OUT=/tmp/cmp.log ./target
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Shared state ────────────────────────────────────────────────────── */
static FILE *cmplog_file = NULL;

/* ── Buffered writer for high-frequency IR callbacks ────────────────── */
#define BUFFER_SIZE (256 * 1024)  /* 256KB write buffer */

static char cmplog_buffer[BUFFER_SIZE];
static size_t cmplog_buf_pos = 0;

static void flush_buffer(void) {
    if (cmplog_buf_pos == 0) return;
    if (cmplog_file) {
        fwrite(cmplog_buffer, 1, cmplog_buf_pos, cmplog_file);
    }
    cmplog_buf_pos = 0;
}

/* Append a CMP line to the buffer. n is the comparison width in bytes.
 * a and b are the operands, result is their signed comparison result.
 * Returns immediately when cmplog_file is NULL (no --cmplog flag). */
static inline void buffer_cmp(uint64_t a, uint64_t b, size_t n) {
    if (!cmplog_file) return;

    if (cmplog_buf_pos + 80 > BUFFER_SIZE) {
        flush_buffer();
    }

    static const char hex[] = "0123456789abcdef";
    char *p = cmplog_buffer + cmplog_buf_pos;

    *p++ = 'C'; *p++ = 'M'; *p++ = 'P'; *p++ = ' ';

    for (size_t i = 0; i < n; i++) {
        uint8_t byte = (uint8_t)(a >> (i * 8));
        *p++ = hex[byte >> 4];
        *p++ = hex[byte & 0xf];
    }

    *p++ = ' ';

    for (size_t i = 0; i < n; i++) {
        uint8_t byte = (uint8_t)(b >> (i * 8));
        *p++ = hex[byte >> 4];
        *p++ = hex[byte & 0xf];
    }

    *p++ = ' ';

    int64_t result;
    if (a < b) result = -1;
    else if (a > b) result = 1;
    else result = 0;
    p += sprintf(p, "%ld %zu\n", (long)result, n);

    cmplog_buf_pos = (size_t)(p - cmplog_buffer);
}

/* ── fprintf-based writer for low-frequency libc interceptors ───────── */
static void log_cmp(const void *a, const void *b, size_t n, int result) {
    if (!cmplog_file || !a || !b || n == 0) return;

    /* Only log comparisons where operands differ (result != 0) */
    if (result == 0) return;

    size_t log_n = n > 64 ? 64 : n;
    fprintf(cmplog_file, "CMP ");
    for (size_t i = 0; i < log_n; i++)
        fprintf(cmplog_file, "%02x", ((const unsigned char *)a)[i]);
    fprintf(cmplog_file, " ");
    for (size_t i = 0; i < log_n; i++)
        fprintf(cmplog_file, "%02x", ((const unsigned char *)b)[i]);
    fprintf(cmplog_file, " %d %zu\n", result, n);
}

/* ── Lifecycle ───────────────────────────────────────────────────────── */

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

/* ── Symbol-based interception: cached libc function pointers ───────── */
typedef int (*cmp_fn)(const void *, const void *, size_t);
typedef int (*str_cmp_fn)(const char *, const char *);
typedef int (*strn_cmp_fn)(const char *, const char *, size_t);
typedef void *(*chr_fn)(const void *, int, size_t);
typedef void *(*memmem_fn)(const void *, size_t, const void *, size_t);
typedef char *(*str_str_fn)(const char *, const char *);

static cmp_fn real_memcmp = NULL;
static str_cmp_fn real_strcmp = NULL;
static strn_cmp_fn real_strncmp = NULL;
static chr_fn real_memchr = NULL;
static str_cmp_fn real_strcasecmp = NULL;
static strn_cmp_fn real_strncasecmp = NULL;
static memmem_fn real_memmem = NULL;
static str_str_fn real_strstr = NULL;
static str_str_fn real_strcasestr = NULL;

static void init_real_funcs(void) {
    if (!real_memcmp) real_memcmp = (cmp_fn)dlsym(RTLD_NEXT, "memcmp");
    if (!real_strcmp) real_strcmp = (str_cmp_fn)dlsym(RTLD_NEXT, "strcmp");
    if (!real_strncmp) real_strncmp = (strn_cmp_fn)dlsym(RTLD_NEXT, "strncmp");
    if (!real_memchr) real_memchr = (chr_fn)dlsym(RTLD_NEXT, "memchr");
    if (!real_strcasecmp) real_strcasecmp = (str_cmp_fn)dlsym(RTLD_NEXT, "strcasecmp");
    if (!real_strncasecmp) real_strncasecmp = (strn_cmp_fn)dlsym(RTLD_NEXT, "strncasecmp");
    if (!real_memmem) real_memmem = (memmem_fn)dlsym(RTLD_NEXT, "memmem");
    if (!real_strstr) real_strstr = (str_str_fn)dlsym(RTLD_NEXT, "strstr");
    if (!real_strcasestr) real_strcasestr = (str_str_fn)dlsym(RTLD_NEXT, "strcasestr");
}

static void __attribute__((constructor)) init_cmplog(void) {
    init_real_funcs();
    const char *path = getenv("_CMPLOG_OUT");
    if (path && path[0]) {
        cmplog_file = fopen(path, "a");
    }
    install_crash_handlers();
}

static void __attribute__((destructor)) fini_cmplog(void) {
    flush_and_close();
}

/* ═══════════════════════════════════════════════════════════════════════
 * Layer 1: libc function interposition (PLT-level)
 * ═══════════════════════════════════════════════════════════════════════ */

int memcmp(const void *a, const void *b, size_t n) {
    int result = real_memcmp(a, b, n);
    log_cmp(a, b, n, result);
    return result;
}

int strcmp(const char *a, const char *b) {
    int result = real_strcmp(a, b);
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    size_t n = len_a < len_b ? len_a : len_b;
    if (n > 0) log_cmp(a, b, n + 1, result);
    return result;
}

int strncmp(const char *a, const char *b, size_t n) {
    int result = real_strncmp(a, b, n);
    if (n > 0) log_cmp(a, b, n, result);
    return result;
}

void *memchr(const void *s, int c, size_t n) {
    void *result = real_memchr(s, c, n);
    unsigned char needle = (unsigned char)c;
    if (cmplog_file && n > 0) {
        size_t log_n = n > 64 ? 64 : n;
        log_cmp(s, &needle, log_n, result ? 0 : -1);
    }
    return result;
}

int strcasecmp(const char *a, const char *b) {
    int result = real_strcasecmp(a, b);
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    size_t n = len_a < len_b ? len_a : len_b;
    if (n > 0) log_cmp(a, b, n + 1, result);
    return result;
}

int strncasecmp(const char *a, const char *b, size_t n) {
    int result = real_strncasecmp(a, b, n);
    if (n > 0) log_cmp(a, b, n, result);
    return result;
}

void *memmem(const void *haystack, size_t haystacklen,
             const void *needle, size_t needlelen) {
    void *result = real_memmem(haystack, haystacklen, needle, needlelen);
    if (cmplog_file && needle && needlelen > 0 && needlelen <= 64) {
        log_cmp(needle, needle, needlelen, -1);
    }
    return result;
}

char *strstr(const char *haystack, const char *needle) {
    char *result = real_strstr(haystack, needle);
    if (cmplog_file && needle) {
        size_t n = strlen(needle);
        if (n > 0 && n <= 64) log_cmp(needle, needle, n, -1);
    }
    return result;
}

char *strcasestr(const char *haystack, const char *needle) {
    char *result = real_strcasestr(haystack, needle);
    if (cmplog_file && needle) {
        size_t n = strlen(needle);
        if (n > 0 && n <= 64) log_cmp(needle, needle, n, -1);
    }
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Layer 2: Compiler-IR callbacks (Clang -fsanitize-coverage=trace-cmp)
 * ═══════════════════════════════════════════════════════════════════════ */

#define MAX_SWITCH_CASES 256

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

void __sanitizer_cov_trace_switch(uint64_t val, uint64_t *ref) {
    if (!ref) return;
    int64_t count = (int64_t)ref[0];
    if (count <= 0 || count > MAX_SWITCH_CASES) return;
    for (int64_t i = 0; i < count; i++) {
        buffer_cmp(val, ref[2 + i], 8);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API for in-process / direct_lite mode
 * ═══════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default")))
void __cmplog_reset(void) {
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
const char *__cmplog_get_path(void) {
    return getenv("_CMPLOG_OUT");
}

/* Alias for backward compatibility with callers that expect __tracecmp_*
 * symbols (e.g. preload_shims, flush_shims in cmplog.py). */
__attribute__((visibility("default")))
void __tracecmp_flush(void) {
    flush_buffer();
    if (cmplog_file) fflush(cmplog_file);
}

__attribute__((visibility("default")))
void __tracecmp_reset(void) {
    __cmplog_reset();
}

__attribute__((visibility("default")))
const char *__tracecmp_get_path(void) {
    return __cmplog_get_path();
}
