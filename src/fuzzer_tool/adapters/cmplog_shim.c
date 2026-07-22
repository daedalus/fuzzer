/* cmplog_shim.c — LD_PRELOAD shim for comparison tracing.

Intercepts memcmp, strcmp, strncmp, memchr, strcasecmp, strncasecmp,
memmem, strstr, and strcasestr, logging operand pairs to a file for
the fuzzer to consume. This enables the fuzzer to discover magic
bytes, protocol constants, and case-insensitive patterns that blind
mutation cannot find.

Intercepted functions:
  - memcmp / strcmp / strncmp / memchr       — standard comparisons
  - strcasecmp / strncasecmp                — case-insensitive string comparisons
  - memmem / strstr / strcasestr            — substring search functions

Protocol: each comparison writes a line to _CMPLOG_OUT:
  CMP <hex_operand1> <hex_operand2> <cmp_result>\n

Usage:
  LD_PRELOAD=./cmplog_shim.so _CMPLOG_OUT=/tmp/cmp.log ./target
*/

#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *cmplog_file = NULL;

/* Cached real function pointers — resolved once, not per call */
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
}

static void flush_and_close(void) {
    if (cmplog_file) {
        fflush(cmplog_file);
        fclose(cmplog_file);
        cmplog_file = NULL;
    }
}

static void __attribute__((destructor)) fini_cmplog(void) {
    flush_and_close();
}

/* Signal handler: flush before dying so we don't lose cmplog data */
static void crash_handler(int sig) {
    flush_and_close();
    /* Re-raise with default handler */
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

static void log_cmp(const void *a, const void *b, size_t n, int result) {
    if (!cmplog_file || !a || !b || n == 0 || n == 0) return;

    /* Only log comparisons where operands differ (result != 0) */
    if (result == 0) return;

    /* Write: CMP <hex_a> <hex_b> <result> */
    size_t log_n = n > 64 ? 64 : n;
    fprintf(cmplog_file, "CMP ");
    for (size_t i = 0; i < log_n; i++)
        fprintf(cmplog_file, "%02x", ((const unsigned char *)a)[i]);
    fprintf(cmplog_file, " ");
    for (size_t i = 0; i < log_n; i++)
        fprintf(cmplog_file, "%02x", ((const unsigned char *)b)[i]);
    fprintf(cmplog_file, " %d %zu\n", result, n);
    /* No per-call fflush — flush on exit/crash via destructor + signal handler */
}

/* Intercepted functions — use cached real pointers */

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

/* ── Case-insensitive comparisons ────────────────────────────────
 * Closes the ignore_case=true gap: fgrep's case-insensitive mode
 * calls strcasecmp/strncasecmp instead of strcmp/strncmp.
 */

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

/* ── Substring search functions ──────────────────────────────────
 * Closes the fallback-path gap: grep-style tools may call memmem,
 * strstr, or strcasestr instead of their own hand-rolled search.
 * For these we always log the needle — it's the compile-time
 * constant pattern the fuzzer needs to discover.
 */

void *memmem(const void *haystack, size_t haystacklen,
             const void *needle, size_t needlelen) {
    void *result = real_memmem(haystack, haystacklen, needle, needlelen);
    if (cmplog_file && needle && needlelen > 0 && needlelen <= 64) {
        /* Always log the needle — it's the magic pattern the fuzzer needs.
         * Use -1 unconditionally so log_cmp doesn't skip it on match. */
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

/* ── Public API for log file management ────────────────────────
 * Called by the fuzzer via ctypes when cmplog is compiled into
 * the target .so in direct_lite mode.
 */

__attribute__((visibility("default")))
void __cmplog_reset(void) {
    /* Truncate-and-reopen the log file so the fuzzer can read
     * fresh data after each execution without deleting the file
     * the .so still has open. */
    if (cmplog_file) {
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
