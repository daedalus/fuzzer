/* cmplog_shim.c — LD_PRELOAD shim for comparison tracing.

Intercepts memcmp, strcmp, strncmp, memchr and logs operand pairs
to a file for the fuzzer to consume. This enables the fuzzer to
discover magic bytes and protocol constants that blind mutation
cannot find.

Protocol: each comparison writes a line to _CMPLOG_OUT:
  CMP <hex_operand1> <hex_operand2> <cmp_result>\n

Usage:
  LD_PRELOAD=./cmplog_shim.so _CMPLOG_OUT=/tmp/cmp.log ./target
*/

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *cmplog_file = NULL;

static void __attribute__((constructor)) init_cmplog(void) {
    const char *path = getenv("_CMPLOG_OUT");
    if (path && path[0]) {
        cmplog_file = fopen(path, "a");
    }
}

static void __attribute__((destructor)) fini_cmplog(void) {
    if (cmplog_file) {
        fclose(cmplog_file);
        cmplog_file = NULL;
    }
}

static void log_cmp(const void *a, const void *b, size_t n, int result) {
    if (!cmplog_file || !a || !b || n == 0 || n > 4096) return;

    /* Only log comparisons where operands differ (result != 0) */
    if (result == 0) return;

    /* Write: CMP <hex_a> <hex_b> <result> */
    fprintf(cmplog_file, "CMP ");
    for (size_t i = 0; i < n && i < 64; i++)
        fprintf(cmplog_file, "%02x", ((const unsigned char *)a)[i]);
    fprintf(cmplog_file, " ");
    for (size_t i = 0; i < n && i < 64; i++)
        fprintf(cmplog_file, "%02x", ((const unsigned char *)b)[i]);
    fprintf(cmplog_file, " %d %zu\n", result, n);
    fflush(cmplog_file);
}

/* Intercepted functions */

typedef int (*cmp_fn)(const void *, const void *, size_t);
typedef int (*str_cmp_fn)(const char *, const char *);
typedef int (*strn_cmp_fn)(const char *, const char *, size_t);
typedef void *(*chr_fn)(const void *, int, size_t);

int memcmp(const void *a, const void *b, size_t n) {
    cmp_fn real = (cmp_fn)dlsym(RTLD_NEXT, "memcmp");
    int result = real(a, b, n);
    log_cmp(a, b, n, result);
    return result;
}

int strcmp(const char *a, const char *b) {
    str_cmp_fn real = (str_cmp_fn)dlsym(RTLD_NEXT, "strcmp");
    int result = real(a, b);
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    size_t n = len_a < len_b ? len_a : len_b;
    if (n > 0) log_cmp(a, b, n + 1, result);
    return result;
}

int strncmp(const char *a, const char *b, size_t n) {
    strn_cmp_fn real = (strn_cmp_fn)dlsym(RTLD_NEXT, "strncmp");
    int result = real(a, b, n);
    if (n > 0) log_cmp(a, b, n, result);
    return result;
}

void *memchr(const void *s, int c, size_t n) {
    chr_fn real = (chr_fn)dlsym(RTLD_NEXT, "memchr");
    void *result = real(s, c, n);
    /* Log the search — operand is the byte being sought */
    unsigned char needle = (unsigned char)c;
    if (cmplog_file && n > 0 && n <= 4096) {
        log_cmp(s, &needle, n < 1 ? 1 : (n > 64 ? 64 : 1), result ? 0 : -1);
    }
    return result;
}
