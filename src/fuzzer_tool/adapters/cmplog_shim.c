/* cmplog_shim.c — Unified LD_PRELOAD shim for all fuzzer instrumentation.
 *
 * Three layers, one shared .so:
 *
 * 1. Symbol-based: intercepts libc comparison functions via dlsym(RTLD_NEXT)
 *    (memcmp/strcmp/strncmp/memchr/strcasecmp/strncasecmp/memmem/strstr/
 *    strcasestr) — catches explicit library calls at the PLT level.
 *
 * 2. Compiler-IR-based: implements Clang's -fsanitize-coverage=trace-cmp
 *    callbacks (__sanitizer_cov_trace_cmp{1,2,4,8}, trace_const_cmp*,
 *    trace_switch) — catches inlined/folded comparisons.
 *
 * 3. AFL edge coverage: provides __afl_map_shm/__afl_map_edge/__afl_map_reset
 *    for AFL-style SHM bitmap coverage with Morris probabilistic counting.
 *    __sanitizer_cov_trace_pc_guard delegates to __afl_map_edge when SHM
 *    is attached, providing the same coverage for Clang trace-pc-guard targets.
 *
 * Layers 1+2 write to _CMPLOG_OUT (CMP line format).
 * Layer 3 writes to __AFL_SHM_ID (SHM segment, via __afl_area).
 *
 * Usage:
 *   LD_PRELOAD=./cmplog_shim.so _CMPLOG_OUT=/tmp/cmp.log ./target
 *   LD_PRELOAD=./cmplog_shim.so __AFL_SHM_ID=<n> ./target
 *
 * When loaded via LD_PRELOAD, runtime symbols shadow the target's compiled-in
 * copies (from -include afl_shim.c).  When the shim is not loaded, the target's
 * own compiled-in fallback works independently.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Shared state
 * ═══════════════════════════════════════════════════════════════════════ */
static FILE *cmplog_file = NULL;

/* ═══════════════════════════════════════════════════════════════════════
 * Layer 3: AFL edge coverage (SHM bitmap + Morris counting)
 * ═══════════════════════════════════════════════════════════════════════ */
static uint32_t __afl_map_size = 65536;
static uint32_t __afl_map_mask = 65535;
static uint8_t *__afl_area = NULL;
static uint32_t __afl_prev_loc = 0;

#define MORRIS_A 30
#define MORRIS_BITS 8
#define MORRIS_MAX_V ((1 << MORRIS_BITS) - 1)
static uint32_t morris_threshold[MORRIS_MAX_V + 1];
static uint32_t morris_rng = 0x2545F491;

static inline uint32_t xorshift32(void) {
    morris_rng ^= morris_rng << 13;
    morris_rng ^= morris_rng >> 17;
    morris_rng ^= morris_rng << 5;
    return morris_rng;
}

static void morris_init(void) {
    morris_threshold[0] = UINT32_MAX;
    for (int i = 1; i <= MORRIS_MAX_V; i++)
        morris_threshold[i] = (uint64_t)morris_threshold[i - 1] * MORRIS_A / (MORRIS_A + 1);
}

__attribute__((weak, visibility("default")))
void __afl_map_shm(void) {
    char *id = getenv("__AFL_SHM_ID");
    if (!id) return;
    int shmid = atoi(id);
    if (shmid <= 0) return;
    char *size_str = getenv("AFL_MAP_SIZE");
    if (size_str) {
        uint32_t s = atoi(size_str);
        if (s > 0 && (s & (s - 1)) == 0) {
            __afl_map_size = s;
            __afl_map_mask = s - 1;
        }
    }
    void *p = shmat(shmid, NULL, 0);
    if (p == (void *)-1) return;
    __afl_area = (uint8_t *)p;
}

static inline void __afl_map_edge(uint32_t cur_loc) {
    if (__afl_area) {
        uint32_t idx = (__afl_prev_loc ^ cur_loc) & __afl_map_mask;
        uint8_t c = __afl_area[idx];
        if (c < MORRIS_MAX_V && xorshift32() < morris_threshold[c])
            __afl_area[idx] = c + 1;
    }
    __afl_prev_loc = cur_loc >> 1;
}

__attribute__((weak, visibility("default")))
void __afl_map_reset(void) {
    if (__afl_area)
        memset(__afl_area, 0, __afl_map_size);
    __afl_prev_loc = 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Layer 2: Buffered writer for compiler-IR callbacks (trace-cmp)
 * ═══════════════════════════════════════════════════════════════════════ */
#define BUFFER_SIZE (256 * 1024)
static char cmplog_buffer[BUFFER_SIZE];
static size_t cmplog_buf_pos = 0;

static void flush_buffer(void) {
    if (cmplog_buf_pos == 0) return;
    if (cmplog_file) fwrite(cmplog_buffer, 1, cmplog_buf_pos, cmplog_file);
    cmplog_buf_pos = 0;
}

static inline void buffer_cmp(uint64_t a, uint64_t b, size_t n) {
    if (!cmplog_file) return;
    if (cmplog_buf_pos + 96 > BUFFER_SIZE) flush_buffer();
    static const char hex[] = "0123456789abcdef";
    char *p = cmplog_buffer + cmplog_buf_pos;
    *p++ = 'C'; *p++ = 'M'; *p++ = 'P'; *p++ = ' ';
    for (size_t i = 0; i < n; i++) {
        uint8_t byte = (uint8_t)(a >> (i * 8));
        *p++ = hex[byte >> 4]; *p++ = hex[byte & 0xf];
    }
    *p++ = ' ';
    for (size_t i = 0; i < n; i++) {
        uint8_t byte = (uint8_t)(b >> (i * 8));
        *p++ = hex[byte >> 4]; *p++ = hex[byte & 0xf];
    }
    *p++ = ' ';
    int64_t result = (a < b) ? -1 : (a > b) ? 1 : 0;
    p += sprintf(p, "%ld %zu", (long)result, n);
    // Optional PC field: __builtin_return_address(0) gives the instruction
    // address after the call to this callback.  The first level of inlining
    // gives the trace_cmp caller; with LTO this is the comparison site.
    *p++ = ' ';
    p += sprintf(p, "%p", __builtin_return_address(0));
    *p++ = '\n';
    cmplog_buf_pos = (size_t)(p - cmplog_buffer);
}

/* ── fprintf writer for low-frequency libc interceptors (Layer 1) ─────── */
static void log_cmp(const void *a, const void *b, size_t n, int result) {
    if (!cmplog_file || !a || !b || n == 0 || result == 0) return;
    size_t log_n = n > 64 ? 64 : n;
    fprintf(cmplog_file, "CMP ");
    for (size_t i = 0; i < log_n; i++)
        fprintf(cmplog_file, "%02x", ((const unsigned char *)a)[i]);
    fprintf(cmplog_file, " ");
    for (size_t i = 0; i < log_n; i++)
        fprintf(cmplog_file, "%02x", ((const unsigned char *)b)[i]);
    fprintf(cmplog_file, " %d %zu\n", result, n);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════════ */
static void flush_and_close(void) {
    flush_buffer();
    if (cmplog_file) { fclose(cmplog_file); cmplog_file = NULL; }
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

/* ── libc function pointers (Layer 1) ─────────────────────────────────── */
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
    morris_init();
    __afl_map_shm();
    const char *cmplog_path = getenv("_CMPLOG_OUT");
    if (cmplog_path && cmplog_path[0])
        cmplog_file = fopen(cmplog_path, "a");
    install_crash_handlers();
}

static void __attribute__((destructor)) fini_cmplog(void) {
    flush_and_close();
}

/* ═══════════════════════════════════════════════════════════════════════
 * Layer 1: libc function interposition (PLT-level)
 * ═══════════════════════════════════════════════════════════════════════ */
int memcmp(const void *a, const void *b, size_t n) {
    int result = real_memcmp(a, b, n); log_cmp(a, b, n, result); return result;
}
int strcmp(const char *a, const char *b) {
    int result = real_strcmp(a, b);
    size_t na = strlen(a), nb = strlen(b), n = na < nb ? na : nb;
    if (n > 0) log_cmp(a, b, n + 1, result); return result;
}
int strncmp(const char *a, const char *b, size_t n) {
    int result = real_strncmp(a, b, n); if (n > 0) log_cmp(a, b, n, result); return result;
}
void *memchr(const void *s, int c, size_t n) {
    void *result = real_memchr(s, c, n);
    unsigned char needle = (unsigned char)c;
    if (cmplog_file && n > 0) log_cmp(s, &needle, n > 64 ? 64 : n, result ? 0 : -1);
    return result;
}
int strcasecmp(const char *a, const char *b) {
    int result = real_strcasecmp(a, b);
    size_t na = strlen(a), nb = strlen(b), n = na < nb ? na : nb;
    if (n > 0) log_cmp(a, b, n + 1, result); return result;
}
int strncasecmp(const char *a, const char *b, size_t n) {
    int result = real_strncasecmp(a, b, n); if (n > 0) log_cmp(a, b, n, result); return result;
}
void *memmem(const void *h, size_t hl, const void *n, size_t nl) {
    void *result = real_memmem(h, hl, n, nl);
    if (cmplog_file && n && nl > 0 && nl <= 64) log_cmp(n, n, nl, -1); return result;
}
char *strstr(const char *h, const char *n) {
    char *result = real_strstr(h, n);
    if (cmplog_file && n) { size_t nl = strlen(n); if (nl > 0 && nl <= 64) log_cmp(n, n, nl, -1); }
    return result;
}
char *strcasestr(const char *h, const char *n) {
    char *result = real_strcasestr(h, n);
    if (cmplog_file && n) { size_t nl = strlen(n); if (nl > 0 && nl <= 64) log_cmp(n, n, nl, -1); }
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Layer 2: Compiler-IR callbacks (Clang -fsanitize-coverage=trace-cmp)
 * ═══════════════════════════════════════════════════════════════════════ */
#define MAX_SWITCH_CASES 256
void __sanitizer_cov_trace_cmp1(uint8_t a, uint8_t b) { buffer_cmp(a, b, 1); }
void __sanitizer_cov_trace_cmp2(uint16_t a, uint16_t b) { buffer_cmp(a, b, 2); }
void __sanitizer_cov_trace_cmp4(uint32_t a, uint32_t b) { buffer_cmp(a, b, 4); }
void __sanitizer_cov_trace_cmp8(uint64_t a, uint64_t b) { buffer_cmp(a, b, 8); }
void __sanitizer_cov_trace_const_cmp1(uint8_t a, uint8_t b) { buffer_cmp(a, b, 1); }
void __sanitizer_cov_trace_const_cmp2(uint16_t a, uint16_t b) { buffer_cmp(a, b, 2); }
void __sanitizer_cov_trace_const_cmp4(uint32_t a, uint32_t b) { buffer_cmp(a, b, 4); }
void __sanitizer_cov_trace_const_cmp8(uint64_t a, uint64_t b) { buffer_cmp(a, b, 8); }
void __sanitizer_cov_trace_switch(uint64_t val, uint64_t *ref) {
    if (!ref) return;
    int64_t count = (int64_t)ref[0];
    if (count <= 0 || count > MAX_SWITCH_CASES) return;
    for (int64_t i = 0; i < count; i++) buffer_cmp(val, ref[2 + i], 8);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Layer 3: trace-pc-guard callback (delegates to AFL edge coverage)
 *
 * When the shim is LD_PRELOAD'd, __sanitizer_cov_trace_pc_guard shadows
 * the target's compiled-in copy (from -include afl_shim.c).  We delegate
 * to __afl_map_edge for edge-pair-hashed coverage in the AFL SHM bitmap.
 *
 * When the shim is NOT loaded, the target's compiled-in fallback handles
 * SHM coverage independently — both paths work.
 * ═══════════════════════════════════════════════════════════════════════ */
__attribute__((weak, visibility("default")))
void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!guard || *guard == 0) return;
    __afl_map_edge(*guard);
}
__attribute__((weak, visibility("default")))
void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    (void)start; (void)stop;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API for in-process / direct_lite mode
 * ═══════════════════════════════════════════════════════════════════════ */
__attribute__((visibility("default")))
void __cmplog_reset(void) {
    if (cmplog_file) {
        flush_buffer();
        const char *path = getenv("_CMPLOG_OUT");
        if (path && path[0]) { fclose(cmplog_file); cmplog_file = fopen(path, "w"); }
    }
}
__attribute__((visibility("default")))
const char *__cmplog_get_path(void) { return getenv("_CMPLOG_OUT"); }

__attribute__((visibility("default")))
void __tracecmp_flush(void) { flush_buffer(); if (cmplog_file) fflush(cmplog_file); }
__attribute__((visibility("default")))
void __tracecmp_reset(void) { __cmplog_reset(); }
__attribute__((visibility("default")))
const char *__tracecmp_get_path(void) { return __cmplog_get_path(); }
