/* Ground-truth edge tracer for tools/edge_diagnostic.py matrix --ground-truth.
 *
 * Build a target with this file in place of afl_shim.c:
 *
 *   clang -fsanitize-coverage=trace-pc-guard -O2 -fno-omit-frame-pointer \
 *       -c vendor/fuzzgoat/fuzzgoat.c -o /tmp/fg.o
 *   clang -O2 -fno-omit-frame-pointer -include tools/ground_truth_tracer.c \
 *       -o /tmp/fuzzgoat_gt targets/fuzzgoat_read.c /tmp/fg.o -lm -ldl
 *
 * and every coverage event is appended to $GT_OUT as three little-endian
 * uint64 words: (previous location, current location, call site). A location
 * is the guard's sequential index or the raw id the wrapper passed to
 * __afl_map_edge; the call site is the same frame-pointer hop
 * __afl_get_caller_ctx() takes, made load-base relative so it survives ASLR.
 *
 * It computes no ids at all. That is the point: the shim's id function is the
 * thing under test, so the reference must not share it. Distinct
 * (prev, cur) pairs are the context-free edges the target really took;
 * comparing their count with the distinct ids a shim build reports over the
 * same inputs measures how many the id function merged.
 *
 * Unbuffered on purpose: fuzzgoat aborts on its planted bugs, and a
 * buffered stream would lose the tail of exactly those executions.
 * Not a shim: no SHM, no forkserver, one input per process. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1  /* Dl_info / dladdr */
#endif
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static FILE *__gt_out;
static uint64_t __gt_prev;

__attribute__((constructor)) static void __gt_open(void) {
    const char *p = getenv("GT_OUT");
    __gt_out = fopen(p ? p : "/dev/null", "ab");
    if (__gt_out) setvbuf(__gt_out, NULL, _IONBF, 0);
}

__attribute__((always_inline)) static inline uint64_t __gt_site(void) {
    void **fp = (void **)__builtin_frame_address(0);
    if (!fp) return 0;
    uintptr_t cur = (uintptr_t)fp, c = (uintptr_t)fp[0];
    if (c <= cur || c - cur > (4u << 20)) return 0;
    uintptr_t ra = (uintptr_t)((void **)c)[1];
    Dl_info info;
    if (ra && dladdr((void *)ra, &info) && info.dli_fbase) ra -= (uintptr_t)info.dli_fbase;
    return ra;
}

__attribute__((always_inline)) static inline void __afl_map_edge(uint32_t cur) {
    uint64_t rec[3] = {__gt_prev, cur, __gt_site()};
    if (__gt_out) fwrite(rec, sizeof rec[0], 3, __gt_out);
    __gt_prev = cur;
}

void __sanitizer_cov_trace_pc_guard(uint32_t *g) {
    if (g && *g) __afl_map_edge(*g);
}

void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    static uint32_t n;
    if (start == stop || *start) return;
    for (uint32_t *g = start; g < stop; g++) *g = ++n;
}
