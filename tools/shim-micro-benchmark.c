/*
 * Per-edge microbenchmark for afl_shim.c hot path.
 *
 * Compiled with -include src/fuzzer_tool/adapters/afl_shim.c so every shim
 * static/global is visible in this TU.  Hand-attaches a fake coverage
 * segment (no SysV shm, no env) and drives the shim in a loop over a
 * synthetic edge stream, reporting ns/call (min of N reps).
 *
 * Usage: bench <mode> <map_entries> <distinct_locs> <iters> <reps> [guard_callbacks_per_exec] [proc_overhead_us [target_work_us]]
 *   guard_callbacks_per_exec = guard callbacks per execution (e.g. ffmpeg_read ≈ 41,000).
 *   Default = distinct (each synthetic edge = one guard). Set to real guard count for realistic eps.
 *   proc_overhead_us: default=50. Process-spawn overhead (fork/exec/ASAN, µs).
 *   target_work_us:   default=0. Additional target work per edge (µs). ~9.8 for 453 µs/exec @41k guards.
 *   mode: full  __afl_map_loc
 *         probe __afl_probe_insert (home_slot preview)
 *         ctx   __afl_get_caller_ctx
 *         mix   __afl_guard_mix
 *         dist  __afl_probe_distance
 *         empty loop overhead
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint8_t *g_base;
static uint32_t *g_locs;
static uint32_t  g_distinct;

/* memory barrier: stops the compiler hoisting an invariant call */
#define BARRIER() __asm__ __volatile__("" ::: "memory")

static void attach(uint32_t map, int touched) {
    size_t bytes = 32 + (size_t)map * 8 + 16;
    if (touched) bytes += 8 + (((size_t)map + 63) / 64) * 8;
    g_base = (uint8_t *)calloc(1, bytes);
    __afl_area        = (struct __afl_entry *)(g_base + 32);
    __afl_map_size    = map;
    __afl_stack_depth = (uint32_t *)(g_base + 0);
    __afl_gen_word    = (uint32_t *)(g_base + 4);
    __afl_path_hash   = (uint64_t *)(g_base + 8);
    __afl_edge_count  = (uint64_t *)(g_base + 16);
    __afl_dropped     = (uint64_t *)(g_base + 24);
#if __AFL_EDGE
    if (touched) __afl_touched = (uint64_t *)(g_base + 32 + (size_t)map * 8 + 16 + 8);
#endif
#if __AFL_DISTANCE_MODE
    if (getenv("BENCH_DIST")) {
        uint32_t cap = 1u << 20;              /* slot capacity, power of two */
        uint8_t *d = (uint8_t *)calloc(1, 4 + (size_t)cap * 16);
        __afl_dist_count = (uint32_t *)d;
        *__afl_dist_count = cap;
        __afl_dist_table = (struct __afl_dist_entry *)(d + 4);
        for (uint32_t i = 0; i < cap; i++) {
            /* Leave every other slot empty so __afl_probe_distance terminates
             * (it breaks on key==0). */
            if (i & 1u) {
                __afl_dist_table[i].key = (uint64_t)(i * 2654435761u + 1);
                __afl_dist_table[i].dist = i & 0xff;
                __afl_dist_table[i].node_idx = i & 0xffff;
            }
        }
    }
#endif
}

/* One pass over the whole stream; returns ns/call. */
static double run(const char *mode, uint64_t iters) {
    volatile uint64_t sink = 0;
    uint32_t distinct = g_distinct;
    uint32_t *locs = g_locs;

    uint64_t t0 = now_ns();
    if (strcmp(mode, "empty") == 0) {
        for (uint64_t i = 0; i < iters; i++) { sink += locs[i % distinct]; BARRIER(); }
    } else if (strcmp(mode, "ctx") == 0) {
#if __AFL_CTX_SENSITIVE
        for (uint64_t i = 0; i < iters; i++) { sink += __afl_get_caller_ctx(); BARRIER(); }
#else
        for (uint64_t i = 0; i < iters; i++) { sink += locs[i % distinct]; }
#endif
    } else if (strcmp(mode, "mix") == 0) {
        for (uint64_t i = 0; i < iters; i++) { sink += __afl_guard_mix(locs[i % distinct]); }
    } else if (strcmp(mode, "probe") == 0) {
        for (uint64_t i = 0; i < iters; i++) {
            uint32_t id = locs[i % distinct];
            __afl_probe_insert(id, __afl_home_slot(id), __AFL_PROBE_MAX, 1);
        }
    } else if (strcmp(mode, "dist") == 0) {
#if __AFL_DISTANCE_MODE
        for (uint64_t i = 0; i < iters; i++) __afl_probe_distance((uint64_t)locs[i % distinct]);
#endif
    } else {
        for (uint64_t i = 0; i < iters; i++) { __afl_map_loc(locs[i % distinct]); BARRIER(); }
    }
    uint64_t t1 = now_ns();
    (void)sink;
    return (double)(t1 - t0) / (double)iters;
}

int main(int argc, char **argv) {
    const char *mode  = argc > 1 ? argv[1] : "full";
    uint32_t map      = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 0) : 8192;
    uint32_t distinct = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 0) : 2000;
    uint64_t iters    = argc > 4 ? strtoull(argv[4], NULL, 0) : 50000000ull;
    int reps                    = argc > 5 ? atoi(argv[5]) : 5;
    uint32_t guard_callbacks_per_exec = argc > 6 ? (uint32_t)strtoul(argv[6], NULL, 0) : distinct;
    uint32_t proc_overhead_us         = argc > 7 ? (uint32_t)strtoul(argv[7], NULL, 0) : 50;
    double target_work_us             = argc > 8 ? atof(argv[8]) : 0.0;

    attach(map, 0);
    g_distinct = distinct;

    uint32_t *locs = (uint32_t *)malloc((size_t)distinct * 4);
    uint64_t s = 0x123456789abcdefULL;
    for (uint32_t i = 0; i < distinct; i++) {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        locs[i] = (uint32_t)(z ^ (z >> 31)) | 1u;
    }
    g_locs = locs;

    /* Warm + steady state using the SAME operation being timed. */
    for (int r = 0; r < reps; r++) {
        for (uint32_t i = 0; i < distinct; i++) {
            uint32_t id = locs[i];
            if (strcmp(mode, "probe") == 0)
                __afl_probe_insert(id, __afl_home_slot(id), __AFL_PROBE_MAX, 1);
            else if (strcmp(mode, "dist") == 0) {
#if __AFL_DISTANCE_MODE
                __afl_probe_distance((uint64_t)id);
#endif
            } else
                __afl_map_loc(id);
        }
    }

    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        double v = run(mode, iters);
        if (v < best) best = v;
    }

    uint32_t occ = 0;
    for (uint32_t i = 0; i < map; i++) if (__afl_area[i].edge_id) occ++;

    double edges_per_sec = 1e9 / best;
    double shim_us = (best * guard_callbacks_per_exec) / 1000.0;  /* µs shim cost per exec */
    double total_us = shim_us + proc_overhead_us + (target_work_us * guard_callbacks_per_exec);
    double real_eps = total_us > 0 ? 1e6 / total_us : 0.0;
    printf("%-7s map=%-7u distinct=%-7u ctx=%-2d occ=%-7u  %8.3f ns/call  %10.0f edges/s  %12.0f eps(ceil)  %8.1f eps(real)\n",
           mode, map, distinct, (int)(__AFL_CTX_BITS + 0), occ, best,
           edges_per_sec,
           edges_per_sec / guard_callbacks_per_exec,
           real_eps);
    free(locs);
    free(g_base);
    return 0;
}
