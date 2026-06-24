/* sancov_shim.c — LD_PRELOAD shim for sanitizer coverage.

Intercepts __sanitizer_cov_trace_pc_guard calls from Clang-instrumented
binaries and writes edge indices to a shared bitmap file. Provides
coverage feedback without AFL instrumentation or ptrace.

This is for Clang-compiled targets with -fsanitize-coverage=trace-pc-guard
that lack their own coverage collection (no AFL, no libFuzzer harness).

Usage:
  LD_PRELOAD=./sancov_shim.so _COV_BITMAP_OUT=/tmp/cov.bin ./target
*/

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAP_SIZE 65536

static uint8_t bitmap[MAP_SIZE] = {0};
static char bitmap_path[256] = {0};
static int bitmap_fd = -1;

static void __attribute__((constructor)) init_sancov(void) {
    const char *path = getenv("_COV_BITMAP_OUT");
    if (path && path[0]) {
        strncpy(bitmap_path, path, sizeof(bitmap_path) - 1);
    }
}

static void __attribute__((destructor)) fini_sancov(void) {
    if (bitmap_path[0]) {
        FILE *f = fopen(bitmap_path, "wb");
        if (f) {
            fwrite(bitmap, 1, MAP_SIZE, f);
            fclose(f);
        }
    }
}

/* The guard function receives a pointer to a guard variable.
   We use the guard's address to derive an edge index. */
void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!guard) return;

    /* Use the guard address to derive an edge index.
       The guard is at a unique location per edge, so
       its address is a good hash source. */
    uintptr_t addr = (uintptr_t)guard;
    uint32_t idx = (uint32_t)(addr ^ (addr >> 12)) % MAP_SIZE;

    /* Increment counter (saturating at 255) */
    if (bitmap[idx] < 255)
        bitmap[idx]++;
}

/* Called once per module with the range of guard variables.
   We don't need to do anything special here. */
void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    (void)start;
    (void)stop;
}
