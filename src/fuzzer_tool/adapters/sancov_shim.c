/* sancov_shim.c — LD_PRELOAD shim for sanitizer coverage.

Intercepts __sanitizer_cov_trace_pc_guard calls from Clang-instrumented
binaries and writes edge indices to a shared bitmap file. Provides
coverage feedback without AFL instrumentation or ptrace.

Uses *guard (the guard variable's value, set by the compiler at init)
as the edge index — this is stable across ASLR runs, unlike the guard's
address. Writes bitmap on exit and on crash via signal handler.

Usage:
  LD_PRELOAD=./sancov_shim.so _COV_BITMAP_OUT=/tmp/cov.bin ./target
*/

#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAP_SIZE 65536

static uint8_t bitmap[MAP_SIZE] = {0};
static char bitmap_path[256] = {0};

static void write_bitmap(void) {
    if (!bitmap_path[0]) return;
    FILE *f = fopen(bitmap_path, "wb");
    if (f) {
        fwrite(bitmap, 1, MAP_SIZE, f);
        fclose(f);
    }
}

static void __attribute__((constructor)) init_sancov(void) {
    const char *path = getenv("_COV_BITMAP_OUT");
    if (path && path[0]) {
        strncpy(bitmap_path, path, sizeof(bitmap_path) - 1);
    }
}

static void __attribute__((destructor)) fini_sancov(void) {
    write_bitmap();
}

/* Signal handler: flush bitmap before crashing */
static void crash_handler(int sig) {
    write_bitmap();
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

/* The guard function: use *guard (the compiler-set value) as edge index.
   Unlike (uintptr_t)guard which varies with ASLR, *guard is stable
   across runs for the same edge. */
void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!guard) return;

    /* Static init: install crash handlers on first call */
    static int handlers_installed = 0;
    if (!handlers_installed) {
        install_crash_handlers();
        handlers_installed = 1;
    }

    /* Use *guard value as edge index — stable across ASLR */
    uint32_t idx = (*guard) % MAP_SIZE;

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
