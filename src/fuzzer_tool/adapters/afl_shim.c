/*
 * Sparse 8-byte edge entry shim for in-process fuzzing.
 *
 * Replaces the traditional AFL fixed-size byte bitmap with an open-addressing
 * hash table of 8-byte entries {edge_id, count}.  Each stored edge is uniquely
 * identified by its full 32-bit edge_id (prev_loc ^ cur_loc) so there are no
 * silent bucket collisions.  AFL_MAP_SIZE is the number of hash table entries
 * (not bytes).  SHM size = AFL_MAP_SIZE * sizeof(struct __afl_entry) + 24
 * (24 bytes front header: stack_depth + pad + path_hash + edge_count).
 *
 * Provides:
 *   - __afl_map_shm()     — attach to SHM segment
 *   - __afl_map_edge()    — record an edge via open-addressing hash table
 *   - __afl_map_reset()   — zero all entries between iterations
 *   - __sanitizer_cov_trace_pc_guard()      — compiler-inserted edge coverage
 *   - __sanitizer_cov_trace_pc_guard_init() — compiler-inserted edge coverage
 *   - __sancov_lowest_stack()               — LLVM stack depth tracking
 *
 * Metadata layout (24 bytes at front of SHM):
 *   offset 0: uint32 stack_depth   (max stack depth in bytes)
 *   offset 4: uint32 _pad
 *   offset 8: uint64 path_hash     (rolling: hash = hash * 31 ^ edge_id)
 *   offset 16: uint64 edge_count   (monotonic new-slot insertion count)
 *   offset 24+:  edge table ({edge_id, count} × map_size entries)
 *
 * Compile target with:
 *   gcc -O2 -g -shared -fPIC -include afl_shim.c -o target.so target.c -lpng -lz
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>

/* ── 8-byte hash table entry ──────────────────────────────────────────
 * edge_id == 0 means empty slot.  count is a simple saturating counter
 * (no Morris probabilistic counting needed with 32-bit range).          */
struct __afl_entry {
    uint32_t edge_id;
    uint32_t count;
};

/* Front header size (stack_depth + pad + path_hash + edge_count) */
#define SHM_HEADER_SIZE 24

/* Default number of hash table entries.  AFL_MAP_SIZE directly sets
 * __afl_map_size (number of entries, not bytes).  Default 8192 entries:
 * edge table = 8192 × 8 = 65536 bytes, header = 24 bytes, total = 65560. */
static uint32_t __afl_map_size  = 8192;

struct __afl_entry *__afl_area   = NULL;
uint32_t           __afl_prev_loc = 0;

/* Metadata pointers (front header, before the edge table) */
static uint32_t *__afl_stack_depth = NULL;   /* offset 0: uint32 */
static uint64_t *__afl_path_hash   = NULL;   /* offset 8: uint64 */
static uint64_t *__afl_edge_count  = NULL;   /* offset 16: uint64 */

/* Per-iteration state */
static uint64_t  __afl_path_hash_acc = 0;       /* rolling path hash accumulator */
static uint32_t  __afl_max_stack_depth = 0;     /* max stack depth this iteration */
static uint64_t  __afl_iter_edge_count = 0;     /* new-slot insertions this iteration */
static uint64_t  __afl_total_edge_count = 0;    /* cumulative, never reset across iterations */

/* ── SHM attachment ──────────────────────────────────────────────────── */

__attribute__((visibility("default")))
void __afl_map_shm(void) {
    char *id = getenv("__AFL_SHM_ID");
    if (!id) return;
    int shmid = atoi(id);
    if (shmid <= 0) return;

    /* Read map size from environment.  AFL_MAP_SIZE is the number of
     * hash table entries (not bytes).  The Python side allocates SHM as
     * AFL_MAP_SIZE * sizeof(struct __afl_entry) + SHM_HEADER_SIZE bytes.   */
    char *size_str = getenv("AFL_MAP_SIZE");
    if (size_str) {
        uint32_t s = (uint32_t)atoi(size_str);
        if (s > 0)
            __afl_map_size = s;
    }

    /* SHM was allocated as header bytes + table bytes */
    void *p = shmat(shmid, NULL, 0);
    if (p == (void *)-1) return;

    /* Edge table starts after the front header */
    uint8_t *base = (uint8_t *)p;
    __afl_area = (struct __afl_entry *)(base + SHM_HEADER_SIZE);

    /* Set up metadata pointers in front header (offsets 0/8/16) */
    __afl_stack_depth = (uint32_t *)(base + 0);
    __afl_path_hash   = (uint64_t *)(base + 8);
    __afl_edge_count  = (uint64_t *)(base + 16);
}

/* ── Edge recording (open-addressing hash table) ───────────────────────
 *
 * Hash: edge_id = prev_loc ^ cur_loc
 * Probe: linear probing from edge_id % map_size until we find a matching
 *        edge_id or an empty slot (edge_id == 0).                       */

__attribute__((visibility("default"), always_inline))
static inline void __afl_map_edge(uint32_t cur_loc) {
    if (!__afl_area) return;

    uint32_t edge_id = __afl_prev_loc ^ cur_loc;
    uint32_t pos     = edge_id % __afl_map_size;

    /* Linear probe: at most map_size iterations guarantees we either
     * find the edge or hit an empty slot. */
    for (uint32_t i = 0; i < __afl_map_size; i++) {
        uint32_t idx = (pos + i) % __afl_map_size;
        uint32_t eid = __afl_area[idx].edge_id;

        if (eid == 0) {                              /* empty slot — claim */
            __afl_area[idx].edge_id = edge_id;
            __afl_area[idx].count   = 1;
            __afl_iter_edge_count++;                 /* track per-iteration new-slot insertion */
            __afl_total_edge_count++;                /* track cumulative across-reset count */
            if (__afl_edge_count)                    /* write CUMULATIVE count live to SHM header */
                *__afl_edge_count = __afl_total_edge_count;
            break;
        }
        if (eid == edge_id) {                        /* existing edge — bump */
            if (__afl_area[idx].count < UINT32_MAX)
                __afl_area[idx].count++;
            break;
        }
        /* else: hash collision, keep probing */
    }

    /* Accumulate rolling path hash: hash = hash * 31 ^ edge_id */
    __afl_path_hash_acc = (__afl_path_hash_acc * 31) ^ edge_id;
    if (__afl_path_hash)
        *__afl_path_hash = __afl_path_hash_acc;

    __afl_prev_loc = cur_loc >> 1;
}

/* ── Compiler-inserted edge coverage callbacks ──────────────────────── */

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!guard || *guard == 0) return;
    __afl_map_edge(*guard);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    static uint32_t guard_counter;
    if (start == stop || *start) return;
    for (uint32_t *g = start; g < stop; g++)
        *g = ++guard_counter;
}

/* ── LLVM stack depth tracking ────────────────────────────────────────
 * When the target is compiled with -fsanitize=address, ASAN provides
 * __sancov_lowest_stack as a TLS symbol — we must NOT define it or
 * the linker fails with TLS/non-TLS type mismatch.  Our definition
 * is used only for non-ASAN builds (standalone, ptrace mode, etc.). */

#if !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
__attribute__((visibility("default")))
void __sancov_lowest_stack(uint32_t addr) {
    /* addr is the stack address of the current instrumentation point.
     * Track the minimum (deepest) stack address seen this iteration. */
    if (__afl_max_stack_depth == 0 || addr < __afl_max_stack_depth) {
        __afl_max_stack_depth = addr;
    }
}
#endif

/* ── Reset (zero all entries between iterations) ───────────────────────
 * Also writes accumulated metadata (stack_depth, path_hash) to the
 * metadata region so Python can read them, then resets accumulators. */

__attribute__((visibility("default")))
void __afl_map_reset(void) {
    if (__afl_area) {
        memset(__afl_area, 0, __afl_map_size * sizeof(struct __afl_entry));

        /* Write metadata before resetting accumulators */
        if (__afl_stack_depth) {
            *__afl_stack_depth = __afl_max_stack_depth;
        }
        if (__afl_path_hash) {
            *__afl_path_hash = __afl_path_hash_acc;
        }
        if (__afl_edge_count) {
            *__afl_edge_count = __afl_total_edge_count;
        }
    }
    __afl_prev_loc = 0;
    __afl_path_hash_acc = 0;
    __afl_max_stack_depth = 0;
    __afl_iter_edge_count = 0;
}

/* ── Crash signal handler ─────────────────────────────────────────────
 * Uses sigsetjmp/siglongjmp instead of the traditional restore-and-re-raise
 * approach because glibc's abort() resets the handler to SIG_DFL after the
 * first SIGABRT and re-raises — killing the process before the fuzzer can
 * recover.  siglongjmp escapes the signal handler entirely, jumping back to
 * __afl_guarded_call which can then report the crash via its return value.
 *
 * This is needed for crashes from pre-compiled code (libasan, libc) that
 * bypasses the abort() preprocessor override below.  The override covers
 * the target's own source (FFmpeg av_assert0, etc.).                         */

static sigjmp_buf __afl_jmp_buf;
static struct sigaction __afl_old_segv;
static struct sigaction __afl_old_abrt;

static void __afl_crash_handler(int sig) {
    siglongjmp(__afl_jmp_buf, sig);
}

static void __afl_install_crash_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = __afl_crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, &__afl_old_segv);
    sigaction(SIGABRT, &sa, &__afl_old_abrt);
}

/* ── Guarded call wrapper ─────────────────────────────────────────────
 * InProcessRunner's direct_lite mode calls this instead of calling the
 * fuzz entry function directly.  __afl_guarded_call sets up a sigsetjmp
 * buffer, calls the entry function, and returns normally.  If a signal
 * (SIGSEGV/SIGABRT) fires during the call, __afl_crash_handler does
 * siglongjmp back here, and we return a negative signal-indicating value.
 *
 * Returns the entry function's return value on success, or -sig (negated
 * signal number, e.g. -6 for SIGABRT, -11 for SIGSEGV) on crash.  This
 * matches the signal-indicating exit code convention used throughout the
 * fuzzer (run_target_stdin etc. return -sig on signal).                      */

__attribute__((visibility("default")))
int __afl_guarded_call(int (*entry)(const uint8_t *, size_t),
                       const uint8_t *data, size_t size) {
    int sig;
    if ((sig = sigsetjmp(__afl_jmp_buf, 1)) == 0)
        return entry(data, size);
    /* sig = signal number from __afl_crash_handler's siglongjmp */
    return -(int)sig;
}

/* ── abort() override (all builds) ───────────────────────────────────
 * Intercept libc's abort() for fuzzing: instead of killing the process,
 * write a marker to stderr and return.  Libraries like FFmpeg call abort()
 * on internal assertion failures (~1600 av_assert0 sites), which would
 * flood the fuzzer with false crashes.  This override catches ALL abort()
 * sources — macros, direct calls in library .c files, etc.
 *
 * In ASAN builds, the __asan_default_options shim (loaded before libasan
 * by the fuzzer) sets halt_on_error=0:abort_on_error=0, so ASAN-detected
 * errors write the report to stderr and return normally — abort() is no
 * longer part of ASAN's own error path.  This makes it safe to override
 * abort() unconditionally: library-internal assertions (av_assert0) are
 * caught, while ASAN bugs produce their full report via stderr capture
 * and the SanitizerReport pipeline.
 *
 * The noreturn warning is suppressed because we intentionally override
 * the behavior for the fuzzing use case.                                  */

/* Override libc abort() for all fuzzing builds.
 *
 * Instead of killing the process, write a marker to stderr and return.
 * Libraries like FFmpeg call abort() on internal assertion failures
 * (~1600 av_assert0 sites), which would flood the fuzzer with false
 * crash detections.
 *
 * A macro + static helper avoids the GCC "noreturn function does return"
 * warning that would fire if we defined void abort(void) directly
 * (stdlib.h declares abort() as __noreturn__).  The preprocessor
 * replaces all abort() calls with __afl_shim_abort() before the compiler
 * sees the declaration mismatch.                                           */
static void __afl_shim_abort(void) {
    static const char msg[] = "[shim] abort() intercepted\n";
    write(STDERR_FILENO, msg, sizeof(msg) - 1);
}
#define abort() __afl_shim_abort()

/* Auto-attach when loaded */
__attribute__((constructor))
static void __afl_auto_init(void) {
    __afl_map_shm();
    __afl_install_crash_handlers();
}
