/*
 * Minimal AFL-style coverage shim for in-process fuzzing.
 *
 * Provides:
 *   - __afl_map_shm()     — attach to AFL SHM bitmap
 *   - __afl_map_edge()    — record an edge: hash(prev, cur) -> bitmap[idx]
 *                           using Morris probabilistic counting (a=30)
 *   - __afl_map_reset()   — zero the bitmap between iterations
 *   - __sanitizer_cov_trace_pc_guard()      — compiler-inserted edge coverage
 *   - __sanitizer_cov_trace_pc_guard_init() — compiler-inserted edge coverage
 *
 * Morris counting: instead of incrementing the counter by 1 each time,
 * increment with probability (a/(a+1))^c where c is the current value.
 * This gives logarithmic growth — counter v represents approximately
 * a * ((1+1/a)^v - 1) hits. Values stay in [0, 255] but provide
 * frequency information across a much wider range than simple counters.
 *
 * Compile target with:
 *   gcc -O2 -g -shared -fPIC -include afl_shim.c -o target.so target.c -lpng -lz
 *
 * For compiler-inserted edge coverage (clang):
 *   clang -O2 -g -fsanitize-coverage=trace-pc-guard -include afl_shim.c \
 *       -shared -fPIC -o target.so target.c -lpng -lz
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>

/* Default size, overridden at runtime from AFL_MAP_SIZE env var */
static uint32_t __afl_map_size = 65536;
static uint32_t __afl_map_mask = 65535;

uint8_t *__afl_area = NULL;
uint32_t __afl_prev_loc = 0;

/* ── Morris probabilistic counting (a=30) ─────────────────────────────── */
#define MORRIS_A 30
#define MORRIS_BITS 8
#define MORRIS_MAX_V ((1 << MORRIS_BITS) - 1)  /* 255 */

/* threshold[v] = UINT32_MAX * (a/(a+1))^v, precomputed once.
 * A random 32-bit value < threshold[v] triggers the increment. */
static uint32_t morris_threshold[MORRIS_MAX_V + 1];
static uint32_t rng_state = 0x2545F491;  /* xorshift32 seed */

static inline uint32_t xorshift32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static void morris_init(void) {
    morris_threshold[0] = UINT32_MAX;
    for (int i = 1; i <= MORRIS_MAX_V; i++)
        morris_threshold[i] = (uint64_t)morris_threshold[i - 1] * MORRIS_A / (MORRIS_A + 1);
}

/* ── SHM attachment ────────────────────────────────────────────────────── */

__attribute__((visibility("default")))
void __afl_map_shm(void) {
    char *id = getenv("__AFL_SHM_ID");
    if (!id) return;
    int shmid = atoi(id);
    if (shmid <= 0) return;

    /* Read actual map size from environment (set by fuzzer) */
    char *size_str = getenv("AFL_MAP_SIZE");
    if (size_str) {
        uint32_t s = atoi(size_str);
        if (s > 0 && (s & (s - 1)) == 0) {  /* must be power of 2 */
            __afl_map_size = s;
            __afl_map_mask = s - 1;
        }
    }

    void *p = shmat(shmid, NULL, 0);
    if (p == (void *)-1) return;
    __afl_area = (uint8_t *)p;
}

/* ── Edge recording ────────────────────────────────────────────────────── */

__attribute__((visibility("default"), always_inline))
static inline void __afl_map_edge(uint32_t cur_loc) {
    if (__afl_area) {
        uint32_t idx = (__afl_prev_loc ^ cur_loc) & __afl_map_mask;
        uint8_t c = __afl_area[idx];
        if (c < MORRIS_MAX_V && xorshift32() < morris_threshold[c])
            __afl_area[idx] = c + 1;
    }
    __afl_prev_loc = cur_loc >> 1;
}

/* ── Compiler-inserted edge coverage callbacks ────────────────────────
 * When the target is compiled with -fsanitize-coverage=trace-pc-guard,
 * Clang inserts calls to __sanitizer_cov_trace_pc_guard at every edge.
 * We delegate to __afl_map_edge() which handles SHM attachment, edge
 * hashing, and Morris counting. No new bitmap logic needed.
 *
 * Guard values: the compiler assigns each guard a unique nonzero uint32_t
 * at init time. We use *guard directly as cur_loc in the existing hash
 * scheme: hash(prev_loc, *guard) & map_mask → bitmap[idx].
 */

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!guard || *guard == 0) return;
    __afl_map_edge(*guard);
}

/* Called once per module with the range of guard variables.
 * The compiler sets each guard to a unique nonzero value at init time.
 * We don't need to do anything — the guard values are already assigned. */
__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    (void)start;
    (void)stop;
}

__attribute__((visibility("default")))
void __afl_map_reset(void) {
    if (__afl_area)
        memset(__afl_area, 0, __afl_map_size);
    __afl_prev_loc = 0;
}

/* Auto-attach when loaded */
__attribute__((constructor))
static void __afl_auto_init(void) {
    morris_init();
    __afl_map_shm();
}
