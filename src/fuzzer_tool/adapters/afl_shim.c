/*
 * Sparse 8-byte edge entry shim for in-process fuzzing.
 *
 * Replaces the traditional AFL fixed-size byte bitmap with an open-addressing
 * hash table of 8-byte entries {edge_id, count}.  Each stored edge is uniquely
 * identified by its full 32-bit edge_id (prev_loc ^ cur_loc) so there are no
 * silent bucket collisions.  The map size (AFL_MAP_SIZE, default 8192) is the
 * number of hash table entries, not bytes — SHM allocation is
 *   map_size * sizeof(struct __afl_entry)  bytes.
 *
 * Provides:
 *   - __afl_map_shm()     — attach to SHM segment
 *   - __afl_map_edge()    — record an edge via open-addressing hash table
 *   - __afl_map_reset()   — zero all entries between iterations
 *   - __sanitizer_cov_trace_pc_guard()      — compiler-inserted edge coverage
 *   - __sanitizer_cov_trace_pc_guard_init() — compiler-inserted edge coverage
 *
 * Compile target with:
 *   gcc -O2 -g -shared -fPIC -include afl_shim.c -o target.so target.c -lpng -lz
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>

/* ── 8-byte hash table entry ──────────────────────────────────────────
 * edge_id == 0 means empty slot.  count is a simple saturating counter
 * (no Morris probabilistic counting needed with 32-bit range).          */
struct __afl_entry {
    uint32_t edge_id;
    uint32_t count;
};

/* Default number of hash table entries.  AFL_MAP_SIZE is in bytes
 * (traditional AFL semantics).  The shim divides by sizeof(entry)
 * to get the number of hash table slots.  64KB / 8 = 8192 entries.    */
static uint32_t __afl_map_size  = 8192;

struct __afl_entry *__afl_area   = NULL;
uint32_t           __afl_prev_loc = 0;

/* ── SHM attachment ──────────────────────────────────────────────────── */

__attribute__((visibility("default")))
void __afl_map_shm(void) {
    char *id = getenv("__AFL_SHM_ID");
    if (!id) return;
    int shmid = atoi(id);
    if (shmid <= 0) return;

    /* Read map size from environment.
     * AFL_MAP_SIZE is in bytes (traditional AFL semantics).
     * The shim divides by sizeof(__afl_entry) to get the number of
     * hash table slots.  Default 64KB / 8 = 8192 entries.               */
    char *size_str = getenv("AFL_MAP_SIZE");
    if (size_str) {
        uint32_t s_bytes = (uint32_t)atoi(size_str);
        if (s_bytes >= sizeof(struct __afl_entry)) {
            uint32_t s = s_bytes / sizeof(struct __afl_entry);
            if (s > 0)
                __afl_map_size = s;
        }
    }

    /* SHM was allocated as map_size * sizeof(struct __afl_entry) bytes */
    void *p = shmat(shmid, NULL, 0);
    if (p == (void *)-1) return;
    __afl_area = (struct __afl_entry *)p;
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
            break;
        }
        if (eid == edge_id) {                        /* existing edge — bump */
            if (__afl_area[idx].count < UINT32_MAX)
                __afl_area[idx].count++;
            break;
        }
        /* else: hash collision, keep probing */
    }

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

/* ── Reset (zero all entries between iterations) ─────────────────────── */

__attribute__((visibility("default")))
void __afl_map_reset(void) {
    if (__afl_area)
        memset(__afl_area, 0, __afl_map_size * sizeof(struct __afl_entry));
    __afl_prev_loc = 0;
}

/* Auto-attach when loaded */
__attribute__((constructor))
static void __afl_auto_init(void) {
    __afl_map_shm();
}
