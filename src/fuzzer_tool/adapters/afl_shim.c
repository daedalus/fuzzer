/*
 * Minimal AFL-style coverage shim for in-process fuzzing.
 *
 * Provides:
 *   - __afl_map_shm()     — attach to AFL SHM bitmap
 *   - __afl_map_edge()    — record an edge: hash(prev, cur) -> bitmap[idx]++
 *   - __afl_map_reset()   — zero the bitmap between iterations
 *
 * Compile target with:
 *   gcc -O2 -g -shared -fPIC -include afl_shim.c -o target.so target.c -lpng -lz
 * Or include this header and call __afl_map_shm() once at startup.
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

__attribute__((visibility("default"), always_inline))
static inline void __afl_map_edge(uint32_t cur_loc) {
    if (__afl_area) {
        uint32_t idx = (__afl_prev_loc ^ cur_loc) & __afl_map_mask;
        __afl_area[idx]++;
    }
    __afl_prev_loc = cur_loc >> 1;
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
    __afl_map_shm();
    fprintf(stderr, "[shim] map_size=%u map_mask=%u area=%p\n",
            __afl_map_size, __afl_map_mask, (void *)__afl_area);
}
