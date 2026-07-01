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
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#define AFL_MAP_SIZE 65536

uint8_t *__afl_area = NULL;
uint32_t __afl_prev_loc = 0;

__attribute__((visibility("default")))
void __afl_map_shm(void) {
    char *id = getenv("__AFL_SHM_ID");
    if (!id) return;
    int shmid = atoi(id);
    if (shmid <= 0) return;
    void *p = shmat(shmid, NULL, 0);
    if (p == (void *)-1) return;
    __afl_area = (uint8_t *)p;
}

__attribute__((visibility("default"), always_inline))
static inline void __afl_map_edge(uint32_t cur_loc) {
    if (__afl_area) {
        uint32_t idx = (__afl_prev_loc ^ cur_loc) & (AFL_MAP_SIZE - 1);
        __afl_area[idx]++;
    }
    __afl_prev_loc = cur_loc >> 1;
}

__attribute__((visibility("default")))
void __afl_map_reset(void) {
    if (__afl_area)
        memset(__afl_area, 0, AFL_MAP_SIZE);
    __afl_prev_loc = 0;
}

/* Auto-attach when loaded */
__attribute__((constructor))
static void __afl_auto_init(void) {
    __afl_map_shm();
}
