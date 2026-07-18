/* fuzz_shm.c — SHM-based fuzz target for persistent in-process execution.

Compiles as a shared library (.so) that can be loaded via ctypes.
Reads input from SHM, executes the fgrep search pipeline, writes
coverage to SHM. Eliminates fork+exec entirely for maximum EPS.

Compile:
  gcc -O2 -g -mavx2 -shared -fPIC \
      -include ../src/fuzzer_tool/adapters/afl_shim.c \
      -I../fgrep/include -I../fgrep/src \
      -o fuzz_shm.so fuzz_shm.c \
      ../fgrep/src/regex_engine.c ../fgrep/src/simd.c ../fgrep/src/cpu.c \
      ../fgrep/src/output.c ../fgrep/src/search.c ../fgrep/src/bmh_simd.c \
      ../fgrep/src/io.c ../fgrep/src/fileutil.c -lpthread

Usage via ctypes:
  import ctypes
  lib = ctypes.CDLL('./fuzz_shm.so')
  lib.fuzz_shm_init(input_shm_id, output_shm_id, map_size)
  lib.fuzz_shm_run(data_ptr, data_len)  # reads from input SHM, writes to output SHM
  lib.fuzz_shm_cleanup()
*/
#include "fgrep.h"
#include "search.h"
#include "regex_engine.h"
#include "simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/shm.h>
#include <stdint.h>

extern void __afl_map_edge(unsigned int cur_loc);

/* Patterns for fuzzing */
static const char *patterns[] = {
    "test", ".", "[a-z]+\\d+", "(?:ab|cd){3}",
    "\\b\\w+\\b", "^$", ".", "X{10,20}",
};
#define NUM_PATTERNS (sizeof(patterns) / sizeof(patterns[0]))

/* SHM state */
static uint8_t *input_shm = NULL;
static uint8_t *output_shm = NULL;
static size_t input_shm_size = 0;
static size_t output_shm_size = 0;

/* Initialize SHM regions for data transfer */
__attribute__((visibility("default")))
int fuzz_shm_init(int input_shmid, int output_shmid, size_t map_size) {
    input_shm = shmat(input_shmid, NULL, 0);
    output_shm = shmat(output_shmid, NULL, 0);
    if (input_shm == (void *)-1 || output_shm == (void *)-1) {
        return -1;
    }
    input_shm_size = map_size;
    output_shm_size = map_size;
    return 0;
}

/* Run fuzz target: data is the input bytes directly (not from SHM) */
__attribute__((visibility("default")))
int fuzz_shm_run(const uint8_t *data, size_t len) {
    if (!data || len == 0) return -1;

    __afl_map_edge(0x1000);

    if (len < 4) { __afl_map_edge(0x1003); return 0; }

    /* Config bytes */
    uint8_t pat_idx = data[0] % NUM_PATTERNS;
    bool ignore_case = (data[1] & 0x01) != 0;
    bool invert_match = (data[1] & 0x02) != 0;
    bool count_only = (data[1] & 0x04) != 0;
    bool fixed_string = (data[1] & 0x08) != 0;
    bool line_number = (data[1] & 0x10) != 0;

    __afl_map_edge(0x1100 + pat_idx);
    __afl_map_edge(ignore_case ? 0x1201 : 0x1200);
    __afl_map_edge(fixed_string ? 0x1301 : 0x1300);
    __afl_map_edge(count_only ? 0x1401 : 0x1400);

    const char *pattern_data = (const char *)(data + 4);
    size_t data_len = len - 4;

    /* Compile pattern */
    fgrep_pattern_t pat;
    fgrep_status_t st = fgrep_pattern_compile(&pat, patterns[pat_idx], fixed_string, ignore_case);
    if (st != FGREP_OK) { __afl_map_edge(0x1500); return 0; }

    fgrep_options_t opts = {
        .fixed_string = fixed_string,
        .ignore_case = ignore_case,
        .invert_match = invert_match,
        .count_only = count_only,
        .line_number = line_number,
        .color = false,
        .max_count = 0,
    };

    FILE *devnull = fopen("/dev/null", "w");
    if (!devnull) { fgrep_pattern_destroy(&pat); return 0; }

    fgrep_stats_t stats = {0};
    fgrep_search_ctx_t ctx = {
        .opts = &opts,
        .pattern = &pat,
        .stats = &stats,
        .output = devnull,
        .output_mutex = NULL,
    };

    __afl_map_edge(0x1600);
    size_t match_count;
    search_data(pattern_data, data_len, "<fuzz>", &ctx, &match_count);

    __afl_map_edge(match_count > 0 ? 0x1701 : 0x1700);
    fclose(devnull);
    fgrep_pattern_destroy(&pat);
    __afl_map_edge(0x1fff);
    return 0;
}

/* Cleanup SHM */
__attribute__((visibility("default")))
void fuzz_shm_cleanup(void) {
    if (input_shm && input_shm != (void *)-1) {
        shmdt(input_shm);
        input_shm = NULL;
    }
    if (output_shm && output_shm != (void *)-1) {
        shmdt(output_shm);
        output_shm = NULL;
    }
}
