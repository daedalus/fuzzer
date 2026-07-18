/* Fuzz target for the full fgrep search pipeline.
 *
 * Exercises search_data() end-to-end: pattern compilation, SIMD search
 * (AVX2 fixed-string path), regex matching, output formatting.
 *
 * Compile with AFL edge coverage:
 *   gcc -O2 -g -fsanitize=address -mavx2 \
 *       -include ../src/fuzzer_tool/adapters/afl_shim.c \
 *       -I../fgrep/include -I../fgrep/src \
 *       -o fuzz_search_pipeline fuzz_search_pipeline.c \
 *       ../fgrep/src/regex_engine.c ../fgrep/src/simd.c ../fgrep/src/cpu.c \
 *       ../fgrep/src/output.c ../fgrep/src/search.c ../fgrep/src/bmh_simd.c \
 *       ../fgrep/src/io.c ../fgrep/src/fileutil.c -lpthread
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

extern void __afl_map_edge(unsigned int cur_loc);

static const char *patterns[] = {
    "test", ".", "[a-z]+\\d+", "(?:ab|cd){3}",
    "\\b\\w+\\b", "^$", ".", "X{10,20}",
};
#define NUM_PATTERNS (sizeof(patterns) / sizeof(patterns[0]))

int main(int argc, char **argv) {
    unsigned char buf[65536];
    size_t n;

    __afl_map_edge(0x1000);
    if (argc == 2) {
        __afl_map_edge(0x1001);
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 1;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        rewind(f);
        if (sz > (long)sizeof(buf)) sz = sizeof(buf);
        n = fread(buf, 1, (size_t)sz, f);
        fclose(f);
    } else {
        __afl_map_edge(0x1002);
        n = fread(buf, 1, sizeof(buf), stdin);
    }
    if (n < 4) { __afl_map_edge(0x1003); return 0; }

    /* Config bytes select mode */
    uint8_t pat_idx = buf[0] % NUM_PATTERNS;
    bool ignore_case = (buf[1] & 0x01) != 0;
    bool invert_match = (buf[1] & 0x02) != 0;
    bool count_only = (buf[1] & 0x04) != 0;
    bool fixed_string = (buf[1] & 0x08) != 0;
    bool line_number = (buf[1] & 0x10) != 0;

    __afl_map_edge(0x1100 + pat_idx);
    __afl_map_edge(ignore_case ? 0x1201 : 0x1200);
    __afl_map_edge(fixed_string ? 0x1301 : 0x1300);
    __afl_map_edge(count_only ? 0x1401 : 0x1400);

    const char *data = (const char *)(buf + 4);
    size_t data_len = n - 4;

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
    search_data(data, data_len, "<fuzz>", &ctx, &match_count);

    __afl_map_edge(match_count > 0 ? 0x1701 : 0x1700);
    fclose(devnull);
    fgrep_pattern_destroy(&pat);
    __afl_map_edge(0x1fff);
    return 0;
}
