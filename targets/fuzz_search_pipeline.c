/* Fuzz target for the full fgrep search pipeline.
 *
 * Exercises search_data() end-to-end: pattern compilation, SIMD search
 * (AVX2 fixed-string path), regex matching, output formatting.
 * This is the highest-value target — it covers the real code path that
 * processes untrusted file content.
 *
 * The input is split: bytes 0-3 select config, rest is the file content.
 *
 * Compile:
 *   gcc -O2 -g -fsanitize=address -I../fgrep/include -I../fgrep/src \
 *       -o fuzz_search_pipeline fuzz_search_pipeline.c \
 *       ../fgrep/src/regex_engine.c ../fgrep/src/simd.c ../fgrep/src/cpu.c \
 *       ../fgrep/src/output.c ../fgrep/src/search.c ../fgrep/src/bmh_simd.c \
 *       -lpthread
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

/* Patterns for fuzzing — covers fixed, regex, and edge cases */
static const char *patterns[] = {
    "test",           /* simple fixed string */
    ".",              /* match any char regex */
    "[a-z]+\\d+",    /* mixed class + escape */
    "(?:ab|cd){3}",  /* non-capturing alternation */
    "\\b\\w+\\b",   /* word boundary + word class */
    "^$",            /* empty line anchor */
    ".",              /* single dot — matches everything */
    "X{10,20}",     /* bounded quantifier */
};
#define NUM_PATTERNS (sizeof(patterns) / sizeof(patterns[0]))

int main(int argc, char **argv) {
    unsigned char buf[65536];
    size_t n;

    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 1;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        rewind(f);
        if (sz > (long)sizeof(buf)) sz = sizeof(buf);
        n = fread(buf, 1, (size_t)sz, f);
        fclose(f);
    } else {
        n = fread(buf, 1, sizeof(buf), stdin);
    }
    if (n < 4) return 0;

    /* Config bytes select mode */
    uint8_t pat_idx = buf[0] % NUM_PATTERNS;
    bool ignore_case = (buf[1] & 0x01) != 0;
    bool invert_match = (buf[1] & 0x02) != 0;
    bool count_only = (buf[1] & 0x04) != 0;
    bool fixed_string = (buf[1] & 0x08) != 0;
    bool line_number = (buf[1] & 0x10) != 0;
    bool use_color = false; /* no tty in fuzzer */

    const char *data = (const char *)(buf + 4);
    size_t data_len = n - 4;

    /* Compile pattern */
    fgrep_pattern_t pat;
    fgrep_status_t st = fgrep_pattern_compile(&pat, patterns[pat_idx], fixed_string, ignore_case);
    if (st != FGREP_OK) return 0;

    /* Build options */
    fgrep_options_t opts = {
        .fixed_string = fixed_string,
        .ignore_case = ignore_case,
        .invert_match = invert_match,
        .count_only = count_only,
        .line_number = line_number,
        .color = use_color,
        .max_count = 0,
    };

    /* Suppress output — /dev/null */
    FILE *devnull = fopen("/dev/null", "w");
    if (!devnull) {
        fgrep_pattern_destroy(&pat);
        return 0;
    }

    fgrep_stats_t stats = {0};
    fgrep_search_ctx_t ctx = {
        .opts = &opts,
        .pattern = &pat,
        .stats = &stats,
        .output = devnull,
        .output_mutex = NULL,
    };

    /* Fuzz the full search pipeline */
    size_t match_count;
    search_data(data, data_len, "<fuzz>", &ctx, &match_count);

    fclose(devnull);
    fgrep_pattern_destroy(&pat);
    return 0;
}
