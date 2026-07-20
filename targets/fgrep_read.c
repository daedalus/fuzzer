/* Fuzz target for fgrep — searches in-memory data with patterns.
 *
 * Tests three code paths from the fgrep codebase:
 *   1. SIMD-accelerated fixed-string search (search_data with fixed_string)
 *   2. POSIX regex search (search_data with regex pattern)
 *   3. Boyer-Moore-Horspool engine (kwset_engine_search)
 *
 * Input format: two bytes split-mode | one byte patlen | pattern | text
 *   byte 0: mode (0=fixed, 1=regex, 2=kwset, 3=all-three)
 *   byte 1-2: pattern length (16-bit LE, clamped to 256)
 *   bytes 3..3+plen-1: pattern
 *   bytes 3+plen..: text to search
 *
 * Returns 1 on crashes (bugs in fgrep), 0 on clean execution.
 * Compile standalone:
 *   gcc -O2 -g -mavx2 -o targets/fgrep_read targets/fgrep_read.c \
 *       ../fgrep/src/search.c ../fgrep/src/regex_engine.c \
 *       ../fgrep/src/simd.c ../fgrep/src/cpu.c ../fgrep/src/output.c \
 *       ../fgrep/src/kwset_engine.c ../fgrep/src/bmh_simd.c \
 *       -I../fgrep/include -I../fgrep/src -lpthread -fsanitize=address
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

/* fgrep headers */
#include "fgrep.h"
#include "search.h"
#include "regex_engine.h"
#include "output.h"
#include "kwset_engine.c"
#include "bmh_simd.c"
#include "simd.c"
#include "cpu.c"
#include "output.c"
#include "regex_engine.c"
#include "search.c"

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

__attribute__((visibility("default")))
int fuzz_fgrep(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);
    if (size < 4) { __afl_map_edge(0x1001); return 0; }

    unsigned char mode = buf[0];
    size_t plen = (size_t)buf[1] | ((size_t)buf[2] << 8);
    if (plen > 256) plen = 256;
    if (3 + plen >= size) { __afl_map_edge(0x1002); return 0; }

    const char *pattern = (const char *)buf + 3;
    const char *text = (const char *)buf + 3 + plen;
    size_t text_len = size - 3 - plen;

    __afl_map_edge(0x1000 + mode);

    /* Null-terminate pattern for C APIs — pattern can contain any bytes */
    char pat_buf[257];
    memcpy(pat_buf, pattern, plen);
    pat_buf[plen] = '\0';

    /* Suppress output during fuzzing */
    FILE *devnull = fopen("/dev/null", "w");
    if (!devnull) devnull = stderr;

    if (mode == 0 || mode == 3) {
        /* Mode 0: Fixed-string SIMD search via search_data */
        __afl_map_edge(0x1100);
        fgrep_options_t opts = {
            .fixed_string = true,
            .count_only = false,
            .color = false,
            .line_number = false,
            .max_count = 0,
        };
        fgrep_pattern_t pat;
        fgrep_status_t st = fgrep_pattern_compile(&pat, pat_buf, true, false);
        if (st == FGREP_OK) {
            fgrep_stats_t stats = {0};
            pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
            fgrep_search_ctx_t ctx = {
                .opts = &opts,
                .pattern = &pat,
                .stats = &stats,
                .output = devnull,
                .output_mutex = &mtx,
            };
            size_t match_count = 0;
            search_data(text, text_len, "<fuzz>", &ctx, &match_count);
            __afl_map_edge(0x1101);
            pthread_mutex_destroy(&mtx);
            fgrep_pattern_destroy(&pat);
        }
    }

    if (mode == 1 || mode == 3) {
        /* Mode 1: Regex search via search_data */
        __afl_map_edge(0x1200);
        fgrep_options_t opts = {
            .fixed_string = false,
            .count_only = false,
            .color = false,
            .line_number = false,
            .max_count = 0,
        };
        fgrep_pattern_t pat;
        fgrep_status_t st = fgrep_pattern_compile(&pat, pat_buf, false, false);
        if (st == FGREP_OK) {
            fgrep_stats_t stats = {0};
            pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
            fgrep_search_ctx_t ctx = {
                .opts = &opts,
                .pattern = &pat,
                .stats = &stats,
                .output = devnull,
                .output_mutex = &mtx,
            };
            size_t match_count = 0;
            search_data(text, text_len, "<fuzz>", &ctx, &match_count);
            __afl_map_edge(0x1201);
            pthread_mutex_destroy(&mtx);
            fgrep_pattern_destroy(&pat);
        }
    }

    if (mode == 2 || mode == 3) {
        /* Mode 2: Boyer-Moore-Horspool engine */
        __afl_map_edge(0x1300);
        if (plen > 0 && plen <= (int)text_len) {
            kwset_engine_t ks;
            kwset_engine_init(&ks, pat_buf, (int)plen);
            kwset_engine_search(&ks, text, (int)text_len);
            __afl_map_edge(0x1301);
            kwset_engine_free(&ks);
        }
    }

    if (devnull && devnull != stderr) fclose(devnull);
    return 0;
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_fgrep(buf, len);
    }
    return 0;
}
#else
int main(int argc, char **argv) {
    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 1;
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        rewind(f);
        unsigned char *buf = malloc(size);
        if (buf) {
            fread(buf, 1, size, f);
            int rc = fuzz_fgrep(buf, size);
            free(buf);
            fclose(f);
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) return fuzz_fgrep(buf, n);
    }
    return 0;
}
#endif
