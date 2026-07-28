/* Fuzz target for fgrep — consolidated from fgrep_read + fuzz_* targets.
 *
 * Modes:
 *   0 — fixed-string SIMD search via search_data (fgrep_read.c)
 *   1 — regex search via search_data (fgrep_read.c)
 *   2 — Boyer-Moore-Horspool kwset engine (fgrep_read.c)
 *   3 — all-three (0+1+2)
 *   4 — pattern-match-only API (from fuzz_pattern_match.c)
 *   5 — adversarial regex compile+match (from fuzz_regex_compile.c)
 *   6 — full search pipeline with flags (from fuzz_search_pipeline.c)
 *
 * Input format varies per mode — see each handler for details.
 *
 * Compile standalone:
 *   gcc -O2 -g -mavx2 -o targets/fgrep_read targets/fgrep_read.c \
 *       -I../fgrep/include -I../fgrep/src -lpthread -fsanitize=address
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <stdbool.h>

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
#include "io.c"
#include "fileutil.c"
#include "search.c"

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* Pre-defined patterns for modes 4 and 6 */
static const char *fgrep_patterns[] = {
    ".", ".*", ".+", "[a-z]+", "(a|b|c){1,5}",
    "\\d+\\.\\d+", "^[[:alpha:]]+$", "(?:ab)+",
    "(?:(?:x|y){2,}){1,}", "\\bword\\b",
};
#define FGREP_NUM_PATTERNS (sizeof(fgrep_patterns) / sizeof(fgrep_patterns[0]))

/* ── Mode 4: Pattern-match-only (was fuzz_pattern_match.c) ───────────
 *
 * Input: byte 0 = pattern index (bit 7 = ignore_case), rest = data
 * Uses fgrep_pattern_compile() + fgrep_pattern_match() — NOT search_data().
 */
static int mode_pattern_match(const unsigned char *buf, size_t size,
                              FILE *devnull) {
    (void)devnull;
    __afl_map_edge(0x1400);
    if (size < 2) { __afl_map_edge(0x1401); return 0; }

    uint8_t pat_idx = buf[0] % FGREP_NUM_PATTERNS;
    bool ignore_case = (buf[0] & 0x80) != 0;
    const char *pattern = fgrep_patterns[pat_idx];
    const char *data = (const char *)(buf + 1);
    size_t data_len = size - 1;

    __afl_map_edge(0x1410 + pat_idx);
    __afl_map_edge(ignore_case ? 0x1421 : 0x1420);

    fgrep_pattern_t pat;
    fgrep_status_t st = fgrep_pattern_compile(&pat, pattern, false, ignore_case);
    if (st != FGREP_OK) { __afl_map_edge(0x1430); return 0; }

    __afl_map_edge(0x1440);
    size_t ms, ml;
    fgrep_pattern_match(&pat, data, data_len, &ms, &ml);
    fgrep_pattern_destroy(&pat);
    __afl_map_edge(0x14ff);
    return 0;
}

/* ── Mode 5: Adversarial regex compile+match (was fuzz_regex_compile.c) ─
 *
 * Input: all bytes = pattern (up to 65536 bytes)
 * Tests fixed-string, regex, and regex+ignore-case compile+match.
 */
static int mode_regex_compile(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1500);
    if (size == 0) { __afl_map_edge(0x1501); return 0; }

    char pattern[65537];
    size_t copy = size < 65536 ? size : 65536;
    memcpy(pattern, buf, copy);
    pattern[copy] = '\0';

    fgrep_pattern_t pat;

    /* Fixed-string compile */
    __afl_map_edge(0x1510);
    fgrep_status_t st = fgrep_pattern_compile(&pat, pattern, true, false);
    if (st == FGREP_OK) {
        __afl_map_edge(0x1511);
        fgrep_pattern_destroy(&pat);
    } else {
        __afl_map_edge(0x1512);
    }

    /* Regex compile + match */
    __afl_map_edge(0x1520);
    st = fgrep_pattern_compile(&pat, pattern, false, false);
    if (st == FGREP_OK) {
        __afl_map_edge(0x1521);
        size_t ms, ml;
        fgrep_pattern_match(&pat, pattern, copy, &ms, &ml);
        fgrep_pattern_destroy(&pat);
    } else {
        __afl_map_edge(0x1522);
    }

    /* Regex + ignore-case compile + match */
    __afl_map_edge(0x1530);
    st = fgrep_pattern_compile(&pat, pattern, false, true);
    if (st == FGREP_OK) {
        __afl_map_edge(0x1531);
        size_t ms, ml;
        fgrep_pattern_match(&pat, pattern, copy, &ms, &ml);
        fgrep_pattern_destroy(&pat);
    } else {
        __afl_map_edge(0x1532);
    }

    __afl_map_edge(0x15ff);
    return 0;
}

/* ── Mode 6: Full search pipeline with flags (was fuzz_search_pipeline.c) ─
 *
 * Input: byte 0 = pattern index, byte 1 = flags, bytes 2-3 = reserved,
 *        bytes 4+ = data
 * Flags: bit 0 = ignore_case, bit 1 = invert_match, bit 2 = count_only,
 *        bit 3 = fixed_string, bit 4 = line_number
 */
static int mode_pipeline_flags(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1600);
    if (size < 4) { __afl_map_edge(0x1601); return 0; }

    uint8_t pat_idx = buf[0] % FGREP_NUM_PATTERNS;
    bool ignore_case = (buf[1] & 0x01) != 0;
    bool invert_match = (buf[1] & 0x02) != 0;
    bool count_only = (buf[1] & 0x04) != 0;
    bool fixed_string = (buf[1] & 0x08) != 0;
    bool line_number = (buf[1] & 0x10) != 0;

    __afl_map_edge(0x1610 + pat_idx);
    __afl_map_edge(ignore_case ? 0x1621 : 0x1620);
    __afl_map_edge(fixed_string ? 0x1631 : 0x1630);
    __afl_map_edge(count_only ? 0x1641 : 0x1640);

    const char *data = (const char *)(buf + 4);
    size_t data_len = size - 4;

    fgrep_pattern_t pat;
    fgrep_status_t st = fgrep_pattern_compile(&pat, fgrep_patterns[pat_idx],
                                               fixed_string, ignore_case);
    if (st != FGREP_OK) { __afl_map_edge(0x1650); return 0; }

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

    __afl_map_edge(0x1660);
    size_t match_count;
    search_data(data, data_len, "<fuzz>", &ctx, &match_count);

    __afl_map_edge(match_count > 0 ? 0x1671 : 0x1670);
    fclose(devnull);
    fgrep_pattern_destroy(&pat);
    __afl_map_edge(0x16ff);
    return 0;
}

/* ── Main fuzz entry ────────────────────────────────────────────── */

__attribute__((visibility("default")))
int fuzz_fgrep(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);

    /* Modes 4 and 5 handle their own input layout */
    if (size > 0 && buf[0] == 4) return mode_pattern_match(buf, size, NULL);
    if (size > 0 && buf[0] == 5) return mode_regex_compile(buf, size);
    if (size > 0 && buf[0] == 6) return mode_pipeline_flags(buf, size);

    /* Modes 0-3: traditional layout with pattern+text split */
    if (size < 4) { __afl_map_edge(0x1001); return 0; }

    unsigned char mode = buf[0];
    size_t plen = (size_t)buf[1] | ((size_t)buf[2] << 8);
    if (plen > 256) plen = 256;
    if (3 + plen >= size) { __afl_map_edge(0x1002); return 0; }

    const char *pattern = (const char *)buf + 3;
    const char *text = (const char *)buf + 3 + plen;
    size_t text_len = size - 3 - plen;

    __afl_map_edge(0x1000 + mode);

    /* Null-terminate pattern for C APIs */
    char pat_buf[257];
    memcpy(pat_buf, pattern, plen);
    pat_buf[plen] = '\0';

    /* Suppress output during fuzzing */
    FILE *devnull = fopen("/dev/null", "w");
    if (!devnull) devnull = stderr;

    if (mode == 0 || mode == 3) {
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

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_fgrep(buf, size);
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
