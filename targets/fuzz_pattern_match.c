/* Fuzz target for fgrep pattern matching.
 *
 * Compiles a fixed regex pattern, then fuzzes the match function
 * with adversarial input data. Covers both the fixed-string fast path
 * (SIMD memchr + memcmp) and the regexec slow path.
 *
 * Compile with AFL edge coverage:
 *   gcc -O2 -g -fsanitize=address -include ../src/fuzzer_tool/adapters/afl_shim.c \
 *       -I../fgrep/include -I../fgrep/src \
 *       -o fuzz_pattern_match fuzz_pattern_match.c \
 *       ../fgrep/src/regex_engine.c ../fgrep/src/simd.c ../fgrep/src/cpu.c
 */
#include "fgrep.h"
#include "regex_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern void __afl_map_edge(unsigned int cur_loc);

static const char *patterns[] = {
    ".", ".*", ".+", "[a-z]+", "(a|b|c){1,5}",
    "\\d+\\.\\d+", "^[[:alpha:]]+$", "(?:ab)+",
    "(?:(?:x|y){2,}){1,}", "\\bword\\b",
};
#define NUM_PATTERNS (sizeof(patterns) / sizeof(patterns[0]))

static int fuzz_pattern_match(const unsigned char *buf, size_t n) {
    __afl_map_edge(0x1000);
    if (n < 2) { __afl_map_edge(0x1003); return 0; }

    __afl_map_edge(0x1100 + (buf[0] % NUM_PATTERNS));
    uint8_t pat_idx = buf[0] % NUM_PATTERNS;
    const char *pattern = patterns[pat_idx];
    const char *data = (const char *)(buf + 1);
    size_t data_len = n - 1;

    fgrep_pattern_t pat;
    bool ignore_case = (buf[0] & 0x80) != 0;
    __afl_map_edge(ignore_case ? 0x1201 : 0x1200);
    fgrep_status_t st = fgrep_pattern_compile(&pat, pattern, false, ignore_case);
    if (st != FGREP_OK) { __afl_map_edge(0x1202); return 0; }

    __afl_map_edge(0x1300);
    size_t ms, ml;
    fgrep_pattern_match(&pat, data, data_len, &ms, &ml);

    __afl_map_edge(0x1fff);
    fgrep_pattern_destroy(&pat);
    return 0;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_pattern_match(buf, size);
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_pattern_match(buf, len);
    }
    return 0;
}
#else
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
    return fuzz_pattern_match(buf, n);
}
#endif
