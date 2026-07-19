/* Fuzz target for fgrep regex compilation.
 *
 * Exercises fgrep_pattern_compile() with user-supplied patterns.
 * The primary attack surface is glibc's regcomp() with adversarial
 * regex patterns — backtracking bombs, quantifier nesting, class
 * intersections, etc.
 *
 * Compile with AFL edge coverage:
 *   gcc -O2 -g -fsanitize=address -include ../src/fuzzer_tool/adapters/afl_shim.c \
 *       -I../fgrep/include -I../fgrep/src \
 *       -o fuzz_regex_compile fuzz_regex_compile.c \
 *       ../fgrep/src/regex_engine.c ../fgrep/src/simd.c ../fgrep/src/cpu.c
 */
#include "fgrep.h"
#include "regex_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern void __afl_map_edge(unsigned int cur_loc);

static int fuzz_regex_compile(const unsigned char *buf, size_t n) {
    __afl_map_edge(0x1000);
    if (n == 0) { __afl_map_edge(0x1003); return 0; }

    /* Null-terminate for regcomp safety */
    char pattern[65537];
    size_t copy = n < 65536 ? n : 65536;
    memcpy(pattern, buf, copy);
    pattern[copy] = '\0';

    /* Try both fixed-string and regex compilation paths */
    fgrep_pattern_t pat;

    /* Fixed-string mode */
    __afl_map_edge(0x1100);
    fgrep_status_t st = fgrep_pattern_compile(&pat, pattern, true, false);
    if (st == FGREP_OK) {
        __afl_map_edge(0x1101);
        fgrep_pattern_destroy(&pat);
    } else {
        __afl_map_edge(0x1102);
    }

    /* Regex mode — high value, exercises regcomp with adversarial patterns */
    __afl_map_edge(0x1200);
    st = fgrep_pattern_compile(&pat, pattern, false, false);
    if (st == FGREP_OK) {
        __afl_map_edge(0x1201);
        size_t ms, ml;
        fgrep_pattern_match(&pat, pattern, copy, &ms, &ml);
        fgrep_pattern_destroy(&pat);
    } else {
        __afl_map_edge(0x1202);
    }

    /* Regex + ignore-case variant */
    __afl_map_edge(0x1300);
    st = fgrep_pattern_compile(&pat, pattern, false, true);
    if (st == FGREP_OK) {
        __afl_map_edge(0x1301);
        size_t ms, ml;
        fgrep_pattern_match(&pat, pattern, copy, &ms, &ml);
        fgrep_pattern_destroy(&pat);
    } else {
        __afl_map_edge(0x1302);
    }

    __afl_map_edge(0x1fff);
    return 0;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_regex_compile(buf, size);
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_regex_compile(buf, len);
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
    return fuzz_regex_compile(buf, n);
}
#endif
