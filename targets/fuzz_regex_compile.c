/* Fuzz target for fgrep regex compilation.
 *
 * Exercises fgrep_pattern_compile() with user-supplied patterns.
 * The primary attack surface is glibc's regcomp() with adversarial
 * regex patterns — backtracking bombs, quantifier nesting, class
 * intersections, etc.
 *
 * Compile:
 *   gcc -O2 -g -fsanitize=address -I../fgrep/include -I../fgrep/src \
 *       -o fuzz_regex_compile fuzz_regex_compile.c \
 *       ../fgrep/src/regex_engine.c ../fgrep/src/simd.c ../fgrep/src/cpu.c
 */
#include "fgrep.h"
#include "regex_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    if (n == 0) return 0;

    /* Null-terminate for regcomp safety */
    char pattern[65537];
    memcpy(pattern, buf, n);
    pattern[n] = '\0';

    /* Try both fixed-string and regex compilation paths */
    fgrep_pattern_t pat;

    /* Fixed-string mode — low complexity, but exercises strdup/strlen path */
    fgrep_status_t st = fgrep_pattern_compile(&pat, pattern, true, false);
    if (st == FGREP_OK) {
        fgrep_pattern_destroy(&pat);
    }

    /* Regex mode — high value, exercises regcomp with adversarial patterns */
    st = fgrep_pattern_compile(&pat, pattern, false, false);
    if (st == FGREP_OK) {
        /* Also exercise match with the compiled pattern against itself */
        size_t ms, ml;
        fgrep_pattern_match(&pat, pattern, n, &ms, &ml);
        fgrep_pattern_destroy(&pat);
    }

    /* Regex + ignore-case variant */
    st = fgrep_pattern_compile(&pat, pattern, false, true);
    if (st == FGREP_OK) {
        size_t ms, ml;
        fgrep_pattern_match(&pat, pattern, n, &ms, &ml);
        fgrep_pattern_destroy(&pat);
    }

    return 0;
}
