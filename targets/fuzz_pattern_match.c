/* Fuzz target for fgrep pattern matching.
 *
 * Compiles a fixed regex pattern, then fuzzes the match function
 * with adversarial input data. Covers both the fixed-string fast path
 * (SIMD memchr + memcmp) and the regexec slow path.
 *
 * The data is split: first byte selects mode, rest is the match target.
 *
 * Compile:
 *   gcc -O2 -g -fsanitize=address -I../fgrep/include -I../fgrep/src \
 *       -o fuzz_pattern_match fuzz_pattern_match.c \
 *       ../fgrep/src/regex_engine.c ../fgrep/src/simd.c ../fgrep/src/cpu.c
 */
#include "fgrep.h"
#include "regex_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Patterns chosen to exercise different regex engine paths */
static const char *patterns[] = {
    ".",           /* single char class */
    ".*",          /* greedy star */
    ".+",          /* greedy plus */
    "[a-z]+",      /* character class range */
    "(a|b|c){1,5}", /* alternation + bounded quantifier */
    "\\d+\\.\\d+",  /* escape sequences */
    "^[[:alpha:]]+$", /* POSIX class */
    "(?:ab)+",     /* non-capturing group */
    "(?:(?:x|y){2,}){1,}", /* nested groups + quantifiers */
    "\\bword\\b",  /* word boundaries */
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
    if (n < 2) return 0;

    /* Use first byte to pick pattern index */
    uint8_t pat_idx = buf[0] % NUM_PATTERNS;
    const char *pattern = patterns[pat_idx];
    const char *data = (const char *)(buf + 1);
    size_t data_len = n - 1;

    /* Compile the pattern */
    fgrep_pattern_t pat;
    fgrep_status_t st = fgrep_pattern_compile(&pat, pattern, false, (buf[0] & 0x80) != 0);
    if (st != FGREP_OK) return 0;

    /* Fuzz the match function */
    size_t ms, ml;
    fgrep_pattern_match(&pat, data, data_len, &ms, &ml);

    fgrep_pattern_destroy(&pat);
    return 0;
}
