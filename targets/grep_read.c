/* Fuzz target for GNU Grep — in-process harness over the vendored engines.
 *
 * This target links GNU Grep's own matcher code (the gnulib DFA engine in
 * lib/dfa.c and grep's Commentz-Walter kwset in src/kwset.c) out of
 * vendor/grep and drives it directly.  It does NOT exec /usr/bin/grep.
 *
 * Why that matters: the previous version of this file fork/exec'd the
 * system grep binary once per execution.  Two consequences, both bad:
 *
 *   1. No coverage of grep.  The SHM bitmap only ever saw this wrapper's
 *      own basic blocks, because the code under test ran in a different
 *      process image with no instrumentation.  Every edge this target
 *      reported was an edge of the harness.
 *   2. A process spawn per execution.  That is the same fork/exec cost
 *      that docs/handover/handover_boltzmann_ab_2026-08-30.md §1 uses to
 *      disqualify the `locked` target set for cost-sensitive arms — so a
 *      .so in the `direct_lite` set reintroduced exactly what that set
 *      exists to avoid.  Measured: ~1.3 ms of matcher time per execution
 *      against a 13-26 ms cell budget, i.e. over 90% of the campaign was
 *      spawn overhead.
 *
 * Requires the vendored source:
 *
 *   tools/vendor_grep.sh        # download, configure, build libgreputils.a
 *   tools/build_targets.sh
 *
 * Input format (unchanged from the exec wrapper, so existing corpora and
 * dictionaries still parse):
 *
 *   byte 0:              mode
 *   byte 1:              pattern length (clamped to 255)
 *   bytes 2..2+plen-1:   pattern
 *   bytes 2+plen..:      subject text
 *
 * The pattern is passed to the engines as (pointer, length) rather than as
 * a C string.  The exec wrapper had to NUL-terminate it for argv, which
 * silently truncated every pattern at its first zero byte and made that
 * whole region of the input space unreachable.
 *
 * What the mode byte selects is an *engine configuration*, not a grep
 * command-line flag.  The exec wrapper's modes named CLI flags (-G, -E,
 * -w, -v ...); several of those are implemented in grep.c's output layer
 * rather than in the matchers, and this harness has no output layer.
 * Claiming to test "-v" by naming a mode MODE_INVERT would assert
 * coverage this target does not have.  The modes below say what is
 * actually exercised.
 *
 * Compile standalone:
 *   gcc -O2 -g -include vendor/grep/config.h \
 *       -I vendor/grep -I vendor/grep/lib -I vendor/grep/src \
 *       -o targets/grep_read targets/grep_read.c vendor/grep/src/kwset.c \
 *       vendor/grep/lib/libgreputils.a
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>
#include <setjmp.h>
#include <regex.h>

#include "localeinfo.h"
#include "dfa.h"
#include "kwset.h"

#ifdef GREP_HAVE_PCRE2
#  define PCRE2_CODE_UNIT_WIDTH 8
#  include <pcre2.h>
#endif

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* ── Engine error trapping ─────────────────────────────────────────────
 *
 * dfaerror() is declared _Noreturn and the DFA engine calls it for every
 * malformed pattern — which, under a fuzzer, is most of them.  grep itself
 * exits the process there.  A persistent in-process target cannot, so the
 * harness longjmps back to the call site instead.
 *
 * g_trap_active guards the jump buffer: dfaerror can in principle be
 * reached outside a protected region, and longjmping to a stale frame is
 * undefined behaviour that would show up as an unreproducible crash.
 * Aborting is the honest outcome there, and it is a real bug report.
 */
static jmp_buf g_trap;
static volatile bool g_trap_active;

_Noreturn void dfaerror(const char *mesg) {
    (void)mesg;
    if (g_trap_active) {
        g_trap_active = false;
        longjmp(g_trap, 1);
    }
    abort();
}

void dfawarn(const char *mesg) {
    (void)mesg;
    __afl_map_edge(0x1300);
}

/* Engine configurations selected by the mode byte. */
enum grep_mode {
    MODE_DFA_BASIC    = 0,  /* DFA, POSIX basic syntax   (grep -G's matcher) */
    MODE_DFA_EXTENDED = 1,  /* DFA, POSIX extended       (grep -E's matcher) */
    MODE_KWSET        = 2,  /* Commentz-Walter kwset     (grep -F's matcher) */
    MODE_PCRE         = 3,  /* PCRE2, when built with it (grep -P's matcher) */
    MODE_KWSET_FOLD   = 4,  /* kwset with a case-folding translation table */
    MODE_DFA_ICASE    = 5,  /* DFA, basic syntax, RE_ICASE */
    MODE_DFA_EOL_NUL  = 6,  /* DFA, extended, NUL as end-of-line */
    MODE_DFA_MUST     = 7,  /* DFA, extended, plus dfamust/dfasuperset */
    MODE_DFA_ANCHOR   = 8,  /* DFA, extended, DFA_ANCHOR */
    MODE_MAX          = 9,
};

/* Case-folding translation table for kwsalloc, as grep builds in
 * searchutils.c.  Static so the pointer stays valid for the kwset's
 * lifetime — kwsalloc stores the pointer, it does not copy the table. */
static char g_fold_table[256];
static bool g_fold_ready;

static const char *fold_table(void) {
    if (!g_fold_ready) {
        for (int i = 0; i < 256; i++) {
            g_fold_table[i] = (i >= 'A' && i <= 'Z') ? (char)(i - 'A' + 'a') : (char)i;
        }
        g_fold_ready = true;
    }
    return g_fold_table;
}

/* Run the DFA engine over one subject buffer.
 *
 * dfaexec requires a writable buffer one byte longer than the subject,
 * with a sentinel newline stored at *end; it reads that byte.  Handing it
 * the fuzzer's own input buffer would be an out-of-bounds write.
 */
static int run_dfa(const unsigned char *pat, size_t plen,
                   const unsigned char *text, size_t tlen,
                   reg_syntax_t syntax, int dfaopts, bool want_must) {
    struct localeinfo li;
    init_localeinfo(&li);

    struct dfa *d = dfaalloc();
    if (!d) return 0;

    char *buf = malloc(tlen + 1);
    if (!buf) { free(d); return 0; }
    if (tlen) memcpy(buf, text, tlen);
    buf[tlen] = '\n';                    /* sentinel dfaexec requires */

    int matched = 0;
    g_trap_active = true;
    if (setjmp(g_trap) == 0) {
        dfasyntax(d, &li, syntax, dfaopts);

        if (want_must) {
            /* dfamust must run against a parsed-but-not-compiled DFA, so
             * this path parses explicitly and then passes a null pattern
             * to dfacomp, per the contract in dfa.h. */
            dfaparse((const char *)pat, plen, d);
            struct dfamust *dm = dfamust(d);
            if (dm) {
                __afl_map_edge(0x1301);
                dfamustfree(dm);
            }
            dfacomp(NULL, 0, d, true);
        } else {
            dfacomp((const char *)pat, plen, d, true);
        }
        g_trap_active = false;

        if (dfasupported(d)) {
            __afl_map_edge(0x1302);
            bool backref = false;
            idx_t count = 0;
            char *m = dfaexec(d, buf, buf + tlen, true, &count, &backref);
            matched = (m != NULL);
            if (backref) __afl_map_edge(0x1303);

            /* dfasuperset returns a borrowed pointer owned by d; it must
             * not be freed here.  dfafree(d) releases it. */
            if (want_must) {
                struct dfa *sup = dfasuperset(d);
                if (sup) {
                    __afl_map_edge(0x1304);
                    bool sbackref = false;
                    dfaexec(sup, buf, buf + tlen, true, NULL, &sbackref);
                }
            }
        }
    } else {
        /* Malformed pattern: the engine rejected it. */
        __afl_map_edge(0x1305);
    }
    g_trap_active = false;

    dfafree(d);
    free(d);          /* dfa.h: pass the dfaalloc pointer to free() after dfafree */
    free(buf);
    return matched;
}

static int run_kwset(const unsigned char *pat, size_t plen,
                     const unsigned char *text, size_t tlen, bool fold) {
    kwset_t kws = kwsalloc(fold ? fold_table() : NULL);
    if (!kws) return 0;

    kwsincr(kws, (const char *)pat, plen);
    kwsprep(kws);
    __afl_map_edge(0x1310 + (kwswords(kws) & 0xf));

    int matched = 0;
    if (tlen) {
        struct kwsmatch km;
        ptrdiff_t off = kwsexec(kws, (const char *)text, tlen, &km, true);
        matched = (off >= 0);
    }
    kwsfree(kws);
    return matched;
}

#ifdef GREP_HAVE_PCRE2
static int run_pcre(const unsigned char *pat, size_t plen,
                    const unsigned char *text, size_t tlen) {
    int errcode = 0;
    PCRE2_SIZE erroffset = 0;
    pcre2_code *re = pcre2_compile((PCRE2_SPTR)pat, plen, 0,
                                   &errcode, &erroffset, NULL);
    if (!re) { __afl_map_edge(0x1320); return 0; }

    pcre2_match_data *md = pcre2_match_data_create_from_pattern(re, NULL);
    int rc = -1;
    if (md) {
        rc = pcre2_match(re, (PCRE2_SPTR)text, tlen, 0, 0, md, NULL);
        pcre2_match_data_free(md);
    }
    pcre2_code_free(re);
    __afl_map_edge(rc >= 0 ? 0x1321 : 0x1322);
    return rc >= 0;
}
#endif

__attribute__((visibility("default")))
int fuzz_grep(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x1000);
    if (size < 3) { __afl_map_edge(0x1001); return 0; }

    unsigned char mode = buf[0];
    size_t plen = buf[1];
    if (plen > 255) plen = 255;
    if (2 + plen >= size) { __afl_map_edge(0x1002); return 0; }

    const unsigned char *pattern = buf + 2;
    const unsigned char *text = buf + 2 + plen;
    size_t text_len = size - 2 - plen;

    mode %= MODE_MAX;
    __afl_map_edge(0x1100 + mode);

    int matched = 0;
    switch (mode) {
        case MODE_DFA_BASIC:
            matched = run_dfa(pattern, plen, text, text_len,
                              RE_SYNTAX_GREP, 0, false);
            break;
        case MODE_DFA_EXTENDED:
            matched = run_dfa(pattern, plen, text, text_len,
                              RE_SYNTAX_EGREP, 0, false);
            break;
        case MODE_KWSET:
            matched = run_kwset(pattern, plen, text, text_len, false);
            break;
        case MODE_PCRE:
#ifdef GREP_HAVE_PCRE2
            matched = run_pcre(pattern, plen, text, text_len);
#else
            /* Built without PCRE2. Fall back to the extended DFA rather
             * than returning early: a silent no-op mode would let a
             * ninth of the input space score as covered while executing
             * nothing. */
            __afl_map_edge(0x1323);
            matched = run_dfa(pattern, plen, text, text_len,
                              RE_SYNTAX_EGREP, 0, false);
#endif
            break;
        case MODE_KWSET_FOLD:
            matched = run_kwset(pattern, plen, text, text_len, true);
            break;
        case MODE_DFA_ICASE:
            matched = run_dfa(pattern, plen, text, text_len,
                              RE_SYNTAX_GREP | RE_ICASE, 0, false);
            break;
        case MODE_DFA_EOL_NUL:
            matched = run_dfa(pattern, plen, text, text_len,
                              RE_SYNTAX_EGREP, DFA_EOL_NUL, false);
            break;
        case MODE_DFA_MUST:
            matched = run_dfa(pattern, plen, text, text_len,
                              RE_SYNTAX_EGREP, 0, true);
            break;
        case MODE_DFA_ANCHOR:
            matched = run_dfa(pattern, plen, text, text_len,
                              RE_SYNTAX_EGREP, DFA_ANCHOR, false);
            break;
        default:
            matched = run_dfa(pattern, plen, text, text_len,
                              RE_SYNTAX_GREP, 0, false);
            break;
    }

    __afl_map_edge(matched ? 0x1200 : 0x1201);
    return 0;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_grep(buf, size);
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(1000)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fuzz_grep(buf, len);
    }
    return 0;
}
#else
int main(int argc, char **argv) {
    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 1;
        fseek(f, 0, SEEK_END);
        long s = ftell(f);
        rewind(f);
        unsigned char *buf = malloc(s);
        if (buf) {
            fread(buf, 1, s, f);
            int rc = fuzz_grep(buf, s);
            free(buf);
            fclose(f);
            return rc;
        }
        fclose(f);
        return 1;
    } else {
        unsigned char buf[65536];
        size_t n = fread(buf, 1, sizeof(buf), stdin);
        if (n > 0) return fuzz_grep(buf, n);
    }
    return 0;
}
#endif
