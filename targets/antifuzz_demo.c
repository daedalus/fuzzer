/* antifuzz_demo.c — benchmark target hardened with the AntiFuzz techniques.
 *
 * Reproduces the four countermeasures from AntiFuzz (Güler et al., USENIX
 * Security '19) around a single real ASAN bug, so the fuzzer's own defeats
 * (hardened crash oracle, coverage-noise probe, --antifuzz-evade preload)
 * can be measured against each one. Each technique is gated by an env var
 * (default on) so a benchmark can isolate them:
 *
 *   AF_COVERAGE=0   no hash-keyed fake edges        (defeat: coverage-noise probe)
 *   AF_CRASH=0      no crash masking                (defeat: hardened oracle + ASAN)
 *   AF_SPEED=0      no delay on malformed input     (defeat: --antifuzz-evade sleep)
 *   AF_PTRACE=0     no self-ptrace anti-debug       (defeat: --antifuzz-evade ptrace)
 *   AF_HASHCMP=0    magic via memcmp, not a hash    (defeat: cmplog)
 *
 * AF_CRASH and AF_PTRACE are process-wide, so only the executable applies
 * them; the .so (direct_lite) runs AF_COVERAGE/AF_SPEED/AF_HASHCMP only.
 *
 * The bug: input beginning with the 4-byte magic "crsh" overflows a stack
 * buffer (ASAN stack-buffer-overflow). The magic is checked §4.4-style via a
 * byte hash rather than a direct compare.
 *
 * Build is wired in tools/build_targets.sh (ASAN + afl_shim, and a .so with
 * fuzz_shm_run for direct_lite), same as asan_target.c.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <time.h>
#include <unistd.h>

#define MAGIC "crsh"
#define MAGIC_LEN 4
#define FAKE_FUNCS 64        /* fake-edge table size (bounded: Hard Rule 54) */
#define FAKE_CHAIN 8         /* fake funcs called per input */
#define DELAY_MS 250         /* §4.3 delay on malformed input */

static int env_on(const char *name) {
    const char *v = getenv(name);
    return !(v && strcmp(v, "0") == 0);
}

/* ── §4.4  input-byte hash (stands in for a SHA on the magic compare) ──── */
static uint32_t byte_hash(const unsigned char *buf, size_t len) {
    uint32_t h = 2166136261u;  /* FNV-1a */
    for (size_t i = 0; i < len; i++) {
        h ^= buf[i];
        h *= 16777619u;
    }
    return h;
}

/* ── §4.1  fake edges keyed on the input hash ──────────────────────────
 * Each fake function is a distinct basic block. A hash-chosen chain makes
 * nearly every input light up new edges, which is exactly what the
 * coverage-noise probe detects (one-byte tail variants -> distinct sets).  */
static volatile uint32_t fake_sink;

#define FAKE(n)                                                   \
    __attribute__((noinline)) static void fake_##n(uint32_t s) { \
        fake_sink += (s ^ (n)) * 2654435761u;                    \
    }
#define FAKE8(b) FAKE(b##0) FAKE(b##1) FAKE(b##2) FAKE(b##3) \
                 FAKE(b##4) FAKE(b##5) FAKE(b##6) FAKE(b##7)
FAKE8(0) FAKE8(1) FAKE8(2) FAKE8(3) FAKE8(4) FAKE8(5) FAKE8(6) FAKE8(7)
#undef FAKE8
#undef FAKE

typedef void (*fake_fn)(uint32_t);
#define FREF(n) fake_##n
#define FREF8(b) FREF(b##0), FREF(b##1), FREF(b##2), FREF(b##3), \
                 FREF(b##4), FREF(b##5), FREF(b##6), FREF(b##7)
static const fake_fn fake_table[FAKE_FUNCS] = {
    FREF8(0), FREF8(1), FREF8(2), FREF8(3),
    FREF8(4), FREF8(5), FREF8(6), FREF8(7),
};
#undef FREF8
#undef FREF

static void fake_coverage(uint32_t h) {
    for (int i = 0; i < FAKE_CHAIN; i++) {
        fake_table[(h >> i) % FAKE_FUNCS](h + i);
    }
}

/* ── §4.2  crash masking ──────────────────────────────────────────────── */
static sigjmp_buf af_jmp;
static volatile sig_atomic_t af_armed;

static void af_handler(int sig) {
    (void)sig;
    if (af_armed) {
        af_armed = 0;
        siglongjmp(af_jmp, 1);
    }
}

static void mask_crashes(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = af_handler;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);

    /* Self-test fake crash: deliberately fault and recover, proving the
     * handler swallows crashes (AntiFuzz §4.2). */
    af_armed = 1;
    if (sigsetjmp(af_jmp, 1) == 0) {
        volatile int *p = NULL;
        *p = 1;  /* fake SIGSEGV, caught by af_handler */
    }
    af_armed = 0;
}

/* ── §4.3  delay on malformed input ────────────────────────────────────── */
static void delay_malformed(void) {
    struct timespec ts = {DELAY_MS / 1000, (DELAY_MS % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

/* ── the actual bug ──────────────────────────────────────────────────────
 * Heap overflow (not stack): ASAN reports it precisely, and in a non-ASAN
 * build it neither corrupts the return address nor loops, so a masked crash
 * returns cleanly instead of re-faulting forever.                           */
__attribute__((noinline)) static void crash(const unsigned char *buf, size_t len) {
    char *p = malloc(MAGIC_LEN);
    memset(p, 'A', len);  /* heap-buffer-overflow when len > MAGIC_LEN */
    fake_sink += p[0];
    free(p);
}

static int magic_ok(const unsigned char *buf) {
    if (!env_on("AF_HASHCMP")) return memcmp(buf, MAGIC, MAGIC_LEN) == 0;

    return byte_hash(buf, MAGIC_LEN) == byte_hash((const unsigned char *)MAGIC, MAGIC_LEN);
}

__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t len) {
    uint32_t h = byte_hash(buf, len);
    if (env_on("AF_COVERAGE")) fake_coverage(h);

    /* §4.4: magic compared by hash, not bytes. AF_HASHCMP=0 exposes it to
     * cmplog via memcmp, so the bug stays reachable for the other gates. */
    int valid = len >= MAGIC_LEN && magic_ok(buf);
    if (!valid) {
        if (env_on("AF_SPEED")) delay_malformed();
        return 0;
    }

    crash(buf, len);  /* magic matched: overflow */
    return 0;
}

/* Process-wide anti-debug and crash masking run once, at startup, as in
 * AntiFuzz. Kept out of fuzz_shm_run: in direct_lite that would trace and
 * re-handle the fuzzer's own process on every call.                       */
int main(void) {
    if (env_on("AF_PTRACE") && ptrace(PTRACE_TRACEME, 0, NULL, NULL) == -1) {
        _exit(0);  /* being traced -> refuse to run (anti-debug) */
    }

    if (env_on("AF_CRASH")) mask_crashes();

    unsigned char buf[256];
    ssize_t n = read(0, buf, sizeof(buf));
    if (n <= 0) return 0;
    return fuzz_shm_run(buf, (size_t)n);
}
