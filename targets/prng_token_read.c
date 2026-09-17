/* prng_token_read.c — a CWE-338 (predictable PRNG) demonstration target.
 *
 * Models the common vulnerable pattern where a service authenticates a
 * request by comparing an attacker-supplied field against a freshly drawn
 * "random" token from a weak, fully linear generator (Boost's taus88 here,
 * three xor-combined LFSRs). Because the generator's state evolves linearly
 * over GF(2), three-to-four observed outputs pin its entire 96-bit state,
 * after which every future token is known in advance.
 *
 * How the fuzzer breaks it (see core/prng_state_recovery.py +
 * core/prng_state_learner.py, methodology ported from Gegell's Factorio RNG
 * writeup):
 *   1. Each call draws one token and compares it to a 4-byte input field.
 *      The comparison is a plain `uint32_t == uint32_t`, so trace-cmp logs
 *      the token as a comparison operand at a single, stable PC.
 *   2. The generator is a SINGLE static instance that persists across
 *      in-process (direct_lite) iterations, so the tokens logged across one
 *      cmplog drain are genuinely consecutive outputs of one stream.
 *   3. PRNGStateLearner groups operands by PC, recovers the 96-bit state
 *      from >= 4 consecutive tokens (a fourth drives the coincidental-fit
 *      rate to zero), and predicts the next draw. The `prng_predict`
 *      operator writes that prediction into the input field.
 *   4. The next execution's token then equals the field and the
 *      otherwise-unreachable authenticated branch (abort) is taken.
 *
 * The generator is seeded once, from a source the fuzzer cannot read
 * (time ^ pid), precisely so that hitting the branch requires *recovering*
 * the state rather than reading a compile-time constant. Consecutive draws
 * are still consecutive regardless of the seed, which is all recovery needs.
 *
 * Depends on nothing outside libc; builds as a .so alongside nop_target via
 * build_simple_so_targets in tools/build_targets.sh.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/* ── Boost taus88: three linear_feedback_shift_engine<uint32_t,...>
 *    components XOR-combined. Parameters (w,k,q,s) per component match
 *    core/prng_state_recovery.py::TAUS88_PARAMS exactly:
 *      (32,31,13,12), (32,29,2,4), (32,28,3,17)
 *    so a state recovered on the Python side reproduces this stream. ── */

static uint32_t s_a, s_b, s_c;
static int seeded = 0;

static uint32_t lfsr_step(uint32_t v, int k, int q, int s) {
    /* uint32_t arithmetic wraps mod 2**32, which is the word mask for w=32. */
    uint32_t b = (((v << q) ^ v)) >> (k - s);
    uint32_t mask = 0xFFFFFFFFu << (32 - k);
    return ((v & mask) << s) ^ b;
}

static void taus88_step(void) {
    s_a = lfsr_step(s_a, 31, 13, 12);
    s_b = lfsr_step(s_b, 29, 2, 4);
    s_c = lfsr_step(s_c, 28, 3, 17);
}

static uint32_t taus88_output(void) {
    return s_a ^ s_b ^ s_c;
}

static void seed_once(void) {
    if (seeded) return;
    /* Unknown to the fuzzer, but constant within this process so the stream
     * is a single continuous sequence across in-process iterations. LFSR
     * components must be non-zero (an all-zero LFSR is a fixed point). */
    uint32_t s = (uint32_t)time(NULL) ^ (uint32_t)getpid();
    s_a = s | 0x1u;
    s_b = (s * 2654435761u) | 0x2u;   /* Knuth multiplicative spread */
    s_c = (s * 40503u) | 0x4u;
    seeded = 1;
}

/* ── Fuzz entry point. In-process signature:
 *      int LLVMFuzzerTestOneInput(const uint8_t *buf, size_t len) ── */
__attribute__((visibility("default")))
int LLVMFuzzerTestOneInput(const uint8_t *buf, size_t len) {
    seed_once();

    /* Advance and draw one token per execution. */
    taus88_step();
    uint32_t token = taus88_output();

    if (len < 4) {
        return 0;
    }

    /* Read the 4-byte field the client presents as its "authentication"
     * token, little-endian to match the operand encoding the shim logs and
     * the value PRNGStateLearner predicts. */
    uint32_t presented = (uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
                         ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24);

    /* The vulnerable check. `token` is the leaked operand; because it is a
     * plain integer comparison the compiler emits a trace-cmp callback that
     * records `token` at this PC on every call. */
    if (presented == token) {
        /* Authenticated with a value the client could not have known without
         * predicting the generator: the bug we want the fuzzer to reach. */
        abort();
    }

    /* A little extra structure so coverage is not degenerate. */
    if (len >= 8 && buf[4] == 0x55 && buf[5] == 0xAA) {
        return 2;
    }
    return 1;
}
