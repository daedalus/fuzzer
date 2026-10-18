/* Differential fuzz target: brainflayer's wide-SIMD hashers vs. their
 * scalar references.
 *
 * brainflayer hashes many keys at once, one per lane of a GCC vector
 * (AVX-512F = 16, AVX2 = 8, SSE2 = 4 lanes), falling back to a scalar
 * tail for the remainder. That transposed, hand-vectorised code is a
 * classic home for lane-crossing, byte-swap and message-padding bugs that
 * only bite at particular lane counts or particular input bytes. This
 * target pits it against an independent scalar reference per input and
 * aborts on the first disagreement:
 *
 *   SHA-256   reference: OpenSSL (sha256_upub / sha256_cpub)
 *             under test: sha256_upub_n / sha256_cpub_n   (brainflayer SIMD)
 *   RIPEMD-160 reference: brainflayer scalar ripemd160_256()
 *             under test: ripemd160_256_n                 (brainflayer SIMD)
 *
 * The SIMD width is CPU-detected once (and capped by the
 * BRAINFLAYER_{SHA256,RIPEMD}_LANES env vars); feeding a record count that
 * is not a multiple of the width also exercises the x16->x8->x4->scalar
 * cascade in one exec. None of the hashers validate their input — the
 * SHA-256 paths hash a fixed 65 bytes (and a 33-byte compressed form
 * derived from bytes 0,1..32,64), RIPEMD hashes a fixed 32 — so every byte
 * string is a legal corpus of records and no input is out of bounds.
 *
 * Input layout (no mode byte; every exec runs all three families):
 *   the buffer is read as a sequence of fixed-size records, independently
 *   for each family: floor(size/65) uncompressed-pubkey records for the two
 *   SHA-256 surfaces, floor(size/32) records for RIPEMD. Record counts are
 *   capped at HASH_DIFF_MAX_RECS so an adversarial length cannot allocate
 *   or spin without bound.
 *
 * A mismatch calls abort(), reported by the runner as a crash. Agreement
 * is the expected result and returns 0.
 *
 * Vendor the sources first (clones daedalus/brainflayer into
 * vendor/brainflayer/):
 *   tools/vendor_brainflayer.sh
 * Then build via the normal path:
 *   tools/build_targets.sh
 *
 * NOTE: `-include afl_shim.c` applies to EVERY .c on the command line, so
 * sha256_pub.c and ripemd160_256.c are compiled SEPARATELY, without the
 * shim — same discipline as lz4_read.c. See compile_brainflayer_hash_objects()
 * in build_targets.sh.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "sha256_pub.h"
#include "ripemd160_256.h"

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* Bound the per-exec work: at most this many records per family. 65 B each
 * for SHA, 32 B each for RIPEMD — a few hundred KB of input, a few hundred
 * KB of digests. direct_lite runs in-process, so an unbounded count would
 * OOM or stall the whole campaign, not one test case. */
#define HASH_DIFF_MAX_RECS 2048u

/* Landmark the lane cascade so inputs that reach the wider vectors get
 * distinct coverage from ones that only hit the scalar tail. */
static void hash_diff_landmark_width(unsigned base, size_t n) {
    if (n >= 16) { __afl_map_edge(base + 4); }
    else if (n >= 8) { __afl_map_edge(base + 3); }
    else if (n >= 4) { __afl_map_edge(base + 2); }
    else if (n >= 1) { __afl_map_edge(base + 1); }
}

/* SHA-256 of 65-byte records: OpenSSL (per record) vs brainflayer SIMD. */
static void fuzz_sha256(const unsigned char *buf, size_t size) {
    size_t n = size / 65u;
    if (n == 0) { return; }
    if (n > HASH_DIFF_MAX_RECS) { n = HASH_DIFF_MAX_RECS; }
    __afl_map_edge(0x9300);
    hash_diff_landmark_width(0x9310, n);

    const unsigned char (*recs)[65] = (const unsigned char (*)[65])buf;
    unsigned char (*simd)[32] = malloc(n * 32u);
    if (!simd) { return; }

    /* Uncompressed-key SHA-256. */
    sha256_upub_n(simd, recs, n);
    for (size_t i = 0; i < n; i++) {
        unsigned char ref[32];
        sha256_upub(ref, recs[i]);
        if (memcmp(ref, simd[i], 32) != 0) { __afl_map_edge(0x9301); abort(); }
    }

    /* Compressed-form SHA-256 (parity byte + x, derived from the same 65). */
    sha256_cpub_n(simd, recs, n);
    for (size_t i = 0; i < n; i++) {
        unsigned char ref[32];
        sha256_cpub(ref, recs[i]);
        if (memcmp(ref, simd[i], 32) != 0) { __afl_map_edge(0x9302); abort(); }
    }

    free(simd);
    __afl_map_edge(0x9303);
}

/* RIPEMD-160 of 32-byte records: brainflayer scalar vs brainflayer SIMD. */
static void fuzz_ripemd160(const unsigned char *buf, size_t size) {
    size_t n = size / 32u;
    if (n == 0) { return; }
    if (n > HASH_DIFF_MAX_RECS) { n = HASH_DIFF_MAX_RECS; }
    __afl_map_edge(0x9400);
    hash_diff_landmark_width(0x9410, n);

    unsigned char (*simd)[20] = malloc(n * 20u);
    if (!simd) { return; }

    ripemd160_256_n(buf, simd, n);
    for (size_t i = 0; i < n; i++) {
        unsigned char ref[20];
        ripemd160_256(buf + i * 32u, ref);
        if (memcmp(ref, simd[i], 20) != 0) { __afl_map_edge(0x9401); abort(); }
    }

    free(simd);
    __afl_map_edge(0x9402);
}

__attribute__((visibility("default")))
int fuzz_hash_simd_diff(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x9000);
    if (size == 0) { return 0; }
    fuzz_sha256(buf, size);
    fuzz_ripemd160(buf, size);
    return 0;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_hash_simd_diff(buf, size);
}
