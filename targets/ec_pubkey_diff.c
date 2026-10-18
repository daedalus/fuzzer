/* Differential fuzz target: brainflayer's precomputed-table public-key
 * derivation vs. the reference libsecp256k1 ec_pubkey_create.
 *
 * Both sides answer the same question — given a 32-byte secret scalar k,
 * what is the uncompressed public key k*G? — by independent code:
 *
 *   reference : secp256k1_ec_pubkey_create() + _serialize() from a pinned
 *               libsecp256k1 (commit acb7f97, the one brainflayer's own
 *               submodule pins), i.e. the stock constant-window ecmult_gen.
 *   under test: brainflayer's secp256k1_ec_pubkey_create_precomp() from
 *               ec_pubkey_fast.c — a hand-rolled signed-window ladder over
 *               a NUMS-offset precomputed table (Courtois/Song/Castellucci).
 *
 * The two share nothing but the curve and the field/group arithmetic
 * headers, so for a valid key they MUST produce byte-identical 65-byte
 * output. A disagreement is a real bug in the fast path — exactly the
 * class a crash-only single-implementation target cannot see. The oracle
 * is a memcmp; a mismatch calls abort(), which the runner reports as a
 * crash.
 *
 * Oracle gate: the reference rejects k = 0 and k >= n (returns 0). The
 * fast path does not range-check, so for a rejected key the two legitimately
 * differ (the fast path reduces mod n implicitly via the ladder). We only
 * compare when the reference accepts the key; a rejected key is expected,
 * not a crash, and returns 0.
 *
 * Input layout:
 *   bytes 0..31 : the secret scalar k, big-endian (a short input is
 *                 right-aligned into a zero-padded 32-byte scalar, so the
 *                 empty/low-byte scalars the mutators love still parse)
 *   bytes 32..  : ignored
 *
 * The naf_scalar / montgomery / secp_fields mutators already craft scalars
 * at the n/p/(n+-1)/2 edges; they operate on the first 32 bytes here just
 * as they do on secp256k1_read's payload.
 *
 * Vendor the sources first (clones daedalus/brainflayer + the pinned
 * libsecp256k1 into vendor/brainflayer/, writes the build config):
 *   tools/vendor_brainflayer.sh
 *
 * Then build via the normal path, which handles ASAN/cmplog variants:
 *   tools/build_targets.sh
 *
 * NOTE: `-include afl_shim.c` applies to EVERY .c on the command line, so
 * the library sources (ec_pubkey_fast.c, mmapf.c, the reference
 * secp256k1.c) are compiled SEPARATELY, without the shim — same discipline
 * as lz4_read.c / secp256k1_read.c. See compile_brainflayer_ec_objects()
 * in build_targets.sh.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "ec_pubkey_fast.h"
#include "secp256k1/include/secp256k1.h"

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* Window for the precomputed table. 8 bits -> 32 windows x 256 entries,
 * built once at first call; small enough that init is instant and the
 * table is well under 1 MB, large enough to exercise a multi-window
 * ladder. The window does not change the result, only the schedule. */
#define EC_DIFF_WINDOW 8

/* Built once, reused across every in-process exec (direct_lite). */
static int ec_diff_ready = 0;
static secp256k1_context *ec_diff_ctx = NULL;

static int ec_diff_init(void) {
    if (ec_diff_ready) { return ec_diff_ctx != NULL; }
    ec_diff_ready = 1;
    /* NULL filename -> build the table in an anonymous mapping. */
    if (secp256k1_ec_pubkey_precomp_table(EC_DIFF_WINDOW, NULL) != 0) {
        return 0;
    }
    ec_diff_ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    return ec_diff_ctx != NULL;
}

__attribute__((visibility("default")))
int fuzz_ec_pubkey_diff(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x9000);
    if (size == 0) { __afl_map_edge(0x9001); return 0; }
    if (!ec_diff_init()) { __afl_map_edge(0x9002); return 0; }

    /* Right-align a short input into a zero-padded 32-byte scalar. */
    unsigned char seckey[32];
    memset(seckey, 0, sizeof(seckey));
    size_t n = size < 32u ? size : 32u;
    memcpy(seckey + (32u - n), buf, n);

    /* Reference side. A rejected key (0 or >= group order) is the gate:
     * nothing to compare, and not a crash. */
    secp256k1_pubkey pk;
    if (secp256k1_ec_pubkey_create(ec_diff_ctx, &pk, seckey) != 1) {
        __afl_map_edge(0x9003);
        return 0;
    }
    __afl_map_edge(0x9004);

    unsigned char ref[65];
    size_t ref_len = sizeof(ref);
    if (secp256k1_ec_pubkey_serialize(ec_diff_ctx, ref, &ref_len, &pk,
                                      SECP256K1_EC_UNCOMPRESSED) != 1
        || ref_len != 65) {
        /* Serialization of a just-created key cannot fail; if it ever
         * does, that is itself a reference-side bug worth surfacing. */
        __afl_map_edge(0x9005);
        abort();
    }

    /* Key accepted by the reference, so the fast path must agree. */
    unsigned char got[65];
    int got_len = 0;
    secp256k1_ec_pubkey_create_precomp(got, &got_len, seckey);

    /* Landmark a couple of structural facts so distinct valid keys get
     * distinct coverage even when they agree. */
    __afl_map_edge(0x9100 + (unsigned)(ref[64] & 0x1));   /* y parity */
    __afl_map_edge(0x9200 + (unsigned)(seckey[0]));       /* top scalar byte */

    if (got_len != 65 || got[0] != 0x04 ||
        memcmp(ref, got, 65) != 0) {
        __afl_map_edge(0x9006);
        abort();   /* fast path disagrees with the reference — a real bug */
    }
    __afl_map_edge(0x9007);
    return 0;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_ec_pubkey_diff(buf, size);
}
