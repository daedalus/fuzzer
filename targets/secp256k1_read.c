/* Fuzz target for libsecp256k1 (vendored, v0.8.0) — exercises the input
 * parsing and verification surfaces of the library: DER and compact ECDSA
 * signatures, compressed/uncompressed public keys, x-only keys with BIP-340
 * Schnorr verification, recoverable signatures, and ECDH.
 *
 * Modeled on lz4_read.c: same __afl_map_edge landmark scheme, same
 * fuzz_shm_run entry point for direct_lite mode, same "bound the input"
 * discipline. secp256k1_context_static is used throughout — no allocations,
 * no context setup, nothing a malformed input can corrupt.
 *
 * Input layout:
 *   byte 0: mode selector (bit flags, each bit arms one parse surface)
 *   byte 1: param (recovery recid is taken from the low 2 bits)
 *   bytes 2..: payload fed to the armed surfaces
 *
 * All parse/verify entry points are bounds-safe by contract: they return 0
 * on malformed input and never read out of bounds. A SIGSEGV here is a real
 * bug in the library, not a malformed-input artifact. Returning 0 is the
 * expected behaviour and is NOT treated as a crash.
 *
 * Vendor the libsecp256k1 sources first (extracts to vendor/secp256k1/):
 *   tools/vendor_secp256k1.sh
 *
 * Then build via the normal path, which handles ASAN/cmplog variants:
 *   tools/build_targets.sh
 *
 * Manual build (what build_targets.sh does under the hood) — the library
 * objects are compiled SEPARATELY, without the shim:
 *   clang -O2 -g -fPIC -DENABLE_MODULE_ECDH -DENABLE_MODULE_RECOVERY \
 *       -DENABLE_MODULE_EXTRAKEYS -DENABLE_MODULE_SCHNORRSIG \
 *       -I vendor/secp256k1/src -I vendor/secp256k1/include \
 *       -c vendor/secp256k1/src/secp256k1.c -o /tmp/secp256k1.o
 *   (repeat for src/precomputed_ecmult.c and src/precomputed_ecmult_gen.c)
 *   clang -O2 -g -shared -fPIC \
 *       -I vendor/secp256k1/src -I vendor/secp256k1/include \
 *       -include src/fuzzer_tool/adapters/afl_shim.c \
 *       -o targets/secp256k1_read.so targets/secp256k1_read.c \
 *       /tmp/secp256k1.o /tmp/precomputed_ecmult.o \
 *       /tmp/precomputed_ecmult_gen.o -Wl,--export-dynamic
 *
 * NOTE: `-include afl_shim.c` applies to EVERY .c on the command line, so
 * the vendored sources must not be passed alongside the wrapper — doing so
 * emits __afl_map_shm / __afl_area / __afl_guarded_call into all four
 * objects and the link fails with multiple-definition errors (see the same
 * note in lz4_read.c).
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "secp256k1.h"
#include "secp256k1_ecdh.h"
#include "secp256k1_recovery.h"
#include "secp256k1_extrakeys.h"
#include "secp256k1_schnorrsig.h"

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* Cap the payload so no path does unbounded work on adversarial lengths. */
#define SECP256K1_FUZZ_MAX_IN (256u)

/* Fixed verification keys (well-known points; parsed per path with the
 * static context). FIXED_XONLY is the x coordinate of G. */
static const unsigned char FIXED_PUBKEY_COMPRESSED[33] = {
    0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62,
    0x95, 0xce, 0x87, 0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
    0xd9, 0xc5, 0x9f, 0x28, 0x15, 0xb1, 0x6f, 0x81, 0x79
};
static const unsigned char FIXED_XONLY[32] = {
    0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95,
    0xce, 0x87, 0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9,
    0xc5, 0x9f, 0x28, 0x15, 0xb1, 0x6f, 0x81, 0x79
};
static const unsigned char FIXED_SECKEY[32] = { 0x01 };
static const unsigned char FIXED_MSG[32] = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
    0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20
};

/* ── ECDSA DER signature parse ───────────────────────────────────── */
static int fuzz_ecdsa_der(const secp256k1_context *ctx,
                          const unsigned char *buf, size_t size) {
    __afl_map_edge(0x3100);

    /* DER parsing is length-sensitive (strict DER, trailing garbage fails),
     * so probe a few truncation points rather than just the full length.
     * Parsing into the same object repeatedly is invalid: libsecp256k1
     * zeroes the object on parse failure, so a failed iteration would
     * clobber a previously parsed value. Keep the last successful parse
     * in `kept` and never touch the loop-scratch object afterwards. */
    secp256k1_ecdsa_signature sig, kept;
    int parsed = 0;
    size_t l = size;
    while (1) {
        if (secp256k1_ecdsa_signature_parse_der(ctx, &sig, buf, l)) {
            parsed = 1;
            kept = sig;
            __afl_map_edge(0x3101);
        } else {
            __afl_map_edge(0x3102);
        }
        if (l == 0) break;
        l = (l > 8 ? l - 8 : 0);
    }
    if (!parsed) { __afl_map_edge(0x3103); return 0; }

    /* Round-trip: serialize back to DER, then compact form + normalize. */
    unsigned char out[72];
    size_t outlen = sizeof(out);
    if (secp256k1_ecdsa_signature_serialize_der(ctx, out, &outlen, &kept)) {
        __afl_map_edge(0x3104);
        secp256k1_ecdsa_signature sig2;
        if (secp256k1_ecdsa_signature_parse_der(ctx, &sig2, out, outlen)) {
            __afl_map_edge(0x3105);
            if (secp256k1_ecdsa_signature_normalize(ctx, &sig2, &kept)) {
                __afl_map_edge(0x3106);
            }
            /* Verify against the fixed pubkey: garbage sigs just return 0. */
            secp256k1_pubkey pk;
            if (secp256k1_ec_pubkey_parse(ctx, &pk, FIXED_PUBKEY_COMPRESSED, 33)) {
                int v = secp256k1_ecdsa_verify(ctx, &kept, FIXED_MSG, &pk);
                __afl_map_edge(0x3107 + (v ? 1u : 0u));
            }
        }
    }
    __afl_map_edge(0x3109);
    return 0;
}

/* ── Compact ECDSA signature parse (64 bytes) ────────────────────── */
static int fuzz_ecdsa_compact(const secp256k1_context *ctx,
                              const unsigned char *buf, size_t size) {
    __afl_map_edge(0x3200);
    if (size < 64) { __afl_map_edge(0x3201); return 0; }

    secp256k1_ecdsa_signature sig;
    if (!secp256k1_ecdsa_signature_parse_compact(ctx, &sig, buf)) {
        __afl_map_edge(0x3202);
        return 0;
    }
    __afl_map_edge(0x3203);

    unsigned char out[72];
    size_t outlen = sizeof(out);
    if (secp256k1_ecdsa_signature_serialize_der(ctx, out, &outlen, &sig)) {
        __afl_map_edge(0x3204);
    }
    secp256k1_ecdsa_signature sig2;
    if (secp256k1_ecdsa_signature_normalize(ctx, &sig2, &sig)) {
        __afl_map_edge(0x3205);
    }
    __afl_map_edge(0x3206);
    return 0;
}

/* ── Public key parse + serialize round-trip + ECDH ──────────────── */
static int fuzz_pubkey(const secp256k1_context *ctx,
                       const unsigned char *buf, size_t size) {
    __afl_map_edge(0x3300);

    /* Valid keys are 33 (compressed) or 65 (uncompressed) bytes; probe
     * truncations too so the length checks get real coverage. Same
     * keep-the-last-successful-parse discipline as fuzz_ecdsa_der: a
     * failed parse zeroes the object, so the loop must not clobber it. */
    secp256k1_pubkey pk, kept;
    int parsed = 0;
    size_t l = size;
    while (1) {
        if (secp256k1_ec_pubkey_parse(ctx, &pk, buf, l)) {
            parsed = 1;
            kept = pk;
            __afl_map_edge(0x3301);
        } else {
            __afl_map_edge(0x3302);
        }
        if (l == 0) break;
        l = (l > 8 ? l - 8 : 0);
    }
    if (!parsed) { __afl_map_edge(0x3303); return 0; }

    unsigned char out[65];
    size_t outlen = sizeof(out);
    if (secp256k1_ec_pubkey_serialize(ctx, out, &outlen, &kept, SECP256K1_EC_COMPRESSED)) {
        __afl_map_edge(0x3304);
    }
    outlen = sizeof(out);
    if (secp256k1_ec_pubkey_serialize(ctx, out, &outlen, &kept, SECP256K1_EC_UNCOMPRESSED)) {
        __afl_map_edge(0x3305);
    }
    /* ECDH against the fixed seckey (default SHA-256 hash function). */
    if (secp256k1_ecdh(ctx, out, &kept, FIXED_SECKEY, NULL, NULL)) {
        __afl_map_edge(0x3306);
    }
    __afl_map_edge(0x3307);
    return 0;
}

/* ── BIP-340 Schnorr verify (x-only key + 64-byte sig) ───────────── */
static int fuzz_schnorr(const secp256k1_context *ctx,
                        const unsigned char *buf, size_t size) {
    __afl_map_edge(0x3400);
    if (size < 64) { __afl_map_edge(0x3401); return 0; }

    secp256k1_xonly_pubkey pk;
    if (!secp256k1_xonly_pubkey_parse(ctx, &pk, FIXED_XONLY)) {
        __afl_map_edge(0x3402);
        return 0;
    }
    __afl_map_edge(0x3403);

    /* sig = first 64 bytes; msg = the remainder (arbitrary length; NULL
     * when empty, per the schnorrsig API contract). */
    const unsigned char *msg = (size > 64) ? buf + 64 : NULL;
    int v = secp256k1_schnorrsig_verify(ctx, buf, msg, size - 64, &pk);
    __afl_map_edge(0x3404 + (v ? 1u : 0u));
    return 0;
}

/* ── Recoverable signature: parse compact + pubkey recovery ──────── */
static int fuzz_recovery(const secp256k1_context *ctx,
                         const unsigned char *buf, size_t size, unsigned recid) {
    __afl_map_edge(0x3500);
    if (size < 64) { __afl_map_edge(0x3501); return 0; }

    secp256k1_ecdsa_recoverable_signature rsig;
    if (!secp256k1_ecdsa_recoverable_signature_parse_compact(ctx, &rsig, buf, (int)(recid & 3u))) {
        __afl_map_edge(0x3502);
        return 0;
    }
    __afl_map_edge(0x3503);

    secp256k1_pubkey pk;
    if (secp256k1_ecdsa_recover(ctx, &pk, &rsig, FIXED_MSG)) {
        __afl_map_edge(0x3504);
    }
    __afl_map_edge(0x3505);
    return 0;
}

__attribute__((visibility("default")))
int fuzz_secp256k1(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x3000);
    if (size < 2) { __afl_map_edge(0x3001); return 0; }

    unsigned char mode = buf[0];
    unsigned char recid = buf[1];
    const unsigned char *payload = buf + 2;
    size_t plen = size - 2;
    if (plen > SECP256K1_FUZZ_MAX_IN) plen = SECP256K1_FUZZ_MAX_IN;

    const secp256k1_context *ctx = secp256k1_context_static;

    /* Each bit arms one parse surface; multiple can run per input so the
     * fuzzer can learn cross-surface behaviour in a single execution. */
    if (mode & 0x01u) { __afl_map_edge(0x3002); fuzz_ecdsa_der(ctx, payload, plen); }
    if (mode & 0x02u) { __afl_map_edge(0x3003); fuzz_pubkey(ctx, payload, plen); }
    if (mode & 0x04u) { __afl_map_edge(0x3004); fuzz_schnorr(ctx, payload, plen); }
    if (mode & 0x08u) { __afl_map_edge(0x3005); fuzz_recovery(ctx, payload, plen, recid); }
    if (mode & 0x10u) { __afl_map_edge(0x3006); fuzz_ecdsa_compact(ctx, payload, plen); }
    __afl_map_edge(0x3007);
    return 0;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_secp256k1(buf, size);
}
