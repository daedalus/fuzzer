/* Fuzz target for OpenSSL libcrypto (vendored, 3.5.x LTS) — DER decoders
 * for X.509 certificates, CRLs, CSRs, PKCS#7 and keys, plus the printers
 * and re-encoders reachable from a parsed object.
 *
 * Modeled on sqlite_read.c: same __afl_map_edge landmark scheme, same
 * fuzz_shm_run entry point for direct_lite mode, same "bound the work"
 * discipline.
 *
 * Input layout: NONE. The input IS the DER blob, fed to every decoder.
 *
 * No mode-selector byte (unlike lz4_read.c): the DER mutators are gated on
 *
 *   core/operator_registry.py: _sniff_der(d) — SEQUENCE/SET tag at d[0]
 *
 * A prefix byte would shift the tag to offset 1, the sniffer would never
 * fire, and certificates would be mutated as flat bytes. Each decoder
 * rejects foreign structures in a few bytes, so trying all of them costs
 * little and lets one corpus cover every surface.
 *
 * Contract under test: d2i_* return NULL on malformed input; printers and
 * i2d_* handle any object d2i_* accepted. A SIGSEGV or ASAN report here is
 * a library bug; a NULL return is a pass.
 *
 * Bounds (direct_lite runs in-process: a leak or hang kills the campaign):
 *   - input capped at OPENSSL_FUZZ_MAX_IN
 *   - every decoded object freed on every path
 *   - error queue cleared per exec (it grows per failed decode otherwise)
 *   - printers write to a null BIO: no I/O, no buffer growth
 *
 * Vendor the sources first (extracts to vendor/openssl/):
 *   tools/vendor_openssl.sh
 *
 * Then build via the normal path, which configures libcrypto.a out of tree
 * under $FUZZ_BUILD_ROOT/openssl/build<suffix>/ and handles ASAN/cmplog:
 *   tools/build_targets.sh
 *
 * NOTE: `-include afl_shim.c` applies to EVERY .c on the command line, so
 * libcrypto is built in its own pass, without the shim (see lz4_read.c).
 */
#include <stddef.h>
#include <stdint.h>

#include <openssl/bio.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pkcs7.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* Certificates and chains rarely exceed this; bounds bignum print cost. */
#define OPENSSL_FUZZ_MAX_IN (16u * 1024u)

/* IJON (--ijon): element counts are not coverage -- a certificate with 3
 * or 30 extensions walks the same decoder edges. One max slot per count,
 * direct (no hashing, no aliasing in 512 slots), value min(n, CAP) + 1:
 * bounded (no reward for bloat past CAP), and n = 0 still registers.
 * A negative n (absent stack: unsigned PKCS#7, CRL without revoked list)
 * is not a count and sets nothing. Inert without --ijon; compiled out
 * without the shim; the helper adds no guard sites. */
enum ossl_ijon_slot {
    OSSL_IJON_CERT_EXT = 0,
    OSSL_IJON_CERT_SUBJ,
    OSSL_IJON_CERT_ISS,
    OSSL_IJON_CRL_REV,
    OSSL_IJON_CRL_EXT,
    OSSL_IJON_REQ_ATTR,
    OSSL_IJON_REQ_SUBJ,
    OSSL_IJON_P7_SIGNERS,
};
#define OSSL_IJON_CAP 256
#if defined(__AFL_EDGE) && __AFL_EDGE
__AFL_NO_COV static void ossl_ijon(enum ossl_ijon_slot slot, int n) {
    if (n < 0) return;
    ijon_max((uint32_t)slot, (uint64_t)(n < OSSL_IJON_CAP ? n : OSSL_IJON_CAP) + 1u);
}
#else
#define ossl_ijon(slot, n) ((void)0)
#endif

/* Shared sink for all printers; created once in openssl_init(). */
static BIO *g_null_bio;

static int openssl_init(void) {
    if (g_null_bio) return 1;

    /* No config file: the fuzzer's environment must not steer the target. */
    OPENSSL_init_crypto(OPENSSL_INIT_NO_LOAD_CONFIG, NULL);
    g_null_bio = BIO_new(BIO_s_null());
    return g_null_bio != NULL;
}

/* Re-encode a decoded object; i2d must accept anything d2i accepted. */
#define ROUND_TRIP(i2d_fn, obj, edge)                 \
    do {                                              \
        unsigned char *der_out = NULL;                \
        int n = i2d_fn((obj), &der_out);              \
        __afl_map_edge((edge) + (n > 0 ? 1u : 0u));   \
        OPENSSL_free(der_out);                        \
    } while (0)

/* ── X.509 certificate ───────────────────────────────────────────── */
static void fuzz_x509(const unsigned char *buf, size_t size) {
    const unsigned char *p = buf;
    X509 *x = d2i_X509(NULL, &p, (long)size);
    if (!x) { __afl_map_edge(0x6100); return; }
    __afl_map_edge(0x6101);
    ossl_ijon(OSSL_IJON_CERT_EXT, X509_get_ext_count(x));
    ossl_ijon(OSSL_IJON_CERT_SUBJ, X509_NAME_entry_count(X509_get_subject_name(x)));
    ossl_ijon(OSSL_IJON_CERT_ISS, X509_NAME_entry_count(X509_get_issuer_name(x)));

    X509_print(g_null_bio, x);
    ROUND_TRIP(i2d_X509, x, 0x6102);

    /* Public key decode: SPKI algorithm dispatch (RSA/EC/DSA/ED*...). */
    EVP_PKEY *pk = X509_get_pubkey(x);
    __afl_map_edge(0x6104 + (pk ? 1u : 0u));
    EVP_PKEY_free(pk);

    /* Extension cache: basicConstraints, keyUsage, SKID/AKID, policies. */
    __afl_map_edge(0x6106 + (X509_check_ca(x) > 0 ? 1u : 0u));
    X509_free(x);
}

/* ── X.509 CRL ───────────────────────────────────────────────────── */
static void fuzz_crl(const unsigned char *buf, size_t size) {
    const unsigned char *p = buf;
    X509_CRL *crl = d2i_X509_CRL(NULL, &p, (long)size);
    if (!crl) { __afl_map_edge(0x6200); return; }
    __afl_map_edge(0x6201);
    ossl_ijon(OSSL_IJON_CRL_REV, sk_X509_REVOKED_num(X509_CRL_get_REVOKED(crl)));
    ossl_ijon(OSSL_IJON_CRL_EXT, X509_CRL_get_ext_count(crl));

    X509_CRL_print(g_null_bio, crl);
    ROUND_TRIP(i2d_X509_CRL, crl, 0x6202);
    X509_CRL_free(crl);
}

/* ── PKCS#10 certificate request ─────────────────────────────────── */
static void fuzz_req(const unsigned char *buf, size_t size) {
    const unsigned char *p = buf;
    X509_REQ *req = d2i_X509_REQ(NULL, &p, (long)size);
    if (!req) { __afl_map_edge(0x6300); return; }
    __afl_map_edge(0x6301);
    ossl_ijon(OSSL_IJON_REQ_ATTR, X509_REQ_get_attr_count(req));
    ossl_ijon(OSSL_IJON_REQ_SUBJ, X509_NAME_entry_count(X509_REQ_get_subject_name(req)));

    X509_REQ_print(g_null_bio, req);
    ROUND_TRIP(i2d_X509_REQ, req, 0x6302);
    X509_REQ_free(req);
}

/* ── PKCS#7 / CMS-like container ─────────────────────────────────── */
static void fuzz_pkcs7(const unsigned char *buf, size_t size) {
    const unsigned char *p = buf;
    PKCS7 *p7 = d2i_PKCS7(NULL, &p, (long)size);
    if (!p7) { __afl_map_edge(0x6400); return; }
    __afl_map_edge(0x6401);
    ossl_ijon(OSSL_IJON_P7_SIGNERS, sk_PKCS7_SIGNER_INFO_num(PKCS7_get_signer_info(p7)));

    PKCS7_print_ctx(g_null_bio, p7, 0, NULL);
    ROUND_TRIP(i2d_PKCS7, p7, 0x6402);
    PKCS7_free(p7);
}

/* ── Keys: PKCS#8 / traditional private, SPKI public ─────────────── */
static void fuzz_keys(const unsigned char *buf, size_t size) {
    const unsigned char *p = buf;
    EVP_PKEY *priv = d2i_AutoPrivateKey(NULL, &p, (long)size);
    __afl_map_edge(0x6500 + (priv ? 1u : 0u));
    EVP_PKEY_free(priv);

    p = buf;
    EVP_PKEY *pub = d2i_PUBKEY(NULL, &p, (long)size);
    __afl_map_edge(0x6502 + (pub ? 1u : 0u));
    EVP_PKEY_free(pub);
}

static int fuzz_openssl(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x6000);
    if (size == 0 || size > OPENSSL_FUZZ_MAX_IN) { __afl_map_edge(0x6001); return 0; }
    if (!openssl_init()) { __afl_map_edge(0x6002); return 0; }

    fuzz_x509(buf, size);
    fuzz_crl(buf, size);
    fuzz_req(buf, size);
    fuzz_pkcs7(buf, size);
    fuzz_keys(buf, size);

    ERR_clear_error();
    return 0;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_openssl(buf, size);
}
