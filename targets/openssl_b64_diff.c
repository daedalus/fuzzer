/* Differential fuzz target for OpenSSL's base64 encoder (vendored 4.1.x):
 * AVX2 path vs scalar path, through the public EVP API.
 *
 * Why: OpenSSL 4.0 added an AVX2 encoder (crypto/evp/enc_b64_avx2.c),
 * dispatched from EVP_EncodeBlock / EVP_EncodeUpdate on
 * OPENSSL_ia32cap_P[2] bit 5. openssl/openssl#33652 (CVE-2026-121481)
 * fixes length bugs around it. Both paths must emit identical bytes.
 *
 *   input ──┬─ AVX2 bit cleared ─► scalar encoder ─┐
 *           │                                      ├─ memcmp ─► SIGABRT
 *           └─ AVX2 bit set ─────► AVX2 encoder ───┘
 *   scalar output ─► EVP_Decode* ─► must equal input ─► else SIGABRT
 *
 * Modeled on libbase64_read.c (differential, crash on mismatch). The
 * dispatch exists only when libcrypto is built with asm (encode.c gates it
 * on OPENSSL_CPUID_OBJ), so build_targets.sh links the build_asm tree.
 *
 * Input layout:
 *   byte 0: stream ctx flags (bit 0 no newlines, bit 1 SRP alphabet)
 *   byte 1: stream chunk size for EVP_EncodeUpdate (0 = one call)
 *   bytes 2..: payload to encode
 *
 * Buffers are heap-allocated at the exact contract size per exec, so an
 * AVX2 over-read of the input or over-write of the output is an ASAN
 * report, not silent slack in a static array.
 *
 * Build: tools/vendor_openssl.sh, then tools/build_targets.sh.
 */
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <openssl/crypto.h>
#include <openssl/evp.h>

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* libcrypto internals, resolved from the static archive:
 * the CPU capability vector (crypto/cpuid.c) and the ctx flag setter
 * (include/crypto/evp.h). */
extern unsigned int OPENSSL_ia32cap_P[];
void evp_encode_ctx_set_flags(EVP_ENCODE_CTX *ctx, unsigned int flags);

/* Values from include/crypto/evp.h (internal header). */
#define B64_CTX_NO_NEWLINES 1u
#define B64_CTX_SRP_ALPHABET 2u
#define B64_CTX_FLAG_MASK (B64_CTX_NO_NEWLINES | B64_CTX_SRP_ALPHABET)

/* encode.c: OPENSSL_ia32cap_P[2] & (1u << 5) selects encode_base64_avx2. */
#define B64_CAP_WORD 2
#define B64_CAP_AVX2 (1u << 5)

#define B64_DIFF_HDR 2u
#define B64_DIFF_MAX_IN (4096u)

/* Decided once: setting the bit on a CPU without AVX2 would SIGILL. */
static int g_ready;
static int g_has_avx2;

static void b64_init(void) {
    if (g_ready) return;

    /* Runs OPENSSL_cpuid_setup, which fills OPENSSL_ia32cap_P. */
    OPENSSL_init_crypto(OPENSSL_INIT_NO_LOAD_CONFIG, NULL);
    g_has_avx2 = __builtin_cpu_supports("avx2");
    g_ready = 1;
}

/* raise(SIGABRT), not abort(): afl_shim.c rewrites abort() in this TU
 * into a logged return (unless --abort-is-crash), which would turn every
 * mismatch into a warning. The raised signal is a crash in all modes. */
static void b64_mismatch(unsigned edge) {
    __afl_map_edge(edge);
    raise(SIGABRT);
}

static void b64_set_avx2(int on) {
    if (on) OPENSSL_ia32cap_P[B64_CAP_WORD] |= B64_CAP_AVX2;
    else OPENSSL_ia32cap_P[B64_CAP_WORD] &= ~B64_CAP_AVX2;
}

/* Fault injection for the oracle's falsification test only: change one
 * output byte to another base64 digit so both oracles must fire. */
static void b64_corrupt(unsigned char *out, size_t len) {
#ifdef B64_DIFF_CORRUPT_SCALAR
    if (len) out[0] = (out[0] == 'A') ? 'B' : 'A';
#else
    (void)out;
    (void)len;
#endif
}

/* ── EVP_EncodeBlock: one shot, no newlines, standard alphabet ──────── */
static size_t enc_block(const unsigned char *in, size_t n, unsigned char *out) {
    /* n == 0 passes NULL: the zero-length case #33652 makes defined. */
    return (size_t)EVP_EncodeBlock(out, n ? in : NULL, (int)n);
}

/* ── EVP_EncodeUpdate/Final in chunks, with ctx flags ───────────────── */
static size_t enc_stream(const unsigned char *in, size_t n, unsigned char *out,
                         unsigned flags, size_t chunk) {
    EVP_ENCODE_CTX *ctx = EVP_ENCODE_CTX_new();
    if (!ctx) return 0;

    EVP_EncodeInit(ctx);
    evp_encode_ctx_set_flags(ctx, flags);

    size_t total = 0, pos = 0;
    if (chunk == 0) chunk = n ? n : 1;
    while (pos < n) {
        size_t take = (n - pos < chunk) ? n - pos : chunk;
        int outl = 0;
        if (!EVP_EncodeUpdate(ctx, out + total, &outl, in + pos, (int)take)) {
            __afl_map_edge(0xB201);
            break;
        }
        total += (size_t)outl;
        pos += take;
    }

    int outl = 0;
    EVP_EncodeFinal(ctx, out + total, &outl);
    total += (size_t)outl;
    EVP_ENCODE_CTX_free(ctx);
    return total;
}

/* Exact EVP_EncodeBlock output size: 4 * ceil(n / 3) + NUL. */
static size_t block_cap(size_t n) { return (n + 2) / 3 * 4 + 1; }

static void diff_block(const unsigned char *in, size_t n) {
    __afl_map_edge(0xB100);
    size_t cap = block_cap(n);
    unsigned char *ref = malloc(cap), *got = malloc(cap), *dec = malloc(cap);
    if (!ref || !got || !dec) goto out;

    b64_set_avx2(0);
    size_t ref_len = enc_block(in, n, ref);
    b64_corrupt(ref, ref_len);

    /* Independent arithmetic oracle on the length. */
    if (ref_len != cap - 1) b64_mismatch(0xB101);

    if (g_has_avx2) {
        b64_set_avx2(1);
        size_t got_len = enc_block(in, n, got);
        if (got_len != ref_len) b64_mismatch(0xB102);
        if (memcmp(ref, got, ref_len + 1) != 0) b64_mismatch(0xB103);
        __afl_map_edge(0xB104);
    }

    /* Round trip: DecodeBlock keeps pad bytes, so 3 * len / 4 out. */
    int dlen = EVP_DecodeBlock(dec, ref, (int)ref_len);
    if (dlen != (int)(ref_len / 4 * 3)) b64_mismatch(0xB105);
    if (n && memcmp(dec, in, n) != 0) b64_mismatch(0xB106);

out:
    free(ref);
    free(got);
    free(dec);
}

/* Stream decode of the scalar output must give the input back. */
static void stream_roundtrip(const unsigned char *enc, size_t len,
                             const unsigned char *in, size_t n) {
    EVP_ENCODE_CTX *ctx = EVP_ENCODE_CTX_new();
    unsigned char *dec = malloc(len / 4 * 3 + 3);
    if (!ctx || !dec) goto out;

    EVP_DecodeInit(ctx);
    int outl = 0, fin = 0;
    if (EVP_DecodeUpdate(ctx, dec, &outl, enc, (int)len) < 0) b64_mismatch(0xB301);
    if (EVP_DecodeFinal(ctx, dec + outl, &fin) != 1) b64_mismatch(0xB302);
    if ((size_t)(outl + fin) != n) b64_mismatch(0xB303);
    if (n && memcmp(dec, in, n) != 0) b64_mismatch(0xB304);

out:
    EVP_ENCODE_CTX_free(ctx);
    free(dec);
}

static void diff_stream(const unsigned char *in, size_t n, unsigned flags, size_t chunk) {
    __afl_map_edge(0xB200 + flags);
    size_t cap = EVP_ENCODE_LENGTH(n);
    unsigned char *ref = malloc(cap), *got = malloc(cap);
    if (!ref || !got) goto out;

    b64_set_avx2(0);
    size_t ref_len = enc_stream(in, n, ref, flags, chunk);
    b64_corrupt(ref, ref_len);

    if (g_has_avx2) {
        b64_set_avx2(1);
        size_t got_len = enc_stream(in, n, got, flags, chunk);
        if (got_len != ref_len) b64_mismatch(0xB210);
        if (memcmp(ref, got, ref_len) != 0) b64_mismatch(0xB211);
        __afl_map_edge(0xB212);
    }

    /* The decoder speaks only the standard alphabet. */
    if (!(flags & B64_CTX_SRP_ALPHABET)) stream_roundtrip(ref, ref_len, in, n);

out:
    free(ref);
    free(got);
}

static int fuzz_b64(const unsigned char *buf, size_t size) {
    __afl_map_edge(0xB000);
    if (size < B64_DIFF_HDR) { __afl_map_edge(0xB001); return 0; }
    size_t n = size - B64_DIFF_HDR;
    if (n > B64_DIFF_MAX_IN) { __afl_map_edge(0xB002); return 0; }
    b64_init();

    unsigned flags = buf[0] & B64_CTX_FLAG_MASK;
    size_t chunk = buf[1];

    /* Exact-size copy: any read past n is an ASAN report. */
    unsigned char *in = malloc(n ? n : 1);
    if (!in) return 0;
    memcpy(in, buf + B64_DIFF_HDR, n);

    unsigned saved = OPENSSL_ia32cap_P[B64_CAP_WORD];
    diff_block(in, n);
    diff_stream(in, n, flags, chunk);
    OPENSSL_ia32cap_P[B64_CAP_WORD] = saved;

    free(in);
    return 0;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_b64(buf, size);
}
