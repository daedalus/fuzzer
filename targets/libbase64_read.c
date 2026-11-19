/* Differential fuzz target for libbase64 (aklomp/base64, v0.5.3).
 *
 * libbase64 ships one portable "plain" codec and up to six x86 SIMD codecs
 * (SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX-512 VBMI) that process 12-48 bytes
 * per iteration and fall back to a scalar tail. That lane-wise shuffle code
 * is where off-by-one tails, bad-character masks and out-of-bounds stores
 * live. The `flags` argument of base64_encode/base64_decode forces a given
 * codec, so one exec can run the same input through every codec this CPU
 * supports and pit each against the plain codec. The first disagreement
 * calls abort(), reported by the runner as a crash.
 *
 * Modeled on hash_simd_diff.c (differential, abort on mismatch) and
 * lz4_read.c (same __afl_map_edge landmark scheme, same fuzz_shm_run entry
 * point for direct_lite mode).
 *
 * Input layout:
 *   byte 0: mode selector (low 2 bits)
 *     0 -> decode:         every codec vs plain, one shot
 *     1 -> encode:         every codec vs plain, then decode round trip
 *     2 -> stream decode:  chunked base64_stream_decode vs one shot
 *     3 -> stream encode:  chunked base64_stream_encode vs one shot
 *   byte 1 (modes 2 and 3 only): chunk-size seed
 *   rest: payload
 *
 * base64 has no magic number, so the mode byte cannot hide a structure-aware
 * mutator's sniffer (see AGENTS.md "Add a new fuzz target" step 6). The
 * decode payload is ASCII text; the encode payload is arbitrary bytes.
 *
 * What counts as a disagreement:
 *   - decode return code differs from plain (1 ok / 0 bad character)
 *   - decode returned 1 and the output length or bytes differ from plain.
 *     After a decode error the partial output is unspecified, so it is not
 *     compared.
 *   - encode output length or bytes differ from plain
 *   - decode(encode(x)) != x
 *   - chunked streaming differs from the one-shot result
 * Input is copied, and output buffers are heap-allocated, at the exact
 * size (outputs: documented maximum + B64_FUZZ_SLACK, default 0), so ASAN
 * reports reads past the input and stores past the bound in C code. The
 * x86-64 default SSSE3/AVX/AVX2 encoders are inline asm, which ASAN does not
 * instrument: there the differential is the only oracle.
 *
 * Vendor the sources first (extracts to vendor/libbase64/):
 *   tools/vendor_libbase64.sh
 *
 * Then build via the normal path, which handles ASAN/cmplog variants:
 *   tools/build_targets.sh
 *
 * NOTE: `-include afl_shim.c` applies to EVERY .c on the command line, so
 * the vendored sources are compiled SEPARATELY, without the shim, one object
 * per codec (each needs its own -m flag) — same discipline as lz4_read.c.
 * See compile_libbase64_objects() in build_targets.sh.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "libbase64.h"

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

/* Bound the per-exec work: direct_lite runs in-process, so an adversarial
 * length must not stall the campaign. */
#define B64_FUZZ_MAX_IN   (1u * 1024u * 1024u)
/* Stream chunk cap. Small enough that most splits land inside a 4-char
 * quantum (3 of 4 offsets), large enough to enter every SIMD main loop
 * (largest entry threshold: AVX-512 encode, 64 bytes). 4096 left a typical
 * payload unsplit ~95% of the time. */
#define B64_FUZZ_MAX_CHUNK 128u

/* Splits only within the first SPAN bytes; the rest goes in one call. A
 * deeper split runs the same resume path, so this bounds the per-exec cost
 * (~9 ns per extra call per codec) without losing split shapes. */
#define B64_FUZZ_SPLIT_SPAN 512u

#ifndef B64_FUZZ_SLACK
#define B64_FUZZ_SLACK 0u
#endif

/* IJON (--ijon): SIMD codecs validate a block branch-free, so edges cannot
 * see where the first bad char sits in it, nor whether an input of a given
 * tail residue was long enough to enter every main loop. Max slots, one
 * band per signal and mode (no hashing, no aliasing in 512 slots):
 *
 *   [0, 128)    first bad char, decode modes:  (mode >> 1) * 64 + pos % 64
 *   [128, 384)  payload length, every mode:    128 + mode * 64 + len % 64
 *
 * Value min(x, CAP) + 1: bounded (no reward for growth past 4 AVX-512
 * blocks), and pos 0 still registers. Inert without --ijon; compiled out
 * without the shim. */
#define B64_IJON_LANES   64u
#define B64_IJON_LEN_BASE (2u * B64_IJON_LANES)
#define B64_IJON_CAP     (4u * B64_IJON_LANES)
#if defined(__AFL_EDGE) && __AFL_EDGE
#define FUZZ_IJON_LANE(base, x) \
    ijon_max((base) + (uint32_t)((x) % B64_IJON_LANES), \
             (uint64_t)((x) < B64_IJON_CAP ? (x) : B64_IJON_CAP) + 1u)
#else
#define FUZZ_IJON_LANE(base, x) ((void)0)
#endif

enum b64_mode {
    B64_MODE_DECODE = 0,
    B64_MODE_ENCODE = 1,
    B64_MODE_STREAM_DECODE = 2,
    B64_MODE_STREAM_ENCODE = 3,
    B64_MODE_MASK = 3,
};

/* Codec table. plain is entry 0 and the reference for every comparison. */
struct b64_codec {
    int flag;
    const char *name;
    int usable; /* filled by b64_probe(): built in AND supported by this CPU */
};

static struct b64_codec codecs[] = {
    { BASE64_FORCE_PLAIN, "plain", 0 },
    { BASE64_FORCE_SSSE3, "ssse3", 0 },
    { BASE64_FORCE_SSE41, "sse41", 0 },
    { BASE64_FORCE_SSE42, "sse42", 0 },
    { BASE64_FORCE_AVX, "avx", 0 },
    { BASE64_FORCE_AVX2, "avx2", 0 },
    { BASE64_FORCE_AVX512, "avx512", 0 },
};
#define B64_NCODECS (sizeof(codecs) / sizeof(codecs[0]))

static int b64_probed;
static unsigned b64_nusable;

/* Forcing a codec the CPU lacks raises SIGILL, so gate on cpuid. Forcing a
 * codec that was not compiled in is a documented no-op: decode returns -1,
 * which is how the build-time half of "usable" is detected. */
static int b64_cpu_has(int flag) {
#if defined(__x86_64__) || defined(__i386__)
    switch (flag) {
    case BASE64_FORCE_PLAIN:  return 1;
    case BASE64_FORCE_SSSE3:  return __builtin_cpu_supports("ssse3");
    case BASE64_FORCE_SSE41:  return __builtin_cpu_supports("sse4.1");
    case BASE64_FORCE_SSE42:  return __builtin_cpu_supports("sse4.2");
    case BASE64_FORCE_AVX:    return __builtin_cpu_supports("avx");
    case BASE64_FORCE_AVX2:   return __builtin_cpu_supports("avx2");
    case BASE64_FORCE_AVX512:
        return __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vbmi");
    default: return 0;
    }
#else
    return flag == BASE64_FORCE_PLAIN;
#endif
}

static void b64_probe(void) {
    if (b64_probed) { return; }
    b64_probed = 1;

    static const char probe_in[] = "QUJDRA==";
    char out[16];

    for (size_t i = 0; i < B64_NCODECS; i++) {
        if (!b64_cpu_has(codecs[i].flag)) { continue; }

        struct base64_state st;
        size_t outlen = 0;
        base64_stream_decode_init(&st, codecs[i].flag);
        int rc = base64_stream_decode(&st, probe_in, sizeof(probe_in) - 1, out, &outlen);
        if (rc == -1) { continue; }

        codecs[i].usable = 1;
        b64_nusable++;
        __afl_map_edge(0xA010 + (unsigned)i);
    }
}

/* Max decoded size of n input chars; max encoded size of n input bytes. */
static size_t b64_dec_cap(size_t n) { return (n * 3u) / 4u + B64_FUZZ_SLACK; }
static size_t b64_enc_cap(size_t n) { return 4u * ((n + 2u) / 3u) + B64_FUZZ_SLACK; }

/* malloc(0) may return NULL; every buffer here is at least one byte. */
static unsigned char *b64_alloc(size_t n) { return malloc(n ? n : 1u); }

#if defined(__AFL_EDGE) && __AFL_EDGE
/* Base64 alphabet plus pad; anything else is a bad char. */
static const unsigned char b64_alpha[256] = {
    ['A' ... 'Z'] = 1, ['a' ... 'z'] = 1, ['0' ... '9'] = 1,
    ['+'] = 1, ['/'] = 1, ['='] = 1,
};

/* IJON: lane of the first bad char of a rejected decode. All chars valid
 * (bad length or pad) -> pos = size. */
static void b64_ijon_bad(unsigned mode, const unsigned char *buf, size_t size, int rc) {
    if (rc) { return; }

    size_t pos = 0;
    while (pos < size && b64_alpha[buf[pos]]) { pos++; }
    FUZZ_IJON_LANE((mode >> 1) * B64_IJON_LANES, pos);
}
#else
#define b64_ijon_bad(mode, buf, size, rc) ((void)0)
#endif

static void b64_mismatch(unsigned edge) {
    __afl_map_edge(edge);
    abort();
}

/* ── Decode: every codec vs plain ─────────────────────────────────────── */
static int fuzz_decode(const unsigned char *buf, size_t size) {
    __afl_map_edge(0xA100);
    size_t cap = b64_dec_cap(size);
    unsigned char *ref = b64_alloc(cap);
    unsigned char *got = b64_alloc(cap);
    if (!ref || !got) { free(ref); free(got); return 0; }

    size_t ref_len = 0;
    int ref_rc = base64_decode((const char *)buf, size, (char *)ref, &ref_len, BASE64_FORCE_PLAIN);
    __afl_map_edge(ref_rc ? 0xA101 : 0xA102);
    b64_ijon_bad(B64_MODE_DECODE, buf, size, ref_rc);

    for (size_t i = 1; i < B64_NCODECS; i++) {
        if (!codecs[i].usable) { continue; }

        size_t got_len = 0;
        int rc = base64_decode((const char *)buf, size, (char *)got, &got_len, codecs[i].flag);
        if (rc != ref_rc) { b64_mismatch(0xA110); }
        if (!ref_rc) { continue; }

        /* Landmark per codec so inputs that reach a wide-vector main loop
         * get distinct coverage from ones that only hit the scalar tail. */
        __afl_map_edge(0xA120 + (unsigned)i);
        if (got_len != ref_len) { b64_mismatch(0xA111); }
        if (memcmp(ref, got, ref_len) != 0) { b64_mismatch(0xA112); }
    }
    free(ref);
    free(got);
    return 0;
}

/* ── Encode: every codec vs plain, then round trip ────────────────────── */
static int fuzz_encode(const unsigned char *buf, size_t size) {
    __afl_map_edge(0xA200);
    size_t ecap = b64_enc_cap(size);
    unsigned char *ref = b64_alloc(ecap);
    unsigned char *got = b64_alloc(ecap);
    unsigned char *back = b64_alloc(b64_dec_cap(ecap));
    if (!ref || !got || !back) { free(ref); free(got); free(back); return 0; }

    size_t ref_len = 0;
    base64_encode((const char *)buf, size, (char *)ref, &ref_len, BASE64_FORCE_PLAIN);

    for (size_t i = 1; i < B64_NCODECS; i++) {
        if (!codecs[i].usable) { continue; }

        size_t got_len = 0;
        base64_encode((const char *)buf, size, (char *)got, &got_len, codecs[i].flag);
        __afl_map_edge(0xA220 + (unsigned)i);
        if (got_len != ref_len) { b64_mismatch(0xA211); }
        if (memcmp(ref, got, ref_len) != 0) { b64_mismatch(0xA212); }
    }

    /* Round trip through every codec: decode(encode(x)) must give x back. */
    for (size_t i = 0; i < B64_NCODECS; i++) {
        if (!codecs[i].usable) { continue; }

        size_t back_len = 0;
        int rc = base64_decode((const char *)ref, ref_len, (char *)back, &back_len, codecs[i].flag);
        __afl_map_edge(0xA240 + (unsigned)i);
        if (rc != 1) { b64_mismatch(0xA213); }
        if (back_len != size) { b64_mismatch(0xA214); }
        if (size && memcmp(back, buf, size) != 0) { b64_mismatch(0xA215); }
    }
    free(ref);
    free(got);
    free(back);
    return 0;
}

/* Next chunk length: 1..B64_FUZZ_MAX_CHUNK, driven by the seed byte so the
 * fuzzer controls where the stream is split (a split inside a 4-char quantum
 * or a SIMD block is the interesting case). */
static size_t b64_next_chunk(unsigned *state, size_t pos, size_t left) {
    if (pos >= B64_FUZZ_SPLIT_SPAN) { return left; }
    *state = *state * 1103515245u + 12345u;
    size_t n = 1u + ((*state >> 16) % B64_FUZZ_MAX_CHUNK);
    return n < left ? n : left;
}

/* ── Stream decode: chunked vs one shot ───────────────────────────────── */
static int fuzz_stream_decode(const unsigned char *buf, size_t size, unsigned seed) {
    __afl_map_edge(0xA300);
    size_t cap = b64_dec_cap(size);
    unsigned char *ref = b64_alloc(cap);
    unsigned char *got = b64_alloc(cap);
    if (!ref || !got) { free(ref); free(got); return 0; }

    size_t ref_len = 0;
    int ref_rc = base64_decode((const char *)buf, size, (char *)ref, &ref_len, BASE64_FORCE_PLAIN);
    b64_ijon_bad(B64_MODE_STREAM_DECODE, buf, size, ref_rc);

    for (size_t i = 0; i < B64_NCODECS; i++) {
        if (!codecs[i].usable) { continue; }

        struct base64_state st;
        base64_stream_decode_init(&st, codecs[i].flag);
        unsigned lcg = seed + (unsigned)i;
        size_t pos = 0, got_len = 0;
        int rc = 1;

        while (pos < size && rc == 1) {
            size_t n = b64_next_chunk(&lcg, pos, size - pos);
            size_t part = 0;
            rc = base64_stream_decode(&st, (const char *)buf + pos, n, (char *)got + got_len, &part);
            got_len += part;
            pos += n;
            __afl_map_edge(0xA310 + (unsigned)(n & 0xF));
        }

        /* Mirror base64_decode(): a trailing partial quantum is an error once
         * the stream ends, e.g. "QUJ". The stream call alone returns 1. */
        if (rc == 1 && st.bytes != 0) { rc = 0; }

        __afl_map_edge(0xA320 + (unsigned)i);
        if (rc != ref_rc) { b64_mismatch(0xA330); }
        if (!ref_rc) { continue; }
        if (got_len != ref_len) { b64_mismatch(0xA331); }
        if (memcmp(ref, got, ref_len) != 0) { b64_mismatch(0xA332); }
    }
    free(ref);
    free(got);
    return 0;
}

/* ── Stream encode: chunked vs one shot ───────────────────────────────── */
static int fuzz_stream_encode(const unsigned char *buf, size_t size, unsigned seed) {
    __afl_map_edge(0xA400);
    size_t cap = b64_enc_cap(size);
    unsigned char *ref = b64_alloc(cap);
    unsigned char *got = b64_alloc(cap);
    if (!ref || !got) { free(ref); free(got); return 0; }

    size_t ref_len = 0;
    base64_encode((const char *)buf, size, (char *)ref, &ref_len, BASE64_FORCE_PLAIN);

    for (size_t i = 0; i < B64_NCODECS; i++) {
        if (!codecs[i].usable) { continue; }

        struct base64_state st;
        base64_stream_encode_init(&st, codecs[i].flag);
        unsigned lcg = seed + (unsigned)i;
        size_t pos = 0, got_len = 0;

        while (pos < size) {
            size_t n = b64_next_chunk(&lcg, pos, size - pos);
            size_t part = 0;
            base64_stream_encode(&st, (const char *)buf + pos, n, (char *)got + got_len, &part);
            got_len += part;
            pos += n;
            __afl_map_edge(0xA410 + (unsigned)(n & 0xF));
        }
        size_t tail = 0;
        base64_stream_encode_final(&st, (char *)got + got_len, &tail);
        got_len += tail;

        __afl_map_edge(0xA420 + (unsigned)i);
        if (got_len != ref_len) { b64_mismatch(0xA431); }
        if (memcmp(ref, got, ref_len) != 0) { b64_mismatch(0xA432); }
    }
    free(ref);
    free(got);
    return 0;
}

/* Run one mode on an exact-size heap copy of the payload. The runner's
 * buffer can extend past the payload (or not be heap at all), so a codec
 * reading src[len] stayed invisible to ASAN; the copy puts a redzone there. */
static int b64_run(unsigned mode, const unsigned char *payload, size_t plen, unsigned seed) {
    unsigned char *in = b64_alloc(plen);
    if (!in) { return 0; }
    memcpy(in, payload, plen);
    FUZZ_IJON_LANE(B64_IJON_LEN_BASE + mode * B64_IJON_LANES, plen);

    int rc;
    switch (mode) {
    case B64_MODE_DECODE:        rc = fuzz_decode(in, plen); break;
    case B64_MODE_ENCODE:        rc = fuzz_encode(in, plen); break;
    case B64_MODE_STREAM_DECODE: rc = fuzz_stream_decode(in, plen, seed); break;
    default:                     rc = fuzz_stream_encode(in, plen, seed); break;
    }
    free(in);
    return rc;
}

__attribute__((visibility("default")))
int fuzz_libbase64(const unsigned char *buf, size_t size) {
    b64_probe();
    __afl_map_edge(0xA000);
    if (size < 1) { __afl_map_edge(0xA001); return 0; }

    unsigned mode = buf[0] & B64_MODE_MASK;
    const unsigned char *payload = buf + 1;
    size_t plen = size - 1;

    __afl_map_edge(0xA002 + mode);

    if (mode == B64_MODE_DECODE || mode == B64_MODE_ENCODE) {
        if (plen > B64_FUZZ_MAX_IN) { plen = B64_FUZZ_MAX_IN; }
        return b64_run(mode, payload, plen, 0);
    }

    /* Stream modes spend one more byte on the chunk-size seed. */
    if (plen < 1) { __afl_map_edge(0xA007); return 0; }
    unsigned seed = payload[0];
    payload++;
    plen--;
    if (plen > B64_FUZZ_MAX_IN) { plen = B64_FUZZ_MAX_IN; }

    return b64_run(mode, payload, plen, seed);
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_libbase64(buf, size);
}
