/* Differential fuzz target: brainflayer's sliced bloom filter (bloom.c).
 *
 * brainflayer checks every candidate hash160 against a bloom filter file.
 * A sliced filter ends in a 16-byte trailer ("BLFSLICE", version, log2bits,
 * k) that bloom_probe() hands to bloom_init(), which derives k bit windows
 * of the 160-bit hash. Window extraction (bloom_slice) and the AVX2 gather
 * lookups (bloom_chk_n) are shift/lane code where an off-by-one silently
 * drops matches — a false negative no crash would ever reveal. Checks, per
 * record, aborting on the first disagreement:
 *
 *   1. bloom_slice()  == bf_ref_slice()  independent MSB-first bit walk
 *   2. bloom_chk()    == 1 after bloom_set()          no false negatives
 *   3. bloom_chk_n()  == bloom_chk()  scalar, then AVX2 when the CPU has it
 *
 * Input layout (trailer first, so the magic sits at offset 0):
 *
 *   [0..16)   trailer as bloom_trailer() writes it
 *   [16..)    hash160 records, 20 bytes each, capped at BF_MAX_RECS
 *
 * Trailer gate mirrors bloom_probe(): magic + version here, log2bits/k by
 * the library's bloom_init(). log2bits = 0 selects the legacy 1 GiB filter
 * (on disk that is chosen by file size instead).
 *
 * The filter is an anonymous MAP_NORESERVE mapping (up to 2^37 bytes of
 * address space at log2bits = 40), unmapped every exec; only touched pages
 * get backed: at most 2 * BF_MAX_RECS * BLOOM_SLICES_MAX of them.
 *
 * Vendor first (tools/vendor_brainflayer.sh), build via tools/build_targets.sh.
 * bloom.c is compiled separately, without the shim — see
 * compile_brainflayer_bloom_objects() in build_targets.sh.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "bloom.h"

/* AFL edge coverage — provided by afl_shim.c */
extern void __afl_map_edge(unsigned int cur_loc);

#define BF_REC_SZ 20u
#define BF_WORDS 5u
#define BF_HASH_BITS 160u
/* 32 records + 32 complemented probes keeps an exec to ~10k page faults. */
#define BF_MAX_RECS 32u
#define BF_MAX_QUERIES (2u * BF_MAX_RECS)

/* Reference window: bits [off, off + width) of the five words read MSB
 * first, wrapping at 160. E.g. off = 30, width = 4: the low 2 bits of
 * h[0] then the top 2 of h[1]. Bit by bit, no shifts shared with bloom.c. */
static uint64_t bf_ref_slice(const uint32_t *h, unsigned off, unsigned width) {
    uint64_t v = 0;
    for (unsigned i = 0; i < width; i++) {
        unsigned p = (off + i) % BF_HASH_BITS;
        v = v << 1 | ((h[p / 32u] >> (31u - p % 32u)) & 1u);
    }
    return v;
}

/* Check 1: every window against the reference (sliced filters only; the
 * legacy BH20..BH24 windows are not contiguous by design). */
static void bf_check_slices(const bloom_t *b, const uint32_t *h) {
    uint32_t w[7];
    memcpy(w, h, BF_REC_SZ);
    w[5] = h[0];
    w[6] = h[1];

    for (unsigned i = 0; i < b->k; i++) {
        unsigned off = i * BF_HASH_BITS / b->k;
        if (bloom_slice(b, w, i) != bf_ref_slice(h, off, b->log2bits)) {
            __afl_map_edge(0x9501);
            abort();
        }
    }
}

/* Check 3: batch lookup against per-hash bloom_chk, one SIMD mode. */
static void bf_check_batch(bloom_t *b, const uint32_t *q, size_t nq, int simd) {
    unsigned char hit[BF_MAX_QUERIES];
    b->simd = simd;
    bloom_chk_n(b, q, nq, hit);

    for (size_t i = 0; i < nq; i++) {
        if (hit[i] != bloom_chk(b, q + BF_WORDS * i)) {
            __afl_map_edge(0x9503 + (unsigned)simd);
            abort();
        }
    }
}

/* Validate the trailer; 0 and b set up when the filter is usable. */
static int bf_parse_trailer(bloom_t *b, const unsigned char *t) {
    if (memcmp(t, BLOOM_MAGIC, 8) != 0 || t[8] != BLOOM_VERSION) { return -1; }
    __afl_map_edge(0x9510);
    if (bloom_init(b, t[9], t[10]) != 0) { return -1; }
    __afl_map_edge(b->legacy ? 0x9511 : 0x9512);
    return 0;
}

static void bf_run(bloom_t *b, const uint32_t *q, size_t n) {
    if (!b->legacy) {
        for (size_t i = 0; i < n; i++) { bf_check_slices(b, q + BF_WORDS * i); }
    }

    /* Check 2: members are always found. */
    for (size_t i = 0; i < n; i++) { bloom_set(b, q + BF_WORDS * i); }
    for (size_t i = 0; i < n; i++) {
        if (bloom_chk(b, q + BF_WORDS * i) != 1) {
            __afl_map_edge(0x9502);
            abort();
        }
    }

    /* Records plus their complements, so batches see misses too. */
    bf_check_batch(b, q, 2 * n, 0);
    if (__builtin_cpu_supports("avx2")) { bf_check_batch(b, q, 2 * n, 1); }
}

__attribute__((visibility("default")))
int fuzz_brainflayer_read(const unsigned char *buf, size_t size) {
    __afl_map_edge(0x9500);
    if (size < BLOOM_TRAILER_SZ) { return 0; }

    bloom_t b;
    if (bf_parse_trailer(&b, buf) != 0) { return 0; }

    size_t n = (size - BLOOM_TRAILER_SZ) / BF_REC_SZ;
    if (n == 0) { return 0; }
    if (n > BF_MAX_RECS) { n = BF_MAX_RECS; }

    /* Word-aligned copy: bloom.c reads hash160s as uint32_t[5]. */
    uint32_t q[BF_MAX_QUERIES * BF_WORDS];
    memcpy(q, buf + BLOOM_TRAILER_SZ, n * BF_REC_SZ);
    for (size_t i = 0; i < n * BF_WORDS; i++) { q[n * BF_WORDS + i] = ~q[i]; }

    void *bits = mmap(NULL, b.bits_sz, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (bits == MAP_FAILED) { return 0; }
    b.bits = bits;

    bf_run(&b, q, n);
    munmap(bits, b.bits_sz);
    __afl_map_edge(0x9520);
    return 0;
}

/* Standard in-process entry point for fuzzer-tool .so mode */
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_brainflayer_read(buf, size);
}
