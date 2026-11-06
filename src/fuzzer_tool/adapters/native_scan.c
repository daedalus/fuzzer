/*
 * Per-exec scan kernels, called through ctypes by adapters/native_scan.py.
 *
 * Each replaces a chain of small numpy calls whose fixed dispatch cost
 * dominated on small maps (fuzzgoat: ~15 numpy calls per coverage scan).
 *
 *   SHM edge table --fs_live / fs_live_touched--> ids[], counts[]
 *   ids[], counts[] --fs_fold--> virgin[], maxc[] updated + candidates
 *   cmplog counters --fs_drain_counts / fs_drain_sites--> deltas, zeroed
 *
 * Layouts mirror adapters/shm.py (_ENTRY_DTYPE) and adapters/cmp_shm.py
 * (_SITE). Pure code, no state: not linked into targets.
 */

#include <stdint.h>

#define GEN_SHIFT 56          /* generation byte: top byte of the 8-byte entry */
#define COUNT_MASK 0xFFFFFFu  /* hit count below the generation byte */
#define BUCKET_CLAMP 255u     /* bucket table has 256 rows */

enum { ST_TRANSITIONS, ST_OLD_MOVED, ST_GAIN, ST_FIELDS };

/* Live entry: generation byte matches and edge_id != 0. */
static inline long take(uint64_t w, uint8_t gen, uint32_t *ids, uint32_t *counts, long k) {
    uint32_t id = (uint32_t)w;
    if ((uint8_t)(w >> GEN_SHIFT) != gen || id == 0)
        return k;
    ids[k] = id;
    counts[k] = (uint32_t)(w >> 32);
    return k + 1;
}

/* Walk the whole table; ascending slot order. Returns live count. */
long fs_live(const uint64_t *table, long n, uint32_t gen, uint32_t *ids, uint32_t *counts) {
    uint8_t g = (uint8_t)gen;
    long k = 0;
    for (long i = 0; i < n; i++)
        k = take(table[i], g, ids, counts, k);
    return k;
}

/* Visit only slots whose touched bit is set; padding bits past n dropped. */
long fs_live_touched(const uint64_t *table, long n, uint32_t gen, const uint64_t *words,
                     long nwords, uint32_t *ids, uint32_t *counts) {
    uint8_t g = (uint8_t)gen;
    long k = 0;
    for (long w = 0; w < nwords; w++) {
        uint64_t bits = words[w];
        while (bits) {
            long slot = w * 64 + __builtin_ctzll(bits);
            bits &= bits - 1;
            if (slot >= n)
                return k;
            k = take(table[slot], g, ids, counts, k);
        }
    }
    return k;
}

/*
 * Fold one exec into the virgin bucket map and per-edge maxima.
 *
 * Candidates are ids whose virgin byte was 0 before the fold: every new id
 * is one (virgin bits are only set for seen ids), so the caller's set
 * lookup runs on these alone. A bucket move on a non-candidate is an old
 * edge moving (ST_OLD_MOVED). Max gain needs cnt >= prev * factor in
 * double: prev * 1.5 wraps u32 near the 24-bit ceiling.
 *
 * Returns the candidate count, or -1 (nothing written) when an id is past
 * either array: the caller grows them and retries.
 */
long fs_fold(const uint32_t *ids, const uint32_t *counts, long n, uint8_t *virgin, long vsize,
             uint32_t *maxc, long msize, const uint8_t *bucket, double factor,
             uint32_t *cand_ids, uint8_t *cand_bits, uint64_t *stats) {
    uint32_t hi = 0;
    for (long i = 0; i < n; i++)
        hi = ids[i] > hi ? ids[i] : hi;
    if (n && ((long)hi >= vsize || (long)hi >= msize))
        return -1;

    uint64_t trans = 0, moved = 0, gain = 0;
    long nc = 0;
    for (long i = 0; i < n; i++) {
        uint32_t id = ids[i];
        uint32_t c = counts[i] & COUNT_MASK;
        uint8_t bits = bucket[c > BUCKET_CLAMP ? BUCKET_CLAMP : c];
        uint8_t prev = virgin[id];

        if (prev == 0) {
            cand_ids[nc] = id;
            cand_bits[nc++] = bits;
        }
        if (bits & (uint8_t)~prev) {
            trans++;
            moved |= prev != 0;
            virgin[id] = prev | bits;
        }

        uint32_t pm = maxc[id];
        gain += pm && (double)c >= (double)pm * factor;
        if (c > pm)
            maxc[id] = c;
    }

    stats[ST_TRANSITIONS] = trans;
    stats[ST_OLD_MOVED] = moved;
    stats[ST_GAIN] = gain;
    return nc;
}

/* Per-callback counters: emit (index, fired, hit) where fired != 0; zero them. */
long fs_drain_counts(uint64_t *fired, uint64_t *hit, long n, int64_t *idx, int64_t *f_out,
                     int64_t *h_out) {
    long k = 0;
    for (long i = 0; i < n; i++) {
        if (!fired[i])
            continue;
        idx[k] = i;
        f_out[k] = (int64_t)fired[i];
        h_out[k++] = (int64_t)hit[i];
        fired[i] = hit[i] = 0;
    }
    return k;
}

/* Mirrors cmp_shm._SITE: {pc u64, fired u64, hit u64, id u32, pad u32}. */
struct site {
    uint64_t pc, fired, hit;
    uint32_t id, pad;
};

/* Site table: emit (slot, fired, hit) where fired != 0; zero counters, keep keys. */
long fs_drain_sites(struct site *sites, long n, int64_t *slots, int64_t *f_out, int64_t *h_out) {
    long k = 0;
    for (long i = 0; i < n; i++) {
        if (!sites[i].fired)
            continue;
        slots[k] = i;
        f_out[k] = (int64_t)sites[i].fired;
        h_out[k++] = (int64_t)sites[i].hit;
        sites[i].fired = sites[i].hit = 0;
    }
    return k;
}
