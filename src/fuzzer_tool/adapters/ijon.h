/*
 * IJON annotations (Aschermann et al., "IJON: Exploring Deep State Spaces
 * via Fuzzing", S&P 2020). Include from target or library sources; the
 * runtime lives in afl_shim.c, linked into the wrapper only. Runtime API
 * and macros follow upstream (RUB-SysSec/ijon), so annotated sources port
 * unchanged.
 *
 *   IJON_SET(x)        value x of this site is coverage (once per run)
 *   IJON_INC(x)        same, but every call counts (hit-count buckets)
 *   IJON_MAX(x)        keep the input reaching the largest x at this site
 *   IJON_MIN(x)        keep the input reaching the smallest x at this site
 *   IJON_CMP(x, y)     coverage per number of bits where x and y differ
 *   IJON_DIST(x, y)    keep the input bringing x closest to y
 *   IJON_STRDIST(x, y) coverage per matched prefix length, per call stack
 *   IJON_CTX(x)        evaluate x with this site xored into the state
 *   IJON_BITS(x)       leading zero bits of 32-bit x (0 for x == 0)
 *
 * Example, a maze: IJON_SET(ijon_hashint(x, y)) rewards each new cell; a
 * checksum: IJON_CMP(computed, stored) rewards each extra matching bit.
 *
 * Deviation: ijon_xor_state/push_state affect IJON_SET/INC ids only, not
 * every real edge as upstream's compiler pass does.
 */
#ifndef FUZZER_TOOL_IJON_H
#define FUZZER_TOOL_IJON_H

#include <stddef.h>
#include <stdint.h>

void ijon_map_set(uint32_t addr);
void ijon_map_inc(uint32_t addr);
void ijon_max(uint32_t addr, uint64_t value);
void ijon_min(uint32_t addr, uint64_t value);

void ijon_xor_state(uint32_t value);
void ijon_push_state(uint32_t value);
void ijon_enable_feedback(void);
void ijon_disable_feedback(void);

uint32_t ijon_strdist(char *a, char *b);
uint32_t ijon_memdist(char *a, char *b, size_t len);
uint64_t ijon_simple_hash(uint64_t x);
uint32_t ijon_hashint(uint32_t old, uint32_t val);
uint32_t ijon_hashstr(uint32_t old, char *val);
uint32_t ijon_hashmem(uint32_t old, char *val, size_t len);
uint32_t ijon_hashstack(void);

/* Per-site id. Upstream calls ijon_hashstr(__LINE__, __FILE__) on every
 * annotation; this FNV-1a fold is constant at -O1+ for a literal path. */
static inline uint32_t _ijon_site(uint32_t line, const char *file) {
    uint32_t h = 2166136261u;
    for (; *file; file++) h = (h ^ (uint8_t)*file) * 16777619u;
    return h ^ (line * 0x9E3779B1u);
}

#define _IJON_SITE _ijon_site(__LINE__, __FILE__)
#define _IJON_CONCAT2(x, y) x##y
#define _IJON_CONCAT(x, y) _IJON_CONCAT2(x, y)
#define _IJON_UNIQ_NAME() _IJON_CONCAT(_ijon_tmp_, __LINE__)
#define _IJON_ABS_DIST(x, y) ((x) < (y) ? (y) - (x) : (x) - (y))

#define IJON_SET(x) ijon_map_set(_IJON_SITE ^ (uint32_t)(x))
#define IJON_INC(x) ijon_map_inc(_IJON_SITE ^ (uint32_t)(x))
#define IJON_MAX(x) ijon_max(_IJON_SITE, (uint64_t)(x))
#define IJON_MIN(x) ijon_min(_IJON_SITE, (uint64_t)(x))
#define IJON_CMP(x, y) IJON_INC(__builtin_popcountll((uint64_t)(x) ^ (uint64_t)(y)))
#define IJON_DIST(x, y) IJON_MIN(_IJON_ABS_DIST((uint64_t)(x), (uint64_t)(y)))
#define IJON_STRDIST(x, y) IJON_SET(ijon_hashint(ijon_hashstack(), ijon_strdist(x, y)))
#define IJON_BITS(x) ((uint32_t)(x) == 0 ? 0 : __builtin_clz((uint32_t)(x)))

/* Upstream's IJON_CTX, with its name typos fixed. */
#define IJON_CTX(x)                                         \
    ({                                                      \
        uint32_t _ijon_h = _IJON_SITE;                      \
        ijon_xor_state(_ijon_h);                            \
        __typeof__(x) _IJON_UNIQ_NAME() = (x);              \
        ijon_xor_state(_ijon_h);                            \
        _IJON_UNIQ_NAME();                                  \
    })

#endif /* FUZZER_TOOL_IJON_H */
