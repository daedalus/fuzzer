/* arith_gates.c -- synthetic target: can arith_mid bisect what other recombination ops cannot?
 *
 * Three narrow integer windows over disjoint fields. Ground truth is known by
 * construction: gate g is solved iff its field x lies in [W, W + WIN); bit g of
 * the fuzz_gates() return value reports it (the shim build's exit code is not
 * reliable: score with the GATES_PLAIN library).
 *
 * Each window has a coverage gradient of distance bands on both sides:
 *   x <  W        fires edge  below[floor(log2(W - x))]
 *   x >= W + WIN  fires edge  above[floor(log2(x - W - WIN + 1))]
 * A seed closer to the window than any before it reaches a new band, so seeds
 * below and past the window both exist in a corpus. The mean of such a pair
 * halves the interval around W, which no bit- or byte-position recombination
 * of the pair does.
 *
 *   gate  field           W           WIN
 *   ----  --------------  ----------  ---
 *   G0    buf[0]          'K'         1     (sanity control)
 *   G1    BE32 buf[1..4]  0x12345678  16
 *   G2    LE32 buf[5..8]  0x00ABCDEF  16
 *   G3    BE16 buf[9..10] 0x1357      4
 *
 * Intended solution: 'K', 12 34 56 78, EF CD AB 00, 13 57 (any window value).
 *
 * Build (coverage build mirrors targets/compcov_gates.c; -O2, clang):
 *   clang -O2 -g -fsanitize-coverage=trace-pc-guard \
 *     -include src/fuzzer_tool/adapters/afl_shim.c \
 *     -o <out>/arith_gates targets/arith_gates.c -ldl
 * Ground-truth library (no shim, no coverage), used by
 * tools/arith_mid_experiment.py:
 *   clang -O2 -shared -fPIC -DGATES_PLAIN -o <out>/arith_gates_ref.so targets/arith_gates.c
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef GATES_PLAIN
static void __afl_map_edge(unsigned int cur_loc) { (void)cur_loc; }
#else
extern void __afl_map_edge(unsigned int cur_loc);
#endif

#define GATES_MIN_LEN 11
#define GATES 4
#define BANDS 32
#define GATE_EDGE(g) __afl_map_edge(0x7100u + (unsigned)(g))
#define BAND_EDGE(g, side, j) \
    __afl_map_edge(0x7200u + 0x80u * (unsigned)(g) + BANDS * (unsigned)(side) + (unsigned)(j))

/* Band edges hit per gate (bit side*BANDS + j), for the ground-truth library. */
static uint64_t g_edges[GATES];

static inline uint32_t be32(const unsigned char *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

static inline uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline int floor_log2(uint32_t v) { return 31 - __builtin_clz(v); }

/* Fires the distance-band edge of gate g; returns 1 inside the window. */
static int window(int g, uint32_t x, uint32_t w, uint32_t win) {
    int side, j;
    if (x >= w && x - w < win) return 1;
    side = x >= w;
    j = floor_log2(side ? x - w - win + 1 : w - x);
    BAND_EDGE(g, side, j);
    g_edges[g] |= 1ull << (side * BANDS + j);
    return 0;
}

__attribute__((visibility("default")))
int fuzz_gates(const unsigned char *buf, size_t len) {
    int mask = 0;
    for (int g = 0; g < GATES; g++) g_edges[g] = 0;
    if (len < GATES_MIN_LEN) return 0;

    /* G0: plain one-byte compare. */
    if (buf[0] == 'K') { GATE_EDGE(0); mask |= 1 << 0; }

    if (window(1, be32(buf + 1), 0x12345678u, 16)) { GATE_EDGE(1); mask |= 1 << 1; }
    if (window(2, le32(buf + 5), 0x00ABCDEFu, 16)) { GATE_EDGE(2); mask |= 1 << 2; }
    if (window(3, ((uint32_t)buf[9] << 8) | buf[10], 0x1357u, 4)) {
        GATE_EDGE(3);
        mask |= 1 << 3;
    }
    return mask;
}

#ifdef GATES_PLAIN
/* Band edges hit on the last fuzz_gates() call, one mask per gate. */
__attribute__((visibility("default")))
const uint64_t *gates_edges(void) { return g_edges; }
#endif

__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_gates(buf, size);
}

#ifndef GATES_PLAIN
int main(int argc, char **argv) {
    static unsigned char buf[65536];
    size_t n = 0;
    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 255;
        n = fread(buf, 1, sizeof(buf), f);
        fclose(f);
    } else {
        n = fread(buf, 1, sizeof(buf), stdin);
    }
    return fuzz_gates(buf, n);
}
#endif
