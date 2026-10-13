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
 * Extended gates (only evaluated when len >= EXT_MIN_LEN = 24, so every 11..23
 * byte input behaves exactly as before and earlier synthetic results still
 * reproduce):
 *
 *   gate  field                      solved iff
 *   ----  -------------------------  ---------------------------------------
 *   G4    BE32 buf[11..14]           x in [0x50000000, +16): a window PAST both
 *                                    seeds. Only the low side has a distance
 *                                    gradient (x < W fires below[log2(W-x)]);
 *                                    beyond the window one flat edge fires, so
 *                                    nothing between two seeds below W reaches
 *                                    it: only t > 1 extrapolation (line_blend,
 *                                    nm_reflect alpha = 2) can.
 *   G5    BE16 buf[15..16] = f1,     f1 >= 0xC000 fires edge A, f2 <= 0x3FFF
 *         BE16 buf[17..18] = f2      fires edge B, solved iff both. A seed with
 *                                    f1 high / f2 high and one with f1 low /
 *                                    f2 low each own one edge; the answer is a
 *                                    per-field combination (max of f1, min of
 *                                    f2), which field_minmax draws directly.
 *   G6    LE32 buf[20..23] = x       (7*x + 13) % 1009 == 500, evaluated in
 *                                    64 bits so the compare is an 8-byte pair
 *                                    (v, 500): the --affine-mod target, and the
 *                                    probe for the cmplog 8-byte hash filter.
 *                                    Not reachable by any pair operator.
 *
 * G4/G5 are scored by tools/arith_mid_experiment.py --ext; G6 needs a cmplog
 * build (see the build line below).
 *
 * Build (coverage build mirrors targets/compcov_gates.c; -O2, clang):
 *   clang -O2 -g -fsanitize-coverage=trace-pc-guard \
 *     -include src/fuzzer_tool/adapters/afl_shim.c \
 *     -o <out>/arith_gates targets/arith_gates.c -ldl
 * G6 only produces a cmplog record on a build that logs comparisons:
 *   clang -O2 -g -fsanitize-coverage=trace-pc-guard,trace-cmp -D__AFL_CMPLOG=1 \
 *     -include src/fuzzer_tool/adapters/afl_shim.c \
 *     -o <out>/arith_gates_cmp targets/arith_gates.c -ldl
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
#define EXT_MIN_LEN 24
#define GATES 7
#define BANDS 32
#define GATE_EDGE(g) __afl_map_edge(0x7100u + (unsigned)(g))
#define G4_WINDOW 0x50000000u
#define G5_EDGE_A 0x7180u
#define G5_EDGE_B 0x7181u
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
    if (len < EXT_MIN_LEN) return mask;

    /* G4: window past both seeds. Gradient below the window only. */
    {
        uint32_t x = be32(buf + 11);
        if (x < G4_WINDOW) {
            int j = floor_log2(G4_WINDOW - x);
            BAND_EDGE(4, 0, j);
            g_edges[4] |= 1ull << j;
        } else if (x - G4_WINDOW < 16) {
            GATE_EDGE(4);
            mask |= 1 << 4;
        } else {
            BAND_EDGE(4, 1, 0); /* overshoot: one flat edge, no gradient */
            g_edges[4] |= 1ull << BANDS;
        }
    }

    /* G5: conjunction of two independently rewarded fields. */
    {
        uint32_t f1 = ((uint32_t)buf[15] << 8) | buf[16];
        uint32_t f2 = ((uint32_t)buf[17] << 8) | buf[18];
        int a = f1 >= 0xC000u, b = f2 <= 0x3FFFu;
        if (a) { __afl_map_edge(G5_EDGE_A); g_edges[5] |= 1ull << 0; }
        if (b) { __afl_map_edge(G5_EDGE_B); g_edges[5] |= 1ull << 1; }
        if (a && b) { GATE_EDGE(5); mask |= 1 << 5; }
    }

    /* G6: modular-affine compare; the 64-bit v keeps the compare 8 bytes wide. */
    {
        uint64_t v = (7ull * le32(buf + 20) + 13ull) % 1009ull;
        if (v == 500ull) { GATE_EDGE(6); mask |= 1 << 6; }
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
