/* gf2_gates.c -- synthetic target: GF(2)-affine gates with no visible compare of the input.
 *
 * Ground truth is known by construction: gate g is solved iff operand[g] == target[g];
 * bit g of the fuzz_gates() return value reports it (the shim build's exit code is not
 * reliable: score with the GATES_PLAIN library).
 *
 * Every operand is an affine map over GF(2) of its input field, so the gate is a linear
 * system. The operand (a parity bit or a syndrome) never occurs in the input, so a
 * compare-to-input substitution has nothing to find.
 *
 *   gate  field           operand                              target   random hit
 *   ----  --------------  -----------------------------------  -------  ----------
 *   G0    buf[0]          buf[0]                               'K'      2^-8 (control)
 *   G1    BE16 buf[1..2]  parity(x & 0xB2D4)                   1        2^-1
 *   G2    BE32 buf[3..6]  8-bit syndrome of x, ^ 0x6E          0x5A     2^-8
 *   G3    LE32 buf[7..10] 16-bit syndrome of x, ^ 0x2B91       0xC3A5   2^-16
 *
 * The syndrome rows are full rank, so a solution exists and the field's low bits are free
 * (G2: 2^24 solutions, G3: 2^16).
 *
 * Build (coverage build mirrors targets/arith_gates.c; -O2, clang):
 *   clang -O2 -g -fsanitize-coverage=trace-pc-guard \
 *     -include src/fuzzer_tool/adapters/afl_shim.c \
 *     -o <out>/gf2_gates targets/gf2_gates.c -ldl
 * Ground-truth library (no shim, no coverage), used by tools/gf2_gates_experiment.py:
 *   cc -O2 -shared -fPIC -DGATES_PLAIN -o <out>/gf2_gates_ref.so targets/gf2_gates.c
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
#define GATE_EDGE(g) __afl_map_edge(0x7300u + (unsigned)(g))

#define G1_MASK 0xB2D4u
#define G2_ROWS 8
#define G3_ROWS 16

static const uint32_t SYNDROME[G3_ROWS] = {
    0x7de9fc02u, 0x7ee9b8c9u, 0x47682bebu, 0x0a7dc563u,
    0x7add219du, 0xb07e1af5u, 0xb4d4d8fbu, 0x3798a188u,
    0x78d9d9b6u, 0x00ee6fc7u, 0xf9728652u, 0x268251c9u,
    0x7eeaacfdu, 0x9c27cc92u, 0x0423e37du, 0x45e9da34u,
};

/* Operand and target of each gate on the last fuzz_gates() call. */
static uint32_t g_operand[GATES];
static uint32_t g_target[GATES] = {'K', 1u, 0x5Au, 0xC3A5u};

static inline uint32_t be16(const unsigned char *p) { return ((uint32_t)p[0] << 8) | p[1]; }

static inline uint32_t be32(const unsigned char *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

static inline uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* Bit j of the result is the parity of x & SYNDROME[j]. */
static uint32_t syndrome(uint32_t x, int rows) {
    uint32_t s = 0;
    for (int j = 0; j < rows; j++) s |= (uint32_t)__builtin_parity(x & SYNDROME[j]) << j;
    return s;
}

__attribute__((visibility("default")))
int fuzz_gates(const unsigned char *buf, size_t len) {
    int mask = 0;
    for (int g = 0; g < GATES; g++) g_operand[g] = 0;
    if (len < GATES_MIN_LEN) return 0;

    g_operand[0] = buf[0];
    g_operand[1] = (uint32_t)__builtin_parity(be16(buf + 1) & G1_MASK);
    g_operand[2] = syndrome(be32(buf + 3), G2_ROWS) ^ 0x6Eu;
    g_operand[3] = syndrome(le32(buf + 7), G3_ROWS) ^ 0x2B91u;

    for (int g = 0; g < GATES; g++) {
        if (g_operand[g] != g_target[g]) continue;
        GATE_EDGE(g);
        mask |= 1 << g;
    }
    return mask;
}

#ifdef GATES_PLAIN
/* Operand and target of each gate on the last fuzz_gates() call. */
__attribute__((visibility("default")))
const uint32_t *gates_operand(void) { return g_operand; }

__attribute__((visibility("default")))
const uint32_t *gates_target(void) { return g_target; }
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
