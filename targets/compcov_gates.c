/* compcov_gates.c -- synthetic target: does COMPCOV solve coverage cmplog cannot?
 *
 * Six independent gates over disjoint input fields. Ground truth is known by
 * construction: each gate has an intended solution, a distinct "solved" edge,
 * and a bit in the return value / exit code (bit g <=> gate g solved).
 *
 * The split that matters is *where the compared operand comes from*:
 *
 *   cmplog (input-to-state) solves a comparison when one operand appears in
 *   the input, so it can be substituted for the other. COMPCOV solves nothing
 *   by itself; it turns a wide compare into a ladder of one-byte-per-rung
 *   synthetic edges, so ordinary novelty search can climb it. The two are
 *   separable exactly when the compared value is *derived* from the input:
 *   the derived value is not in the input (cmplog blind), but its low bytes
 *   still track the input's low bytes (COMPCOV has a gradient).
 *
 *   gate  compare                                   cmplog  COMPCOV  expected
 *   ----  ----------------------------------------  ------  -------  --------------------------
 *   G0    buf[0] == 'K'                              --      --      every arm (sanity control)
 *   G1    memcmp(buf+1, 8-byte magic)                YES     (yes)   cmplog arms; baseline never
 *   G2    (u32 * K) == C            K odd, volatile  no      L1+     COMPCOV only: 3-rung ladder
 *   G3    (sum of 16 bytes) == 0x0A57                no      L1+     COMPCOV only: 2-rung ladder
 *   G4    avalanche(u32) == C                        no      no      NOBODY (negative control):
 *                                                                    low byte carries no gradient,
 *                                                                    COMPCOV only mints noise edges
 *   G5    strncmp(rot13(buf+33,12), "HelloWorld!!")  no      L2 only COMPCOV level 2 only: libc
 *                                                                    layer never fires at L1
 *
 * Why G2 uses a volatile multiplier: at -O2 LLVM can rewrite `x*K == C` as
 * `x == C*K^-1` (K odd => invertible mod 2^32). That would put the *input*
 * operand into the compare and hand the gate to cmplog. volatile K keeps the
 * multiply in the IR, which is what a real derived-value check looks like.
 *
 * Intended solution (see SOLUTION below; tools/compcov_gates_experiment.py
 * builds it and asserts the checker reports all six bits):
 *   buf[0]      = 'K'
 *   buf[1..8]   = de ad be ef 13 37 c0 de
 *   buf[9..12]  = 0x1F2E3D4C little-endian
 *   buf[13..28] = any 16 bytes summing to 0x0A57
 *   buf[29..32] = 0x5A17C0DE little-endian
 *   buf[33..44] = "UryybJbeyq!!"   (rot13 of "HelloWorld!!")
 *
 * Build (mirrors tools/build_targets.sh --tracecmp; -O2, clang):
 *   clang -O2 -g -fno-omit-frame-pointer \
 *     -fsanitize-coverage=trace-cmp,trace-div,trace-gep,trace-pc-guard \
 *     <NOBUILTIN_CMP> -D__AFL_CMPLOG=1 -include src/fuzzer_tool/adapters/afl_shim.c \
 *     -o targets/compcov_gates targets/compcov_gates.c -ldl
 * Ground-truth checker (no shim, no coverage, no folding):
 *   gcc -O0 -DGATES_PLAIN -o targets/compcov_gates_check targets/compcov_gates.c
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GATES_PLAIN
static void __afl_map_edge(unsigned int cur_loc) { (void)cur_loc; }
#else
extern void __afl_map_edge(unsigned int cur_loc);
#endif

#define GATES_MIN_LEN 64
#define GATE_EDGE(g) __afl_map_edge(0x7000u + (unsigned)(g))

static volatile uint32_t G2_K = 0x9E3779B1u;

static inline uint32_t ld32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint32_t avalanche(uint32_t x) {
    uint32_t h = x * 0x85EBCA6Bu;
    h ^= h >> 15;
    h *= 0xC2B2AE35u;
    h ^= h >> 13;
    return h;
}

static inline char rot13(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return (char)((c - 'A' + 13) % 26 + 'A');
    if (c >= 'a' && c <= 'z') return (char)((c - 'a' + 13) % 26 + 'a');
    return (char)c;
}

__attribute__((visibility("default")))
int fuzz_gates(const unsigned char *buf, size_t len) {
    int mask = 0;
    if (len < GATES_MIN_LEN) return 0;

    /* G0: plain one-byte compare. Plain edge coverage already sees it. */
    if (buf[0] == 'K') { GATE_EDGE(0); mask |= 1 << 0; }

    /* G1: input-to-state. The magic sits verbatim in the input's other half
     * of the compare, so cmplog can substitute it in one step. */
    static const unsigned char magic[8] = {0xde, 0xad, 0xbe, 0xef, 0x13, 0x37, 0xc0, 0xde};
    if (memcmp(buf + 1, magic, 8) == 0) { GATE_EDGE(1); mask |= 1 << 1; }

    /* G2: derived value. u32*K is not in the input; its low byte depends only
     * on buf[9], its low 2 bytes only on buf[9..10], and so on. */
    if ((uint32_t)(ld32(buf + 9) * G2_K) == 0x7D454D8Cu) { GATE_EDGE(2); mask |= 1 << 2; }

    /* G3: derived value, many-to-one. A 16-bit sum; nudging any byte moves it
     * by the same delta, so partial progress is climbable. */
    uint32_t s = 0;
    for (int i = 13; i < 29; i++) s += buf[i];
    if ((uint16_t)s == 0x0A57u) { GATE_EDGE(3); mask |= 1 << 3; }

    /* G4: negative control. Full avalanche: matching the low byte of the hash
     * says nothing about the input, so there is no gradient to climb. */
    if (avalanche(ld32(buf + 29)) == 0x46044640u) { GATE_EDGE(4); mask |= 1 << 4; }

    /* G5: derived byte string through the libc layer. The compared buffer is
     * rot13(input), so cmplog logs a pair whose "input side" never occurs in
     * the input; COMPCOV level 2 walks the match one byte at a time. */
    char tmp[16];
    for (int i = 0; i < 12; i++) tmp[i] = rot13(buf[33 + i]);
    tmp[12] = '\0';
    if (strncmp(tmp, "HelloWorld!!", 12) == 0) { GATE_EDGE(5); mask |= 1 << 5; }

    return mask;
}

__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    return fuzz_gates(buf, size);
}

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
    int mask = fuzz_gates(buf, n);
#ifdef GATES_PLAIN
    /* Ground-truth ladder depth, independent of any fuzzer feedback:
     *   d1 = prefix bytes of the G1 magic matched (0..8)
     *   d2 = low bytes of (u32*K) matching the G2 constant (0..4)
     *   d3 = 0 none, 1 low byte of the sum matches, 2 full 16-bit match
     *   d4 = low bytes of avalanche() matching the G4 constant (0..4)
     *   d5 = prefix bytes of rot13(buf+33) matching "HelloWorld!!" (0..12) */
    int d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0;
    if (n >= GATES_MIN_LEN) {
        static const unsigned char mg[8] = {0xde, 0xad, 0xbe, 0xef, 0x13, 0x37, 0xc0, 0xde};
        while (d1 < 8 && buf[1 + d1] == mg[d1]) d1++;
        uint32_t p2 = ld32(buf + 9) * 0x9E3779B1u, c2 = 0x7D454D8Cu;
        while (d2 < 4 && ((p2 >> (8 * d2)) & 0xFF) == ((c2 >> (8 * d2)) & 0xFF)) d2++;
        uint32_t s3 = 0;
        for (int i = 13; i < 29; i++) s3 += buf[i];
        d3 = ((s3 & 0xFF) == 0x57) ? (((uint16_t)s3 == 0x0A57u) ? 2 : 1) : 0;
        uint32_t p4 = avalanche(ld32(buf + 29)), c4 = 0x46044640u;
        while (d4 < 4 && ((p4 >> (8 * d4)) & 0xFF) == ((c4 >> (8 * d4)) & 0xFF)) d4++;
        while (d5 < 12 && rot13(buf[33 + d5]) == "HelloWorld!!"[d5]) d5++;
    }
    printf("%d %d %d %d %d %d\n", mask, d1, d2, d3, d4, d5);
#endif
    return mask;
}
