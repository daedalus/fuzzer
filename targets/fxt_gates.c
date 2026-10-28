/* fxt_gates.c -- synthetic target: one gate per chapter of J. Arndt, "Matters Computational".
 *
 * Ground truth is known by construction: gate g is solved iff its field is valid and
 * operand[g] == target[g]; bit g of the fuzz_gates() return value reports it. The shim
 * build's exit code carries nothing: score with the GATES_PLAIN library.
 *
 * Each gate reads its own field of the input (offsets are cumulative, in gate order) and
 * runs the book's algorithm on it. The operand never occurs in the input, so a
 * compare-to-input substitution has nothing to find. Gates with a structural
 * precondition (a permutation, a Dyck word, ...) fire a second, "valid" edge when it
 * holds, then a "solved" edge when the operand hits the target.
 *
 * Chapter 5 (conventions) has no algorithm of its own: its ranking/unranking appears in
 * the rank gates (6-17). Chapters 1-4 and 21-42 are one gate each; chapter 23 has two.
 * tools/fxt_gates_ref.py is the independent Python reference (gate names and sections).
 * G45 perm_sqrt is not a book chapter: it extends chapter 2 (permutation cycle structure).
 *
 * Classes (what a solver needs, not what the fuzzer does):
 *   BIT    GF(2)-affine            G0 G25 G43
 *   RING   integer/modular maps    G1 G2(add, xor) G3 G22-G24 G26-G30 G37
 *   MONO   monotone, bisectable    G31-G34 G35(decreasing) G36(low nibble) G38(per byte)
 *   RANK   structure + rank        G4-G19 G21 G45
 *   HASH   search only             G20 G39-G42 G44 (no gradient)
 *
 * Build (coverage build mirrors targets/gf2_gates.c; -O2, clang):
 *   clang -O2 -g -fsanitize-coverage=trace-pc-guard \
 *     -include src/fuzzer_tool/adapters/afl_shim.c \
 *     -o <out>/fxt_gates targets/fxt_gates.c -ldl
 * Ground-truth library (no shim, no coverage):
 *   cc -O2 -shared -fPIC -DGATES_PLAIN -o <out>/fxt_gates_ref.so targets/fxt_gates.c
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

#define GATES 46
#define INPUT_LEN 204
#define EDGE_BASE 0x7400u
#define VALID_EDGE(g) __afl_map_edge(EDGE_BASE + 2u * (unsigned)(g))
#define SOLVED_EDGE(g) __afl_map_edge(EDGE_BASE + 2u * (unsigned)(g) + 1u)

typedef uint64_t u64;
typedef uint32_t u32;
typedef int64_t i64;
typedef int32_t i32;
typedef unsigned __int128 u128;

/* Gate function: returns 1 and sets *op when the field is valid, else 0. */
typedef int (*gate_fn)(const unsigned char *p, u64 *op);

static inline u32 le16(const unsigned char *p) { return (u32)p[0] | ((u32)p[1] << 8); }
static inline u32 le32(const unsigned char *p) { return le16(p) | (le16(p + 2) << 16); }
static inline u64 le64(const unsigned char *p) { return le32(p) | ((u64)le32(p + 4) << 32); }
static inline i32 s8(unsigned char v) { return (int8_t)v; }

static u64 gcd64(u64 a, u64 b) {
    while (b) { u64 t = a % b; a = b; b = t; }
    return a;
}

static u64 binom(unsigned n, unsigned k) {
    u64 r = 1;
    if (k > n) return 0;
    for (unsigned i = 1; i <= k; i++) r = (u64)((u128)r * (n - k + i) / i);
    return r;
}

static u64 fact(unsigned n) { return n ? n * fact(n - 1) : 1; }

/* Decodes n 3-bit entries of x; returns 1 iff they are a permutation of 0..n-1. */
static int perm3(u32 x, unsigned n, unsigned *p) {
    unsigned seen = 0;
    for (unsigned i = 0; i < n; i++) {
        p[i] = (x >> (3 * i)) & 7u;
        if (p[i] >= n || (seen >> p[i] & 1u)) return 0;
        seen |= 1u << p[i];
    }
    return 1;
}

/* ---- Part I: low level ---- */

static int g_gray(const unsigned char *p, u64 *op) { u32 x = le32(p); *op = x ^ (x >> 1); return 1; }

static int g_popcount(const unsigned char *p, u64 *op) { *op = (u64)__builtin_popcountll(le64(p)); return 1; }

static int g_negabinary(const unsigned char *p, u64 *op) {
    u32 x = le32(p);
    *op = (u32)((x + 0xAAAAAAAAu) ^ 0xAAAAAAAAu);
    return 1;
}

/* Newton iteration for the inverse modulo 2^32, odd x only. */
static int g_inv_mod_2n(const unsigned char *p, u64 *op) {
    u32 x = le32(p), y = x;
    if (!(x & 1u)) return 0;
    for (int i = 0; i < 5; i++) y *= 2u - x * y;
    *op = y;
    return 1;
}

static int g_perm_order(const unsigned char *p, u64 *op) {
    unsigned q[8], seen = 0;
    u64 order = 1;
    if (!perm3(le32(p), 8, q)) return 0;
    for (unsigned s = 0; s < 8; s++) {
        unsigned n = 0, i = s;
        while (!(seen >> i & 1u)) { seen |= 1u << i; i = q[i]; n++; }
        if (n) order = order / gcd64(order, n) * n;
    }
    *op = order;
    return 1;
}

static int g_inversions(const unsigned char *p, u64 *op) {
    u64 n = 0;
    for (int i = 0; i < 8; i++)
        for (int j = i + 1; j < 8; j++) n += p[i] > p[j];
    *op = n;
    return 1;
}

static int g_heap(const unsigned char *p, u64 *op) {
    unsigned sum = 0;
    for (int i = 0; i < 8; i++) {
        sum += p[i];
        if (2 * i + 1 < 8 && p[i] < p[2 * i + 1]) return 0;
        if (2 * i + 2 < 8 && p[i] < p[2 * i + 2]) return 0;
    }
    *op = sum & 0xFFu;
    return 1;
}

/* ---- Part II: combinatorial generation ---- */

/* Colex rank of a 6-subset of {0..31}: sum of C(pos_i, i+1). */
static int g_comb_colex(const unsigned char *p, u64 *op) {
    u32 x = le32(p);
    u64 r = 0;
    unsigned i = 0;
    if (__builtin_popcount(x) != 6) return 0;
    for (unsigned b = 0; b < 32; b++)
        if (x >> b & 1u) r += binom(b, ++i);
    *op = r;
    return 1;
}

/* Product of the parts of the composition of 16 whose cut points are the set bits. */
static int g_composition(const unsigned char *p, u64 *op) {
    u32 x = le16(p) & 0x7FFFu;
    u64 prod = 1;
    int last = -1;
    for (int i = 0; i < 15; i++) {
        if (!(x >> i & 1u)) continue;
        prod *= (u64)(i - last);
        last = i;
    }
    *op = prod * (u64)(15 - last);
    return 1;
}

/* Position of each nibble in the lex-min de Bruijn sequence B(2,4). */
static int g_debruijn(const unsigned char *p, u64 *op) {
    static const unsigned char SEQ[16] = {0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1};
    unsigned char pos[16];
    u32 x = le32(p);
    u64 r = 0;
    for (unsigned t = 0; t < 16; t++) {
        unsigned w = 0;
        for (unsigned k = 0; k < 4; k++) w = (w << 1) | SEQ[(t + k) & 15u];
        pos[w] = (unsigned char)t;
    }
    for (unsigned i = 0; i < 8; i++) r |= (u64)pos[(x >> (4 * i)) & 15u] << (4 * i);
    *op = r;
    return 1;
}

static int g_mixed_radix(const unsigned char *p, u64 *op) {
    static const unsigned R[6] = {2, 3, 4, 5, 6, 7};
    u32 x = le32(p);
    u64 v = 0, w = 1;
    for (unsigned i = 0; i < 6; i++) {
        unsigned d = (x >> (3 * i)) & 7u;
        if (d >= R[i]) return 0;
        v += d * w;
        w *= R[i];
    }
    *op = v;
    return 1;
}

static int g_perm_lex(const unsigned char *p, u64 *op) {
    unsigned q[8];
    u64 r = 0;
    if (!perm3(le32(p), 8, q)) return 0;
    for (unsigned i = 0; i < 8; i++) {
        unsigned smaller = 0;
        for (unsigned j = i + 1; j < 8; j++) smaller += q[j] < q[i];
        r += smaller * fact(7 - i);
    }
    *op = r;
    return 1;
}

/* 1 iff the permutation is a fixed-point-free involution. */
static int g_involution(const unsigned char *p, u64 *op) {
    unsigned q[8], ok = 1;
    if (!perm3(le32(p), 8, q)) return 0;
    for (unsigned i = 0; i < 8; i++) ok &= (q[q[i]] == i) & (q[i] != i);
    *op = ok;
    return 1;
}

static int g_kperm_lex(const unsigned char *p, u64 *op) {
    static const unsigned W[4] = {210, 30, 5, 1};
    u32 x = le16(p);
    unsigned a[4], used = 0;
    u64 r = 0;
    for (unsigned i = 0; i < 4; i++) {
        a[i] = (x >> (3 * i)) & 7u;
        if (used >> a[i] & 1u) return 0;
        used |= 1u << a[i];
    }
    used = 0;
    for (unsigned i = 0; i < 4; i++) {
        unsigned c = 0;
        for (unsigned v = 0; v < a[i]; v++) c += !(used >> v & 1u);
        r += c * W[i];
        used |= 1u << a[i];
    }
    *op = r;
    return 1;
}

static u64 multinomial(const unsigned *cnt) {
    unsigned n = cnt[0] + cnt[1] + cnt[2] + cnt[3];
    u64 r = fact(n);
    for (int c = 0; c < 4; c++) r /= fact(cnt[c]);
    return r;
}

/* Lex rank of an arrangement of {0,0,1,1,2,2,3,3}. */
static int g_multiset_perm(const unsigned char *p, u64 *op) {
    u32 x = le16(p);
    unsigned s[8], cnt[4] = {0, 0, 0, 0};
    u64 r = 0;
    for (unsigned i = 0; i < 8; i++) cnt[s[i] = (x >> (2 * i)) & 3u]++;
    if (cnt[0] != 2 || cnt[1] != 2 || cnt[2] != 2 || cnt[3] != 2) return 0;
    for (unsigned i = 0; i < 8; i++) {
        for (unsigned c = 0; c < s[i]; c++) {
            if (!cnt[c]) continue;
            cnt[c]--;
            r += multinomial(cnt);
            cnt[c]++;
        }
        cnt[s[i]]--;
    }
    *op = r;
    return 1;
}

/* Zeckendorf value of a word with no two adjacent ones. */
static int g_fibonacci(const unsigned char *p, u64 *op) {
    u32 x = le32(p);
    u64 w[32], v = 0;
    if (x & (x >> 1)) return 0;
    w[0] = 1;
    w[1] = 2;
    for (int i = 2; i < 32; i++) w[i] = w[i - 1] + w[i - 2];
    for (int i = 0; i < 32; i++)
        if (x >> i & 1u) v += w[i];
    *op = v;
    return 1;
}

/* Ways to finish rem parentheses from depth to 0 without going below 0.
 * Reflection principle: C(rem, u) - C(rem, u - 1), u = (rem - depth) / 2 closing steps
 * beyond the descent. Closed form: the recursion is exponential on a deep valid prefix. */
static u64 ballot(unsigned rem, unsigned depth) {
    unsigned u;
    if (depth > rem || ((rem - depth) & 1u)) return 0;
    u = (rem - depth) / 2;
    return binom(rem, u) - (u ? binom(rem, u - 1) : 0);
}

/* Lex rank of a Dyck word of length 16, bit 1 = '(' < ')' = bit 0. */
static int g_dyck_lex(const unsigned char *p, u64 *op) {
    u32 x = le16(p);
    int depth = 0;
    u64 r = 0;
    for (unsigned i = 0; i < 16; i++) {
        if (!(x >> i & 1u)) r += ballot(15 - i, (unsigned)depth + 1);
        depth += (x >> i & 1u) ? 1 : -1;
        if (depth < 0) return 0;
    }
    *op = r;
    return depth == 0;
}

/* Sum of squares of a partition of 20 into at most 8 nibble parts, non-increasing. */
static int g_partition(const unsigned char *p, u64 *op) {
    u32 x = le32(p);
    unsigned prev = 15, sum = 0;
    u64 sq = 0;
    for (unsigned i = 0; i < 8; i++) {
        unsigned v = (x >> (4 * i)) & 15u;
        if (v > prev) return 0;
        prev = v;
        sum += v;
        sq += v * v;
    }
    if (sum != 20) return 0;
    *op = sq;
    return 1;
}

static u64 bell_tail(unsigned rem, unsigned k) {
    if (!rem) return 1;
    return k * bell_tail(rem - 1, k) + bell_tail(rem - 1, k + 1);
}

/* Lex rank of a restricted growth string of length 8. */
static int g_set_part_rgs(const unsigned char *p, u64 *op) {
    u32 x = le32(p);
    unsigned k = 0;
    u64 r = 0;
    for (unsigned i = 0; i < 8; i++) {
        unsigned a = (x >> (3 * i)) & 7u;
        if (a > k) return 0;
        for (unsigned v = 0; v < a; v++) r += bell_tail(7 - i, v == k ? k + 1 : k);
        if (a == k) k++;
    }
    *op = r;
    return 1;
}

/* Necklace representative: smallest rotation of a 24-bit word. */
static int g_necklace(const unsigned char *p, u64 *op) {
    u32 x = le32(p) & 0xFFFFFFu, best = x;
    for (unsigned r = 1; r < 24; r++) {
        u32 y = ((x << r) | (x >> (24 - r))) & 0xFFFFFFu;
        if (y < best) best = y;
    }
    *op = best;
    return 1;
}

/* Rows of the order-16 Hadamard matrix from the m-sequence s[t+4] = s[t+1]^s[t]. */
static void hadamard16(int8_t h[16][16]) {
    unsigned char s[15] = {1, 0, 0, 0};
    for (int t = 4; t < 15; t++) s[t] = s[t - 3] ^ s[t - 4];
    for (int j = 0; j < 16; j++) h[0][j] = 1;
    for (int i = 0; i < 15; i++) {
        h[i + 1][0] = 1;
        for (int j = 0; j < 15; j++) h[i + 1][j + 1] = s[(i + j) % 15] ? -1 : 1;
    }
}

/* Largest |correlation| of the +-1 vector of the field with a Hadamard row. */
static int g_hadamard_lfsr(const unsigned char *p, u64 *op) {
    int8_t h[16][16];
    u32 x = le16(p);
    int best = 0;
    hadamard16(h);
    for (int i = 0; i < 16; i++) {
        int dot = 0;
        for (int j = 0; j < 16; j++) dot += h[i][j] * ((x >> j & 1u) ? -1 : 1);
        if (dot < 0) dot = -dot;
        if (dot > best) best = dot;
    }
    *op = (u64)best;
    return 1;
}

/* Tail and cycle length of the walk from 0 in a successor table on 6 vertices. */
static int g_rho_path(const unsigned char *p, u64 *op) {
    u32 x = le32(p);
    unsigned succ[6], first[6] = {~0u, ~0u, ~0u, ~0u, ~0u, ~0u}, v = 0, t = 0;
    for (unsigned i = 0; i < 6; i++)
        if ((succ[i] = (x >> (3 * i)) & 7u) >= 6) return 0;
    while (first[v] == ~0u) { first[v] = t++; v = succ[v]; }
    *op = ((u64)first[v] << 8) | (t - first[v]);
    return 1;
}

/* ---- Part III: fast transforms ---- */

/* Length-4 complex DFT of int8 pairs: re(X1) and im(X2). */
static int g_dft4(const unsigned char *p, u64 *op) {
    i32 re[4], im[4];
    for (int j = 0; j < 4; j++) { re[j] = s8(p[2 * j]); im[j] = s8(p[2 * j + 1]); }
    i32 x1re = re[0] + im[1] - re[2] - im[3];
    i32 x2im = im[0] - im[1] + im[2] - im[3];
    *op = ((u64)(u32)(x1re & 0xFFFF) << 16) | (u32)(x2im & 0xFFFF);
    return 1;
}

/* Cyclic convolution of two 4-byte sequences, entry 1. */
static int g_convolution(const unsigned char *p, u64 *op) {
    const unsigned char *a = p, *b = p + 4;
    *op = (u64)a[0] * b[1] + (u64)a[1] * b[0] + (u64)a[2] * b[3] + (u64)a[3] * b[2];
    return 1;
}

/* Walsh-Hadamard transform (natural order) of 8 int8: entries 3 and 6. */
static int g_walsh(const unsigned char *p, u64 *op) {
    i32 y[8];
    for (int i = 0; i < 8; i++) y[i] = s8(p[i]);
    for (int len = 1; len < 8; len <<= 1)
        for (int i = 0; i < 8; i += 2 * len)
            for (int j = i; j < i + len; j++) {
                i32 a = y[j], b = y[j + len];
                y[j] = a + b;
                y[j + len] = a - b;
            }
    *op = ((u64)(u32)(y[3] & 0xFFFF) << 16) | (u32)(y[6] & 0xFFFF);
    return 1;
}

/* Reed-Muller (algebraic normal form) transform of a 16-entry truth table. */
static int g_reed_muller(const unsigned char *p, u64 *op) {
    u32 f = le16(p);
    for (int i = 0; i < 4; i++)
        for (u32 m = 0; m < 16; m++)
            if (m >> i & 1u) f ^= ((f >> (m ^ (1u << i))) & 1u) << m;
    *op = f;
    return 1;
}

/* Non-normalized Haar transform of 8 bytes: total sum and the last level-1 difference. */
static int g_haar(const unsigned char *p, u64 *op) {
    i32 x[8], h[8];
    for (int i = 0; i < 8; i++) x[i] = p[i];
    for (int n = 8; n > 1; n >>= 1) {
        for (int i = 0; i < n / 2; i++) {
            h[n / 2 + i] = x[2 * i] - x[2 * i + 1];
            x[i] = x[2 * i] + x[2 * i + 1];
        }
    }
    h[0] = x[0];
    *op = ((u64)(u32)h[0] << 16) | (u32)(h[7] & 0xFFFF);
    return 1;
}

/* Discrete Hartley transform of 4 int8 (cas kernel is +-1): H1 * H2. */
static int g_hartley4(const unsigned char *p, u64 *op) {
    static const int CAS[4] = {1, 1, -1, -1};
    i32 h[4];
    for (int k = 0; k < 4; k++) {
        h[k] = 0;
        for (int j = 0; j < 4; j++) h[k] += s8(p[j]) * CAS[(j * k) & 3];
    }
    *op = (u32)(h[1] * h[2]);
    return 1;
}

/* Length-4 NTT modulo 257 (root 16), all four outputs packed. */
static int g_ntt4(const unsigned char *p, u64 *op) {
    u64 r = 0;
    for (unsigned k = 0; k < 4; k++) {
        u32 acc = 0, w = 1, step = 1;
        for (unsigned i = 0; i < (k & 3u); i++) step = step * 16u % 257u;
        for (unsigned j = 0; j < 4; j++) { acc = (acc + p[j] * w) % 257u; w = w * step % 257u; }
        r |= (u64)acc << (9 * k);
    }
    *op = r;
    return 1;
}

/* Integer (S-transform) wavelet: final average and one difference per level. */
static int g_wavelet_s(const unsigned char *p, u64 *op) {
    i32 x[8], d1[4], d2[2], d3, n = 8;
    for (int i = 0; i < 8; i++) x[i] = p[i];
    for (int i = 0; i < n / 2; i++) { d1[i] = x[2 * i] - x[2 * i + 1]; x[i] = (x[2 * i] + x[2 * i + 1]) >> 1; }
    n = 4;
    for (int i = 0; i < n / 2; i++) { d2[i] = x[2 * i] - x[2 * i + 1]; x[i] = (x[2 * i] + x[2 * i + 1]) >> 1; }
    d3 = x[0] - x[1];
    x[0] = (x[0] + x[1]) >> 1;
    *op = (u32)x[0] | ((u64)(u32)(d3 & 0xFF) << 8) | ((u64)(u32)(d2[0] & 0xFF) << 16) |
          ((u64)(u32)(d1[3] & 0xFF) << 24);
    return 1;
}

/* ---- Part IV: fast arithmetic ---- */

/* Casting out 255 and 257: residues of a*b. */
static int g_cast_out(const unsigned char *p, u64 *op) {
    u64 m = (u64)le32(p) * le32(p + 4);
    *op = ((m % 255u) << 9) | (m % 257u);
    return 1;
}

static u64 isqrt64(u64 x) {
    u64 r = 0, bit = 1ull << 62;
    while (bit > x) bit >>= 2;
    while (bit) {
        if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; }
        else r >>= 1;
        bit >>= 2;
    }
    return r;
}

static int g_isqrt(const unsigned char *p, u64 *op) { *op = isqrt64(le64(p)); return 1; }

/* Third Newton iterate for sqrt(n) started at n. */
static int g_newton(const unsigned char *p, u64 *op) {
    u64 n = le32(p), x = n;
    if (!n) return 0;
    for (int i = 0; i < 3; i++) x = (x + n / x) >> 1;
    *op = x;
    return 1;
}

/* Integer AGM of two non-zero 32-bit words. */
static int g_agm(const unsigned char *p, u64 *op) {
    u64 a = le32(p), b = le32(p + 4);
    if (!a || !b) return 0;
    for (int i = 0; i < 64; i++) {
        u64 d = a > b ? a - b : b - a;
        if (d <= 1) break;
        u64 na = (a + b) >> 1, nb = isqrt64(a * b);
        a = na;
        b = nb;
    }
    *op = (a + b) >> 1;
    return 1;
}

/* log2 in Q16 by repeated squaring of the mantissa. */
static int g_log2_q16(const unsigned char *p, u64 *op) {
    u32 x = le32(p);
    u64 m, frac = 0;
    unsigned n;
    if (!x) return 0;
    n = 31u - (unsigned)__builtin_clz(x);
    m = (u64)(u32)(x << (31u - n));
    for (int i = 0; i < 16; i++) {
        unsigned bit = 0;
        m = (m * m) >> 31;
        if (m >> 32) { m >>= 1; bit = 1; }
        frac = (frac << 1) | bit;
    }
    *op = ((u64)n << 16) | frac;
    return 1;
}

/* 16-step CORDIC rotation in Q16; angle is the field in radians/65536. */
static int g_cordic(const unsigned char *p, u64 *op) {
    static const i32 ATAN[16] = {51472, 30386, 16055, 8150, 4091, 2047, 1024, 512,
                                 256,   128,   64,    32,   16,   8,    4,    2};
    i32 z = (i32)le16(p), x = 39797, y = 0;
    for (int i = 0; i < 16; i++) {
        i32 d = z >= 0 ? 1 : -1, nx = x - d * (y >> i), ny = y + d * (x >> i);
        x = nx;
        y = ny;
        z -= d * ATAN[i];
    }
    *op = ((u64)(u32)(x & 0xFFFF) << 16) | (u32)(y & 0xFFFF);
    return 1;
}

/* Numerator of the partial sum of 1/k! to n terms over the common denominator n!. */
static int g_binary_split(const unsigned char *p, u64 *op) {
    unsigned n = p[0] & 15u;
    u64 s = 0;
    for (unsigned k = 0; k <= n; k++) s += fact(n) / fact(k);
    *op = s;
    return 1;
}

/* Chebyshev T_n(x) modulo 2^32 by the three-term recurrence. */
static int g_chebyshev(const unsigned char *p, u64 *op) {
    unsigned n = p[0];
    u32 x = le16(p + 1), t0 = 1, t1 = x;
    if (!n) { *op = 1; return 1; }
    for (unsigned i = 1; i < n; i++) { u32 t = 2u * x * t1 - t0; t0 = t1; t1 = t; }
    *op = t1;
    return 1;
}

/* Vandermonde sum of C(n,k)C(m,k) from the 2F1 term ratio; equals C(n+m, n). */
static int g_vandermonde(const unsigned char *p, u64 *op) {
    unsigned n = p[0], m = p[1];
    u64 t = 1, s = 1;
    if (n > 33 || m > 33) return 0;
    for (unsigned k = 0; k < n && k < m; k++) {
        t = (u64)((u128)t * (n - k) * (m - k) / ((u128)(k + 1) * (k + 1)));
        s += t;
    }
    *op = s;
    return 1;
}

/* Number of Euclid steps (continued fraction length) of (p, q). */
static int g_euclid_steps(const unsigned char *p, u64 *op) {
    u32 a = le16(p), b = le16(p + 2), steps = 0;
    if (!b) return 0;
    while (b) { u32 t = a % b; a = b; b = t; steps++; }
    *op = steps;
    return 1;
}

/* Sum of bit j's weight 2^ceil(j/2) over the set bits of m (m < 2^16): gather the
 * even bits (weight 2^i at j = 2i) and the odd bits (2^(i+1) at j = 2i+1). */
static u32 even_bits(u32 x) {
    x &= 0x5555u;
    x = (x | x >> 1) & 0x3333u;
    x = (x | x >> 2) & 0x0F0Fu;
    return (x | x >> 4) & 0x00FFu;
}

static i32 rs_weight(u32 m) { return (i32)(even_bits(m) + 2u * even_bits(m >> 1)); }

/* Partial sum of the Golay-Rudin-Shapiro sequence over k < n, n < 2^16, branch-free.
 * P_{j+1} = P_j + x^(2^j) Q_j, Q_{j+1} = P_j - x^(2^j) Q_j, P_j(1) = 2^ceil(j/2): each
 * set bit j of n adds one whole P_j block, negated when the bits above j hold an odd
 * number of adjacent 11 pairs. A suffix XOR of n & n>>1 gives those parities at once.
 * The O(n) loop was 29 of 31.5 us per input; a load-time table re-ran in every
 * forkserver child (the shim's constructor runs first and forks from inside it). */
static int g_rudin_shapiro(const unsigned char *p, u64 *op) {
    u32 n = le16(p), par = n & (n >> 1);
    par ^= par >> 1;
    par ^= par >> 2;
    par ^= par >> 4;
    par ^= par >> 8;
    u32 neg = n & (par >> 1);
    *op = (u32)(rs_weight(n & ~neg) - rs_weight(neg));
    return 1;
}

/* ---- Part V: finite fields ---- */

/* a^((p+1)/4) mod p for p = 2^31 - 1, the square-root exponent. */
static int g_modexp_sqrt(const unsigned char *p, u64 *op) {
    const u64 P = 0x7FFFFFFFull;
    u64 a = le32(p) % P;
    for (int i = 0; i < 29; i++) a = a * a % P;
    *op = a;
    return 1;
}

static u32 pmulmod(u32 a, u32 b, u32 poly) {
    u32 r = 0;
    while (b) {
        if (b & 1u) r ^= a;
        b >>= 1;
        a <<= 1;
        if (a >> 15 & 1u) a ^= poly;
    }
    return r;
}

static u32 ppow(u32 e, u32 poly) {
    u32 r = 1, base = 2;
    while (e) {
        if (e & 1u) r = pmulmod(r, base, poly);
        base = pmulmod(base, base, poly);
        e >>= 1;
    }
    return r;
}

/* 1 iff x is a primitive polynomial of degree 15 (order of x is 2^15 - 1). */
static int g_primitive_poly(const unsigned char *p, u64 *op) {
    static const u32 Q[3] = {7, 31, 151};
    u32 poly = le16(p);
    if (!(poly >> 15 & 1u) || !(poly & 1u)) return 0;
    *op = 0;
    if (ppow(32767, poly) != 1) return 1;
    for (int i = 0; i < 3; i++)
        if (ppow(32767 / Q[i], poly) == 1) return 1;
    *op = 1;
    return 1;
}

/* CRC-16/CCITT-FALSE over 6 bytes. */
static int g_crc16(const unsigned char *p, u64 *op) {
    u32 crc = 0xFFFFu;
    for (int i = 0; i < 6; i++) {
        crc ^= (u32)p[i] << 8;
        for (int b = 0; b < 8; b++) crc = (crc & 0x8000u) ? ((crc << 1) ^ 0x1021u) & 0xFFFFu : (crc << 1) & 0xFFFFu;
    }
    *op = crc;
    return 1;
}

static u32 gmul(u32 a, u32 b) {
    u32 r = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1u) r ^= a;
        b >>= 1;
        a <<= 1;
        if (a & 0x100u) a ^= 0x11Bu;
    }
    return r;
}

/* 1 iff b is the inverse of a in GF(2^8) mod 0x11B and the trace of a is 1. */
static int g_gf256_trace(const unsigned char *p, u64 *op) {
    u32 a = p[0], t = a, tr = 0;
    if (!a) return 0;
    for (int i = 0; i < 8; i++) { tr ^= t; t = gmul(t, t); }
    *op = (gmul(a, p[1]) == 1u) & (tr == 1u);
    return 1;
}

/* Number of permutations q of 0..7 with q*q == p (cycle-type closed form).
 * Even k: m cycles pair up, (m-1)!! * k^(m/2), 0 when m is odd.
 * Odd k: sum_j m! / ((m-2j)! j! 2^j) * k^j over j merged pairs. */
static int g_perm_sqrt(const unsigned char *p, u64 *op) {
    unsigned q[8], seen = 0, m[9] = {0};
    u64 total = 1;
    if (!perm3(le32(p), 8, q)) return 0;
    for (unsigned s = 0; s < 8; s++) {
        unsigned n = 0, i = s;
        while (!(seen >> i & 1u)) { seen |= 1u << i; i = q[i]; n++; }
        m[n]++;
    }
    for (unsigned k = 1; k <= 8; k++) {
        u64 f = 0;
        if (!m[k]) continue;
        if (!(k & 1u)) {
            if (m[k] & 1u) { total = 0; break; }
            f = 1;
            for (unsigned c = m[k] - 1; c > 1; c -= 2) f *= c;
            for (unsigned j = 0; j < m[k] / 2; j++) f *= k;
        } else {
            for (unsigned j = 0; 2 * j <= m[k]; j++) {
                u64 t = fact(m[k]) / (fact(m[k] - 2 * j) * fact(j) << j);
                for (unsigned c = 0; c < j; c++) t *= k;
                f += t;
            }
        }
        total *= f;
    }
    *op = total;
    return 1;
}

static const gate_fn GATE_FN[GATES] = {
    g_gray, g_popcount, g_negabinary, g_inv_mod_2n, g_perm_order, g_inversions, g_heap,
    g_comb_colex, g_composition, g_debruijn, g_mixed_radix, g_perm_lex, g_involution,
    g_kperm_lex, g_multiset_perm, g_fibonacci, g_dyck_lex, g_partition, g_set_part_rgs,
    g_necklace, g_hadamard_lfsr, g_rho_path, g_dft4, g_convolution, g_walsh, g_reed_muller,
    g_haar, g_hartley4, g_ntt4, g_wavelet_s, g_cast_out, g_isqrt, g_newton, g_agm,
    g_log2_q16, g_cordic, g_binary_split, g_chebyshev, g_vandermonde, g_euclid_steps,
    g_rudin_shapiro, g_modexp_sqrt, g_primitive_poly, g_crc16, g_gf256_trace,
    g_perm_sqrt,
};

/* Field length in bytes; offsets are cumulative and sum to INPUT_LEN. */
static const unsigned char GATE_LEN[GATES] = {
    4, 8, 4, 4, 4, 8, 8, 4, 2, 4, 4, 4, 4, 2, 2, 4, 2, 4, 4, 4, 2, 4, 8, 8, 8, 2, 8, 4, 4,
    8, 8, 8, 4, 8, 4, 2, 1, 3, 2, 4, 2, 4, 2, 6, 2, 4,
};

/* Gates whose structural precondition fires a separate "valid" edge. */
static const unsigned char GATE_STAGED[GATES] = {
    0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1,
};

static const u64 g_target[GATES] = {
    0x7722DD91ull, 0x29ull, 0x1C10032ull, 0x5C2418B3ull,
    0xFull, 0x1Cull, 0x58ull, 0x97973ull,
    0x144ull, 0xE07D1253ull, 0x13AFull, 0x3EE4ull,
    0x1ull, 0x46Full, 0xD1ull, 0x3DEA0Bull,
    0x2DEull, 0x44ull, 0x51Dull, 0xA5C3ull,
    0x10ull, 0x204ull, 0xFFB2FF8Full, 0x54DDull,
    0x7BFEFBull, 0xCFF1ull, 0x3570044ull, 0x28Bull,
    0x4D90D04D2ull, 0xF3BF4A68ull, 0xF0DCull, 0x44444444ull,
    0x13C6EF39ull, 0x1EF06473ull, 0x1BB5BCull, 0xF8073F58ull,
    0x677A818ull, 0xD3EED06Full, 0x72363EAull, 0x15ull,
    0xEDull, 0x4532032Eull, 0x1ull, 0xD78Cull,
    0x1ull, 0x1Aull,
};

/* One solving input: the field of every gate set to a value that opens it. */
__attribute__((unused)) static const unsigned char g_witness[INPUT_LEN] = {
    0xE1, 0x96, 0x3C, 0x5A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0xEE, 0xFF, 0xC0, 0x00,
    0x7B, 0x56, 0x34, 0x12, 0x11, 0x58, 0x7F, 0x00, 0xF0, 0xD0, 0xB0, 0x90, 0x70, 0x50, 0x30, 0x10,
    0xF0, 0xC8, 0xB0, 0x90, 0x80, 0x70, 0x60, 0x10, 0x12, 0x22, 0x40, 0x40, 0x24, 0x09, 0x34, 0x12,
    0xDE, 0xC0, 0xD1, 0x58, 0x03, 0x00, 0x0B, 0xF1, 0xCA, 0x00, 0xC1, 0x54, 0xDE, 0x00, 0xD5, 0x01,
    0xE4, 0x1B, 0x25, 0x42, 0x10, 0x88, 0x1B, 0x4D, 0x45, 0x33, 0x22, 0x01, 0x08, 0x94, 0x41, 0x00,
    0xC3, 0xA5, 0x00, 0x00, 0xC8, 0x7A, 0xD1, 0x58, 0x01, 0x00, 0x0A, 0xEC, 0x1E, 0x05, 0xF9, 0x0C,
    0xFF, 0x64, 0x11, 0xC8, 0x21, 0x05, 0x5B, 0x80, 0xFA, 0x03, 0x0C, 0xA6, 0x2D, 0x07, 0xDF, 0x64,
    0x80, 0x40, 0xEF, 0xBE, 0xC8, 0x0D, 0x4D, 0xB4, 0x05, 0xFA, 0x63, 0x1F, 0x0A, 0xE2, 0x4D, 0x9C,
    0x0C, 0xC8, 0xFF, 0x00, 0xC9, 0x11, 0x63, 0xFA, 0x03, 0x80, 0x40, 0x4D, 0x40, 0xE2, 0x01, 0x00,
    0x40, 0x42, 0x0F, 0x00, 0xEF, 0xCD, 0xAB, 0x90, 0x78, 0x56, 0x34, 0x12, 0xB9, 0x79, 0x37, 0x9E,
    0x34, 0x12, 0xFF, 0x00, 0x34, 0x12, 0x00, 0x7A, 0x5E, 0xEA, 0x15, 0x0D, 0x00, 0x40, 0xAB, 0x0D,
    0xEF, 0xBE, 0x11, 0x0D, 0xF1, 0x6F, 0x2F, 0x45, 0x5B, 0x4A, 0x3C, 0x1B, 0x6A, 0x2F, 0x03, 0x80,
    0x46, 0x58, 0x54, 0x42, 0x4B, 0x21, 0x20, 0x3A,
    0x11, 0xC6, 0xFA, 0x00,
};

/* Operand of each gate on the last fuzz_gates() call (SENTINEL when invalid). */
static u64 g_operand[GATES];
#define OPERAND_INVALID (~0ull)

__attribute__((visibility("default")))
u64 fuzz_gates(const unsigned char *buf, size_t len) {
    u64 mask = 0;
    size_t off = 0;
    for (int g = 0; g < GATES; g++) g_operand[g] = OPERAND_INVALID;
    if (len < INPUT_LEN) return 0;

    for (int g = 0; g < GATES; g++) {
        u64 op = 0;
        const unsigned char *field = buf + off;
        off += GATE_LEN[g];
        if (!GATE_FN[g](field, &op)) continue;
        g_operand[g] = op;
        if (GATE_STAGED[g]) VALID_EDGE(g);
        if (op != g_target[g]) continue;
        SOLVED_EDGE(g);
        mask |= 1ull << g;
    }
    return mask;
}

#ifdef GATES_PLAIN
__attribute__((visibility("default")))
const u64 *gates_operand(void) { return g_operand; }

__attribute__((visibility("default")))
const u64 *gates_target(void) { return g_target; }

__attribute__((visibility("default")))
const unsigned char *gates_witness(void) { return g_witness; }
#endif

__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *buf, size_t size) {
    (void)fuzz_gates(buf, size);
    return 0;
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
    (void)fuzz_gates(buf, n);
    return 0;
}
#endif
