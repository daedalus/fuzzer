// bithacks_diff.c - differential tester for graphics.stanford.edu/~seander/bithacks.html
// ./bithacks_diff exh [lg]      : sweep, prints failing hacks
// ./bithacks_diff [fuzz] < input : fuzz entry (default, no args), aborts on mismatch/UB
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <stdarg.h>
#include <math.h>
#include <unistd.h>

typedef unsigned __int128 u128;
#define MAXID 400
static const char *names[MAXID];
static unsigned long fails[MAXID];
static char first[MAXID][200];
static int fuzzmode = 0;
static int nids = 0;

static void record(int id, const char *fmt, ...) {
    if (fails[id]++ == 0) {
        va_list ap; va_start(ap, fmt);
        vsnprintf(first[id], sizeof first[id], fmt, ap);
        va_end(ap);
    }
    if (fuzzmode && !strstr(names[id],"expect") && !strstr(names[id],"caveat") && !strstr(names[id],"VIOLATED") && !strstr(names[id],"OUTSIDE") && !strstr(names[id],"m&-((signed)")) { fprintf(stderr, "MISMATCH %s: %s\n", names[id], first[id]); abort(); }
}
#define CHK(nm, cond, ...) do { enum { ID = __COUNTER__ }; names[ID] = nm; if (ID + 1 > nids) nids = ID + 1; \
    if (!(cond)) record(ID, __VA_ARGS__); } while (0)

static inline uint32_t h32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
static inline uint64_t h64(uint64_t x) { x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL; x ^= x >> 31; return x; }

// ---- tables copied from the page ----
static const unsigned char BitsSetTable256[256] = {
#define B2(n) n, n+1, n+1, n+2
#define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
#define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
    B6(0), B6(1), B6(1), B6(2) };
static const bool ParityTable256[256] = {
#define P2(n) n, n^1, n^1, n
#define P4(n) P2(n), P2(n^1), P2(n^1), P2(n)
#define P6(n) P4(n), P4(n^1), P4(n^1), P4(n)
    P6(0), P6(1), P6(1), P6(0) };
static const unsigned char BitReverseTable256[256] = {
#define R2(n) n, n + 2*64, n + 1*64, n + 3*64
#define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
    R6(0), R6(2), R6(1), R6(3) };
static const char LogTable256[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7) };
static const int DeBruijnLog[32] = {0,9,1,10,13,21,2,29,11,14,16,18,22,25,3,30,8,12,20,28,15,17,24,7,19,27,23,6,26,5,4,31};
static const int DeBruijnPow2[32] = {0,1,28,2,29,14,24,3,30,22,20,15,25,17,4,8,31,27,13,23,21,19,16,7,26,12,18,6,11,5,10,9};
static const int Mod37BitPosition[] = {32,0,1,26,2,23,27,0,3,16,24,30,28,11,0,13,4,7,17,0,25,22,31,15,29,10,12,6,0,21,14,9,5,20,8,19,18};
static const unsigned short MortonTable256[256] = {
  0x0000,0x0001,0x0004,0x0005,0x0010,0x0011,0x0014,0x0015,0x0040,0x0041,0x0044,0x0045,0x0050,0x0051,0x0054,0x0055,
  0x0100,0x0101,0x0104,0x0105,0x0110,0x0111,0x0114,0x0115,0x0140,0x0141,0x0144,0x0145,0x0150,0x0151,0x0154,0x0155,
  0x0400,0x0401,0x0404,0x0405,0x0410,0x0411,0x0414,0x0415,0x0440,0x0441,0x0444,0x0445,0x0450,0x0451,0x0454,0x0455,
  0x0500,0x0501,0x0504,0x0505,0x0510,0x0511,0x0514,0x0515,0x0540,0x0541,0x0544,0x0545,0x0550,0x0551,0x0554,0x0555,
  0x1000,0x1001,0x1004,0x1005,0x1010,0x1011,0x1014,0x1015,0x1040,0x1041,0x1044,0x1045,0x1050,0x1051,0x1054,0x1055,
  0x1100,0x1101,0x1104,0x1105,0x1110,0x1111,0x1114,0x1115,0x1140,0x1141,0x1144,0x1145,0x1150,0x1151,0x1154,0x1155,
  0x1400,0x1401,0x1404,0x1405,0x1410,0x1411,0x1414,0x1415,0x1440,0x1441,0x1444,0x1445,0x1450,0x1451,0x1454,0x1455,
  0x1500,0x1501,0x1504,0x1505,0x1510,0x1511,0x1514,0x1515,0x1540,0x1541,0x1544,0x1545,0x1550,0x1551,0x1554,0x1555,
  0x4000,0x4001,0x4004,0x4005,0x4010,0x4011,0x4014,0x4015,0x4040,0x4041,0x4044,0x4045,0x4050,0x4051,0x4054,0x4055,
  0x4100,0x4101,0x4104,0x4105,0x4110,0x4111,0x4114,0x4115,0x4140,0x4141,0x4144,0x4145,0x4150,0x4151,0x4154,0x4155,
  0x4400,0x4401,0x4404,0x4405,0x4410,0x4411,0x4414,0x4415,0x4440,0x4441,0x4444,0x4445,0x4450,0x4451,0x4454,0x4455,
  0x4500,0x4501,0x4504,0x4505,0x4510,0x4511,0x4514,0x4515,0x4540,0x4541,0x4544,0x4545,0x4550,0x4551,0x4554,0x4555,
  0x5000,0x5001,0x5004,0x5005,0x5010,0x5011,0x5014,0x5015,0x5040,0x5041,0x5044,0x5045,0x5050,0x5051,0x5054,0x5055,
  0x5100,0x5101,0x5104,0x5105,0x5110,0x5111,0x5114,0x5115,0x5140,0x5141,0x5144,0x5145,0x5150,0x5151,0x5154,0x5155,
  0x5400,0x5401,0x5404,0x5405,0x5410,0x5411,0x5414,0x5415,0x5440,0x5441,0x5444,0x5445,0x5450,0x5451,0x5454,0x5455,
  0x5500,0x5501,0x5504,0x5505,0x5510,0x5511,0x5514,0x5515,0x5540,0x5541,0x5544,0x5545,0x5550,0x5551,0x5554,0x5555 };

static const unsigned int PowersOf10[] = {1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000};
static int multipliers[33], divisors[33];
static void init_tables(void) {
    for (int b = 0; b <= 32; b++) {
        unsigned M = b ? (1U << (32 - b)) : 0;
        multipliers[b] = (int)M;
        divisors[b] = (int)M;
    }
    multipliers[0] = 0; divisors[0] = 1; divisors[1] = (int)~(1U << 31);
}

// mod (1<<s)-1 tables
static const unsigned int Mm[] = {0x00000000,0x55555555,0x33333333,0xc71c71c7,0x0f0f0f0f,0xc1f07c1f,0x3f03f03f,0xf01fc07f,
  0x00ff00ff,0x07fc01ff,0x3ff003ff,0xffc007ff,0xff000fff,0xfc001fff,0xf0003fff,0xc0007fff,
  0x0000ffff,0x0001ffff,0x0003ffff,0x0007ffff,0x000fffff,0x001fffff,0x003fffff,0x007fffff,
  0x00ffffff,0x01ffffff,0x03ffffff,0x07ffffff,0x0fffffff,0x1fffffff,0x3fffffff,0x7fffffff};
static const unsigned int Qm[][6] = {
 {0,0,0,0,0,0},{16,8,4,2,1,1},{16,8,4,2,2,2},{15,6,3,3,3,3},{16,8,4,4,4,4},{15,5,5,5,5,5},
 {12,6,6,6,6,6},{14,7,7,7,7,7},{16,8,8,8,8,8},{9,9,9,9,9,9},{10,10,10,10,10,10},{11,11,11,11,11,11},
 {12,12,12,12,12,12},{13,13,13,13,13,13},{14,14,14,14,14,14},{15,15,15,15,15,15},{16,16,16,16,16,16},
 {17,17,17,17,17,17},{18,18,18,18,18,18},{19,19,19,19,19,19},{20,20,20,20,20,20},{21,21,21,21,21,21},
 {22,22,22,22,22,22},{23,23,23,23,23,23},{24,24,24,24,24,24},{25,25,25,25,25,25},{26,26,26,26,26,26},
 {27,27,27,27,27,27},{28,28,28,28,28,28},{29,29,29,29,29,29},{30,30,30,30,30,30},{31,31,31,31,31,31}};
static unsigned int Rm[32][6];
static void init_R(void) {
    static const unsigned first_rows[8][6] = {
     {0,0,0,0,0,0},{0xffff,0xff,0xf,3,1,1},{0xffff,0xff,0xf,3,3,3},{0x7fff,0x3f,7,7,7,7},
     {0xffff,0xff,0xf,0xf,0xf,0xf},{0x7fff,0x1f,0x1f,0x1f,0x1f,0x1f},{0xfff,0x3f,0x3f,0x3f,0x3f,0x3f},{0x3fff,0x7f,0x7f,0x7f,0x7f,0x7f}};
    for (int i = 0; i < 8; i++) memcpy(Rm[i], first_rows[i], sizeof Rm[i]);
    for (int s = 8; s < 32; s++) {
        unsigned v = (s == 8) ? 0xffff : (s == 9 ? 0x1ff : (s < 16 ? ((1u << s) - 1) : ((1u << s) - 1)));
        if (s == 8) { unsigned r[6] = {0xffff,0xff,0xff,0xff,0xff,0xff}; memcpy(Rm[s], r, sizeof r); continue; }
        for (int k = 0; k < 6; k++) Rm[s][k] = v;
    }
}

// ---- generic popcount template (page's "arbitrary width" version) ----
#define GENPOP(T, fname) static unsigned fname(T v) { T c; \
    v = v - ((v >> 1) & (T)~(T)0/3); \
    v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3); \
    v = (v + (v >> 4)) & (T)~(T)0/255*15; \
    c = (T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * CHAR_BIT; return (unsigned)c; }
GENPOP(uint8_t, pop8g) GENPOP(uint16_t, pop16g) GENPOP(uint32_t, pop32g) GENPOP(uint64_t, pop64g) GENPOP(u128, pop128g)

#define SWAP_ADD(a, b) ((&(a) == &(b)) || (((a) -= (b)), ((b) += (a)), ((a) = (b) - (a))))
#define SWAP_XOR(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))

#define haszero(v) (((v) - 0x01010101UL) & ~(v) & 0x80808080UL)
#define hasvalue(x,n) (haszero((x) ^ (~0UL/255 * (n))))
#define hasless(x,n) (((x)-~0UL/255*(n))&~(x)&~0UL/255*128)
#define countless(x,n) (((~0UL/255*(127+(n))-((x)&~0UL/255*127))&~(x)&~0UL/255*128)/128%255)
#define hasmore(x,n) (((x)+~0UL/255*(127-(n))|(x))&~0UL/255*128)
#define countmore(x,n) (((((x)&~0UL/255*127)+~0UL/255*(127-(n))|(x))&~0UL/255*128)/128%255)
#define likelyhasbetween(x,m,n) ((((x)-~0UL/255*(n))&~(x)&((x)&~0UL/255*127)+~0UL/255*(127-(m)))&~0UL/255*128)
#define hasbetween(x,m,n) ((~0UL/255*(127+(n))-((x)&~0UL/255*127)&~(x)&((x)&~0UL/255*127)+~0UL/255*(127-(m)))&~0UL/255*128)
#define countbetween(x,m,n) (hasbetween(x,m,n)/128%255)

static int ref_log2(uint32_t v) { return v ? 31 - __builtin_clz(v) : -1; }
static uint32_t ref_rev32(uint32_t v) { uint32_t r = 0; for (int i = 0; i < 32; i++) r |= ((v >> i) & 1u) << (31 - i); return r; }
static uint32_t ref_spread16(uint32_t x) { uint32_t z = 0; for (int i = 0; i < 16; i++) z |= ((x >> i) & 1u) << (2 * i); return z; }

// ============ 32-bit checks ============
static void check32(uint32_t v) {
    uint32_t w = h32(v), p = h32(w);
    int sv = (int)v, sw = (int)w;
    int sgn = (sv > 0) - (sv < 0);

    CHK("sign: v>>31 (arith)", (sv >> 31) == (sv < 0 ? -1 : 0), "v=%08x", v);
    CHK("sign: -(int)((unsigned)v>>31)", -(int)((unsigned)sv >> 31) == (sv < 0 ? -1 : 0), "v=%08x", v);
    CHK("sign: +1|(v>>31)", (+1 | (sv >> 31)) == (sv < 0 ? -1 : 1), "v=%08x", v);
    CHK("sign: (v!=0)|-(int)(u>>31)", ((sv != 0) | -(int)((unsigned)sv >> 31)) == sgn, "v=%08x", v);
    CHK("sign: (v!=0)|(v>>31)", ((sv != 0) | (sv >> 31)) == sgn, "v=%08x", v);
    CHK("sign: nonneg 1^(u>>31)", (1 ^ ((unsigned)sv >> 31)) == (sv >= 0), "v=%08x", v);
    CHK("opposite signs (x^y)<0", ((sv ^ sw) < 0) == ((sv < 0) != (sw < 0)), "x=%08x y=%08x", v, w);

    { int mask = sv >> 31; unsigned ref = sv < 0 ? -(unsigned)sv : (unsigned)sv;
      CHK("abs: (v+mask)^mask", (unsigned)((sv + mask) ^ mask) == ref, "v=%08x", v);
      CHK("abs: (v^mask)-mask", (unsigned)((sv ^ mask) - mask) == ref, "v=%08x", v); }

    { int mn = sv < sw ? sv : sw, mx = sv < sw ? sw : sv;
      CHK("min: y^((x^y)&-(x<y))", (sw ^ ((sv ^ sw) & -(sv < sw))) == mn, "x=%d y=%d", sv, sw);
      CHK("max: x^((x^y)&-(x<y))", (sv ^ ((sv ^ sw) & -(sv < sw))) == mx, "x=%d y=%d", sv, sw);
      long long d = (long long)sv - sw; bool ok = d >= INT_MIN && d <= INT_MAX;
      int dd = (int)((unsigned)sv - (unsigned)sw);
      if (ok) {
        CHK("min quick&dirty (precondition holds)", (sw + (dd & (dd >> 31))) == mn, "x=%d y=%d", sv, sw);
        CHK("max quick&dirty (precondition holds)", (sv - (dd & (dd >> 31))) == mx, "x=%d y=%d", sv, sw);
      } else {
        CHK("min quick&dirty (precondition VIOLATED, expected to fail)", (sw + (dd & (dd >> 31))) == mn, "x=%d y=%d", sv, sw);
      } }

    CHK("pow2: v&(v-1)==0 (v=0 caveat)", ((v & (v - 1)) == 0) == (__builtin_popcount(v) == 1), "v=%08x", v);
    CHK("pow2: v && !(v&(v-1))", (v && !(v & (v - 1))) == (__builtin_popcount(v) == 1), "v=%08x", v);

    { int ref = (int)(v << 27) >> 27; int r; struct { signed int x:5; } s; r = s.x = sv;
      CHK("signext const 5 (bitfield)", r == ref, "v=%08x", v); }
    { unsigned b = 1 + p % 32; int ref = (int)(v << (32 - b)) >> (32 - b);
      if (b < 32) { unsigned x = v & ((1U << b) - 1), m = 1U << (b - 1);
        CHK("signext var: (x^m)-m after mask", (int)((x ^ m) - m) == ref, "v=%08x b=%u", v, b); }
      { unsigned m = 32 - b; CHK("signext var: (x<<m)>>m", ((int)(v << m) >> m) == ref, "v=%08x b=%u", v, b); }
      { int r = (int)((unsigned)sv * (unsigned)multipliers[b]) / divisors[b];
        CHK("signext var: 3-op multipliers/divisors", r == ref, "v=%08x b=%u got=%d ref=%d", v, b, r, ref); } }

    { bool f = p & 1; unsigned m = h32(p), ww = v; unsigned ref = f ? (ww | m) : (ww & ~m);
      unsigned a = ww; a ^= (-(unsigned)f ^ a) & m; unsigned b2 = (ww & ~m) | (-(unsigned)f & m);
      CHK("cond set/clear: w^=(-f^w)&m", a == ref, "w=%08x m=%08x f=%d", ww, m, f);
      CHK("cond set/clear: superscalar", b2 == ref, "w=%08x m=%08x f=%d", ww, m, f); }
    { int f = p & 1; int r1 = (int)((unsigned)(f ^ (f - 1)) * (unsigned)sv); int r2 = (sv ^ -f) + f;
      CHK("cond negate: fDontNegate", r1 == (f ? sv : (int)-(unsigned)sv), "v=%d f=%d", sv, f);
      CHK("cond negate: fNegate", r2 == (f ? (int)-(unsigned)sv : sv), "v=%d f=%d", sv, f); }
    { unsigned a = v, b = w, mask = p; CHK("merge bits", (a ^ ((a ^ b) & mask)) == ((a & ~mask) | (b & mask)), "a=%08x b=%08x m=%08x", a, b, mask); }

    // popcount
    { unsigned ref = __builtin_popcount(v), c; uint32_t t;
      c = 0; for (t = v; t; t >>= 1) c += t & 1; CHK("popcount naive", c == ref, "v=%08x", v);
      CHK("popcount table256", (BitsSetTable256[v & 0xff] + BitsSetTable256[(v >> 8) & 0xff] + BitsSetTable256[(v >> 16) & 0xff] + BitsSetTable256[v >> 24]) == ref, "v=%08x", v);
      { unsigned char *pp = (unsigned char *)&v; CHK("popcount table256 (byte ptr)", (BitsSetTable256[pp[0]] + BitsSetTable256[pp[1]] + BitsSetTable256[pp[2]] + BitsSetTable256[pp[3]]) == ref, "v=%08x", v); }
      for (c = 0, t = v; t; c++) t &= t - 1; CHK("popcount Kernighan", c == ref, "v=%08x", v);
      { uint32_t v14 = v & 0x3fff; c = (v14 * 0x200040008001ULL & 0x111111111111111ULL) % 0xf; CHK("popcount 64-bit mult opt1 (<=14 bits)", c == (unsigned)__builtin_popcount(v14), "v=%08x", v14); }
      { uint32_t v15 = v & 0x7fff; c = (v15 * 0x200040008001ULL & 0x111111111111111ULL) % 0xf; CHK("popcount 64-bit mult opt1 on 15-bit input (out of spec; expect fail)", c == (unsigned)__builtin_popcount(v15), "v=%08x", v15); }
      { uint32_t v24 = v & 0xffffff; c = ((v24 & 0xfff) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;
        c += (((v24 & 0xfff000) >> 12) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f; CHK("popcount 64-bit mult opt2 (<=24 bits)", c == (unsigned)__builtin_popcount(v24), "v=%08x", v24); }
      c = ((v & 0xfff) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;
      c += (((v & 0xfff000) >> 12) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;
      c += ((v >> 24) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f; CHK("popcount 64-bit mult opt3 (32 bits)", c == ref, "v=%08x", v);
      { static const int S[] = {1,2,4,8,16}; static const int B[] = {0x55555555,0x33333333,0x0F0F0F0F,0x00FF00FF,0x0000FFFF};
        c = v - ((v >> 1) & B[0]); c = ((c >> S[1]) & B[1]) + (c & B[1]); c = ((c >> S[2]) + c) & B[2]; c = ((c >> S[3]) + c) & B[3]; c = ((c >> S[4]) + c) & B[4];
        CHK("popcount parallel (16 ops)", c == ref, "v=%08x", v); }
      t = v - ((v >> 1) & 0x55555555); t = (t & 0x33333333) + ((t >> 2) & 0x33333333); c = ((t + (t >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
      CHK("popcount best (12 ops)", c == ref, "v=%08x", v);
      CHK("popcount generic<uint32_t>", pop32g(v) == ref, "v=%08x", v); }

    // parity
    { unsigned ref = __builtin_parity(v); bool par = false; uint32_t t = v;
      while (t) { par = !par; t &= t - 1; } CHK("parity naive", par == ref, "v=%08x", v);
      t = v; t ^= t >> 16; t ^= t >> 8; CHK("parity table (32-bit, 1 lookup)", ParityTable256[t & 0xff] == ref, "v=%08x", v);
      { unsigned char *pp = (unsigned char *)&v; CHK("parity table (byte ptr xor)", ParityTable256[pp[0] ^ pp[1] ^ pp[2] ^ pp[3]] == ref, "v=%08x", v); }
      { unsigned char b = v & 0xff; CHK("parity byte 64-bit mult+mod", ((((b * 0x0101010101010101ULL) & 0x8040201008040201ULL) % 0x1FF) & 1) == (unsigned)__builtin_parity(b), "b=%02x", b);
        uint32_t u = b; u ^= u >> 4; u &= 0xf; CHK("parity byte parallel (5 ops)", ((0x6996 >> u) & 1) == (unsigned)__builtin_parity(b), "b=%02x", b); }
      t = v; t ^= t >> 1; t ^= t >> 2; t = (t & 0x11111111U) * 0x11111111U; CHK("parity 32-bit multiply", ((t >> 28) & 1) == ref, "v=%08x", v);
      t = v; t ^= t >> 16; t ^= t >> 8; t ^= t >> 4; t &= 0xf; CHK("parity parallel 0x6996", ((0x6996 >> t) & 1) == ref, "v=%08x", v); }

    // swap bits
    { unsigned i = p & 7, j = 16 + ((p >> 3) & 7), n = 1 + ((p >> 8) % 8), b = v; unsigned mk = (1U << n) - 1;
      unsigned x = ((b >> i) ^ (b >> j)) & mk; unsigned r = b ^ ((x << i) | (x << j));
      unsigned fi = (b >> i) & mk, fj = (b >> j) & mk; unsigned ref = (b & ~((mk << i) | (mk << j))) | (fj << i) | (fi << j);
      CHK("swap bit ranges via XOR", r == ref, "b=%08x i=%u j=%u n=%u", b, i, j, n); }

    // reverse
    { uint32_t ref = ref_rev32(v), vv = v, r = v; int s = sizeof(v) * CHAR_BIT - 1;
      for (vv >>= 1; vv; vv >>= 1) { r <<= 1; r |= vv & 1; s--; } r <<= s; CHK("reverse obvious loop", r == ref, "v=%08x", v);
      uint32_t c = (BitReverseTable256[v & 0xff] << 24) | (BitReverseTable256[(v >> 8) & 0xff] << 16) | (BitReverseTable256[(v >> 16) & 0xff] << 8) | BitReverseTable256[(v >> 24) & 0xff];
      CHK("reverse table opt1", c == ref, "v=%08x", v);
      { uint32_t cc, vv2 = v; unsigned char *pp = (unsigned char *)&vv2, *q = (unsigned char *)&cc; q[3] = BitReverseTable256[pp[0]]; q[2] = BitReverseTable256[pp[1]]; q[1] = BitReverseTable256[pp[2]]; q[0] = BitReverseTable256[pp[3]]; CHK("reverse table opt2", cc == ref, "v=%08x", v); }
      { unsigned char b = v & 0xff; unsigned char rb = ref >> 24 ? 0 : 0; rb = (unsigned char)(ref_rev32(b) >> 24);
        unsigned char r3 = (b * 0x0202020202ULL & 0x010884422010ULL) % 1023; CHK("reverse byte 3 ops (mult+mod)", r3 == rb, "b=%02x", b);
        unsigned char r4 = ((b * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32; CHK("reverse byte 4 ops (64-bit mult)", r4 == rb, "b=%02x", b);
        unsigned char r7 = ((b * 0x0802LU & 0x22110LU) | (b * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16; CHK("reverse byte 7 ops", r7 == rb, "b=%02x", b); }
      uint32_t t = v; t = ((t >> 1) & 0x55555555) | ((t & 0x55555555) << 1); t = ((t >> 2) & 0x33333333) | ((t & 0x33333333) << 2);
      t = ((t >> 4) & 0x0F0F0F0F) | ((t & 0x0F0F0F0F) << 4); t = ((t >> 8) & 0x00FF00FF) | ((t & 0x00FF00FF) << 8); t = (t >> 16) | (t << 16);
      CHK("reverse parallel 5*lg N", t == ref, "v=%08x", v);
      t = v; { unsigned ss = 32; uint32_t mask = ~0; while ((ss >>= 1) > 0) { mask ^= (mask << ss); t = ((t >> ss) & mask) | ((t << ss) & ~mask); } }
      CHK("reverse parallel (computed masks)", t == ref, "v=%08x", v); }

    // modulus (1<<s)-1
    { unsigned s = 1 + p % 31; unsigned d = (1U << s) - 1; unsigned n = v, ref = v % d, m, nn;
      for (m = n, nn = n; nn > d; nn = m) { for (m = 0; nn; nn >>= s) m += nn & d; } m = m == d ? 0 : m;
      CHK("mod (1<<s)-1 loop", m == ref, "n=%08x s=%u got=%u ref=%u", n, s, m, ref);
      m = (n & Mm[s]) + ((n >> s) & Mm[s]);
      for (const unsigned *q = &Qm[s][0], *r = &Rm[s][0]; m > d; q++, r++) m = (m >> *q) + (m & *r);
      unsigned m2 = m == d ? 0 : m; unsigned m3 = m & -((signed)(m - d) >> s);
      CHK("mod (1<<s)-1 parallel (tables)", m2 == ref, "n=%08x s=%u got=%u ref=%u", n, s, m2, ref);
      CHK("mod (1<<s)-1 parallel (m&-((signed)(m-d)>>s) variant)", m3 == ref, "n=%08x s=%u got=%u ref=%u", n, s, m3, ref); }

    // log2 (v != 0)
    if (v) { int ref = ref_log2(v); unsigned r = 0, vv = v;
      while (vv >>= 1) r++; CHK("log2 obvious", (int)r == ref, "v=%08x", v);
      if (!(v >> 31)) { int iv = (int)v; int rr; union { unsigned int u[2]; double d; } t; t.u[1] = 0x43300000; t.u[0] = iv; t.d -= 4503599627370496.0; rr = (t.u[1] >> 20) - 0x3FF;
        CHK("log2 via IEEE double", rr == ref, "v=%08x", v); }
      { unsigned rr, t, tt; if ((tt = v >> 16)) rr = (t = tt >> 8) ? 24 + LogTable256[t] : 16 + LogTable256[tt]; else rr = (t = v >> 8) ? 8 + LogTable256[t] : LogTable256[v];
        CHK("log2 lookup table", (int)rr == ref, "v=%08x", v);
        if ((tt = v >> 24)) rr = 24 + LogTable256[tt]; else if ((tt = v >> 16)) rr = 16 + LogTable256[tt]; else if ((tt = v >> 8)) rr = 8 + LogTable256[tt]; else rr = LogTable256[v];
        CHK("log2 lookup table (input-tuned)", (int)rr == ref, "v=%08x", v); }
      { const unsigned b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000}; const unsigned S[] = {1, 2, 4, 8, 16}; unsigned rr = 0, vv2 = v;
        for (int i = 4; i >= 0; i--) if (vv2 & b[i]) { vv2 >>= S[i]; rr |= S[i]; } CHK("log2 O(lgN) loop", (int)rr == ref, "v=%08x", v); }
      { unsigned rr, shift, vv2 = v; rr = (vv2 > 0xFFFF) << 4; vv2 >>= rr; shift = (vv2 > 0xFF) << 3; vv2 >>= shift; rr |= shift;
        shift = (vv2 > 0xF) << 2; vv2 >>= shift; rr |= shift; shift = (vv2 > 0x3) << 1; vv2 >>= shift; rr |= shift; rr |= (vv2 >> 1);
        CHK("log2 O(lgN) branch-free", (int)rr == ref, "v=%08x", v); }
      { uint32_t vv2 = v; vv2 |= vv2 >> 1; vv2 |= vv2 >> 2; vv2 |= vv2 >> 4; vv2 |= vv2 >> 8; vv2 |= vv2 >> 16;
        CHK("log2 de Bruijn", DeBruijnLog[(uint32_t)(vv2 * 0x07C4ACDDU) >> 27] == ref, "v=%08x", v); }
      { int t = (ref_log2(v) + 1) * 1233 >> 12; int r = t - (v < PowersOf10[t]);
        int r2 = (v >= 1000000000) ? 9 : (v >= 100000000) ? 8 : (v >= 10000000) ? 7 : (v >= 1000000) ? 6 : (v >= 100000) ? 5 : (v >= 10000) ? 4 : (v >= 1000) ? 3 : (v >= 100) ? 2 : (v >= 10) ? 1 : 0;
        CHK("log10 via log2 (1233>>12)", r == r2, "v=%u got=%d ref=%d", v, r, r2); }
      if (__builtin_popcount(v) == 1) {
        CHK("log2 de Bruijn (pow2 only)", DeBruijnPow2[(uint32_t)(v * 0x077CB531U) >> 27] == ref, "v=%08x", v);
        { static const unsigned b[] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000}; unsigned rr = (v & b[0]) != 0; for (int i = 4; i > 0; i--) rr |= ((v & b[i]) != 0) << i;
          CHK("log2 pow2-only bitmask version", (int)rr == ref, "v=%08x", v); } }
      // trailing zeros
      { int ref2 = __builtin_ctz(v); uint32_t vv2 = v; int c;
        vv2 = (vv2 ^ (vv2 - 1)) >> 1; for (c = 0; vv2; c++) vv2 >>= 1; CHK("ctz linear", c == ref2, "v=%08x", v);
        { unsigned cc = 32, t = v; t &= -(int)t; if (t) cc--; if (t & 0x0000FFFF) cc -= 16; if (t & 0x00FF00FF) cc -= 8; if (t & 0x0F0F0F0F) cc -= 4; if (t & 0x33333333) cc -= 2; if (t & 0x55555555) cc -= 1;
          CHK("ctz parallel (needs -(int)v in C)", (int)cc == ref2, "v=%08x", v); }
        { unsigned cc, t = v; if (t & 1) cc = 0; else { cc = 1; if ((t & 0xffff) == 0) { t >>= 16; cc += 16; } if ((t & 0xff) == 0) { t >>= 8; cc += 8; } if ((t & 0xf) == 0) { t >>= 4; cc += 4; } if ((t & 3) == 0) { t >>= 2; cc += 2; } cc -= t & 1; }
          CHK("ctz binary search", (int)cc == ref2, "v=%08x", v); }
        { float f = (float)(v & -v); uint32_t bits; memcpy(&bits, &f, 4); CHK("ctz float cast", (int)(bits >> 23) - 0x7f == ref2, "v=%08x", v); }
        CHK("ctz mod 37 lookup", Mod37BitPosition[(-v & v) % 37] == ref2, "v=%08x", v);
        CHK("ctz de Bruijn mult", DeBruijnPow2[((uint32_t)((v & -v) * 0x077CB531U)) >> 27] == ref2, "v=%08x", v); }
      // next-gen: round up pow2
      { uint64_t ref2 = 1; while (ref2 < v) ref2 <<= 1;
        if (v <= (1u << 31)) {
          unsigned r; if (v > 1) { float f = (float)v; uint32_t bits; memcpy(&bits, &f, 4); unsigned t = 1U << ((bits >> 23) - 0x7f); r = t << (t < v); } else r = 1;
          CHK("roundup pow2 via float (v<=2^31)", r == ref2, "v=%u got=%u ref=%llu", v, r, (unsigned long long)ref2); }
        if (v > 1 && v < (1u << 25)) { float f = (float)(v - 1); uint32_t bits; memcpy(&bits, &f, 4); unsigned r = 1U << ((bits >> 23) - 126);
          CHK("roundup pow2 float quick&dirty (1<v<2^25)", r == ref2, "v=%u", v); }
        if (v >= (1u << 25) && v <= (1u << 31)) { float f = (float)(v - 1); uint32_t bits; memcpy(&bits, &f, 4); unsigned r = 1U << ((bits >> 23) - 126);
          CHK("roundup pow2 float quick&dirty OUTSIDE domain (expect fail)", r == ref2, "v=%u got=%u ref=%llu", v, r, (unsigned long long)ref2); }
        if (v <= (1u << 31)) { uint32_t t = v; t--; t |= t >> 1; t |= t >> 2; t |= t >> 4; t |= t >> 8; t |= t >> 16; t++; CHK("roundup pow2 shift-or", t == ref2, "v=%u", v); } }
    }

    // float log2
    { uint32_t bits = v & 0x7fffffff; int e = bits >> 23;
      if (bits && e != 255) { float f; memcpy(&f, &bits, 4); int ref = ilogbf(f); int x = (int)bits, c = x >> 23;
        if (e) { CHK("float log2 (normals, simple)", (x >> 23) - 127 == ref, "bits=%08x", bits); }
        if (c) c -= 127; else { unsigned t; if ((t = x >> 16)) c = LogTable256[t] - 133; else c = (t = x >> 8) ? LogTable256[t] - 141 : LogTable256[x] - 149; }
        CHK("float log2 (incl. subnormals)", c == ref, "bits=%08x got=%d ref=%d", bits, c, ref);
        if (e) { int r = (p >> 8) % 5; int cc = (int)bits; cc = ((((cc - 0x3f800000) >> r) + 0x3f800000) >> 23) - 127;
          long double lg = log2l((long double)f) / (long double)(1 << r); int refr = (int)floorl(lg);
          CHK("float log2 of pow(2,r)-root", cc == refr, "bits=%08x r=%d got=%d ref=%d", bits, r, cc, refr); } } }

    // interleave
    { uint32_t x = v & 0xffff, y = v >> 16; uint32_t ref = ref_spread16(x) | (ref_spread16(y) << 1); uint32_t z = 0;
      for (int i = 0; i < 16; i++) z |= (x & 1U << i) << i | (y & 1U << i) << (i + 1); CHK("interleave obvious", z == ref, "v=%08x", v);
      z = MortonTable256[y >> 8] << 17 | MortonTable256[x >> 8] << 16 | MortonTable256[y & 0xFF] << 1 | MortonTable256[x & 0xFF]; CHK("interleave table (literal table!)", z == ref, "v=%08x", v);
      { uint32_t xb = x & 0xff, yb = y & 0xff; uint32_t rb = ref_spread16(xb) | (ref_spread16(yb) << 1);
        unsigned short zz = ((xb * 0x0101010101010101ULL & 0x8040201008040201ULL) * 0x0102040810204081ULL >> 49) & 0x5555 | ((yb * 0x0101010101010101ULL & 0x8040201008040201ULL) * 0x0102040810204081ULL >> 48) & 0xAAAA;
        CHK("interleave 64-bit mult (bytes)", zz == rb, "x=%02x y=%02x got=%04x ref=%04x", xb, yb, zz, rb); }
      { static const unsigned B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF}; static const unsigned S[] = {1, 2, 4, 8}; uint32_t xx = x, yy = y;
        xx = (xx | (xx << S[3])) & B[3]; xx = (xx | (xx << S[2])) & B[2]; xx = (xx | (xx << S[1])) & B[1]; xx = (xx | (xx << S[0])) & B[0];
        yy = (yy | (yy << S[3])) & B[3]; yy = (yy | (yy << S[2])) & B[2]; yy = (yy | (yy << S[1])) & B[1]; yy = (yy | (yy << S[0])) & B[0];
        CHK("interleave binary magic numbers", (xx | (yy << 1)) == ref, "v=%08x", v); } }

    // byte tests on 32-bit word
    { bool hz = false; for (int i = 0; i < 4; i++) hz |= ((v >> (8 * i)) & 0xff) == 0;
      bool a = ~((((v & 0x7F7F7F7F) + 0x7F7F7F7F) | v) | 0x7F7F7F7F);
      CHK("hasZeroByte (5 ops)", a == hz, "v=%08x", v);
      bool pre = ((v + 0x7efefeff) ^ ~v) & 0x81010100; CHK("hasZeroByte pretest (no false negatives)", !hz || pre, "v=%08x", v);
      CHK("haszero macro (32-bit v)", (haszero(v) != 0) == hz, "v=%08x", v);
      unsigned n = p & 0xff; bool hv = false; for (int i = 0; i < 4; i++) hv |= ((v >> (8 * i)) & 0xff) == n;
      CHK("hasvalue macro (32-bit x)", (hasvalue(v, n) != 0) == hv, "x=%08x n=%u", v, n);
      unsigned nl = (p >> 8) % 129; int cl = 0; for (int i = 0; i < 4; i++) cl += ((v >> (8 * i)) & 0xff) < nl;
      CHK("hasless macro (32-bit x)", (hasless(v, nl) != 0) == (cl > 0), "x=%08x n=%u", v, nl);
      CHK("countless macro (32-bit x)", (int)countless(v, nl) == cl, "x=%08x n=%u got=%lu ref=%d", v, nl, (unsigned long)countless(v, nl), cl);
      unsigned nm = (p >> 16) % 128; int cm = 0; for (int i = 0; i < 4; i++) cm += ((v >> (8 * i)) & 0xff) > nm;
      CHK("hasmore macro (32-bit x)", (hasmore(v, nm) != 0) == (cm > 0), "x=%08x n=%u", v, nm);
      CHK("countmore macro (32-bit x)", (int)countmore(v, nm) == cm, "x=%08x n=%u got=%lu ref=%d", v, nm, (unsigned long)countmore(v, nm), cm);
      unsigned m = (p >> 24) % 128, nn = m + 1 + (p % (129 - m - 1 > 0 ? 129 - m - 1 : 1)); if (nn > 128) nn = 128; int cb = 0;
      for (int i = 0; i < 4; i++) { unsigned bb = (v >> (8 * i)) & 0xff; cb += bb > m && bb < nn; }
      CHK("hasbetween macro (32-bit x)", (hasbetween(v, m, nn) != 0) == (cb > 0), "x=%08x m=%u n=%u", v, m, nn);
      CHK("countbetween macro (32-bit x)", (int)countbetween(v, m, nn) == cb, "x=%08x m=%u n=%u got=%lu ref=%d", v, m, nn, (unsigned long)countbetween(v, m, nn), cb);
      CHK("likelyhasbetween (no false negatives)", cb == 0 || likelyhasbetween(v, m, nn) != 0, "x=%08x m=%u n=%u", v, m, nn); }

    // next bit permutation
    if (v) { uint64_t c = v & -v, r = (uint64_t)v + c; uint64_t refn = (((r ^ v) >> 2) / c) | r;
      if (refn <= 0xffffffffULL) { unsigned t = v | (v - 1); unsigned w1 = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
        unsigned t2 = (v | (v - 1)) + 1; unsigned w2 = t2 | ((((t2 & -t2) / (v & -v)) >> 1) - 1);
        CHK("next bit permutation (ctz version)", w1 == refn, "v=%08x got=%08x ref=%08llx", v, w1, (unsigned long long)refn);
        CHK("next bit permutation (division version)", w2 == refn, "v=%08x got=%08x ref=%08llx", v, w2, (unsigned long long)refn); } }
}

// ============ 64/128-bit + multi-arg checks (also the fuzz entry) ============
static void check64(uint64_t v, uint64_t w, uint32_t p, uint32_t q) {
    // rank (pos 1..64)
    { unsigned pos = 1 + p % 64; uint64_t r = v >> (sizeof(v) * CHAR_BIT - pos);
      uint64_t ref = __builtin_popcountll(r);
      r = r - ((r >> 1) & ~0UL / 3); r = (r & ~0UL / 5) + ((r >> 2) & ~0UL / 5); r = (r + (r >> 4)) & ~0UL / 17; r = (r * (~0UL / 255)) >> ((sizeof(v) - 1) * CHAR_BIT);
      CHK("rank from MSB", r == ref, "v=%016llx pos=%u got=%llu ref=%llu", (unsigned long long)v, pos, (unsigned long long)r, (unsigned long long)ref); }
    // select
    { unsigned r = 1 + q % 64, rr = r; int cnt = __builtin_popcountll(v); unsigned refs = 64; int seen = 0;
      for (int i = 0; i < 64; i++) if ((v >> (63 - i)) & 1) { if (++seen == (int)r) { refs = i + 1; break; } }
      unsigned s; uint64_t a, b, c, d; unsigned t;
      a = v - ((v >> 1) & ~0UL / 3); b = (a & ~0UL / 5) + ((a >> 2) & ~0UL / 5); c = (b + (b >> 4)) & ~0UL / 0x11; d = (c + (c >> 8)) & ~0UL / 0x101; t = (d >> 32) + (d >> 48);
      s = 64; s -= ((t - rr) & 256) >> 3; rr -= (t & ((t - rr) >> 8)); t = (d >> (s - 16)) & 0xff;
      s -= ((t - rr) & 256) >> 4; rr -= (t & ((t - rr) >> 8)); t = (c >> (s - 8)) & 0xf;
      s -= ((t - rr) & 256) >> 5; rr -= (t & ((t - rr) >> 8)); t = (b >> (s - 4)) & 0x7;
      s -= ((t - rr) & 256) >> 6; rr -= (t & ((t - rr) >> 8)); t = (a >> (s - 2)) & 0x3;
      s -= ((t - rr) & 256) >> 7; rr -= (t & ((t - rr) >> 8)); t = (v >> (s - 1)) & 0x1;
      s -= ((t - rr) & 256) >> 8; s = 65 - s;
      if ((int)r <= cnt) CHK("select bit by rank from MSB (rank<=popcount)", s == refs, "v=%016llx r=%u got=%u ref=%u", (unsigned long long)v, r, s, refs);
      else CHK("select bit by rank (rank>popcount, doc says returns 64)", s == 64, "v=%016llx r=%u cnt=%d got=%u", (unsigned long long)v, r, cnt, s); }
    // generic popcount, wide
    CHK("popcount generic<uint64_t>", pop64g(v) == (unsigned)__builtin_popcountll(v), "v=%016llx", (unsigned long long)v);
    { u128 x = ((u128)v << 64) | w; CHK("popcount generic<u128>", pop128g(x) == (unsigned)(__builtin_popcountll(v) + __builtin_popcountll(w)), "v=%016llx%016llx", (unsigned long long)v, (unsigned long long)w); }
    // 64-bit parity multiply
    { uint64_t t = v; t ^= t >> 1; t ^= t >> 2; t = (t & 0x1111111111111111UL) * 0x1111111111111111UL; CHK("parity 64-bit multiply", ((t >> 60) & 1) == (unsigned)__builtin_parityll(v), "v=%016llx", (unsigned long long)v); }
    // 64-bit word byte tests
    { unsigned long x = v; bool hz = false; for (int i = 0; i < 8; i++) hz |= ((x >> (8 * i)) & 0xff) == 0;
      CHK("hasZeroByte 64-bit (7F7F.. consts)", (bool)~((((x & 0x7F7F7F7F7F7F7F7FUL) + 0x7F7F7F7F7F7F7F7FUL) | x) | 0x7F7F7F7F7F7F7F7FUL) == hz, "x=%016lx", x);
      unsigned nl = q % 129; int cl = 0; for (int i = 0; i < 8; i++) cl += ((x >> (8 * i)) & 0xff) < nl;
      CHK("hasless macro (64-bit x)", (hasless(x, nl) != 0) == (cl > 0), "x=%016lx n=%u", x, nl);
      CHK("countless macro (64-bit x)", (int)countless(x, nl) == cl, "x=%016lx n=%u", x, nl);
      unsigned nm = (q >> 8) % 128; int cm = 0; for (int i = 0; i < 8; i++) cm += ((x >> (8 * i)) & 0xff) > nm;
      CHK("hasmore/countmore macro (64-bit x)", (hasmore(x, nm) != 0) == (cm > 0) && (int)countmore(x, nm) == cm, "x=%016lx n=%u", x, nm);
      unsigned m = (q >> 16) % 128, nn = m + 1 + ((q >> 24) % (128 - m)); int cb = 0; for (int i = 0; i < 8; i++) { unsigned b = (x >> (8 * i)) & 0xff; cb += b > m && b < nn; }
      CHK("hasbetween/countbetween macro (64-bit x)", (hasbetween(x, m, nn) != 0) == (cb > 0) && (int)countbetween(x, m, nn) == cb, "x=%016lx m=%u n=%u", x, m, nn);
      CHK("likelyhasbetween 64-bit (no false negatives)", cb == 0 || likelyhasbetween(x, m, nn) != 0, "x=%016lx m=%u n=%u", x, m, nn); }
}

// ============ edge / documented-behavior probes (ran under -fsanitize=undefined separately) ============
static void edge_probes(void) {
    printf("-- documented-edge probes --\n");
    { unsigned v = 0; unsigned c = 32; unsigned t = v; t &= -(int)t; if (t) c--; printf("ctz parallel v=0 -> %u (doc: 32)\n", c); }
    { unsigned v = 0, c; if (v & 1) c = 0; else { c = 1; if ((v & 0xffff) == 0) { v >>= 16; c += 16; } if ((v & 0xff) == 0) { v >>= 8; c += 8; } if ((v & 0xf) == 0) { v >>= 4; c += 4; } if ((v & 3) == 0) { v >>= 2; c += 2; } c -= v & 1; } printf("ctz binary-search v=0 -> %u (doc: 31)\n", c); }
    { float f = (float)0u; uint32_t b; memcpy(&b, &f, 4); printf("ctz float-cast v=0 -> %d (doc: -127)\n", (int)(b >> 23) - 0x7f); }
    printf("ctz mod37 v=0 -> %d (doc: n/a)\n", Mod37BitPosition[0]);
    printf("ctz deBruijn v=0 -> %d (doc: 0)\n", DeBruijnPow2[0]);
    { uint32_t v = 0; v |= v >> 1; printf("log2 deBruijn v=0 -> %d\n", DeBruijnLog[(uint32_t)(v * 0x07C4ACDDU) >> 27]); }
    { unsigned v = 0, r; unsigned t, tt; if ((tt = v >> 16)) r = 0; else r = (t = v >> 8) ? 8 + LogTable256[t] : LogTable256[v]; printf("log2 table v=0 -> %d (doc: -1)\n", (int)r); }
    { unsigned v = 0; v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16; v++; printf("roundup pow2 v=0 -> %u (doc: 0)\n", v); }
    { unsigned v = 0x80000001u; v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16; v++; printf("roundup pow2 v=2^31+1 -> %u (overflow wraps to 0)\n", v); }
    { int a = 5; int *pa = &a; (void)pa; int x = 7; int r = SWAP_ADD(x, x); printf("SWAP_ADD(x,x) aliasing guard -> x=%d (should stay 7), r=%d\n", x, r); }
    { int x = 7; SWAP_XOR(x, x); printf("SWAP_XOR(x,x) aliasing -> x=%d (doc: breaks, becomes 0)\n", x); }
    { int a[2] = {1, 2}; SWAP_XOR(a[0], a[1]); printf("SWAP_XOR distinct -> %d %d\n", a[0], a[1]); }
    { unsigned bb = 0b00101111; unsigned i = 1, j = 5, n = 3; unsigned x = ((bb >> i) ^ (bb >> j)) & ((1U << n) - 1); unsigned r = bb ^ ((x << i) | (x << j)); printf("swap-bits doc example: %02x (doc expects 11100011 = e3)\n", r); }
    { volatile unsigned pos = 0; unsigned long long v = ~0ULL; unsigned long long r = v >> (64 - pos); printf("rank pos=0: v>>64 -> %llx (UB: shift by 64; x86 gives full word)\n", r); }
    { volatile int x = INT_MIN; int mask = x >> 31; printf("abs(INT_MIN) wrap path -> %u\n", (unsigned)((x + mask) ^ mask)); }
    { volatile unsigned b = 32; unsigned m = (1U << (b - 1)); unsigned mk = (b < 32) ? ((1U << b) - 1) : 0; printf("signext var b=32: mask expr (1U<<b) is UB; m=%x\n", m); (void)mk; }
}

static void report(void) {
    int bad = 0;
    printf("\n== %d checks; failures below ==\n", nids);
    for (int i = 0; i < nids; i++) if (fails[i]) { bad++; printf("FAIL [%s] x%lu  first: %s\n", names[i], fails[i], first[i]); }
    printf("== %d passed, %d failed ==\n", nids - bad, bad);
}

int main(int argc, char **argv) {
    init_tables(); init_R();
    if (argc == 1 || !strcmp(argv[1], "fuzz")) {
        fuzzmode = 1; uint8_t buf[64] = {0}; ssize_t n = 0;
        FILE *f = argc > 2 ? fopen(argv[2], "rb") : stdin; if (!f) return 1;
        n = fread(buf, 1, sizeof buf, f); if (n < 24) return 0;
        uint64_t v, w; uint32_t p, q; memcpy(&v, buf, 8); memcpy(&w, buf + 8, 8); memcpy(&p, buf + 16, 4); memcpy(&q, buf + 20, 4);
        check64(v, w, p, q); return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "full")) { for (uint64_t i = 0; i <= 0xffffffffULL; i++) check32((uint32_t)i); report(); return 0; }
    int lg = argc > 2 ? atoi(argv[2]) : 26;
    uint64_t N = 1ULL << lg;
    // sweep A: sequential low 2^lg (covers all small values / low-bit patterns); sweep B: sequential high; sweep C: hashed
    for (uint64_t i = 0; i < N && i < (1ULL << 32); i++) check32((uint32_t)i);
    for (uint64_t i = 0; i < N; i++) check32((uint32_t)(0xffffffffULL - i));
    for (uint64_t i = 0; i < N; i++) check32(h32((uint32_t)i) ^ ((uint32_t)i << 20 & 0x0f000000u));
    // single/double-bit and run patterns
    for (int i = 0; i < 32; i++) for (int j = 0; j < 32; j++) { check32((1u << i) | (1u << j)); check32(((1u << i) - 1) << j); check32(~((1u << i) | (1u << j))); }
    for (uint64_t i = 0; i < N; i++) { uint64_t a = h64(i), b = h64(i + 0x9e3779b9ULL); check64(a, b, (uint32_t)(a >> 32) ^ (uint32_t)i, (uint32_t)(b >> 32) ^ (uint32_t)(i >> 3)); }
    for (int i = 0; i < 64; i++) for (int j = 0; j < 64; j++) { uint64_t a = (1ULL << i) | (1ULL << j); check64(a, ~a, i * 7 + j, j * 5 + i); check64(~a, a, i + j * 64, i * 64 + j); check64((~0ULL << i) >> j, 0, i, j); }
    // 8/16-bit exhaustive for generic popcount
    for (unsigned i = 0; i < 65536; i++) {
        CHK("popcount generic<uint16_t> (exhaustive)", pop16g(i) == (unsigned)__builtin_popcount(i), "v=%04x", i);
        CHK("popcount generic<uint8_t> (exhaustive)", pop8g(i & 0xff) == (unsigned)__builtin_popcount(i & 0xff), "v=%02x", i & 0xff);
    }
    // exhaustive (v16, b) for 3-op sign extension and variable-width sext
    for (unsigned b = 1; b <= 32; b++) for (unsigned x = 0; x < 65536; x++) {
        int ref = (int)((uint32_t)x << (32 - b)) >> (32 - b); int r = (int)((unsigned)x * (unsigned)multipliers[b]) / divisors[b];
        CHK("signext 3-op (exhaustive v16 x b)", r == ref, "x=%04x b=%u got=%d ref=%d", x, b, r, ref);
    }
    edge_probes();
    report();
    return 0;
}
