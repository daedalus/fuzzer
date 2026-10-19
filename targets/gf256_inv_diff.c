/* gf256_inv_diff.c -- differential tester for the GF(2^8) multiply / inverse code in
 * https://crypto.stackexchange.com/a/12956 (answers 12962 poncho, fgrieu, Khovratovich).
 * Field: AES, x^8+x^4+x^3+x+1 (0x11b).  Code below is copied verbatim from the page except
 * where marked.  Oracle: brute force (the inverse of a is the b with gmul(a,b)==1).
 *
 *   ./gf256_inv_diff exh            : sweep all 256 a / 65536 (a,b), report failing checks
 *   ./gf256_inv_diff [fuzz] [file]  : fuzz entry (stdin or file); buf[0]=a, buf[1]=b; aborts on mismatch
 *
 * Known finding: poncho's ginv_table(1) reads antilog[255] (declared antilog[255]).  The
 * harness checks the index instead of performing the out-of-bounds read, so the result is
 * deterministic with or without ASan.
 *
 * Build: cc -O2 -g -fsanitize=address,undefined -o gf256_inv_diff targets/gf256_inv_diff.c
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXID 64
static const char *names[MAXID];
static unsigned long fails[MAXID];
static char first[MAXID][96];
static int nids = 0, fuzzmode = 0;

static void record(int id, const char *nm, int a, int b, int got, int ref) {
    if (fails[id]++ == 0) snprintf(first[id], sizeof first[id], "a=%02x b=%02x got=%02x ref=%02x", a, b, got, ref);
    if (fuzzmode) { fprintf(stderr, "MISMATCH %s: %s\n", nm, first[id]); abort(); }
}
#define CHK(nm, cond, a, b, got, ref) do { enum { ID = __COUNTER__ }; names[ID] = nm; \
    if (ID + 1 > nids) { nids = ID + 1; } \
    if (!(cond)) { record(ID, nm, a, b, got, ref); } } while (0)

/* ---- verbatim from the page ---- */
static uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t p = 0, carry;
    for (int i = 0; i < 8; i++) {
        if (b & 1) p ^= a;
        carry = a & 0x80; a = a << 1; if (carry) a ^= 0x1b; b = b >> 1;
    }
    return p;
}
static uint8_t log_table[256], antilog[255];
static const uint8_t g = 3;
static void init_log_table(void) {
    log_table[0] = 0;
    for (int i = 0, x = 1; i < 255; x = gmul(x, g), i++) { log_table[x] = i; antilog[i] = x; }
}
static uint8_t gmul_table(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    uint8_t x = log_table[a], y = log_table[b];
    uint8_t log_mult = (x + y) % 255;
    return antilog[log_mult];
}
/* ginv_table: page version has `antilog[255 - x]`, out of bounds for a==1; split here so the
 * index can be checked before use (the read itself is not performed when idx == 255). */
static unsigned ginv_table_idx(uint8_t a) { return 255 - log_table[a]; }
static uint8_t ginv_table(uint8_t a) {              /* page logic, guarded */
    if (a == 0) return 0;
    unsigned idx = ginv_table_idx(a);
    return idx < 255 ? antilog[idx] : 0xEE;          /* 0xEE: marker, never a valid read */
}
static uint8_t ginv_table_fixed(uint8_t a) {        /* fix: reduce mod 255 */
    if (a == 0) return 0;
    return antilog[(255 - log_table[a]) % 255];
}
static int bitlength(unsigned x) { return x ? 32 - __builtin_clz(x) : 0; }
static uint8_t ginv_eea(uint8_t x) {                /* poncho, bitlength() variant */
    uint16_t u1 = 0, u3 = 0x11b, v1 = 1, v3 = x;
    while (v3 != 0) {
        uint16_t t1 = u1, t3 = u3;
        int8_t q = bitlength(u3) - bitlength(v3);
        if (q >= 0) { t1 ^= v1 << q; t3 ^= v3 << q; }
        u1 = v1; u3 = v3; v1 = t1; v3 = t3;
    }
    if (u1 >= 0x100) u1 ^= 0x11b;
    return u1;
}
static uint8_t ginv_pow(uint8_t a) {                /* fgrieu: a^254 */
    uint8_t j, b = a;
    for (j = 14; --j;) b = gmul(b, j & 1 ? b : a);
    return b;
}
static uint8_t ginvt[256];
static int build_iters;
static void gbuildt(void) {                          /* fgrieu: generator 0x03, inverse 0xf6 */
    uint8_t i = 1, j = 1;
    for (;;) { build_iters++; ginvt[i] = j; i = gmul(i, 0x03); if (i == j) break; j = gmul(j, 0xf6); ginvt[j] = i; }
    ginvt[0] = 0;
}

/* ---- oracle ---- */
static uint8_t ref_inv[256];
static void init_ref(void) {
    for (int a = 1; a < 256; a++) for (int b = 1; b < 256; b++) if (gmul(a, b) == 1) { ref_inv[a] = b; break; }
}
static int order(uint8_t e) { uint8_t x = e; int n = 1; while (x != 1) { x = gmul(x, e); n++; } return n; }

static void check_inv(uint8_t a) {
    uint8_t r = ref_inv[a];
    CHK("inv: EEA (poncho)",              ginv_eea(a) == r,        a, 0, ginv_eea(a), r);
    CHK("inv: a^254 (fgrieu)",            ginv_pow(a) == r,        a, 0, ginv_pow(a), r);
    CHK("inv: ginvt table (fgrieu)",      ginvt[a] == r,           a, 0, ginvt[a], r);
    CHK("inv: log/antilog, antilog idx < 255 [BUG a==1]", a == 0 || ginv_table_idx(a) < 255, a, 0, ginv_table_idx(a), 254);
    CHK("inv: log/antilog (poncho, guarded)", ginv_table(a) == r,  a, 0, ginv_table(a), r);
    CHK("inv: log/antilog (mod 255 fix)", ginv_table_fixed(a) == r, a, 0, ginv_table_fixed(a), r);
    CHK("inv: a*inv(a)==1 (a!=0)",        a == 0 || gmul(a, ginv_pow(a)) == 1, a, 0, gmul(a, ginv_pow(a)), 1);
    CHK("inv: inv(inv(a))==a",            ginv_pow(ginv_pow(a)) == a, a, 0, ginv_pow(ginv_pow(a)), a);
}
static void check_mul(uint8_t a, uint8_t b) {
    CHK("mul: gmul_table vs gmul", gmul_table(a, b) == gmul(a, b), a, b, gmul_table(a, b), gmul(a, b));
    CHK("mul: commutative",        gmul(a, b) == gmul(b, a),       a, b, gmul(a, b), gmul(b, a));
}
static void report(void) {
    int bad = 0;
    for (int i = 0; i < nids; i++) {
        if (fails[i]) { bad++; printf("FAIL  %-52s %lu  first: %s\n", names[i], fails[i], first[i]); }
        else printf("ok    %s\n", names[i]);
    }
    printf("%d / %d checks failing\n", bad, nids);
}

int main(int argc, char **argv) {
    init_log_table(); gbuildt(); init_ref();
    if (argc == 1 || strcmp(argv[1], "exh")) {
        fuzzmode = 1;
        const char *path = argc > 2 ? argv[2] : (argc > 1 && strcmp(argv[1], "fuzz") ? argv[1] : NULL);
        FILE *f = path ? fopen(path, "rb") : stdin; if (!f) return 1;
        uint8_t buf[2] = {0}; if (fread(buf, 1, 2, f) < 2) return 0;
        check_inv(buf[0]); check_mul(buf[0], buf[1]);
        return 0;
    }
    for (int a = 0; a < 256; a++) { check_inv(a); for (int b = 0; b < 256; b++) check_mul(a, b); }
    int nz = 0; for (int a = 1; a < 256; a++) nz += ginvt[a] != 0;
    CHK("gen: ord(3)==255",              order(3) == 255,        3, 0, order(3), 255);
    CHK("gen: ord(2)==51 (not primitive)", order(2) == 51,       2, 0, order(2), 51);
    CHK("gen: gmul(3,0xf6)==1",          gmul(3, 0xf6) == 1,     3, 0xf6, gmul(3, 0xf6), 1);
    CHK("gen: inv(2)==0x8d (0x11b>>1)",  ref_inv[2] == 0x8d,     2, 0, ref_inv[2], 0x8d);
    CHK("gen: ginvt fills 255 entries",  nz == 255,              0, 0, nz, 255);
    CHK("gen: inv(0x95)==0x8a",          ref_inv[0x95] == 0x8a,  0x95, 0, ref_inv[0x95], 0x8a);
    report();
    return 0;
}
