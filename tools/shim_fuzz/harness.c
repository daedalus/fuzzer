/* Differential harness: cmplog interposers (afl_shim.c, __AFL_PRELOAD_ONLY) vs libc.
 * Operands sit flush against a PROT_NONE page, so any read past the bytes the real libc
 * call legally touches faults.  Legality is decided by a reference scan, not by trying. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <wchar.h>
#include <wctype.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __AFL_COMPILER
__AFL_FUZZ_INIT();
#else   /* standalone: one testcase from stdin, no AFL needed (used by the regression test) */
static unsigned char sa_buf[1 << 16]; static ssize_t sa_len; static int sa_once;
#define __AFL_INIT()
#define __AFL_FUZZ_TESTCASE_BUF sa_buf
#define __AFL_FUZZ_TESTCASE_LEN sa_len
#define __AFL_LOOP(n) (!sa_once && ((sa_len = read(0, sa_buf, sizeof sa_buf)), (sa_once = 1), sa_len >= 0))
#endif

#define PG 4096
static unsigned char *regA, *regB;   /* [guard][data][guard] */
static void *libc;
static void (*cmp_reset)(void);

static unsigned char *mk(void) {
    unsigned char *r = mmap(NULL, 3 * PG, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (r == MAP_FAILED) abort();
    mprotect(r, PG, PROT_NONE);
    mprotect(r + 2 * PG, PG, PROT_NONE);
    return r;
}
#define MISMATCH() __builtin_trap()

static inline int sgn(int x) { return (x > 0) - (x < 0); }
static inline int lw(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

/* element = 1 (char) or 4 (wchar_t); counts are in elements */
static long get(const unsigned char *p, int w, size_t i) {
    return w == 1 ? p[i] : (long)((const wchar_t *)p)[i];
}
static int in_set(const unsigned char *s, int w, size_t ls, long c) {
    for (size_t i = 0; i < ls; i++) { long x = get(s, w, i); if (!x) return 0; if (x == c) return 1; }
    return 0;
}
static int has_nul(const unsigned char *p, int w, size_t l) {
    for (size_t i = 0; i < l; i++) if (!get(p, w, i)) return 1;
    return 0;
}
/* would strcmp/strncmp/strcasecmp stop inside both buffers? */
static int legal2(const unsigned char *a, size_t la, const unsigned char *b, size_t lb,
                  size_t n, int w, int fold) {
    for (size_t i = 0; i < n; i++) {
        if (i >= la || i >= lb) return 0;
        long x = get(a, w, i), y = get(b, w, i);
        if (fold) { x = lw((int)x); y = lw((int)y); }
        if (x != y || !x) return 1;
    }
    return 1;
}

#include <sys/shm.h>
struct rec_hdr { uint64_t magic; uint32_t gen,int_sites,int_hits,byte_slots,rec_slots,rec_used,byte_used,pad;
                 uint64_t site_dropped,hit_dropped,rec_dropped,byte_dropped,pad2; };
static struct rec_hdr *rec;          /* records segment (REC mode) */
static void *shimh;                  /* handle of the preloaded shim */
#define CB(name, T) ((T)dlsym(shimh ? shimh : RTLD_DEFAULT, name))
static unsigned char *regC;
static void rec_drain(void) { if (!rec) return; rec->rec_used = 0; rec->byte_used = 0; rec->gen++; }
static void rec_scribble(const unsigned char *d, size_t dl) {   /* the fuzzer owns this header */
    if (!rec || dl < 40) return;
    uint32_t *w = (uint32_t *)rec;
    for (int i = 0; i < 8; i++) { uint32_t v; memcpy(&v, d + 4 * (i % (dl / 4)), 4); if (i > 1 || (d[0] & 1)) w[1 + (i % 7)] ^= v; }
    uint32_t v; memcpy(&v, d + 8, 4); rec->rec_used = v; memcpy(&v, d + 12, 4); rec->byte_used = v;
    uint64_t *sp = (uint64_t *)((char *)rec + sizeof *rec);     /* scribble site table */
    for (int i = 0; i < 32; i++) sp[i] ^= ((uint64_t)d[i % dl] << (8 * (i & 7)));
}
#define NF 32
static void run(const unsigned char *in, size_t len) {
    if (len < 8) return;
    unsigned f = in[0] % NF, pl = in[1];
    size_t la = in[2], lb = in[3], nsel = in[4] % 8;
    size_t nv[8] = {0, 1, 0, 0, 0, (size_t)-1, in[5], (size_t)1 << 40};
    const unsigned char *d = in + 8; size_t dl = len - 8;
    int w = (f >= 19 && f <= 22);                       /* wide family */
    int el = w ? 4 : 1;
    if (w) { la &= 63; lb &= 63; }
    size_t mn = la < lb ? la : lb, mx = la > lb ? la : lb;
    nv[2] = mn; nv[3] = mx; nv[4] = la + 1;
    size_t n = nv[nsel];

    unsigned char *A = regA + PG, *B = regB + PG;
    size_t ba = la * el, bb = lb * el;
    if (pl & 1) A += PG - ba; else A += 0;           /* end-flush vs start-of-page */
    if (pl & 2) B += PG - bb;
    if (!(pl & 1) && 0) {}
    for (size_t i = 0; i < ba; i++) A[i] = dl ? d[i % dl] : 0;
    for (size_t i = 0; i < bb; i++) B[i] = dl ? d[(i + la + 7) % dl] : 0;
    if (pl & 4) { if (ba) memset(A + ba - el, 0, el); }      /* terminate */
    if (pl & 8) { if (bb) memset(B + bb - el, 0, el); }
    if (pl & 16 && dl > 1) { /* make a and b share a prefix so compares go deep */
        size_t k = (d[0] % (mn + 1)) * el; memcpy(B, A, k < bb ? k : bb);
    }

    uint64_t rec0 = 0; int scr = 0;
    if (rec) { scr = (in[6] & 0x80) != 0; if (scr) rec_scribble(d, dl); rec0 = (uint64_t)rec->byte_used + rec->byte_dropped; }
    errno = 0x5a5a;
    int expect = 0;
    volatile long rr = 0, rs = 0;                      /* real vs shimmed */
#define REAL(name, T) ((T)dlsym(libc, name))
    switch (f) {
    case 0: { n = n > mn ? mn : n;
        int r = REAL("memcmp", int(*)(const void*,const void*,size_t))(A, B, n);
        int s = memcmp(A, B, n); if (sgn(r) != sgn(s)) MISMATCH(); expect = n > 0 && s != 0; break; }
    case 1: { n = n > mn ? mn : n;
        int r = REAL("bcmp", int(*)(const void*,const void*,size_t))(A, B, n);
        int s = bcmp(A, B, n); if ((r != 0) != (s != 0)) MISMATCH(); expect = n > 0 && s != 0; break; }
    case 2: { n = n > la ? la : n; int c = d[0];
        void *r = REAL("memchr", void*(*)(const void*,int,size_t))(A, c, n);
        void *s = memchr(A, c, n); if (r != s) MISMATCH(); break; }
    case 3: { n = n > la ? la : n; int c = d[0];
        void *r = REAL("memrchr", void*(*)(const void*,int,size_t))(A, c, n);
        void *s = memrchr(A, c, n); if (r != s) MISMATCH(); break; }
    case 4: {
        void *r = REAL("memmem", void*(*)(const void*,size_t,const void*,size_t))(A, la, B, lb);
        void *s = memmem(A, la, B, lb); if (r != s) MISMATCH(); break; }
    case 5: if (!legal2(A, la, B, lb, (size_t)-1, 1, 0)) return;
        rr = REAL("strcmp", int(*)(const char*,const char*))((char*)A, (char*)B);
        rs = strcmp((char*)A, (char*)B); if (sgn(rr) != sgn(rs)) MISMATCH(); expect = rs != 0; break;
    case 6: if (!legal2(A, la, B, lb, n, 1, 0)) return;
        rr = REAL("strncmp", int(*)(const char*,const char*,size_t))((char*)A, (char*)B, n);
        rs = strncmp((char*)A, (char*)B, n); if (sgn(rr) != sgn(rs)) MISMATCH(); expect = rs != 0; break;
    case 7: if (!legal2(A, la, B, lb, (size_t)-1, 1, 1)) return;
        rr = REAL("strcasecmp", int(*)(const char*,const char*))((char*)A, (char*)B);
        rs = strcasecmp((char*)A, (char*)B); if (sgn(rr) != sgn(rs)) MISMATCH(); expect = rs != 0; break;
    case 8: if (!legal2(A, la, B, lb, n, 1, 1)) return;
        rr = REAL("strncasecmp", int(*)(const char*,const char*,size_t))((char*)A, (char*)B, n);
        rs = strncasecmp((char*)A, (char*)B, n); if (sgn(rr) != sgn(rs)) MISMATCH(); expect = rs != 0; break;
    case 9: case 10: case 11: {                    /* strpbrk / strspn / strcspn */
        if (!has_nul(B, 1, lb)) return;
        int ok = 0;
        for (size_t i = 0; i < la && !ok; i++) {
            long c = A[i]; if (!c) ok = 1;
            else if (f == 11 ? !in_set(B, 1, lb, c) : in_set(B, 1, lb, c)) ok = (f == 11) ? 1 : 1;
            if (f == 10 && in_set(B, 1, lb, c)) ok = 0; /* strspn stops at NOT in set */
        }
        if (f == 10) { ok = 0; for (size_t i = 0; i < la && !ok; i++) { long c = A[i]; if (!c || !in_set(B,1,lb,c)) ok = 1; } }
        if (f != 10) { ok = 0; for (size_t i = 0; i < la && !ok; i++) { long c = A[i]; if (!c || in_set(B,1,lb,c)) ok = 1; } }
        if (!ok) return;
        if (f == 9) { void *r = REAL("strpbrk", char*(*)(const char*,const char*))((char*)A,(char*)B);
                      void *s = strpbrk((char*)A,(char*)B); if (r != s) MISMATCH(); }
        if (f == 10) { size_t r = REAL("strspn", size_t(*)(const char*,const char*))((char*)A,(char*)B);
                       size_t s = strspn((char*)A,(char*)B); if (r != s) MISMATCH(); }
        if (f == 11) { size_t r = REAL("strcspn", size_t(*)(const char*,const char*))((char*)A,(char*)B);
                       size_t s = strcspn((char*)A,(char*)B); if (r != s) MISMATCH(); }
        break; }
    case 12: case 13:                              /* strstr / strcasestr: both terminated */
        if (!has_nul(A, 1, la) || !has_nul(B, 1, lb)) return;
        if (f == 12) { void *r = REAL("strstr", char*(*)(const char*,const char*))((char*)A,(char*)B);
                       void *s = strstr((char*)A,(char*)B); if (r != s) MISMATCH(); }
        else { void *r = REAL("strcasestr", char*(*)(const char*,const char*))((char*)A,(char*)B);
               void *s = strcasestr((char*)A,(char*)B); if (r != s) MISMATCH(); }
        break;
    case 19: { n = n > mn ? mn : n;
        int r = REAL("wmemcmp", int(*)(const wchar_t*,const wchar_t*,size_t))((wchar_t*)A,(wchar_t*)B,n);
        int s = wmemcmp((wchar_t*)A,(wchar_t*)B,n); if (sgn(r) != sgn(s)) MISMATCH(); expect = n > 0 && s != 0; break; }
    case 20: if (!legal2(A, la, B, lb, (size_t)-1, 4, 0)) return;
        rr = REAL("wcscmp", int(*)(const wchar_t*,const wchar_t*))((wchar_t*)A,(wchar_t*)B);
        rs = wcscmp((wchar_t*)A,(wchar_t*)B); if (sgn(rr) != sgn(rs)) MISMATCH(); expect = rs != 0; break;
    case 21: if (!legal2(A, la, B, lb, n, 4, 0)) return;
        rr = REAL("wcsncmp", int(*)(const wchar_t*,const wchar_t*,size_t))((wchar_t*)A,(wchar_t*)B,n);
        rs = wcsncmp((wchar_t*)A,(wchar_t*)B,n); if (sgn(rr) != sgn(rs)) MISMATCH(); expect = rs != 0; break;
    case 22: if (!legal2(A, la, B, lb, (size_t)-1, 4, 1)) return;
        rr = REAL("wcscasecmp", int(*)(const wchar_t*,const wchar_t*))((wchar_t*)A,(wchar_t*)B);
        rs = wcscasecmp((wchar_t*)A,(wchar_t*)B); if (sgn(rr) != sgn(rs)) MISMATCH(); expect = rs != 0; break;
    case 23: case 24: case 25: case 26: {                 /* trace_cmp{1,2,4,8} / const_cmp */
        uint64_t x, y; memcpy(&x, d, 8 > dl ? dl : 8); memcpy(&y, d + (dl > 8 ? 8 : 0), 8 > dl ? dl : 8);
        static const char *nm[2][4] = {{"__sanitizer_cov_trace_cmp1","__sanitizer_cov_trace_cmp2","__sanitizer_cov_trace_cmp4","__sanitizer_cov_trace_cmp8"},
                                       {"__sanitizer_cov_trace_const_cmp1","__sanitizer_cov_trace_const_cmp2","__sanitizer_cov_trace_const_cmp4","__sanitizer_cov_trace_const_cmp8"}};
        void (*fn)(uint64_t, uint64_t) = CB(nm[pl & 1][f - 23], void(*)(uint64_t,uint64_t));
        if (fn) fn(x, y); break; }
    case 27: {                                             /* trace_switch: [count][bits][vals...] */
        size_t cnt = la % 65; uint64_t *ref = (uint64_t *)(regC + PG + PG - (2 + cnt) * 8);
        ref[0] = cnt; ref[1] = (pl & 8) ? (uint64_t)d[1] * 0x0101010101ULL : (uint64_t[]){8,16,32,64}[pl & 3];
        for (size_t i = 0; i < cnt; i++) memcpy(&ref[2 + i], d + (i * 8) % (dl > 8 ? dl - 8 : 1), dl >= 8 ? 8 : dl);
        uint64_t v; memcpy(&v, d, dl >= 8 ? 8 : dl);
        void (*fn)(uint64_t, void *) = CB("__sanitizer_cov_trace_switch", void(*)(uint64_t,void*));
        if (fn) fn(v, ref); break; }
    case 28: { uint64_t v; memcpy(&v, d, dl >= 8 ? 8 : dl);
        if (pl & 1) { void (*q)(uint64_t) = CB("__sanitizer_cov_trace_div8", void(*)(uint64_t)); if (q) q(v); }
        else { void (*q)(uint32_t) = CB("__sanitizer_cov_trace_div4", void(*)(uint32_t)); if (q) q((uint32_t)v); }
        break; }
    case 29: { uintptr_t v; memcpy(&v, d, dl >= 8 ? 8 : dl);
        void (*q)(uintptr_t) = CB("__sanitizer_cov_trace_gep", void(*)(uintptr_t)); if (q) q(v); break; }
    case 30: case 31: {                                    /* sanitizer weak hooks, legal operands */
        if (!has_nul(A, 1, la) || !has_nul(B, 1, lb)) return;
        int r = (int)(n * 2654435761u) | (pl & 1); void *pc = (void *)(uintptr_t)(0x400000 + d[0] * 16);
        switch (nsel) {
        case 0: { n = n > mn ? mn : n; void (*q)(void*,const void*,const void*,size_t,int) = CB("__sanitizer_weak_hook_memcmp", void(*)(void*,const void*,const void*,size_t,int)); if (q) q(pc,A,B,n,r); break; }
        case 1: { void (*q)(void*,const char*,const char*,int) = CB("__sanitizer_weak_hook_strcmp", void(*)(void*,const char*,const char*,int)); if (q) q(pc,(char*)A,(char*)B,r); break; }
        case 2: { void (*q)(void*,const char*,const char*,size_t,int) = CB("__sanitizer_weak_hook_strncmp", void(*)(void*,const char*,const char*,size_t,int)); if (q) q(pc,(char*)A,(char*)B,n,r); break; }
        case 3: { void (*q)(void*,const char*,const char*,int) = CB("__sanitizer_weak_hook_strcasecmp", void(*)(void*,const char*,const char*,int)); if (q) q(pc,(char*)A,(char*)B,r); break; }
        case 4: { void (*q)(void*,const char*,const char*,size_t,int) = CB("__sanitizer_weak_hook_strncasecmp", void(*)(void*,const char*,const char*,size_t,int)); if (q) q(pc,(char*)A,(char*)B,n,r); break; }
        case 5: { void (*q)(void*,const char*,const char*,char*) = CB("__sanitizer_weak_hook_strstr", void(*)(void*,const char*,const char*,char*)); if (q) q(pc,(char*)A,(char*)B,(pl&2)?(char*)A:NULL); break; }
        case 6: { void (*q)(void*,const char*,const char*,char*) = CB("__sanitizer_weak_hook_strcasestr", void(*)(void*,const char*,const char*,char*)); if (q) q(pc,(char*)A,(char*)B,(pl&2)?(char*)A:NULL); break; }
        default:{ void (*q)(void*,const void*,size_t,const void*,size_t,void*) = CB("__sanitizer_weak_hook_memmem", void(*)(void*,const void*,size_t,const void*,size_t,void*)); if (q) q(pc,A,la,B,lb,(pl&2)?A:NULL); break; }
        } break; }
    default: return;
    }
    if ((f == 5 || f == 7 || f == 20 || f == 22) && (!get(A, w ? 4 : 1, 0) || !get(B, w ? 4 : 1, 0))) expect = 0; /* empty operand: shim records nothing by design */
    if (errno != 0x5a5a) MISMATCH();
    if (rec && !scr && expect && (uint64_t)rec->byte_used + rec->byte_dropped <= rec0)
        MISMATCH();   /* channel gate: a non-equal libc compare must land in the records segment */                   /* shim must preserve errno */
}

int main(void) {
    libc = dlopen("libc.so.6", RTLD_NOW);
    if (!libc) return 2;
    regA = mk(); regB = mk();
    cmp_reset = (void (*)(void))dlsym(RTLD_DEFAULT, "__cmplog_reset");
    { const char *pre = getenv("AFL_PRELOAD"); shimh = pre ? dlopen(pre, RTLD_NOW | RTLD_NOLOAD) : NULL; }
    regC = mk();
    { const char *id = getenv("_CMPLOG_RECORDS_SHM");
      if (id && *id) { void *p = shmat(atoi(id), NULL, 0); if (p != (void *)-1) { rec = p; rec_drain(); } } }
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;
    while (__AFL_LOOP(20000)) {
        run(buf, (size_t)__AFL_FUZZ_TESTCASE_LEN);
        if (cmp_reset && !getenv("NORESET")) cmp_reset();
        if (rec) { /* restore valid geometry the way a real fuzzer would, then drain */
            rec->magic = 0x3230434552504d43ULL; rec->int_sites = 64; rec->int_hits = 4; rec->byte_slots = 64; rec->rec_slots = 242; rec_drain(); }
    }
    return 0;
}
