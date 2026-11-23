/* p6_flags_diff.c - differential tester for the P6 micro-op flag/alias model in
 * https://nand2mario.github.io/posts/2026/p6_basics/ (2026-11-22).
 *
 * Model: per-register flags (tier 1), U2 publish masks into EFLAGS (tier 2),
 * ADC through MOVEMERGEFLGS, CMPXCHG as the branchless MERGE/CMOV.Z sequence,
 * partial-register merges (AL/AH/AX), 32-bit effective-address wrap.
 * Oracle: the host CPU. Each record is assembled into real x86 and run from a
 * JIT page, so the model is checked against silicon, not against another model.
 *
 *   record (12 bytes): op, ctl, flags_in, disp, a:u32, b:u32
 *   ctl: [1:0] width (8L,16,32,8H)  [3:2] rm reg  [5:4] reg field  [7:6] SIB scale
 *
 *   ./p6_flags_diff exh            : sweep, per-op mismatch counts
 *   ./p6_flags_diff list           : op count, then names
 *   ./p6_flags_diff [fuzz] [file]  : fuzz entry (stdin or file); SIGABRT on mismatch
 *
 * Flags the SDM calls undefined (AF after logic ops, SF/ZF/AF/PF after MUL, ...)
 * are masked out of the comparison, never guessed.
 * Build: cc -O2 -g -o p6_flags_diff targets/p6_flags_diff.c  (-DP6_BUG=1..4 plants a model bug)
 */
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) && defined(__linux__)
#include <sys/mman.h>

#ifndef P6_BUG
#define P6_BUG 0
#endif

#define REC_LEN 12
#define MAX_RECS 64
#define MEM_LEN 16
#define MEM_OFF_MASK 7
#define MAX_ACCESS 4
#define NREGS 4
#define RFLAGS_BASE 0x2u
#define PAGE 4096
#define CODE_MAX 64
#define HASH_A 0x9E3779B1u
#define HASH_B 0x85EBCA6Bu

enum { F_CF = 0x001, F_PF = 0x004, F_AF = 0x010, F_ZF = 0x040, F_SF = 0x080, F_OF = 0x800 };
enum { F_ALL = 0x8D5, F_NOCF = 0x8D4, F_CFOF = 0x801 };
enum { WC_8L, WC_16, WC_32, WC_8H };
enum { K_ADD, K_ADC, K_SUB, K_SBB, K_AND, K_OR, K_XOR };
enum { NO_MEM = -1 };

enum {
    OP_ADD, OP_OR, OP_ADC, OP_SBB, OP_AND, OP_SUB, OP_XOR, OP_CMP,
    OP_INC, OP_DEC, OP_NEG, OP_NOT, OP_MUL, OP_IMUL, OP_ROL1, OP_ROR1,
    OP_BTS, OP_BTR, OP_BTC, OP_CMPXCHG, OP_XADD,
    OP_MINC, OP_MDEC, OP_MADD, OP_MCMPXCHG, NOPS
};

static const char *const opname[NOPS] = {
    "ADD", "OR", "ADC", "SBB", "AND", "SUB", "XOR", "CMP",
    "INC", "DEC", "NEG", "NOT", "MUL", "IMUL", "ROL1", "ROR1",
    "BTS", "BTR", "BTC", "CMPXCHG", "XADD",
    "LOCK_INC_M", "LOCK_DEC_M", "LOCK_ADD_M", "LOCK_CMPXCHG_M"
};

/* OP_ADD..OP_CMP in x86 /r opcode order */
static const int alu_kind[8] = { K_ADD, K_OR, K_ADC, K_SBB, K_AND, K_SUB, K_XOR, K_SUB };

typedef struct { uint32_t r[NREGS]; uint64_t fl; } St;   /* eax ecx edx ebx; offsets used by PRO/EPI */
typedef struct { uint32_t d, f; } V;                     /* value + per-register flags */
typedef struct { uint8_t op, ctl, flags, disp; uint32_t a, b; } Rec;
typedef struct { int wc, d, s, sc, mo; } Ctx;            /* decoded record + model memory offset */

static uint8_t *nmem;        /* native memory operand, below 4 GiB so addr32 reaches it */
static uint32_t nmem32;
static uint8_t *code;
static uint8_t mmem[MEM_LEN];

static int wbits(int wc) { return wc == WC_16 ? 16 : wc == WC_32 ? 32 : 8; }
static uint32_t wmask(int w) { return w == 32 ? 0xFFFFFFFFu : (1u << w) - 1; }
static int is8(int wc) { return wc == WC_8L || wc == WC_8H; }
static int renc(int r, int wc) { return wc == WC_8H ? r + 4 : r; }

static uint32_t rd(const St *s, int r, int wc) {
    if (wc == WC_8H) return (s->r[r] >> 8) & 0xFF;
    return s->r[r] & wmask(wbits(wc));
}

static void wr(St *s, int r, int wc, uint32_t v) {
    uint32_t m = wmask(wbits(wc));

    if (wc == WC_8H) s->r[r] = (s->r[r] & ~0xFF00u) | ((v & 0xFF) << 8);
    else if (wc == WC_32) s->r[r] = v;
    else s->r[r] = (s->r[r] & ~m) | (v & m);
}

/* tier 2: merge only the U2-selected flags into architectural EFLAGS */
static void pub(St *s, uint32_t f, uint32_t prof) {
    s->fl = (s->fl & ~(uint64_t)prof) | (f & prof);
}

/* ZF, SF, PF of a result; sb is its sign bit */
static uint32_t szp(uint32_t res, uint32_t sb) {
    return (res ? 0 : F_ZF) | (res & sb ? F_SF : 0) | (__builtin_parity(res & 0xFF) ? 0 : F_PF);
}

static V alu(int k, int w, uint32_t a, uint32_t b, uint32_t cin) {
    uint32_t m = wmask(w), sb = 1u << (w - 1), res, f = 0;
    uint64_t t;

    a &= m; b &= m;
    if (k == K_ADD || k == K_ADC) {
        t = (uint64_t)a + b + cin; res = t & m;
        if (t >> w) f |= F_CF;
        if ((a ^ b ^ res) & 0x10) f |= F_AF;
        if (((a ^ res) & (b ^ res)) & sb) f |= F_OF;
    } else if (k == K_SUB || k == K_SBB) {
        t = (uint64_t)a - b - cin; res = t & m;
        if ((uint64_t)a < (uint64_t)b + cin) f |= F_CF;
        if ((a ^ b ^ res) & 0x10) f |= F_AF;
        if (((a ^ b) & (a ^ res)) & sb) f |= F_OF;
    } else {
        res = k == K_AND ? a & b : k == K_OR ? a | b : a ^ b;
    }
    return (V){ res, f | szp(res, sb) };
}

/* MOVEMERGEFLGS: attach EFLAGS to a value so the next ALU uop reads its CF */
static V movemerge(const St *s, uint32_t v) { return (V){ v, (uint32_t)s->fl & F_ALL }; }

static void alu_model(St *s, int op, const Ctx *c) {
    int k = alu_kind[op];
    uint32_t a = rd(s, c->d, c->wc), b = rd(s, c->s, c->wc), cin = 0;
    V r;

    if (k == K_ADC || k == K_SBB) {
        V t0 = movemerge(s, a);
        cin = P6_BUG == 2 ? 0 : (t0.f & F_CF) != 0;
        a = t0.d;
    }
    r = alu(k, wbits(c->wc), a, b, cin);
    if (op != OP_CMP) wr(s, c->d, c->wc, r.d);
    pub(s, r.f, F_ALL);
}

static void unary_model(St *s, int op, const Ctx *c) {
    int w = wbits(c->wc);
    uint32_t a = rd(s, c->d, c->wc), prof = F_ALL;
    V r;

    if (op == OP_INC || op == OP_DEC) prof = P6_BUG == 1 ? F_ALL : F_NOCF;
    if (op == OP_INC) r = alu(K_ADD, w, a, 1, 0);
    else if (op == OP_DEC) r = alu(K_SUB, w, a, 1, 0);
    else if (op == OP_NEG) r = alu(K_SUB, w, 0, a, 0);
    else { r.d = ~a; r.f = 0; prof = 0; }
    wr(s, c->d, c->wc, r.d);
    pub(s, r.f, prof);
}

static void mul_model(St *s, int op, const Ctx *c) {
    int w = wbits(c->wc), cf;
    uint32_t a = rd(s, 0, is8(c->wc) ? WC_8L : c->wc), b = rd(s, c->d, c->wc);
    uint64_t lo, hi;

    if (op == OP_IMUL) {
        int64_t x = (int64_t)(int32_t)(a << (32 - w)) >> (32 - w);
        int64_t y = (int64_t)(int32_t)(b << (32 - w)) >> (32 - w);
        int64_t p = x * y, sx = (int64_t)((uint64_t)p << (64 - w)) >> (64 - w);

        lo = (uint64_t)p & wmask(w); hi = ((uint64_t)p >> w) & wmask(w);
        cf = p != sx;
    } else {
        uint64_t p = (uint64_t)a * b;

        lo = p & wmask(w); hi = (p >> w) & wmask(w);
        cf = hi != 0;
    }
    if (w == 8) wr(s, 0, WC_16, (uint32_t)(lo | hi << 8));
    else { wr(s, 0, c->wc, (uint32_t)lo); wr(s, 2, c->wc, (uint32_t)hi); }
    pub(s, cf ? F_CF | F_OF : 0, F_CFOF);
}

static void rot_model(St *s, int op, const Ctx *c) {
    int w = wbits(c->wc);
    uint32_t a = rd(s, c->d, c->wc), m = wmask(w), res, cf, of;

    if (op == OP_ROL1) {
        res = ((a << 1) | (a >> (w - 1))) & m;
        cf = res & 1;
        of = ((res >> (w - 1)) & 1) ^ cf;
    } else {
        res = ((a >> 1) | (a << (w - 1))) & m;
        cf = (res >> (w - 1)) & 1;
        of = ((res >> (w - 1)) ^ (res >> (w - 2))) & 1;
    }
    wr(s, c->d, c->wc, res);
    pub(s, (cf ? F_CF : 0) | (of ? F_OF : 0), F_CFOF);
}

static void bt_model(St *s, const Rec *q, int op, const Ctx *c) {
    int bit = q->disp & (wbits(c->wc) - 1);
    uint32_t v = rd(s, c->d, c->wc), cf = (v >> bit) & 1;

    if (op == OP_BTS) v |= 1u << bit;
    else if (op == OP_BTR) v &= ~(1u << bit);
    else v ^= 1u << bit;
    wr(s, c->d, c->wc, v);
    pub(s, cf ? F_CF : 0, F_CF);
}

static void xadd_model(St *s, const Ctx *c) {
    uint32_t a = rd(s, c->d, c->wc), b = rd(s, c->s, c->wc);
    V r = alu(K_ADD, wbits(c->wc), a, b, 0);

    wr(s, c->s, c->wc, a);
    wr(s, c->d, c->wc, r.d);
    pub(s, r.f, F_ALL);
}

/* 2651..2656 of the MSROM listing */
static void cmpxchg_model(St *s, const Ctx *c) {
    int accwc = is8(c->wc) ? WC_8L : c->wc;
    uint32_t rm = rd(s, c->d, c->wc), src = rd(s, c->s, c->wc);
    V t1 = alu(K_SUB, wbits(c->wc), rd(s, 0, accwc), rm, 0);
    V t0 = { src, t1.f }, t2;

    pub(s, t1.f, F_ALL);
    wr(s, c->d, c->wc, (t1.f & F_ZF) ? t0.d : rm);
    t2 = (V){ rd(s, 0, accwc), t1.f };
    if (P6_BUG == 3) return;
    wr(s, 0, accwc, (t1.f & F_ZF) ? t2.d : rd(s, c->d, c->wc));
}

static uint32_t mget(int mo, int w) {
    uint32_t v = 0;

    memcpy(&v, mmem + mo, w / 8);
    return v;
}

static void mput(int mo, int w, uint32_t v) { memcpy(mmem + mo, &v, w / 8); }

static void mem_model(St *s, int op, const Ctx *c) {
    int w = wbits(c->wc), mo = c->mo;
    uint32_t v, src = rd(s, c->s, c->wc);
    V r;

    if (mo == NO_MEM) return;
    v = mget(mo, w);
    if (op == OP_MCMPXCHG) {
        r = alu(K_SUB, w, rd(s, 0, c->wc), v, 0);
        pub(s, r.f, F_ALL);
        if (r.f & F_ZF) mput(mo, w, src);
        else wr(s, 0, c->wc, v);
        return;
    }
    r = alu(op == OP_MDEC ? K_SUB : K_ADD, w, v, op == OP_MADD ? src : 1, 0);
    mput(mo, w, r.d);
    pub(s, r.f, op == OP_MADD || P6_BUG == 1 ? F_ALL : F_NOCF);
}

static void model_exec(St *s, const Rec *q, const Ctx *c) {
    int op = q->op;

    if (op <= OP_CMP) alu_model(s, op, c);
    else if (op <= OP_NOT) unary_model(s, op, c);
    else if (op <= OP_IMUL) mul_model(s, op, c);
    else if (op <= OP_ROR1) rot_model(s, op, c);
    else if (op <= OP_BTC) bt_model(s, q, op, c);
    else if (op == OP_CMPXCHG) cmpxchg_model(s, c);
    else if (op == OP_XADD) xadd_model(s, c);
    else mem_model(s, op, c);
}

/* flags the SDM defines for this op; the rest are not compared */
static uint32_t cmp_mask(int op) {
    if (op == OP_AND || op == OP_OR || op == OP_XOR) return F_ALL & ~F_AF;
    if (op == OP_MUL || op == OP_IMUL) return F_CFOF;
    if (op >= OP_BTS && op <= OP_BTC) return F_CF | F_ZF;
    return F_ALL;
}

/* ---- native side: assemble one instruction, run it between a save/restore stub ---- */

static const uint8_t PRO[] = {
    0x53,                                  /* push rbx */
    0x49, 0x89, 0xF8,                      /* mov r8, rdi */
    0x41, 0x8B, 0x00, 0x41, 0x8B, 0x48, 0x04,
    0x41, 0x8B, 0x50, 0x08, 0x41, 0x8B, 0x58, 0x0C,   /* mov eax/ecx/edx/ebx, [r8+..] */
    0x41, 0xFF, 0x70, 0x10, 0x9D           /* push [r8+16]; popfq */
};
static const uint8_t EPI[] = {
    0x9C, 0x41, 0x8F, 0x40, 0x10,          /* pushfq; pop [r8+16] */
    0x41, 0x89, 0x00, 0x41, 0x89, 0x48, 0x04,
    0x41, 0x89, 0x50, 0x08, 0x41, 0x89, 0x58, 0x0C,   /* mov [r8+..], eax/ecx/edx/ebx */
    0x5B, 0xC3                             /* pop rbx; ret */
};

static int asm_mem(uint8_t *o, int op, const Ctx *c, uint32_t disp) {
    int n = 0, w8 = is8(c->wc), ext = op == OP_MDEC;
    uint8_t modrm = 0x84 | (op == OP_MINC || op == OP_MDEC ? ext << 3 : renc(c->s, c->wc) << 3);

    o[n++] = 0x67; o[n++] = 0xF0;
    if (c->wc == WC_16) o[n++] = 0x66;
    if (op == OP_MCMPXCHG) { o[n++] = 0x0F; o[n++] = 0xB0 | !w8; }
    else if (op == OP_MADD) o[n++] = !w8;
    else o[n++] = 0xFE | !w8;
    o[n++] = modrm;
    o[n++] = 0x0B | c->sc << 6;            /* SIB: base ebx, index ecx */
    memcpy(o + n, &disp, 4);
    return n + 4;
}

static int asm_reg(uint8_t *o, const Rec *q, const Ctx *c) {
    int n = 0, op = q->op, w8 = is8(c->wc), rm = renc(c->d, c->wc), rg = renc(c->s, c->wc);
    static const uint8_t ext[] = { 0, 1, 3, 2, 4, 5, 0, 1, 5, 6, 7 };   /* from OP_INC */

    if (c->wc == WC_16) o[n++] = 0x66;
    if (op <= OP_CMP) { o[n++] = (op << 3) | !w8; o[n++] = 0xC0 | rg << 3 | rm; return n; }
    if (op >= OP_BTS && op <= OP_BTC) {
        o[n++] = 0x0F; o[n++] = 0xBA; o[n++] = 0xC0 | ext[op - OP_INC] << 3 | rm; o[n++] = q->disp;
        return n;
    }
    if (op == OP_CMPXCHG || op == OP_XADD) {
        o[n++] = 0x0F; o[n++] = (op == OP_XADD ? 0xC0 : 0xB0) | !w8;
        o[n++] = 0xC0 | rg << 3 | rm;
        return n;
    }
    if (op <= OP_DEC) o[n++] = 0xFE | !w8;
    else if (op <= OP_IMUL) o[n++] = 0xF6 | !w8;
    else o[n++] = 0xD0 | !w8;
    o[n++] = 0xC0 | ext[op - OP_INC] << 3 | rm;
    return n;
}

static void run_native(St *n, const Rec *q, const Ctx *c) {
    int len = sizeof PRO;

    len += q->op >= OP_MINC ? asm_mem(code + len, q->op, c, q->b) : asm_reg(code + len, q, c);
    memcpy(code + len, EPI, sizeof EPI);
    ((void (*)(St *))code)(n);
    n->fl &= F_ALL;
}

/* ---- harness ---- */

static int norm_wc(int op, int wc) {
    if (op >= OP_BTS && op <= OP_BTC) return wc == WC_16 ? WC_16 : WC_32;
    if (op >= OP_MINC && wc == WC_8H) return WC_32;
    return wc;
}

/* identical pre-state for model and native; mem forms derive ebx so ea lands in nmem */
static void prep(St *cur, const Rec *q, const Ctx *c) {
    uint32_t h = q->a * HASH_A + q->b, off = q->disp & MEM_OFF_MASK, f = 0;
    int i;

    if (q->op >= OP_MINC) {
        cur->r[c->s] = q->a;
        cur->r[3] = nmem32 + off - (cur->r[1] << c->sc) - q->b;
    } else {
        cur->r[c->d] = q->a;
        if (c->d != c->s) cur->r[c->s] = q->b;
    }
    for (i = 0; i < MEM_LEN; i++) nmem[i] = mmem[i] = (uint8_t)((h + i * HASH_B) >> 24);
    f |= (q->flags & 1 ? F_CF : 0) | (q->flags & 2 ? F_PF : 0) | (q->flags & 4 ? F_AF : 0);
    f |= (q->flags & 8 ? F_ZF : 0) | (q->flags & 16 ? F_SF : 0) | (q->flags & 32 ? F_OF : 0);
    cur->fl = RFLAGS_BASE | f;
}

/* model effective address -> offset into the operand window */
static int model_mo(const St *s, const Rec *q, int sc) {
#if P6_BUG == 4
    uint64_t ea = (uint64_t)s->r[3] + ((uint64_t)s->r[1] << sc) + q->b;
    uint64_t off = ea - nmem32;
#else
    uint32_t ea = s->r[3] + (s->r[1] << sc) + q->b;
    uint32_t off = ea - nmem32;
#endif
    return off > MEM_LEN - MAX_ACCESS ? NO_MEM : (int)off;
}

static int same(const St *m, const St *n, const Rec *q) {
    uint32_t mask = cmp_mask(q->op);

    if (memcmp(m->r, n->r, sizeof m->r)) return 0;
    if ((m->fl ^ n->fl) & mask) return 0;
    return q->op < OP_MINC || !memcmp(mmem, nmem, MEM_LEN);
}

static int step(const Rec *q, St *cur, int fuzz, char *first, size_t flen) {
    Ctx c = { norm_wc(q->op, q->ctl & 3), q->ctl >> 2 & 3, q->ctl >> 4 & 3, q->ctl >> 6, NO_MEM };
    St m, n;

    prep(cur, q, &c);
    m = n = *cur;
    if (q->op >= OP_MINC) c.mo = model_mo(&m, q, c.sc);
    model_exec(&m, q, &c);
    run_native(&n, q, &c);
    *cur = n;
    if (same(&m, &n, q)) return 0;

    if (first && !*first)
        snprintf(first, flen, "w=%d d=%d s=%d fl=%02x a=%08x b=%08x model eax=%08x fl=%03x hw eax=%08x fl=%03x",
                 wbits(c.wc), c.d, c.s, q->flags, q->a, q->b, m.r[0], (unsigned)m.fl, n.r[0], (unsigned)n.fl);
    if (!fuzz) return 1;
    fprintf(stderr, "MISMATCH %s: %s\n", opname[q->op], first ? first : "");
    raise(SIGABRT);
    return 1;
}

static int init_native(void) {
    code = mmap(NULL, PAGE, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    nmem = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
    if (code == MAP_FAILED || nmem == MAP_FAILED) return 0;
    memcpy(code, PRO, sizeof PRO);
    nmem32 = (uint32_t)(uintptr_t)nmem;
    return 1;
}

static const St INIT = { { 0x11223344u, 0x55667788u, 0x99AABBCCu, 0xDDEEFF00u }, RFLAGS_BASE };

static int fuzz_main(const char *path) {
    FILE *f = path ? fopen(path, "rb") : stdin;
    uint8_t buf[REC_LEN * MAX_RECS] = { 0 };
    size_t n, i;
    St cur = INIT;
    char first[160] = "";

    if (!f) return 1;
    n = fread(buf, 1, sizeof buf, f);
    if (path) fclose(f);
    if (!n || !init_native()) return 0;

    for (i = 0; i < (n + REC_LEN - 1) / REC_LEN; i++) {
        const uint8_t *p = buf + i * REC_LEN;
        Rec q = { p[0] % NOPS, p[1], p[2], p[3], 0, 0 };

        memcpy(&q.a, p + 4, 4);
        memcpy(&q.b, p + 8, 4);
        step(&q, &cur, 1, first, sizeof first);
    }
    return 0;
}

static const uint32_t EDGE[] = {
    0, 1, 2, 0x0F, 0x10, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF, 0x7FFFFFFF, 0x80000000u, 0xFFFFFFFFu
};
static const uint8_t FLIN[] = { 0x00, 0x01, 0x3F, 0x2A };

static int exh_main(void) {
    unsigned long cases[NOPS] = { 0 }, bad[NOPS] = { 0 }, total = 0;
    char first[NOPS][160] = { { 0 } };
    size_t ia, ib, fi;
    int op, wc, d, s;
    St cur = INIT;

    if (!init_native()) { fprintf(stderr, "no RWX/MAP_32BIT\n"); return 2; }
    for (op = 0; op < NOPS; op++)
    for (wc = 0; wc < 4; wc++)
    for (d = 0; d < NREGS; d++)
    for (s = 0; s < NREGS; s++)
    for (ia = 0; ia < sizeof EDGE / sizeof *EDGE; ia++)
    for (ib = 0; ib < sizeof EDGE / sizeof *EDGE; ib++)
    for (fi = 0; fi < sizeof FLIN; fi++) {
        Rec q = { (uint8_t)op, (uint8_t)(wc | d << 2 | s << 4), FLIN[fi], (uint8_t)(ia * 3), EDGE[ia], EDGE[ib] };

        cases[op]++;
        bad[op] += step(&q, &cur, 0, first[op], sizeof first[op]);
    }
    for (op = 0; op < NOPS; op++) {
        printf("op %s cases=%lu bad=%lu %s\n", opname[op], cases[op], bad[op], first[op]);
        total += bad[op];
    }
    printf("mismatches=%lu\n", total);
    return total != 0;
}

int main(int argc, char **argv) {
    const char *path;
    int i;

    if (argc > 1 && !strcmp(argv[1], "exh")) return exh_main();
    if (argc > 1 && !strcmp(argv[1], "list")) {
        printf("%d\n", NOPS);
        for (i = 0; i < NOPS; i++) puts(opname[i]);
        return 0;
    }
    path = argc > 2 ? argv[2] : (argc > 1 && strcmp(argv[1], "fuzz") ? argv[1] : NULL);
    return fuzz_main(path);
}

#else
int main(void) { puts("p6_flags_diff: needs linux x86-64"); return 0; }
#endif
