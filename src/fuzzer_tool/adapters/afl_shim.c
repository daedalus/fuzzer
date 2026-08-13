/*
 * Sparse 8-byte edge entry shim for in-process fuzzing.
 *
 * Replaces the traditional AFL fixed-size byte bitmap with an open-addressing
 * hash table of 8-byte entries {edge_id, count}.  Each stored edge is uniquely
 * identified by its full 32-bit edge_id (caller_ctx ^ prev_loc ^ cur_loc,
 * call-stack-sensitive by default — see __afl_get_caller_ctx() below, or
 * plain prev_loc ^ cur_loc if built with -D__AFL_CTX_SENSITIVE=0) so there
 * are no silent bucket collisions.  AFL_MAP_SIZE is the number of hash table entries
 * (not bytes).  SHM size = AFL_MAP_SIZE * sizeof(struct __afl_entry) + 24
 * (24 bytes front header: stack_depth + pad + path_hash + edge_count).
 *
 * Provides:
 *   - __afl_map_shm()     — attach to SHM segment
 *   - __afl_map_edge()    — record an edge via open-addressing hash table
 *   - __afl_map_reset()   — zero all entries between iterations
 *   - __sanitizer_cov_trace_pc_guard()      — compiler-inserted edge coverage
 *   - __sanitizer_cov_trace_pc_guard_init() — compiler-inserted edge coverage
 *   - __sanitizer_cov_trace_pc()            — trace-pc distance builds
 *                                             (__AFL_DISTANCE_MODE)
 *   (no __sancov_lowest_stack definition — the sanitizer runtimes own
 *   that symbol as a TLS variable; see the stack-tracking note below)
 *
 * Metadata layout (24 bytes at front of SHM):
 *   offset 0: uint32 stack_depth   (max stack depth in bytes)
 *   offset 4: uint32 _pad
 *   offset 8: uint64 path_hash     (rolling: hash = hash * 31 ^ edge_id)
 *   offset 16: uint64 edge_count   (monotonic new-slot insertion count)
 *   offset 24+:  edge table ({edge_id, count} × map_size entries)
 *   after table:  distance tail (u64 dist_sum + u64 dist_count) in
 *                 __AFL_DISTANCE_MODE builds
 *
 * Compile target with:
 *   gcc -O2 -g -shared -fPIC -include afl_shim.c -o target.so target.c -lpng -lz
 *
 * Call-stack-sensitive edge hashing (default, see __afl_get_caller_ctx()
 * below) walks one real stack frame via __builtin_return_address(1), so
 * add -fno-omit-frame-pointer for reliable disambiguation at -O2+ (GCC/Clang
 * already default to it at -O0/-O1). Without an intact frame pointer this
 * degrades to ctx==0 — same coverage as before, not corrupted coverage.
 * Add -D__AFL_CTX_SENSITIVE=0 to opt back into the old plain
 * prev_loc^cur_loc hash unconditionally.
 */
#ifdef __AFL_DISTANCE_MODE
/* dladdr/Dl_info need _GNU_SOURCE before any system header. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#ifdef __AFL_DISTANCE_MODE
#include <dlfcn.h>
static void __afl_map_dist_shm(void);
#endif

/* ── 8-byte hash table entry ──────────────────────────────────────────
 * edge_id == 0 means empty slot.  count is a simple saturating counter
 * (no Morris probabilistic counting needed with 32-bit range).          */
struct __afl_entry {
    uint32_t edge_id;
    uint32_t count;
};

#ifndef __AFL_CTX_SENSITIVE
/* Off by default. The caller-context walk (__builtin_return_address(1))
 * dereferences the caller's return-address slot, which only exists if the
 * whole chain keeps frame pointers. clang/gcc on x86-64 omit them by
 * default at -O1/-O2, so enabling this unconditionally segfaults standard
 * builds (observed in the -O1 distance targets: SEGV in the frame walk on
 * every startup edge). Opt in per build with
 * -D__AFL_CTX_SENSITIVE=1 -fno-omit-frame-pointer (on every TU). */
#define __AFL_CTX_SENSITIVE 0
#endif

/* ── Context width ────────────────────────────────────────────────────
 *
 * The context term is XOR'd into edge_id, so its width directly bounds how
 * far context-sensitivity can inflate the number of distinct edge IDs: at
 * most 2^__AFL_CTX_BITS times the context-free count.
 *
 * That bound is the whole point. The guard values assigned by
 * trace_pc_guard_init are small sequential integers, so a context-free
 * edge_id = prev_loc ^ cur_loc lands in a dense range of roughly
 * 2 * guard_count. A full 32-bit context hash scatters those IDs across the
 * entire u32 space, which does not hurt the modulo but does mean the number
 * of LIVE distinct IDs is bounded only by the target's real call-graph
 * fan-in -- a quantity nothing in the build or the sizing path can predict.
 * The fixed-size open-addressing table then saturates, and a saturated
 * table silently drops edges (see the probe loop in __afl_map_edge).
 *
 * 8 bits is the default because it keeps the worst case within the map
 * sizes the per-execution reset can afford (see ShmCoverage.reset_edge_map:
 * the table is memset on every exec, measured at 3.8us for 8K entries and
 * 84.6us for 128K) while still separating the common case this feature
 * exists for -- the same library function reached from a handful of
 * distinct call sites. AFL++'s CTX variants mask for the same reason.
 *
 * Raise it with -D__AFL_CTX_BITS=N if a target genuinely has deep fan-in
 * AND the map has room; check the drop counter (see __afl_diag) rather than
 * guessing. __AFL_CTX_BITS=0 is equivalent to __AFL_CTX_SENSITIVE=0.
 */
#if __AFL_CTX_SENSITIVE
#  ifndef __AFL_CTX_BITS
#    define __AFL_CTX_BITS 8
#  endif
#else
#  ifdef __AFL_CTX_BITS
#    undef __AFL_CTX_BITS
#  endif
#  define __AFL_CTX_BITS 0
#endif

#if __AFL_CTX_BITS < 0 || __AFL_CTX_BITS > 32
#  error "__AFL_CTX_BITS must be in [0, 32]"
#endif

#if __AFL_CTX_BITS >= 32
#  define __AFL_CTX_MASK 0xFFFFFFFFu
#elif __AFL_CTX_BITS == 0
#  define __AFL_CTX_MASK 0u
#else
#  define __AFL_CTX_MASK ((1u << __AFL_CTX_BITS) - 1u)
#endif

/* Static advertisement of the context width, so the Python side can size
 * the coverage map BEFORE the first execution -- at sizing time there is no
 * running target to ask.  The value is encoded in the symbol NAME rather
 * than the symbol's contents so a plain symbol-table scan can read it,
 * reusing the machinery that already detects __cmplog_reset.  See
 * elf.detect_ctx_bits().  Always emitted, including as __afl_ctx_bits_0 for
 * context-free builds, so "no symbol" unambiguously means "shim predates
 * this" rather than "context is off". */
#define __AFL_CAT2(a, b) a##b
#define __AFL_CAT(a, b) __AFL_CAT2(a, b)
__attribute__((visibility("default"), used))
const uint32_t __AFL_CAT(__afl_ctx_bits_, __AFL_CTX_BITS) = __AFL_CTX_BITS;

/* Front header size (stack_depth + pad + path_hash + edge_count) */
#define SHM_HEADER_SIZE 24

/* Default number of hash table entries.  AFL_MAP_SIZE directly sets
 * __afl_map_size (number of entries, not bytes).  Default 8192 entries:
 * edge table = 8192 × 8 = 65536 bytes, header = 24 bytes, total = 65560. */
static uint32_t __afl_map_size  = 8192;

struct __afl_entry *__afl_area   = NULL;
uint32_t           __afl_prev_loc = 0;

/* Set while the SHM map / distance table is being attached. The map/setup
 * code is itself coverage-instrumented (targets -include this file), so its
 * entry fires trace_pc → map_edge before the stack contract a caller-context
 * frame walk expects exists — reading the return address there loads through
 * a bogus frame and segfaults (observed at -O1). No useful context exists
 * during setup, so callbacks skip it. Updated by __afl_map_shm, read by
 * __afl_get_caller_ctx. */
static volatile int __afl_mapping = 0;

/* Metadata pointers (front header, before the edge table) */
static uint32_t *__afl_stack_depth = NULL;   /* offset 0: uint32 */
static uint32_t *__afl_diag        = NULL;   /* offset 4: uint32 (was pad) */
static uint64_t *__afl_path_hash   = NULL;   /* offset 8: uint64 */
static uint64_t *__afl_edge_count  = NULL;   /* offset 16: uint64 */

/* ── Diagnostics word (header offset 4, previously an unused pad) ──────
 *
 *   bits  0..7   __AFL_CTX_BITS this target was built with
 *   bits  8..31  saturating count of edges DROPPED because the open-
 *                addressing probe found no free slot
 *
 * The drop counter closes a self-masking failure. When the table fills, the
 * probe loop in __afl_map_edge runs to completion and returns without
 * recording anything -- the edge is lost, silently. Every occupancy figure
 * the Python side computes is derived from edges it actually received, so a
 * saturated table looks UNDER-occupied from the outside, and
 * EdgeTracker.recommended_map_size() (which triggers on load factor > 0.7)
 * can never fire in precisely the situation it was written for.
 *
 * Counting the drops at the point of loss is the only place the information
 * exists. Increments are non-atomic, which is fine: this is a saturation
 * signal, not an accounting record, and it is only ever compared against
 * zero or used as a magnitude.
 *
 * Deliberately NOT cleared by reset_edge_map(): the header survives the
 * per-execution table wipe, so the counter accumulates over the run.       */
#define __AFL_DIAG_CTX_MASK   0xFFu
#define __AFL_DIAG_DROP_SHIFT 8
#define __AFL_DIAG_DROP_MAX   0xFFFFFFu

__attribute__((always_inline))
static inline void __afl_note_drop(void) {
    if (!__afl_diag) return;
    uint32_t v = *__afl_diag;
    uint32_t drops = v >> __AFL_DIAG_DROP_SHIFT;
    if (drops < __AFL_DIAG_DROP_MAX)
        *__afl_diag = ((drops + 1) << __AFL_DIAG_DROP_SHIFT) | (v & __AFL_DIAG_CTX_MASK);
}

/* Per-iteration state */
static uint64_t  __afl_path_hash_acc = 0;       /* rolling path hash accumulator */
static uint32_t  __afl_max_stack_depth = 0;     /* max stack depth this iteration */
static uint64_t  __afl_iter_edge_count = 0;     /* new-slot insertions this iteration */
static uint64_t  __afl_total_edge_count = 0;    /* cumulative, never reset across iterations */

/* ── SHM attachment ──────────────────────────────────────────────────── */

__attribute__((visibility("default")))
void __afl_map_shm(void) {
    char *id = getenv("__AFL_SHM_ID");
    if (!id) return;
    int shmid = atoi(id);
    if (shmid <= 0) return;

    /* Read map size from environment.  AFL_MAP_SIZE is the number of
     * hash table entries (not bytes).  The Python side allocates SHM as
     * AFL_MAP_SIZE * sizeof(struct __afl_entry) + SHM_HEADER_SIZE bytes.   */
    char *size_str = getenv("AFL_MAP_SIZE");
    if (size_str) {
        uint32_t s = (uint32_t)atoi(size_str);
        if (s > 0)
            __afl_map_size = s;
    }

    /* SHM was allocated as header bytes + table bytes */
    void *p = shmat(shmid, NULL, 0);
    if (p == (void *)-1) return;

    /* Edge table starts after the front header */
    uint8_t *base = (uint8_t *)p;
    __afl_area = (struct __afl_entry *)(base + SHM_HEADER_SIZE);

    /* Set up metadata pointers in front header (offsets 0/8/16) */
    __afl_stack_depth = (uint32_t *)(base + 0);
    __afl_diag        = (uint32_t *)(base + 4);
    __afl_path_hash   = (uint64_t *)(base + 8);
    __afl_edge_count  = (uint64_t *)(base + 16);

    /* Publish the context width so the fuzzer can confirm the map was sized
     * for the binary it is actually running, not the one it inspected.
     * Preserves any drop count already accumulated in the upper bits. */
    *__afl_diag = (*__afl_diag & ~(uint32_t)__AFL_DIAG_CTX_MASK)
                | ((uint32_t)__AFL_CTX_BITS & __AFL_DIAG_CTX_MASK);

#ifdef __AFL_DISTANCE_MODE
    __afl_mapping = 1;  /* map_dist_shm is instrumented; no ctx during setup */
    __afl_map_dist_shm();
    __afl_mapping = 0;
#endif
}

/* ── Call-stack-sensitive context ───────────────────────────────────────
 *
 * Plain prev_loc^cur_loc coverage is call-site-blind: a shared-library
 * function (or any statically-shared code path — inlined helper reused
 * across TUs, common error path, etc.) produces the IDENTICAL edge_id
 * sequence no matter which caller reached it. Two bugs only reachable
 * through different callers of the same library function look like one
 * bug to the fuzzer, and the search never learns that "reach it via
 * caller A" and "reach it via caller B" are different frontiers worth
 * exploring separately.
 *
 * __afl_get_caller_ctx() recovers a cheap 1-level call-stack context: the
 * return address of whoever called the function that CONTAINS the
 * current edge (not the edge's own PC — that's already cur_loc).
 *
 * Both call sites that invoke __afl_map_edge() are real (non-inlined)
 * functions: __sanitizer_cov_trace_pc_guard() and __sanitizer_cov_trace_pc().
 * __afl_map_edge() itself is always_inline, so it never introduces its
 * own stack frame — from the CPU's point of view, this code still runs
 * inside trace_pc_guard/trace_pc's frame regardless of the C-level call
 * boundary. Frame 0 from that vantage is trace_pc_guard's own return
 * address, i.e. the instrumented call site within the CURRENT function —
 * that's redundant with cur_loc, already captured by *guard. Frame 1
 * walks one further: the return address saved in the CURRENT function's
 * own frame, i.e. the call site of whoever called the function this edge
 * lives in. That's the missing signal — which caller reached this shared
 * code — and it stays constant for every edge hit during that one
 * invocation, exactly like AFL++'s CTX instrumentation.
 *
 * Caveats (real, not hidden):
 *   - **Requires an intact frame-pointer chain. Default clang/gcc builds
 *     omit frame pointers (-O1/-O2, x86-64), and the walk then loads the
 *     caller's return address from a slot that does not exist — a SEGV, not
 *     a graceful ctx==0. This is why __AFL_CTX_SENSITIVE defaults to 0;
 *     enabling it demands -fno-omit-frame-pointer on every TU.**
 *   - A tail call elides its own frame, so a tail-called function's true
 *     "caller of my caller" becomes invisible and two distinct call
 *     chains can collapse onto the same ctx. That under-disambiguates
 *     (fewer distinct edges than the ideal) rather than fabricating a
 *     false edge — the same conservative failure mode AFL++ accepts.
 *   - Build with `-D__AFL_CTX_SENSITIVE=0` to fall back to the old
 *     2-term hash exactly (e.g. to keep byte-for-byte corpus/edge_id
 *     compatibility with a pre-context session).                        */

/* __AFL_CTX_SENSITIVE and __AFL_CTX_BITS are configured at the top of this
 * file, because __afl_map_shm() (above) publishes the context width into
 * the SHM header and so needs them already defined. The caveats that
 * govern whether you should turn this on are documented immediately
 * above. */

#if __AFL_CTX_SENSITIVE
__attribute__((visibility("default"), always_inline))
static inline uint32_t __afl_get_caller_ctx(void) {
    if (__afl_mapping) return 0;
    void *fp = __builtin_frame_address(0);
    if (!fp) return 0;  /* frame-pointer-less build: no walkable chain */
    void *ra = __builtin_return_address(1);
    if (!ra) return 0;

    /* Fold to 32 bits via a hash, not a truncation: return addresses in
     * the same binary share high bits (load base + text segment), so a
     * plain cast would collapide distinct call sites into the same low
     * 32 bits far more than a real 64-bit space would. Fibonacci-hashing
     * style mix (splitmix64 finalizer) also spreads return addresses
     * that are only a few bytes apart (adjacent call instructions —
     * common for PLT stubs / thin wrapper callers) into different
     * buckets instead of adjacent ones. ASLR/PIE base differences across
     * runs don't matter here: we only need identical call chains WITHIN
     * one process/session to hash identically, which they do. */
    uint64_t p = (uint64_t)(uintptr_t)ra;
    p ^= p >> 33;
    p *= 0xff51afd7ed558ccdULL;
    p ^= p >> 33;
    p *= 0xc4ceb9fe1a85ec53ULL;
    p ^= p >> 33;
    /* Mask AFTER mixing, never before: the splitmix finalizer is what
     * spreads adjacent call sites apart, so truncating its output keeps
     * that spreading while bounding the ID inflation. Masking the raw
     * return address instead would put neighbouring call instructions in
     * the same bucket, which is exactly what the mixing exists to avoid. */
    return (uint32_t)p & __AFL_CTX_MASK;
}
#endif

/* ── Edge recording (open-addressing hash table) ───────────────────────
 *
 * Hash: edge_id = caller_ctx ^ prev_loc ^ cur_loc  (__AFL_CTX_SENSITIVE=1)
 *       edge_id = prev_loc ^ cur_loc               (__AFL_CTX_SENSITIVE=0)
 * caller_ctx disambiguates identical prev_loc^cur_loc sequences reached
 * through different call chains (e.g. the same shared-library function
 * invoked from two different call sites) — see __afl_get_caller_ctx().
 * Probe: linear probing from edge_id % map_size until we find a matching
 *        edge_id or an empty slot (edge_id == 0).                       */

__attribute__((visibility("default"), always_inline))
static inline void __afl_map_edge(uint32_t cur_loc) {
    if (!__afl_area) return;

#if __AFL_CTX_SENSITIVE
    uint32_t caller_ctx = __afl_get_caller_ctx();
    uint32_t edge_id = caller_ctx ^ __afl_prev_loc ^ cur_loc;
#else
    uint32_t edge_id = __afl_prev_loc ^ cur_loc;
#endif
    uint32_t pos     = edge_id % __afl_map_size;

    /* Linear probe: at most map_size iterations guarantees we either
     * find the edge or hit an empty slot. */
    for (uint32_t i = 0; i < __afl_map_size; i++) {
        uint32_t idx = (pos + i) % __afl_map_size;
        uint32_t eid = __afl_area[idx].edge_id;

        if (eid == 0) {                              /* empty slot — claim */
            __afl_area[idx].edge_id = edge_id;
            __afl_area[idx].count   = 1;
            __afl_iter_edge_count++;                 /* track per-iteration new-slot insertion */
            __afl_total_edge_count++;                /* track cumulative across-reset count */
            if (__afl_edge_count)                    /* write CUMULATIVE count live to SHM header */
                *__afl_edge_count = __afl_total_edge_count;
            break;
        }
        if (eid == edge_id) {                        /* existing edge — bump */
            if (__afl_area[idx].count < UINT32_MAX)
                __afl_area[idx].count++;
            break;
        }
        /* else: hash collision, keep probing */

        /* Last iteration and still nowhere to put it: the table is full and
         * this edge is about to be lost. Count it -- see __afl_diag. */
        if (i == __afl_map_size - 1)
            __afl_note_drop();
    }

    /* Accumulate rolling path hash: hash = hash * 31 ^ edge_id */
    __afl_path_hash_acc = (__afl_path_hash_acc * 31) ^ edge_id;
    if (__afl_path_hash)
        *__afl_path_hash = __afl_path_hash_acc;

    __afl_prev_loc = cur_loc >> 1;
}

/* ── Compiler-inserted edge coverage callbacks ────────────────────────
 * Hidden visibility: PIE builds call these via the PLT, so a libasan
 * LD_PRELOAD (the fuzzer-tool CLI preloads it for ASAN targets) would
 * interpose its own weak stubs over ours.  Hidden visibility forces
 * direct call instructions within the target, bypassing PLT resolution
 * entirely (same pattern as the abort() override below). */

__attribute__((visibility("hidden")))
void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!guard || *guard == 0) return;
    __afl_map_edge(*guard);
}

__attribute__((visibility("hidden")))
void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    static uint32_t guard_counter;
    if (start == stop || *start) return;
    for (uint32_t *g = start; g < stop; g++)
        *g = ++guard_counter;
}

/* ── AFLGo distance channel (__AFL_DISTANCE_MODE builds only) ─────────
 *
 * Distance builds compile the target with -fsanitize-coverage=trace-pc
 * instead of trace-pc-guard, so __sanitizer_cov_trace_pc() receives the
 * PC of every instrumented site.  We (1) record the edge (PC-based),
 * and (2) look up the block's AFLGo distance in a table the fuzzer
 * uploads to a second SHM segment (__AFL_DIST_SHM_ID), accumulating
 * sum/count written to the tail of the coverage SHM at reset.
 *
 * The distance table keys are block addresses relative to the object's
 * load base; the runtime key is pc - dladdr_base.  Entries with
 * key == 0 are empty slots.  Blocks without a table entry do not
 * contribute to the average (AFLGo semantics).                       */

#ifdef __AFL_DISTANCE_MODE

/* Layout must match DistanceTableShm (Python): 4-byte header = SLOT
 * capacity (power of two >= 2x entries — the slack guarantees empty
 * slots so the k == 0 probe break fires on misses instead of scanning
 * the whole table), then 12-byte entries (u64 key, u32 dist) inserted
 * at key % capacity with linear probing, exactly like the lookup
 * below.  Packed keeps the C stride at 12 — without it the struct pads
 * to 16 and every entry misreads. */
struct __afl_dist_entry {
    uint64_t key;
    uint32_t dist;
} __attribute__((packed));

static uint32_t  *__afl_dist_count = NULL;  /* entries + count at segment head */
static struct __afl_dist_entry *__afl_dist_table = NULL;
static uint64_t   __afl_base = 0;           /* dladdr-derived object base */
static uint64_t   __afl_dist_sum = 0;
static uint64_t   __afl_dist_hits = 0;

static void __afl_map_dist_shm(void) {
    char *id = getenv("__AFL_DIST_SHM_ID");
    if (!id) return;
    int shmid = atoi(id);
    if (shmid <= 0) return;
    void *p = shmat(shmid, NULL, 0);
    if (p == (void *)-1) return;
    __afl_dist_count = (uint32_t *)p;
    __afl_dist_table = (struct __afl_dist_entry *)((uint8_t *)p + 4);
}

/* Hidden visibility: same PLT-interposition rationale as the guard
 * callbacks — the CLI's libasan LD_PRELOAD must not shadow this. */
__attribute__((visibility("hidden")))
void __sanitizer_cov_trace_pc(void) {
    uintptr_t pc = (uintptr_t)__builtin_return_address(0);
    if (__afl_base == 0) {
        Dl_info info;
        if (dladdr((void *)pc, &info) && info.dli_fbase)
            __afl_base = (uintptr_t)info.dli_fbase;
        else
            __afl_base = 1;  /* dladdr failed — treat the PC as absolute */
    }
    uint64_t key = (uint64_t)pc - __afl_base;

    /* Edge coverage: PC-based (prev_loc ^ cur_loc, same sparse table). */
    __afl_map_edge((uint32_t)(key >> 1));

    if (!__afl_dist_table || !__afl_dist_count) return;
    uint32_t size = *__afl_dist_count;
    if (size == 0) return;
    uint32_t pos = (uint32_t)(key % size);
    for (uint32_t i = 0; i < size; i++) {
        uint32_t idx = (pos + i) % size;
        uint64_t k = __afl_dist_table[idx].key;
        if (k == 0) break;  /* empty slot — no distance for this block */
        if (k == key) {
            __afl_dist_sum += __afl_dist_table[idx].dist;
            __afl_dist_hits++;
            break;
        }
    }
}

#endif /* __AFL_DISTANCE_MODE */

#ifdef __AFL_DISTANCE_MODE
/* Write the accumulated distance sum/count to the SHM tail (16 bytes
 * past the edge table; the Python side always allocates them).
 * count==0 means "no distance data" for the reader. */
static void __afl_write_distance_tail(void) {
    if (__afl_area) {
        uint64_t *dist_sum = (uint64_t *)((uint8_t *)__afl_area +
                                          __afl_map_size * sizeof(struct __afl_entry));
        *dist_sum = __afl_dist_sum;
        *(dist_sum + 1) = __afl_dist_hits;
    }
}
#endif /* __AFL_DISTANCE_MODE */

/* ── LLVM stack depth tracking ────────────────────────────────────────
 * The sanitizer runtimes (ASAN/TSAN/UBSAN coverage) provide
 * __sancov_lowest_stack as a TLS variable that they call themselves; we
 * must NOT define it — under clang the sanitizer runtime is linked for
 * any sanitizer or sanitize-coverage build, and a function
 * definition with the same name collides with the runtime's TLS
 * variable at link time ("TLS definition ... mismatches non-TLS").
 * Nothing outside the runtimes calls it, so omitting it is safe. */

/* ── Reset (zero all entries between iterations) ───────────────────────
 * Also writes accumulated metadata (stack_depth, path_hash) to the
 * metadata region so Python can read them, then resets accumulators. */

__attribute__((visibility("default")))
void __afl_map_reset(void) {
    if (__afl_area) {
        memset(__afl_area, 0, __afl_map_size * sizeof(struct __afl_entry));

        /* Write metadata before resetting accumulators */
        if (__afl_stack_depth) {
            *__afl_stack_depth = __afl_max_stack_depth;
        }
        if (__afl_path_hash) {
            *__afl_path_hash = __afl_path_hash_acc;
        }
        if (__afl_edge_count) {
            *__afl_edge_count = __afl_total_edge_count;
        }
#ifdef __AFL_DISTANCE_MODE
        __afl_write_distance_tail();
#endif
    }
    __afl_prev_loc = 0;
    __afl_path_hash_acc = 0;
    __afl_max_stack_depth = 0;
    __afl_iter_edge_count = 0;
#ifdef __AFL_DISTANCE_MODE
    __afl_dist_sum = 0;
    __afl_dist_hits = 0;
#endif
}

#ifdef __AFL_DISTANCE_MODE
/* In-process (direct_lite) mode has no process boundary between
 * iterations and nothing calls __afl_map_reset — export a flush that
 * writes the accumulated tail and zeroes the accumulators WITHOUT
 * touching the edge table (the fuzzer reads coverage after the run).
 * The Python runner calls it after each in-process run. */
__attribute__((visibility("default")))
void __afl_dist_flush(void) {
    __afl_write_distance_tail();
    __afl_dist_sum = 0;
    __afl_dist_hits = 0;
}

/* One-shot subprocess runs never call __afl_map_reset — write the tail
 * at process exit instead.  (Persistent/in-process loops call reset per
 * iteration and the destructor only repeats the final values.) */
__attribute__((destructor))
static void __afl_write_distance_tail_exit(void) {
    __afl_write_distance_tail();
}
#endif /* __AFL_DISTANCE_MODE */

/* ── Crash signal handler ─────────────────────────────────────────────
 * Uses sigsetjmp/siglongjmp instead of the traditional restore-and-re-raise
 * approach because glibc's abort() resets the handler to SIG_DFL after the
 * first SIGABRT and re-raises — killing the process before the fuzzer can
 * recover.  siglongjmp escapes the signal handler entirely, jumping back to
 * __afl_guarded_call which can then report the crash via its return value.
 *
 * This is needed for crashes from pre-compiled code (libasan, libc) that
 * bypasses the abort() preprocessor override below.  The override covers
 * the target's own source (FFmpeg av_assert0, etc.).                         */

static sigjmp_buf __afl_jmp_buf;
static struct sigaction __afl_old_handlers[8];
static int __afl_guard_signals[] = {
    SIGSEGV, SIGABRT, SIGFPE, SIGBUS, SIGILL, SIGPIPE, SIGSYS,
};
#define __afl_NUM_GUARD_SIGNALS \
    (int)(sizeof(__afl_guard_signals) / sizeof(__afl_guard_signals[0]))

static void __afl_crash_handler(int sig) {
    siglongjmp(__afl_jmp_buf, sig);
}

static void __afl_install_crash_handlers(void) {
    struct sigaction sa;
    sa.sa_handler = __afl_crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    for (int i = 0; i < __afl_NUM_GUARD_SIGNALS; i++)
        sigaction(__afl_guard_signals[i], &sa, &__afl_old_handlers[i]);
}

/* ── Guarded call wrapper ─────────────────────────────────────────────
 * InProcessRunner's direct_lite mode calls this instead of calling the
 * fuzz entry function directly.  __afl_guarded_call sets up a sigsetjmp
 * buffer, calls the entry function, and returns normally.  If a signal
 * (SIGSEGV/SIGABRT) fires during the call, __afl_crash_handler does
 * siglongjmp back here, and we return a negative signal-indicating value.
 *
 * Returns the entry function's return value on success, or -sig (negated
 * signal number, e.g. -6 for SIGABRT, -11 for SIGSEGV) on crash.  This
 * matches the signal-indicating exit code convention used throughout the
 * fuzzer (run_target_stdin etc. return -sig on signal).                      */

__attribute__((visibility("default")))
int __afl_guarded_call(int (*entry)(const uint8_t *, size_t),
                       const uint8_t *data, size_t size) {
    int sig;
    if ((sig = sigsetjmp(__afl_jmp_buf, 1)) == 0)
        return entry(data, size);
    /* sig = signal number from __afl_crash_handler's siglongjmp */
    return -(int)sig;
}

/* ── abort() override (all builds) ───────────────────────────────────
 * Intercept libc's abort() for fuzzing: instead of killing the process,
 * write a marker to stderr and return.  Libraries like FFmpeg call abort()
 * on internal assertion failures (~1600 av_assert0 sites), which would
 * flood the fuzzer with false crashes.  This override catches ALL abort()
 * sources — macros, direct calls in library .c files, etc.
 *
 * In ASAN builds, the __asan_default_options shim (loaded before libasan
 * by the fuzzer) sets halt_on_error=0:abort_on_error=0, so ASAN-detected
 * errors write the report to stderr and return normally — abort() is no
 * longer part of ASAN's own error path.  This makes it safe to override
 * abort() unconditionally: library-internal assertions (av_assert0) are
 * caught, while ASAN bugs produce their full report via stderr capture
 * and the SanitizerReport pipeline.
 *
 * The noreturn warning is suppressed because we intentionally override
 * the behavior for the fuzzing use case.                                  */

/* Override libc abort() for all fuzzing builds.
 *
 * Instead of killing the process, write a marker to stderr and return.
 * Libraries like FFmpeg call abort() on internal assertion failures
 * (~1600 av_assert0 sites), which would flood the fuzzer with false
 * crash detections.
 *
 * A macro + static helper avoids the GCC "noreturn function does return"
 * warning that would fire if we defined void abort(void) directly
 * (stdlib.h declares abort() as __noreturn__).  The preprocessor
 * replaces all abort() calls with __afl_shim_abort() before the compiler
 * sees the declaration mismatch.                                           */
static void __afl_shim_abort(void) {
    static const char msg[] = "[shim] abort() intercepted\n";
    write(STDERR_FILENO, msg, sizeof(msg) - 1);
}
#define abort() __afl_shim_abort()

/* Auto-attach when loaded */
__attribute__((constructor))
static void __afl_auto_init(void) {
    /* The whole startup window runs on an unusual stack (libc init →
     * constructor), which the caller-context frame walk cannot survive in
     * every build (-O1/-O2 omit frame pointers; observed SEGV in
     * map_shm/install_crash_handlers). suppress ctx until done. */
    __afl_mapping = 1;
    __afl_map_shm();
    __afl_install_crash_handlers();
    __afl_mapping = 0;
}
