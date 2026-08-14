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
 * With -D__AFL_CMPLOG=1 it additionally provides comparison logging
 * (formerly cmplog_shim.c), writing CMP records to $_CMPLOG_OUT:
 *   - libc interposition: memcmp/strcmp/strncmp/memchr/strcasecmp/
 *     strncasecmp/memmem/strstr/strcasestr
 *   - Clang -fsanitize-coverage=trace-cmp callbacks
 *     (__sanitizer_cov_trace_cmp{1,2,4,8}, trace_const_cmp*, trace_switch)
 *   - __cmplog_reset() / __tracecmp_flush() / __tracecmp_reset()
 *
 * ── Why the cmplog layer lives here and not in its own .so ────────────
 *
 * It used to be a separate cmplog_shim.c carrying its own copy of the edge
 * machinery (byte bitmap + Morris counting) behind `weak` definitions of
 * __afl_map_shm/__afl_map_reset/__sanitizer_cov_trace_pc_guard{,_init}.
 * `weak` only loses to a strong definition at STATIC link time. At dynamic
 * link time the first definition in the global lookup scope wins regardless
 * of binding, and LD_PRELOAD precedes dependency .so's -- so a preloaded
 * cmplog_shim.so preempted the target's own __afl_map_shm and the target's
 * __afl_area stayed NULL. Measured on a .so target built without
 * -Wl,-Bsymbolic: __afl_area = 0x7f4c757d6018 without the preload, (nil)
 * with it, i.e. the run recorded zero edges. Three further defects came
 * from the same duplication: the segment was attached twice (its
 * constructor re-called __afl_map_shm), AFL_MAP_SIZE was read as *bytes*
 * there and as *entries* here, and its crash handler restored the previous
 * disposition permanently so the comparison buffer was flushed on the first
 * crash only.
 *
 * One definition of the edge machinery removes all four by construction.
 * The cmplog layer now reuses this file's intrinsics: __afl_map_shm for
 * attachment, __afl_crash_handler for the pre-crash flush (every crash, not
 * just the first), __afl_auto_init for setup, and hidden visibility on the
 * trace-cmp callbacks so no LD_PRELOAD can interpose them.
 *
 * Build modes:
 *   -D__AFL_CMPLOG=1       edge coverage + comparison logging (needs -ldl)
 *   (default)              edge coverage only
 *   -D__AFL_PRELOAD_ONLY   comparison logging only, no edge machinery --
 *                          the LD_PRELOAD artifact for targets that were
 *                          never built with this shim. It deliberately
 *                          defines none of the __afl_* / trace_pc_guard
 *                          symbols, so it cannot shadow an instrumented
 *                          target the way cmplog_shim.so could.
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
/* Unconditional: dladdr/Dl_info (__AFL_DISTANCE_MODE) and RTLD_NEXT /
 * memmem / strcasestr (__AFL_CMPLOG) all need it before any system
 * header, and getting it wrong is a silent implicit-declaration. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

/* ── Build-mode gates ─────────────────────────────────────────────────
 *
 * __AFL_EDGE    the coverage machinery (default on)
 * __AFL_CMPLOG  the comparison-logging layer (default off)
 *
 * cmplog is off by default because it is not free: it defines memcmp,
 * strcmp and friends, so every call in the target routes through an
 * interposer, and the link acquires -ldl. Turning it on per target keeps
 * that cost where it buys something, and keeps __cmplog_reset out of the
 * symbol table of targets that do not have it -- which is what
 * services/fuzzer.py::_detect_cmplog reads to decide whether the target
 * can run in direct_lite mode. A shim that always exported the symbol
 * would make that probe a constant.
 */
#ifdef __AFL_PRELOAD_ONLY
#  undef __AFL_CMPLOG
#  define __AFL_CMPLOG 1
#  define __AFL_EDGE 0
#else
#  define __AFL_EDGE 1
#endif

#ifndef __AFL_CMPLOG
#  define __AFL_CMPLOG 0
#endif

/* ── Keeping the logger out of its own instrumentation ────────────────
 *
 * cmplog_shim.c was a separate translation unit, deliberately compiled
 * without -fsanitize-coverage ("it PROVIDES the callbacks, it does not
 * call them"). Merged in via -include it is part of the TARGET's TU and
 * gets instrumented with everything else.
 *
 * SanitizerCoverage skips functions named __sanitizer_cov_*, so the
 * callbacks themselves are safe -- but the record writer they call is not.
 * It contains comparisons, those comparisons get trace-cmp callbacks, and
 * the callback calls the record writer again: unbounded recursion, which
 * arrives as a stack-overflow SIGSEGV at startup rather than as anything
 * that looks like a coverage bug. Reproduced on
 * `gcc -D__AFL_CMPLOG=1 -fsanitize-coverage=trace-cmp` before this guard.
 *
 * __AFL_NO_COV suppresses instrumentation per function; the re-entrancy
 * flag in __afl_cmplog_ints is the backstop for toolchains without the
 * attribute. Both are cheap, and the failure mode without them is bad
 * enough to justify belt and braces. */
#if defined(__clang__)
#  if defined(__has_attribute) && __has_attribute(no_sanitize)
#    define __AFL_NO_COV __attribute__((no_sanitize("coverage")))
#  endif
#elif defined(__GNUC__)
#  if defined(__has_attribute) && __has_attribute(no_sanitize_coverage)
#    define __AFL_NO_COV __attribute__((no_sanitize_coverage))
#  endif
#endif
#ifndef __AFL_NO_COV
#  define __AFL_NO_COV
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
#include <sys/wait.h>

#if __AFL_CMPLOG
#include <dlfcn.h>
#include <fcntl.h>
static void __afl_cmplog_flush(void);
static void __afl_cmplog_init(void);
static void __afl_cmplog_fini(void);
#endif

#ifdef __AFL_DISTANCE_MODE
#include <dlfcn.h>
static void __afl_map_dist_shm(void);
#endif

#if __AFL_EDGE

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
    /* edge_id == 0 means "empty slot" to the probe loop below, so a valid
     * edge that hashes to 0 would be silently dropped and the slot
     * reclaimed by the next collision. Force it to 1 instead. */
    edge_id |= 1;
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

#endif /* __AFL_EDGE */

/* ══════════════════════════════════════════════════════════════════════
 * Comparison logging (-D__AFL_CMPLOG=1)
 *
 * Two interception layers, one output stream ($_CMPLOG_OUT):
 *
 *   Layer 1  libc interposition via dlsym(RTLD_NEXT) — memcmp/strcmp/...
 *            Catches explicit library calls. Needs -fno-builtin-<fn> on
 *            the target or -O2 folds the call away before it can be seen
 *            (see $NOBUILTIN_CMP in tools/build_targets.sh).
 *   Layer 2  Clang -fsanitize-coverage=trace-cmp callbacks. Catches
 *            inlined/folded integer compares and switch dispatch, which
 *            Layer 1 cannot see at all.
 *
 * Record format, unchanged from cmplog_shim.c so existing logs still
 * parse (core/cmplog.py::collect_tokens):
 *   Layer 1:  CMP <hex a> <hex b> <result> <n>
 *   Layer 2:  CMP <hex a> <hex b> <result> <n> 0x<pc>
 * ══════════════════════════════════════════════════════════════════════ */
#if __AFL_CMPLOG

#define CMPLOG_BUFFER_SIZE (256 * 1024)

/* Longest operand pair written to the cmplog stream, in bytes. The record
 * writer truncates to this, so the interceptors must not promise more than
 * it records -- and memchr sizes a stack buffer from it. */
#define CMPLOG_MAX_OPERAND 64

/* Worst-case record: "CMP " + 2*2*64 hex + 3 separators + result + n + pc
 * + newline. 320 covers it with room to spare. */
#define CMPLOG_MAX_RECORD 320

/* Raw fd, not FILE*.
 *
 * The pre-crash flush runs inside a signal handler, and fwrite/fprintf are
 * not async-signal-safe -- cmplog_shim.c called both from its handler. A
 * raw descriptor plus write(2) is safe there, and drops the stdio lock from
 * a path that runs on every intercepted comparison. O_APPEND so the
 * external truncation in CmplogCollector.reset_log() stays coherent. */
static int    __afl_cmplog_fd  = -1;
static char   __afl_cmplog_buf[CMPLOG_BUFFER_SIZE];
static size_t __afl_cmplog_pos = 0;

__AFL_NO_COV static void __afl_cmplog_flush(void) {
    if (__afl_cmplog_pos == 0 || __afl_cmplog_fd < 0) {
        __afl_cmplog_pos = 0;
        return;
    }
    size_t off = 0;
    while (off < __afl_cmplog_pos) {
        ssize_t w = write(__afl_cmplog_fd, __afl_cmplog_buf + off,
                          __afl_cmplog_pos - off);
        if (w <= 0) break;   /* EINTR/ENOSPC: drop the rest, never spin */
        off += (size_t)w;
    }
    __afl_cmplog_pos = 0;
}

/* ── Async-signal-safe integer formatting ─────────────────────────────
 * sprintf() was used for the result/width/pc fields. Hand-rolling them
 * keeps the whole record writer callable from __afl_crash_handler and
 * removes a printf parse from the hot path. */
static char *__afl_put_i64(char *p, int64_t v) {
    if (v < 0) { *p++ = '-'; v = -v; }
    char tmp[20];
    int n = 0;
    do { tmp[n++] = (char)('0' + (v % 10)); v /= 10; } while (v);
    while (n) *p++ = tmp[--n];
    return p;
}

static char *__afl_put_hex64(char *p, uint64_t v) {
    static const char hex[] = "0123456789abcdef";
    *p++ = '0'; *p++ = 'x';
    char tmp[16];
    int n = 0;
    do { tmp[n++] = hex[v & 0xf]; v >>= 4; } while (v);
    while (n) *p++ = tmp[--n];
    return p;
}

static char *__afl_put_hexbytes(char *p, const unsigned char *b, size_t n) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        *p++ = hex[b[i] >> 4];
        *p++ = hex[b[i] & 0xf];
    }
    return p;
}

/* ── Layer 1 record: two byte buffers ─────────────────────────────────
 * result == 0 is dropped: an already-satisfied comparison is exactly the
 * "looks unsolved but is solved" pollution the pair pool must not carry. */
__AFL_NO_COV static void __afl_cmplog_bytes(const void *a, const void *b, size_t n, int result) {
    if (__afl_cmplog_fd < 0 || !a || !b || n == 0 || result == 0) return;
    size_t k = n > CMPLOG_MAX_OPERAND ? CMPLOG_MAX_OPERAND : n;
    if (__afl_cmplog_pos + CMPLOG_MAX_RECORD > CMPLOG_BUFFER_SIZE)
        __afl_cmplog_flush();
    char *p = __afl_cmplog_buf + __afl_cmplog_pos;
    *p++ = 'C'; *p++ = 'M'; *p++ = 'P'; *p++ = ' ';
    p = __afl_put_hexbytes(p, (const unsigned char *)a, k);
    *p++ = ' ';
    p = __afl_put_hexbytes(p, (const unsigned char *)b, k);
    *p++ = ' ';
    p = __afl_put_i64(p, result);
    *p++ = ' ';
    p = __afl_put_i64(p, (int64_t)n);
    *p++ = '\n';
    __afl_cmplog_pos = (size_t)(p - __afl_cmplog_buf);
}

/* ── Layer 2 record: two integers plus the comparison site ────────────
 * pc is __builtin_return_address(0) from the callback: the instruction
 * after the call, i.e. the comparison site itself once inlined. */
static __thread int __afl_cmplog_busy = 0;

__AFL_NO_COV static inline void __afl_cmplog_ints(uint64_t a, uint64_t b, size_t n, void *pc) {
    if (__afl_cmplog_fd < 0) return;
    /* Backstop for toolchains where __AFL_NO_COV expands to nothing: an
     * instrumented record writer re-enters through its own trace-cmp
     * callbacks and recurses until the stack is gone. Costs one TLS read. */
    if (__afl_cmplog_busy) return;
    __afl_cmplog_busy = 1;
    if (__afl_cmplog_pos + CMPLOG_MAX_RECORD > CMPLOG_BUFFER_SIZE)
        __afl_cmplog_flush();
    unsigned char ab[8], bb[8];
    for (size_t i = 0; i < n; i++) {
        ab[i] = (unsigned char)(a >> (i * 8));
        bb[i] = (unsigned char)(b >> (i * 8));
    }
    char *p = __afl_cmplog_buf + __afl_cmplog_pos;
    *p++ = 'C'; *p++ = 'M'; *p++ = 'P'; *p++ = ' ';
    p = __afl_put_hexbytes(p, ab, n);
    *p++ = ' ';
    p = __afl_put_hexbytes(p, bb, n);
    *p++ = ' ';
    p = __afl_put_i64(p, (a < b) ? -1 : (a > b) ? 1 : 0);
    *p++ = ' ';
    p = __afl_put_i64(p, (int64_t)n);
    *p++ = ' ';
    p = __afl_put_hex64(p, (uint64_t)(uintptr_t)pc);
    *p++ = '\n';
    __afl_cmplog_pos = (size_t)(p - __afl_cmplog_buf);
    __afl_cmplog_busy = 0;
}

/* ── Layer 1: libc interposition ──────────────────────────────────────
 *
 * dlsym(RTLD_NEXT) resolution is lazy rather than constructor-only.
 * cmplog_shim.c resolved everything in its constructor and dereferenced
 * the pointers unconditionally, which is a NULL call for any comparison
 * that happens before the constructor runs. As an LD_PRELOAD object it
 * got to run early enough to mostly get away with it; compiled into the
 * target it is one constructor among many and the ordering is not ours to
 * choose. Each interceptor now resolves on first use, and falls back to a
 * naive implementation if resolution fails or would recurse (dlsym itself
 * calls into the str/mem functions we are interposing -- without the guard
 * the first memcmp would re-enter through dlsym forever). */

static __thread int __afl_in_dlsym = 0;

static void *__afl_next_sym(const char *name) {
    if (__afl_in_dlsym) return NULL;
    __afl_in_dlsym = 1;
    void *p = dlsym(RTLD_NEXT, name);
    __afl_in_dlsym = 0;
    return p;
}

typedef int   (*afl_cmp_fn)(const void *, const void *, size_t);
typedef int   (*afl_str_cmp_fn)(const char *, const char *);
typedef int   (*afl_strn_cmp_fn)(const char *, const char *, size_t);
typedef void *(*afl_chr_fn)(const void *, int, size_t);
typedef void *(*afl_memmem_fn)(const void *, size_t, const void *, size_t);
typedef char *(*afl_str_str_fn)(const char *, const char *);

static afl_cmp_fn      real_memcmp      = NULL;
static afl_str_cmp_fn  real_strcmp      = NULL;
static afl_strn_cmp_fn real_strncmp     = NULL;
static afl_chr_fn      real_memchr      = NULL;
static afl_str_cmp_fn  real_strcasecmp  = NULL;
static afl_strn_cmp_fn real_strncasecmp = NULL;
static afl_memmem_fn   real_memmem      = NULL;
static afl_str_str_fn  real_strstr      = NULL;
static afl_str_str_fn  real_strcasestr  = NULL;

/* Fallbacks. Only reached before the loader can satisfy dlsym, or if the
 * symbol genuinely is not there. Correctness first, speed irrelevant. */
__AFL_NO_COV static int   __afl_fb_lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
__AFL_NO_COV static size_t __afl_fb_len(const char *s) { const char *p = s; while (*p) p++; return (size_t)(p - s); }

__AFL_NO_COV static int __afl_fb_memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *x = a, *y = b;
    for (size_t i = 0; i < n; i++) if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
__AFL_NO_COV static int __afl_fb_strncmp(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char x = (unsigned char)a[i], y = (unsigned char)b[i];
        if (x != y) return x < y ? -1 : 1;
        if (!x) return 0;
    }
    return 0;
}
__AFL_NO_COV static int __afl_fb_strcmp(const char *a, const char *b) {
    return __afl_fb_strncmp(a, b, (size_t)-1);
}
__AFL_NO_COV static int __afl_fb_strncasecmp(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int x = __afl_fb_lower((unsigned char)a[i]), y = __afl_fb_lower((unsigned char)b[i]);
        if (x != y) return x < y ? -1 : 1;
        if (!x) return 0;
    }
    return 0;
}
__AFL_NO_COV static int __afl_fb_strcasecmp(const char *a, const char *b) {
    return __afl_fb_strncasecmp(a, b, (size_t)-1);
}
static void *__afl_fb_memchr(const void *s, int c, size_t n) {
    const unsigned char *p = s;
    for (size_t i = 0; i < n; i++) if (p[i] == (unsigned char)c) return (void *)(p + i);
    return NULL;
}
static void *__afl_fb_memmem(const void *h, size_t hl, const void *n, size_t nl) {
    if (nl == 0) return (void *)h;
    if (hl < nl) return NULL;
    const unsigned char *p = h;
    for (size_t i = 0; i + nl <= hl; i++)
        if (__afl_fb_memcmp(p + i, n, nl) == 0) return (void *)(p + i);
    return NULL;
}
static char *__afl_fb_strstr(const char *h, const char *n) {
    return (char *)__afl_fb_memmem(h, __afl_fb_len(h), n, __afl_fb_len(n));
}
static char *__afl_fb_strcasestr(const char *h, const char *n) {
    size_t nl = __afl_fb_len(n), hl = __afl_fb_len(h);
    if (nl == 0) return (char *)h;
    if (hl < nl) return NULL;
    for (size_t i = 0; i + nl <= hl; i++)
        if (__afl_fb_strncasecmp(h + i, n, nl) == 0) return (char *)(h + i);
    return NULL;
}

#define __AFL_RESOLVE(slot, type, name, fallback)                   \
    do {                                                            \
        if (!(slot)) {                                              \
            (slot) = (type)__afl_next_sym(name);                    \
            if (!(slot)) (slot) = (type)(fallback);                 \
        }                                                           \
    } while (0)

__AFL_NO_COV int memcmp(const void *a, const void *b, size_t n) {
    __AFL_RESOLVE(real_memcmp, afl_cmp_fn, "memcmp", __afl_fb_memcmp);
    int result = real_memcmp(a, b, n);
    __afl_cmplog_bytes(a, b, n, result);
    return result;
}

__AFL_NO_COV int strcmp(const char *a, const char *b) {
    __AFL_RESOLVE(real_strcmp, afl_str_cmp_fn, "strcmp", __afl_fb_strcmp);
    int result = real_strcmp(a, b);
    size_t na = __afl_fb_len(a), nb = __afl_fb_len(b), n = na < nb ? na : nb;
    if (n > 0) __afl_cmplog_bytes(a, b, n + 1, result);
    return result;
}

__AFL_NO_COV int strncmp(const char *a, const char *b, size_t n) {
    __AFL_RESOLVE(real_strncmp, afl_strn_cmp_fn, "strncmp", __afl_fb_strncmp);
    int result = real_strncmp(a, b, n);
    if (n > 0) __afl_cmplog_bytes(a, b, n, result);
    return result;
}

__AFL_NO_COV void *memchr(const void *s, int c, size_t n) {
    __AFL_RESOLVE(real_memchr, afl_chr_fn, "memchr", __afl_fb_memchr);
    void *result = real_memchr(s, c, n);
    /* A one-byte pair (s[0] vs c) is memory-safe but a weak anchor: the
     * input-to-state indexer has a single byte to locate in the input, which
     * matches everywhere and therefore nowhere useful. Materialise the needle
     * into a stack buffer instead, so the haystack side keeps a window worth
     * searching for. Only built when the search failed -- the record writer
     * discards result==0 anyway, so on a successful memchr the memset would
     * be pure cost on what is often a hot loop. */
    if (__afl_cmplog_fd >= 0 && n > 0 && !result) {
        size_t k = n > CMPLOG_MAX_OPERAND ? CMPLOG_MAX_OPERAND : n;
        unsigned char needle[CMPLOG_MAX_OPERAND];
        for (size_t i = 0; i < k; i++) needle[i] = (unsigned char)c;
        __afl_cmplog_bytes(s, needle, k, -1);
    }
    return result;
}

__AFL_NO_COV int strcasecmp(const char *a, const char *b) {
    __AFL_RESOLVE(real_strcasecmp, afl_str_cmp_fn, "strcasecmp", __afl_fb_strcasecmp);
    int result = real_strcasecmp(a, b);
    size_t na = __afl_fb_len(a), nb = __afl_fb_len(b), n = na < nb ? na : nb;
    if (n > 0) __afl_cmplog_bytes(a, b, n + 1, result);
    return result;
}

__AFL_NO_COV int strncasecmp(const char *a, const char *b, size_t n) {
    __AFL_RESOLVE(real_strncasecmp, afl_strn_cmp_fn, "strncasecmp", __afl_fb_strncasecmp);
    int result = real_strncasecmp(a, b, n);
    if (n > 0) __afl_cmplog_bytes(a, b, n, result);
    return result;
}

/* The NULL checks below are deliberate: these are declared nonnull, but a
 * fuzz target reaching them with NULL is a bug we want to log around, not
 * crash inside. GCC warns that a nonnull parameter is compared to NULL. */
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnonnull-compare"
#endif

__AFL_NO_COV void *memmem(const void *h, size_t hl, const void *n, size_t nl) {
    __AFL_RESOLVE(real_memmem, afl_memmem_fn, "memmem", __afl_fb_memmem);
    void *result = real_memmem(h, hl, n, nl);
    /* input-to-state needs one half from the buffer and one to plant;
     * log haystack-vs-needle, not needle-vs-itself.
     *
     * Pass the real outcome: the record writer drops result==0, which is the
     * filter that keeps already-solved comparisons out of the pool. A
     * hardcoded -1 logs a *successful* match as if it were still unsolved.
     *
     * Log min(hl, nl) bytes rather than requiring hl >= nl. Demanding a
     * full-length haystack dropped the case the pool needs most: an input
     * shorter than the token it must contain is exactly the state early
     * fuzzing is in, and it was logging nothing at all there. A needle prefix
     * is a partial anchor; nothing is none. */
    if (__afl_cmplog_fd >= 0 && n && nl > 0 && nl <= CMPLOG_MAX_OPERAND && hl > 0) {
        size_t k = hl < nl ? hl : nl;
        __afl_cmplog_bytes(h, n, k, result ? 0 : -1);
    }
    return result;
}

__AFL_NO_COV char *strstr(const char *h, const char *n) {
    __AFL_RESOLVE(real_strstr, afl_str_str_fn, "strstr", __afl_fb_strstr);
    char *result = real_strstr(h, n);
    if (__afl_cmplog_fd >= 0 && n && h) {
        size_t nl = __afl_fb_len(n);
        /* min(strnlen(h, nl), nl): see memmem. A haystack shorter than the
         * needle is the case worth planting into, not the case to skip. */
        size_t k = 0;
        while (k < nl && h[k]) k++;
        if (k > 0 && nl <= CMPLOG_MAX_OPERAND)
            __afl_cmplog_bytes(h, n, k, result ? 0 : -1);
    }
    return result;
}

__AFL_NO_COV char *strcasestr(const char *h, const char *n) {
    __AFL_RESOLVE(real_strcasestr, afl_str_str_fn, "strcasestr", __afl_fb_strcasestr);
    char *result = real_strcasestr(h, n);
    if (__afl_cmplog_fd >= 0 && n && h) {
        size_t nl = __afl_fb_len(n);
        size_t k = 0;
        while (k < nl && h[k]) k++;
        if (k > 0 && nl <= CMPLOG_MAX_OPERAND)
            __afl_cmplog_bytes(h, n, k, result ? 0 : -1);
    }
    return result;
}

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif

/* ── Layer 2: Clang -fsanitize-coverage=trace-cmp callbacks ───────────
 *
 * Hidden visibility, same rationale as the trace_pc_guard callbacks above:
 * the target calls these directly instead of through the PLT, so no
 * LD_PRELOAD (libasan's weak stubs, an older cmplog_shim.so) can interpose
 * them. Under __AFL_PRELOAD_ONLY they must stay exported -- interposition
 * is the entire point of that build -- so the attribute is conditional. */
#if __AFL_EDGE
#  define __AFL_CMP_VIS __AFL_NO_COV __attribute__((visibility("hidden")))
#else
#  define __AFL_CMP_VIS __AFL_NO_COV __attribute__((visibility("default")))
#endif

#define MAX_SWITCH_CASES 256

__AFL_CMP_VIS void __sanitizer_cov_trace_cmp1(uint8_t a, uint8_t b) {
    __afl_cmplog_ints(a, b, 1, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_cmp2(uint16_t a, uint16_t b) {
    __afl_cmplog_ints(a, b, 2, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_cmp4(uint32_t a, uint32_t b) {
    __afl_cmplog_ints(a, b, 4, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_cmp8(uint64_t a, uint64_t b) {
    __afl_cmplog_ints(a, b, 8, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp1(uint8_t a, uint8_t b) {
    __afl_cmplog_ints(a, b, 1, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp2(uint16_t a, uint16_t b) {
    __afl_cmplog_ints(a, b, 2, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp4(uint32_t a, uint32_t b) {
    __afl_cmplog_ints(a, b, 4, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp8(uint64_t a, uint64_t b) {
    __afl_cmplog_ints(a, b, 8, __builtin_return_address(0));
}

/* GCC declares this builtin as void(unsigned long, void *) and warns on the
 * uint64_t* form cmplog_shim.c used. That never surfaced while the shim was
 * a standalone TU compiled without -fsanitize-coverage; it does now. Match
 * the builtin and cast inside. */
__AFL_CMP_VIS void __sanitizer_cov_trace_switch(uint64_t val, void *cases) {
    if (!cases) return;
    uint64_t *ref = (uint64_t *)cases;
    int64_t count = (int64_t)ref[0];
    if (count <= 0 || count > MAX_SWITCH_CASES) return;
    void *pc = __builtin_return_address(0);
    for (int64_t i = 0; i < count; i++)
        __afl_cmplog_ints(val, ref[2 + i], 8, pc);
}

/* ── Lifecycle ────────────────────────────────────────────────────────
 * Called from __afl_auto_init (edge builds) or the preload-only
 * constructor below. */
__AFL_NO_COV static void __afl_cmplog_init(void) {
    const char *path = getenv("_CMPLOG_OUT");
    if (!path || !path[0]) return;
    __afl_cmplog_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
}

__AFL_NO_COV static void __afl_cmplog_fini(void) {
    __afl_cmplog_flush();
    if (__afl_cmplog_fd >= 0) {
        close(__afl_cmplog_fd);
        __afl_cmplog_fd = -1;
    }
}

/* ── Public API (in-process / direct_lite) ────────────────────────────
 * __cmplog_reset is also the symbol services/fuzzer.py::_detect_cmplog
 * greps for to decide whether a target has the layer compiled in, so it
 * must stay exported and must not exist in non-cmplog builds. */
__AFL_NO_COV __attribute__((visibility("default")))
void __cmplog_reset(void) {
    __afl_cmplog_flush();
    if (__afl_cmplog_fd >= 0) {
        if (ftruncate(__afl_cmplog_fd, 0) != 0) { /* best effort */ }
        lseek(__afl_cmplog_fd, 0, SEEK_SET);
    }
}

__AFL_NO_COV __attribute__((visibility("default")))
const char *__cmplog_get_path(void) { return getenv("_CMPLOG_OUT"); }

__AFL_NO_COV __attribute__((visibility("default")))
void __tracecmp_flush(void) { __afl_cmplog_flush(); }

__AFL_NO_COV __attribute__((visibility("default")))
void __tracecmp_reset(void) { __cmplog_reset(); }

__AFL_NO_COV __attribute__((visibility("default")))
const char *__tracecmp_get_path(void) { return __cmplog_get_path(); }

__attribute__((destructor))
__AFL_NO_COV static void __afl_cmplog_fini_dtor(void) { __afl_cmplog_fini(); }

#endif /* __AFL_CMPLOG */

#if __AFL_EDGE

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
#if __AFL_CMPLOG
    /* Flush before escaping. cmplog_shim.c installed a second handler for
     * this and restored the previous disposition from inside it, so the
     * comparison buffer was flushed on the FIRST crash only -- every later
     * crash in a persistent/direct_lite loop lost up to 256KB of records.
     * Folding it in here runs it on every crash, and __afl_cmplog_flush is
     * write(2)-based precisely so it is legal at this point. */
    __afl_cmplog_flush();
#endif
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

/* ── Forkserver ───────────────────────────────────────────────────────
 *
 * The point of a forkserver is that the ELF load, the dynamic linker, libc
 * init and the target's own constructors happen ONCE. Every subsequent
 * execution is a fork() from a process already sitting at that point, which
 * is why AFL gets several times the throughput of a spawn-per-input loop.
 *
 * fuzz_loader.c used to be called a forkserver while doing fork+exec per
 * input, which pays the whole tax every time: measured 0.99x against
 * posix_spawn on an ASAN target with real static init (93.8 vs 92.6
 * exec/s). The exec is the cost, so the server has to live inside the
 * target — here — not in the loader.
 *
 * Protocol (AFL's, on AFL's fd numbers):
 *   target -> FORKSRV_FD+1 : 4-byte hello, once, after init
 *   loader -> FORKSRV_FD   : 4 bytes, "run one"
 *   target -> FORKSRV_FD+1 : 4-byte child pid
 *   target -> FORKSRV_FD+1 : 4-byte wait status
 *
 * This installs late (default constructor priority), so ASAN's runtime and
 * anything else registered earlier is already up when we fork. Coverage
 * written during that pre-fork init is recorded once and then cleared by
 * the fuzzer's per-exec reset; children never re-record it. That matches
 * AFL and costs nothing — init edges are constant across inputs, so they
 * carry no signal.
 *
 * Absent the control pipe (any normal run of the binary) this is a single
 * failed read and the target runs exactly as before.                       */

#define AFL_FORKSRV_FD 198

static void __afl_start_forkserver(void) {
    char hello[4] = {0, 0, 0, 0};

    /* Opt-in: only enter forkserver mode when the loader explicitly asks
     * for it.  Without this guard, any ct ypes.CDLL()-loaded .so that happens
     * to inherit fds 198/199 from its parent would enter the forkserver loop
     * and hang the loader waiting for a command that never comes. */
    if (!getenv("__AFL_FORKSRV")) return;

    /* No control pipe: not being driven by the loader. */
    if (write(AFL_FORKSRV_FD + 1, hello, 4) != 4) return;

    /* This loop is itself instrumented — it lives in the same translation
     * unit as the target. Left recording, it would (a) write its own edges
     * into the map AFTER the fuzzer's per-exec reset, attributing the
     * server's control flow to whatever input is running, and (b) leave
     * __afl_prev_loc at a different value each iteration, so the same input
     * would produce different edge ids on its first executions.
     * (Both were measured: b'hello' gave 5, then 6, then a stable 6 edges.)
     *
     * Detaching __afl_area suppresses recording through the null check that
     * is already the first line of __afl_map_edge, so the hot path pays
     * nothing extra, and prev_loc is left untouched because that check
     * returns before prev_loc is read or written — every child forks from
     * an identical coverage state. */
    struct __afl_entry *saved_area = __afl_area;
    __afl_area = NULL;

    while (1) {
        char cmd[4];
        if (read(AFL_FORKSRV_FD, cmd, 4) != 4) _exit(0);

        pid_t child = fork();
        if (child < 0) _exit(1);

        if (child == 0) {
            /* Child: restore recording, drop the control pipe, and fall
             * through into main(). */
            __afl_area = saved_area;
            close(AFL_FORKSRV_FD);
            close(AFL_FORKSRV_FD + 1);
            return;
        }

        if (write(AFL_FORKSRV_FD + 1, &child, 4) != 4) _exit(0);

        int status = 0;
        while (waitpid(child, &status, 0) < 0) {
            /* EINTR only — a stray signal must not be read as a crash. */
        }

        if (write(AFL_FORKSRV_FD + 1, &status, 4) != 4) _exit(0);
    }
}

/* Auto-attach when loaded */
__attribute__((constructor))
static void __afl_auto_init(void) {
    /* The whole startup window runs on an unusual stack (libc init →
     * constructor), which the caller-context frame walk cannot survive in
     * every build (-O1/-O2 omit frame pointers; observed SEGV in
     * map_shm/install_crash_handlers). suppress ctx until done. */
    __afl_mapping = 1;
    __afl_map_shm();
#if __AFL_CMPLOG
    /* One constructor, one attachment. cmplog_shim.c had its own, which
     * re-entered __afl_map_shm and shmat'd the segment a second time --
     * measured 2 attachments per exec against 1 for the shim alone. */
    __afl_cmplog_init();
#endif
    __afl_install_crash_handlers();
    __afl_mapping = 0;
    /* Last: the crash handlers must already be installed in the parent so
     * every forked child inherits them. */
    __afl_start_forkserver();
}

#endif /* __AFL_EDGE */

#if !__AFL_EDGE && __AFL_CMPLOG
/* ── Preload-only lifecycle ───────────────────────────────────────────
 *
 * No edge machinery, so no __afl_guarded_call to escape to: the process
 * really is dying and the only job is to get the buffer out first.
 *
 * Hardware faults (SIGSEGV/SIGBUS/SIGFPE) must NOT be re-raised with
 * raise(): raise() produces a *software* signal, and the kernel only
 * populates siginfo_t.si_addr for hardware faults. A ptrace tracer reading
 * PTRACE_GETSIGINFO would then see si_addr=0 instead of the real faulting
 * address, silently defeating fault-address capture and collapsing
 * NULL-deref vs wild-pointer crashes into one dedup bucket. Restore the
 * previous disposition and return instead: the faulting instruction
 * re-executes, faults again in hardware, and the signal is delivered with
 * genuine si_addr intact.
 *
 * SIGABRT is software-generated -- there is no faulting instruction to
 * re-execute, so returning would resume past the abort(). Keep the
 * explicit raise(); it carries no meaningful si_addr anyway. */
static struct sigaction __afl_pre_old_segv;
static struct sigaction __afl_pre_old_abrt;
static struct sigaction __afl_pre_old_bus;
static struct sigaction __afl_pre_old_fpe;

__AFL_NO_COV static void __afl_preload_crash_handler(int sig) {
    __afl_cmplog_flush();
    struct sigaction *old;
    int hardware_fault = 0;
    switch (sig) {
    case SIGSEGV: old = &__afl_pre_old_segv; hardware_fault = 1; break;
    case SIGBUS:  old = &__afl_pre_old_bus;  hardware_fault = 1; break;
    case SIGFPE:  old = &__afl_pre_old_fpe;  hardware_fault = 1; break;
    case SIGABRT: old = &__afl_pre_old_abrt; break;
    default:      signal(sig, SIG_DFL); raise(sig); return;
    }
    sigaction(sig, old, NULL);
    if (hardware_fault)
        return;  /* re-execute the faulting instruction; preserves si_addr */
    raise(sig);
}

__attribute__((constructor))
__AFL_NO_COV static void __afl_preload_init(void) {
    __afl_cmplog_init();
    struct sigaction sa;
    sa.sa_handler = __afl_preload_crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, &__afl_pre_old_segv);
    sigaction(SIGABRT, &sa, &__afl_pre_old_abrt);
    sigaction(SIGBUS,  &sa, &__afl_pre_old_bus);
    sigaction(SIGFPE,  &sa, &__afl_pre_old_fpe);
}
#endif /* !__AFL_EDGE && __AFL_CMPLOG */
