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
 *     strncasecmp/memmem/strstr/strcasestr, plus bcmp, wmemcmp, wcscmp,
 *     wcsncmp, wcscasecmp, strpbrk, strspn, strcspn, memrchr
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
 * Default-on channels (both opt-out with =0):
 *   __AFL_CTX_SENSITIVE=1  call-stack-sensitive edge hashing
 *   __AFL_DISTANCE_MODE=1  AFLGo SHM-tail distance channel (inert until
 *                          the fuzzer uploads a table via __AFL_DIST_SHM_ID)
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
 * below) walks one real stack frame via the saved frame pointer, so
 * build every shim TU with -fno-omit-frame-pointer for reliable
 * disambiguation at -O2+ (GCC/Clang already keep it at -O0/-O1). Without
 * an intact frame pointer the bounds-checked walk yields junk-or-zero
 * context — degraded signal, never a crash. Add -D__AFL_CTX_SENSITIVE=0
 * to opt back into the old plain prev_loc^cur_loc hash unconditionally.
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
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>

#if __AFL_CMPLOG
#include <dlfcn.h>
#include <fcntl.h>
#include <strings.h>
#include <wchar.h>
static void __afl_cmplog_flush(void);
static void __afl_cmplog_init(void);
static void __afl_cmplog_fini(void);
#endif

#ifndef __AFL_DISTANCE_MODE
/* AFLGo distance channel defaults ON; -D__AFL_DISTANCE_MODE=0 opts out.
 * The channel is inert unless the fuzzer uploads a distance table via
 * __AFL_DIST_SHM_ID (directed mode) — without one, sum/count stay 0 and
 * every reader degrades to the no-data path. */
#define __AFL_DISTANCE_MODE 1
#endif

#if __AFL_DISTANCE_MODE
#include <dlfcn.h>
static void __afl_map_dist_shm(void);
static void __afl_map_node_shm(void);
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
/* On by default: call-stack context separates identical edges reached via
 * different callers. The walk below is hardened (bounds-checked single hop,
 * constructor-window guard), so frame-pointer-less builds degrade to noisy
 * or zero context rather than crash -- but meaningful disambiguation needs
 * -fno-omit-frame-pointer on every TU including this one (clang/gcc omit
 * them at -O1/-O2). Pass -D__AFL_CTX_SENSITIVE=0 to fall back to the old
 * plain prev_loc^cur_loc hash exactly. */
#define __AFL_CTX_SENSITIVE 1
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

/* ── n-gram history depth ─────────────────────────────────────────────
 *
 * k = blocks encoded into one edge id: the current block plus its k−1
 * predecessors. k=2 IS the historical behaviour and stays byte-identical:
 * same layout, same exported __afl_prev_loc, same XOR edge ids, so existing
 * corpora and resume state remain valid (docs/ngram_coverage_plan.md,
 * Compatibility Concerns). Only k>2 introduces the ring buffer and the
 * FNV-1a mix, which deliberately changes every edge id.
 */
#ifndef __AFL_NGRAM_K
#define __AFL_NGRAM_K 2
#endif

#if __AFL_NGRAM_K < 2
#error "__AFL_NGRAM_K must be >= 2"
#endif
#if __AFL_NGRAM_K > 4096
#error "__AFL_NGRAM_K too large (ring BSS / index sanity)"
#endif

__attribute__((visibility("default"), used))
const uint32_t __AFL_CAT(__afl_ngram_k_, __AFL_NGRAM_K) = __AFL_NGRAM_K;

/* Front header size (stack_depth + pad + path_hash + edge_count) */
#define SHM_HEADER_SIZE 24

/* Default number of hash table entries.  AFL_MAP_SIZE directly sets
 * __afl_map_size (number of entries, not bytes).  Default 8192 entries:
 * edge table = 8192 × 8 = 65536 bytes, header = 24 bytes, total = 65560. */
static uint32_t __afl_map_size  = 8192;

struct __afl_entry *__afl_area   = NULL;
#if __AFL_NGRAM_K == 2
uint32_t           __afl_prev_loc = 0;
#else
/* k−1 predecessor slots, FIFO via __afl_prev_idx (next-overwrite target =
 * oldest entry). The index is uint32_t on purpose: an 8-bit counter wraps
 * at 256 and mis-addresses rings once k−1 > 255. */
static uint32_t __afl_prev_locs[__AFL_NGRAM_K - 1];
static uint32_t __afl_prev_idx  = 0;
#endif

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
#define __AFL_DIAG_DROP_MAX   0xFFFFu
#define __AFL_DIAG_GEN_SHIFT  24
#define __AFL_DIAG_GEN_MASK   0xFFu

/* Maximum linear-probe distance in __afl_map_edge, for both lookup and
 * insertion. Bounds the per-edge-execution cost to a constant instead of
 * O(map_size) in the worst case.
 *
 * The trade is a drop rate at high load: an edge whose whole window is
 * occupied by other edges is discarded and counted via __afl_note_drop(),
 * so the cost is observable through ShmCoverage.read_dropped_edges().
 * Simulated drop rate against the only target in the tree that exceeds the
 * 8192-entry floor (ffmpeg_read: 201,279 distinct edges in a 262,144-entry
 * map, load 0.77):
 *
 *     window   8 -> 4.43%      window  32 -> 0.40%
 *     window  16 -> 1.60%      window  64 -> 0.04%
 *
 * 64 is the default because a dropped edge is permanently invisible to the
 * fuzzer, not merely delayed, and 0.04% buys nearly all of the cost bound
 * that 16 does. Every other instrumented target sits at ~13% load, where
 * every one of these windows drops nothing at all. Override at build time
 * with -D__AFL_PROBE_MAX=N. */
#ifndef __AFL_PROBE_MAX
#define __AFL_PROBE_MAX 64u
#endif

__attribute__((always_inline))
static inline void __afl_note_drop(void) {
    if (!__afl_diag) return;
    uint32_t v = *__afl_diag;
    uint32_t drops = (v >> __AFL_DIAG_DROP_SHIFT) & 0xFFFF;
    if (drops < __AFL_DIAG_DROP_MAX)
        *__afl_diag = ((drops + 1) << __AFL_DIAG_DROP_SHIFT) | (v & __AFL_DIAG_CTX_MASK) | (v & (0xFFu << __AFL_DIAG_GEN_SHIFT));
}

/* Per-iteration state */
static uint64_t  __afl_path_hash_acc = 0;       /* rolling path hash accumulator */
static uint32_t  __afl_max_stack_depth = 0;     /* max stack depth this iteration */
static uint64_t  __afl_iter_edge_count = 0;     /* new-slot insertions this iteration */
static uint64_t  __afl_total_edge_count = 0;    /* cumulative, never reset across iterations */
static uint8_t   __afl_generation = 0;          /* generation counter for tag-based reset */

/* ── SHM attachment ──────────────────────────────────────────────────── */

__attribute__((visibility("default")))
void __afl_map_shm(void) {
    char *id = getenv("__AFL_SHM_ID");
    if (!id) return;   /* not under the fuzzer — silence is correct here */

    /* Past this point the fuzzer has explicitly asked for coverage, so a
     * failure must not be silent. It used to be: all three early returns
     * below left __afl_area NULL, the target then ran to completion and
     * exited 0 having recorded nothing, and the fuzzer read back an
     * all-zero header indistinguishable from "the child never wrote".
     * That is the whole of the "Loose thread" in
     * docs/edge-coverage-analysis.md -- three sightings across ~50 runs,
     * unresolvable each time because neither side left a trace.
     *
     * write(2) rather than fprintf: this runs from a constructor, before
     * the target's own stdio setup, and may run inside a forkserver child.
     * The wording deliberately contains none of the tokens
     * ExecutionRunner.is_crash() scans stderr for (SIGSEGV, SIGABRT,
     * SIGFPE, SIGBUS, "Segmentation fault", "Aborted"), so a diagnostic
     * cannot be misread as a crashing input. */

    /* strtol, not atoi: atoi cannot fail. It returns 0 for "0", for "" and
     * for "banana" alike, so the only malformed id the old `shmid < 0`
     * guard could catch was an explicitly negative one. Everything else
     * fell through to shmat() and came back as an attach failure
     * ("shmat(0) failed: Invalid argument") rather than as the parse
     * failure it actually was -- a diagnostic pointing at the wrong half
     * of the operation. That is what
     * TestAttachFailureIsLoud::test_unparseable_id_is_reported was seeing:
     * it passes "0", which atoi happily accepts.
     *
     * Reachable outside the test: __AFL_SHM_ID is inherited across exec,
     * so anything that truncates or clobbers the environment yields a
     * malformed id, not a negative one.
     *
     * 0 itself is *not* rejected here. It is syntactically a valid id, and
     * the kernel can hand one out (ipc ids start at seq 0), so refusing it
     * would trade this bug for a rarer one. A bogus 0 still fails loudly,
     * one line down, at shmat. */
    errno = 0;
    char *end = NULL;
    long parsed = strtol(id, &end, 10);
    if (end == id || *end != '\0' || errno == ERANGE || parsed < 0 || parsed > INT_MAX) {
        char msg[128];
        int n = snprintf(msg, sizeof(msg),
                         "__afl_shim: __AFL_SHM_ID=%.32s is not a valid segment id"
                         " -- coverage disabled\n", id);
        if (n > 0) { ssize_t w = write(2, msg, (size_t)n); (void)w; }
        return;
    }
    int shmid = (int)parsed;

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
    if (p == (void *)-1) {
        char msg[160];
        int n = snprintf(msg, sizeof(msg),
                         "__afl_shim: shmat(%d) failed: %.64s"
                         " -- coverage disabled\n", shmid, strerror(errno));
        if (n > 0) { ssize_t w = write(2, msg, (size_t)n); (void)w; }
        return;
    }

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
    *__afl_diag = (*__afl_diag & ~(uint32_t)(__AFL_DIAG_CTX_MASK | (0xFFu << __AFL_DIAG_GEN_SHIFT)))
                | ((uint32_t)__AFL_CTX_BITS & __AFL_DIAG_CTX_MASK);

#if __AFL_DISTANCE_MODE
    __afl_mapping = 1;  /* map_dist_shm is instrumented; no ctx during setup */
    __afl_map_dist_shm();
    __afl_map_node_shm();
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
 *   - **Wants an intact frame-pointer chain. Default clang/gcc builds
 *     omit frame pointers (-O1/-O2, x86-64); the walk below is hardened
 *     (bounds-checked single hop) so such builds do not SEGV, but the
 *     context is then junk-or-zero rather than a real caller. Meaningful
 *     disambiguation demands -fno-omit-frame-pointer on every TU —
 *     tools/build_targets.sh applies it to every shim build.**
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
    /* Safe two-level unwind.  trace-pc-guard fires at basic-block entry,
     * including a function's *entry* block before its prologue has linked
     * the frame pointer.  At that instant the second frame up is not yet
     * established, so __builtin_return_address(1) — documented as possibly
     * crashing for any nonzero argument — walks an unlinked pointer and
     * faults.  Instead we read the saved frame pointer by hand and
     * bounds-check the single unvalidated hop against the current stack
     * window, returning 0 (no context for this edge) rather than
     * dereferencing a wild pointer.  fp[0] is our own saved FP and is
     * always valid; caller_fp[1] is only read after the range check. */
    void **fp = (void **)__builtin_frame_address(0);
    if (!fp) return 0;  /* frame-pointer-less build: no walkable chain */
    uintptr_t cur = (uintptr_t)fp;
    void **caller_fp = (void **)fp[0];          /* saved FP of the frame above */
    uintptr_t cfp = (uintptr_t)caller_fp;
    /* The stack grows down, so a genuine older frame sits at a higher
     * address than ours and within a sane single-hop span (4 MiB covers
     * any realistic frame without risking a wild read).  Anything outside
     * that window is an unlinked/garbage frame — skip context for it. */
    if (cfp <= cur || cfp - cur > (4u << 20)) return 0;
    void *ra = caller_fp[1];                    /* return addr into caller's caller */
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

    uint32_t gen = __afl_generation;
    if (__afl_diag)
        gen = (*__afl_diag >> __AFL_DIAG_GEN_SHIFT) & 0xFF;

#if __AFL_NGRAM_K > 2
    /* FNV-1a over the k−1 ring slots (oldest→newest from __afl_prev_idx)
     * then cur_loc. Order-sensitive and cheap (2 ops/slot); XOR chains go
     * commutative and lose path direction once k>2. */
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < (uint32_t)(__AFL_NGRAM_K - 1); i++) {
        h ^= __afl_prev_locs[(__afl_prev_idx + i) % (__AFL_NGRAM_K - 1)];
        h *= 16777619u;
    }
    h ^= cur_loc;
    h *= 16777619u;
# if __AFL_CTX_SENSITIVE
    uint32_t edge_id = __afl_get_caller_ctx() ^ h;
# else
    uint32_t edge_id = h;
# endif
#else
#if __AFL_CTX_SENSITIVE
    uint32_t caller_ctx = __afl_get_caller_ctx();
    uint32_t edge_id = caller_ctx ^ __afl_prev_loc ^ cur_loc;
#else
    uint32_t edge_id = __afl_prev_loc ^ cur_loc;
#endif
#endif
    /* edge_id == 0 means "empty slot" to the probe loop below, so a valid
     * edge that hashes to 0 would be silently dropped and the slot
     * reclaimed by the next collision. Force it to 1 instead. */
    edge_id |= 1;
    uint32_t pos     = edge_id % __afl_map_size;

    /* Linear probe, bounded to __AFL_PROBE_MAX slots.
     *
     * The bound converts a map_size-iteration worst case into a constant.
     * It is only correct because insertion is bounded by the same constant:
     * an edge is therefore always within __AFL_PROBE_MAX of its home slot
     * or absent, so a bounded lookup can never miss an edge a bounded
     * insert placed. Do not bound one without the other. */
    uint32_t window = __AFL_PROBE_MAX;
    if (window > __afl_map_size) window = __afl_map_size;

    for (uint32_t i = 0; i < window; i++) {
        uint32_t idx = (pos + i) % __afl_map_size;
        uint32_t eid = __afl_area[idx].edge_id;

        if (eid == 0) {                              /* empty slot — claim */
            __afl_area[idx].edge_id = edge_id;
            __afl_area[idx].count   = (gen << 24) | 1;
            __afl_iter_edge_count++;                 /* track per-iteration new-slot insertion */
            __afl_total_edge_count++;                /* track cumulative across-reset count */
            if (__afl_edge_count)                    /* write CUMULATIVE count live to SHM header */
                *__afl_edge_count = __afl_total_edge_count;
            break;
        }
        if (eid == edge_id) {                        /* existing edge */
            if ((__afl_area[idx].count >> 24) == gen) {
                if ((__afl_area[idx].count & 0x00FFFFFFu) < 0x00FFFFFFu)
                    __afl_area[idx].count++;
                break;
            }
            /* Stale entry for this same edge: reclaim IN PLACE.
             *
             * This branch used to fall through and keep probing, which meant
             * every generation inserted a *fresh duplicate* of every edge
             * that fired, in a new slot, while the stale copy was never
             * freed. The table therefore filled at (edges per exec) slots
             * per execution regardless of how few distinct edges the target
             * had, saturated after roughly map_size/edges_per_exec
             * executions, and from then on could claim no slot at all --
             * every subsequent execution reported ZERO current-generation
             * edges, silently ending coverage guidance mid-campaign.
             * Measured pre-fix on an 8192-entry map with 43 distinct edges:
             * saturated at exec ~190, edge visibility 43 -> 0 at exec 200.
             *
             * Reclaiming in place makes table occupancy the union of
             * distinct edges ever seen, which is bounded by the target's
             * guard count -- the behaviour the generation design intended.
             * total_edge_count is deliberately NOT bumped here: this edge
             * already owns a slot, so it is not a newly discovered edge. */
            __afl_area[idx].count = (gen << 24) | 1;
            __afl_iter_edge_count++;
            break;
        }
        /* else: hash collision against a live or stale *different* edge —
         * keep probing. A stale different edge is not reclaimed: its slot
         * still records an edge this target has genuinely reached, and
         * dropping it would lose cumulative coverage. */

        /* Window exhausted and still nowhere to put it. Unlike the old
         * unbounded loop, this does not mean the table is full -- it means
         * this edge's neighbourhood is. Count it either way: a dropped edge
         * is invisible to the fuzzer, and read_dropped_edges() is how that
         * cost is meant to be observed. */
        if (i == window - 1)
            __afl_note_drop();
    }

    /* Accumulate rolling path hash: hash = hash * 31 ^ edge_id */
    __afl_path_hash_acc = (__afl_path_hash_acc * 31) ^ edge_id;
    if (__afl_path_hash)
        *__afl_path_hash = __afl_path_hash_acc;

#if __AFL_NGRAM_K > 2
    __afl_prev_locs[__afl_prev_idx] = cur_loc >> 1;
    __afl_prev_idx = (__afl_prev_idx + 1) % (__AFL_NGRAM_K - 1);
#else
    __afl_prev_loc = cur_loc >> 1;
#endif
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

#if __AFL_DISTANCE_MODE

/* Layout must match DistanceTableShm (Python): 4-byte header = SLOT
 * capacity (power of two >= 2x entries — the slack guarantees empty
 * slots so the k == 0 probe break fires on misses instead of scanning
 * the whole table), then 16-byte entries (u64 key, u32 dist, u32
 * node_idx) inserted at key % capacity with linear probing, exactly
 * like the lookup below. Packed keeps the C stride at 16 — without it
 * the struct pads and every entry misreads. node_idx feeds the
 * K-Scheduler node bitmap; NODE_IDX_NONE-equivalent values fail the
 * bounds check below. */
struct __afl_dist_entry {
    uint64_t key;
    uint32_t dist;
    uint32_t node_idx;
} __attribute__((packed));

static uint32_t  *__afl_dist_count = NULL;  /* entries + count at segment head */
static struct __afl_dist_entry *__afl_dist_table = NULL;
static uint64_t   __afl_base = 0;           /* dladdr-derived object base */
static uint64_t   __afl_dist_sum = 0;
static uint64_t   __afl_dist_hits = 0;

/* K-Scheduler node-visit bitmap (see NodeBitmapShm): u32 size_bytes head,
 * then the payload. Eagerly written on probe hits, read-and-cleared by
 * Python after each execution — no destructor writer. */
static uint8_t   *__afl_node_bitmap = NULL;
static uint32_t   __afl_node_bitmap_bytes = 0;

static void __afl_map_node_shm(void) {
    char *id = getenv("__AFL_NODE_BITMAP_ID");
    if (!id) return;
    int shmid = atoi(id);
    if (shmid < 0) return;
    void *p = shmat(shmid, NULL, 0);
    if (p == (void *)-1) return;
    uint32_t bytes = *(uint32_t *)p;
    if (bytes == 0 || bytes > (1u << 28)) return;  /* insane header: ignore */
    __afl_node_bitmap_bytes = bytes;
    __afl_node_bitmap = (uint8_t *)((uint8_t *)p + 4);
}

static void __afl_map_dist_shm(void) {
    char *id = getenv("__AFL_DIST_SHM_ID");
    if (!id) return;
    int shmid = atoi(id);
    if (shmid < 0) return;
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
            uint32_t nidx = __afl_dist_table[idx].node_idx;
            if (__afl_area && __afl_node_bitmap &&
                nidx < __afl_node_bitmap_bytes * 8u)
                __afl_node_bitmap[nidx >> 3] |= (uint8_t)(1u << (nidx & 7u));
            break;
        }
    }
}

#endif /* __AFL_DISTANCE_MODE */

#if __AFL_DISTANCE_MODE
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
        __afl_generation = (__afl_generation + 1) & 0xFF;

        /* Generation tags are 8 bits, so they repeat every 256 resets. An
         * entry keeps the tag of the last execution in which its edge
         * fired, so an edge that fired once and then went quiet is read as
         * live again exactly 256 executions later -- a ghost edge, credited
         * to an execution that never reached that code.
         *
         * Measured pre-fix: fire edge A once, then run N executions that
         * never fire it, and A reappears in the live set at N = 256, 512,
         * 768 ... The reclaim fix does not help here; it is the tag space
         * that is too small, not the reclaim logic.
         *
         * Wiping the table on wrap bounds staleness to one 256-execution
         * cycle and cannot alias. Cost is one memset per 256 resets --
         * ~86.9us amortised over 256 executions, ~0.34us each, against the
         * 2.2us the generation scheme saves on every other execution. The
         * win that motivated generation tagging is kept; only the aliasing
         * is paid for.
         *
         * The wipe must happen when the counter returns to 0, i.e. covering
         * every entry written under any prior tag. */
        if (__afl_generation == 0) {
            for (uint32_t i = 0; i < __afl_map_size; i++) {
                __afl_area[i].edge_id = 0;
                __afl_area[i].count   = 0;
            }
        }

        if (__afl_diag) {
            *__afl_diag = (*__afl_diag & 0x00FFFFFFu)
                        | (__afl_generation << __AFL_DIAG_GEN_SHIFT);
        }

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
#if __AFL_DISTANCE_MODE
        __afl_write_distance_tail();
#endif
    }
#if __AFL_NGRAM_K > 2
    memset(__afl_prev_locs, 0, sizeof(__afl_prev_locs));
    __afl_prev_idx = 0;
#else
    __afl_prev_loc = 0;
#endif
    __afl_path_hash_acc = 0;
    __afl_max_stack_depth = 0;
    __afl_iter_edge_count = 0;
#if __AFL_DISTANCE_MODE
    __afl_dist_sum = 0;
    __afl_dist_hits = 0;
#endif
}

#if __AFL_DISTANCE_MODE
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

/* ── Per-callback comparison counters ($_CMPLOG_COUNTS) ───────────────
 *
 * The CMP record stream cannot answer "how many comparisons fired, and
 * how many were satisfied", for three structural reasons:
 *
 *   1. No function identity. Every layer-1 interceptor funnels into
 *      __afl_cmplog_bytes and every layer-2 callback into
 *      __afl_cmplog_ints, so a record cannot say which one produced it:
 *      const_cmp is indistinguishable from cmp, and a switch case is
 *      indistinguishable from an 8-byte compare.
 *   2. No multiplicity. The Python side dedups on (op_a, op_b, width)
 *      per batch and again against the running pair set, then truncates
 *      the log -- a comparison that fired a million times with the same
 *      operands reaches the collector once.
 *   3. No satisfied comparisons at all on layer 1. __afl_cmplog_bytes
 *      drops result == 0 on purpose (see its comment): a solved compare
 *      is exactly the pollution the pair pool must not carry. The record
 *      that would prove the compare was satisfied is the one never written.
 *
 * Counting therefore lives in the interceptors, ahead of the record
 * writer, and travels on its own channel. Two counters per site: fired
 * (the interceptor was entered) and asserted (the comparison's predicate
 * held -- operands equal for the cmp family, needle found for the search
 * family, non-empty span for strspn/strcspn, a == b for trace-cmp).
 *
 * The counts go to $_CMPLOG_COUNTS rather than into $_CMPLOG_OUT because
 * the collector caps its read of the record stream at 10k lines per pass
 * and truncates regardless; a CNT record past the cap would be silently
 * dropped, and rotation would eat it too.
 *
 * Dumps are DELTAS, zeroed as they are written. That makes summation on
 * the Python side correct in every execution mode without the reader
 * knowing anything about process lifetimes: a subprocess run dumps once
 * at exit, a direct_lite run dumps repeatedly into the same file, and
 * both simply add up.
 *
 * Counting is off unless _CMPLOG_COUNTS is set: memcmp is hot in most
 * targets and an unread counter is pure overhead. */
enum {
    __AFL_CMP_MEMCMP = 0,
    __AFL_CMP_STRCMP,
    __AFL_CMP_STRNCMP,
    __AFL_CMP_STRCASECMP,
    __AFL_CMP_STRNCASECMP,
    __AFL_CMP_BCMP,
    __AFL_CMP_MEMCHR,
    __AFL_CMP_MEMRCHR,
    __AFL_CMP_MEMMEM,
    __AFL_CMP_STRSTR,
    __AFL_CMP_STRCASESTR,
    __AFL_CMP_STRPBRK,
    __AFL_CMP_STRSPN,
    __AFL_CMP_STRCSPN,
    __AFL_CMP_WMEMCMP,
    __AFL_CMP_WCSNCMP,
    __AFL_CMP_WCSCMP,
    __AFL_CMP_WCSCASECMP,
    __AFL_CMP_TRACE_CMP1,
    __AFL_CMP_TRACE_CMP2,
    __AFL_CMP_TRACE_CMP4,
    __AFL_CMP_TRACE_CMP8,
    __AFL_CMP_TRACE_CONST_CMP1,
    __AFL_CMP_TRACE_CONST_CMP2,
    __AFL_CMP_TRACE_CONST_CMP4,
    __AFL_CMP_TRACE_CONST_CMP8,
    __AFL_CMP_TRACE_SWITCH,
    __AFL_CMP_SITES
};

/* Index-matched to the enum above. Longest is "trace_const_cmp1" (16). */
static const char *const __afl_cmp_names[__AFL_CMP_SITES] = {
    "memcmp",      "strcmp",      "strncmp",     "strcasecmp",
    "strncasecmp", "bcmp",        "memchr",      "memrchr",
    "memmem",      "strstr",      "strcasestr",  "strpbrk",
    "strspn",      "strcspn",     "wmemcmp",     "wcsncmp",
    "wcscmp",      "wcscasecmp",  "trace_cmp1",  "trace_cmp2",
    "trace_cmp4",  "trace_cmp8",  "trace_const_cmp1", "trace_const_cmp2",
    "trace_const_cmp4", "trace_const_cmp8", "trace_switch",
};

static uint64_t __afl_cmp_fired[__AFL_CMP_SITES];
static uint64_t __afl_cmp_hit[__AFL_CMP_SITES];
static int      __afl_cmp_counts_fd = -1;

/* The fd doubles as the enable flag: no output file, no counting. */
#define __AFL_CMP_COUNT(id, satisfied)                                   \
    do {                                                                 \
        if (__afl_cmp_counts_fd >= 0) {                                  \
            __afl_cmp_fired[(id)]++;                                     \
            if (satisfied) __afl_cmp_hit[(id)]++;                        \
        }                                                                \
    } while (0)

__AFL_NO_COV static void __afl_cmplog_flush(void) {
    if (__afl_cmplog_pos == 0) {
        return;
    }
    /* Lazy reopen: if Python rotated the log and closed the fd, reopen
     * from the current _CMPLOG_OUT so the next write doesn't silently
     * drop cmplog records. */
    if (__afl_cmplog_fd < 0) {
        const char *path = getenv("_CMPLOG_OUT");
        if (path && path[0]) {
            __afl_cmplog_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
        }
        if (__afl_cmplog_fd < 0) {
            __afl_cmplog_pos = 0;
            return;
        }
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

/* ── Counter dump: "CNT <name> <fired> <asserted>" ────────────────────
 *
 * Writes the delta since the previous dump and zeroes as it goes, so
 * callers may dump as often as they like without double counting. Only
 * sites that fired are emitted, which keeps the common case (a target
 * touching three of the twenty-seven) to three short lines.
 *
 * write(2) and hand-rolled formatting only: this runs from the crash
 * handler, where stdio is not async-signal-safe. Worst-case line is
 * "CNT " + 16 name + 2 * 20 digits + 2 separators + newline = 63 bytes,
 * so the 64-byte headroom check below cannot under-reserve. */
__AFL_NO_COV static void __afl_cmp_dump_counts(void) {
    if (__afl_cmp_counts_fd < 0) return;
    char buf[2048];
    char *p = buf;
    for (int i = 0; i < __AFL_CMP_SITES; i++) {
        if (__afl_cmp_fired[i] == 0) continue;
        if ((size_t)(p - buf) + 64 > sizeof(buf)) break;  /* next dump gets the rest */
        const char *nm = __afl_cmp_names[i];
        *p++ = 'C'; *p++ = 'N'; *p++ = 'T'; *p++ = ' ';
        while (*nm) *p++ = *nm++;
        *p++ = ' ';
        p = __afl_put_i64(p, (int64_t)__afl_cmp_fired[i]);
        *p++ = ' ';
        p = __afl_put_i64(p, (int64_t)__afl_cmp_hit[i]);
        *p++ = '\n';
        __afl_cmp_fired[i] = 0;
        __afl_cmp_hit[i]   = 0;
    }
    size_t len = (size_t)(p - buf), off = 0;
    while (off < len) {
        ssize_t w = write(__afl_cmp_counts_fd, buf + off, len - off);
        if (w <= 0) break;   /* EINTR/ENOSPC: drop the rest, never spin */
        off += (size_t)w;
    }
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
typedef int   (*afl_wchar_cmp_fn)(const wchar_t *, const wchar_t *, size_t);
typedef int   (*afl_wchar_cmp_2arg_fn)(const wchar_t *, const wchar_t *);
typedef void *(*afl_memrchr_fn)(const void *, int, size_t);
typedef char *(*afl_strpbrk_fn)(const char *, const char *);
typedef size_t(*afl_strspn_fn)(const char *, const char *);
typedef size_t(*afl_strcspn_fn)(const char *, const char *);
typedef int   (*afl_bcmp_fn)(const void *, const void *, size_t);

static afl_cmp_fn      real_memcmp      = NULL;
static afl_str_cmp_fn  real_strcmp      = NULL;
static afl_strn_cmp_fn real_strncmp     = NULL;
static afl_chr_fn      real_memchr      = NULL;
static afl_str_cmp_fn  real_strcasecmp  = NULL;
static afl_strn_cmp_fn real_strncasecmp = NULL;
static afl_memmem_fn   real_memmem      = NULL;
static afl_str_str_fn  real_strstr      = NULL;
static afl_str_str_fn  real_strcasestr  = NULL;
static afl_wchar_cmp_fn      real_wmemcmp     = NULL;
static afl_wchar_cmp_fn      real_wcsncmp     = NULL;
static afl_wchar_cmp_2arg_fn real_wcscmp      = NULL;
static afl_wchar_cmp_2arg_fn real_wcscasecmp  = NULL;
static afl_memrchr_fn  real_memrchr     = NULL;
static afl_strpbrk_fn real_strpbrk      = NULL;
static afl_strspn_fn  real_strspn       = NULL;
static afl_strcspn_fn real_strcspn      = NULL;
static afl_bcmp_fn    real_bcmp        = NULL;

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

/* ── Fallbacks: wide-char, set-scan, memrchr, bcmp ───────────────────
 * Same contract as the existing fallbacks: correct first, speed
 * irrelevant. Only reached before the loader can satisfy dlsym, or if
 * the symbol genuinely is not present in the target's libc. */

__AFL_NO_COV static int __afl_fb_bcmp(const void *a, const void *b, size_t n) {
    const unsigned char *x = a, *y = b;
    for (size_t i = 0; i < n; i++) if (x[i] != y[i]) return 1;
    return 0;
}

__AFL_NO_COV static int __afl_fb_wmemcmp(const wchar_t *a, const wchar_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}

__AFL_NO_COV static size_t __afl_fb_wcslen(const wchar_t *s) {
    const wchar_t *p = s;
    while (*p) p++;
    return (size_t)(p - s);
}

__AFL_NO_COV static int __afl_fb_wcsncmp(const wchar_t *a, const wchar_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        wchar_t x = a[i], y = b[i];
        if (x != y) return x < y ? -1 : 1;
        if (!x) return 0;
    }
    return 0;
}

__AFL_NO_COV static int __afl_fb_wcscmp(const wchar_t *a, const wchar_t *b) {
    return __afl_fb_wcsncmp(a, b, (size_t)-1);
}

__AFL_NO_COV static wchar_t __afl_fb_wlower(wchar_t c) {
    return (c >= L'A' && c <= L'Z') ? c + (L'a' - L'A') : c;
}

__AFL_NO_COV static int __afl_fb_wcscasecmp(const wchar_t *a, const wchar_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        wchar_t x = __afl_fb_wlower(a[i]), y = __afl_fb_wlower(b[i]);
        if (x != y) return x < y ? -1 : 1;
        if (!x) return 0;
    }
    return 0;
}

__AFL_NO_COV static char *__afl_fb_strpbrk(const char *s, const char *accept) {
    for (const char *p = s; *p; p++)
        for (const char *q = accept; *q; q++)
            if (*p == *q) return (char *)p;
    return NULL;
}

__AFL_NO_COV static size_t __afl_fb_strspn(const char *s, const char *accept) {
    size_t n = 0;
    for (const char *p = s; *p; p++) {
        int found = 0;
        for (const char *q = accept; *q; q++)
            if (*p == *q) { found = 1; break; }
        if (!found) break;
        n++;
    }
    return n;
}

__AFL_NO_COV static size_t __afl_fb_strcspn(const char *s, const char *reject) {
    size_t n = 0;
    for (const char *p = s; *p; p++) {
        int found = 0;
        for (const char *q = reject; *q; q++)
            if (*p == *q) { found = 1; break; }
        if (found) break;
        n++;
    }
    return n;
}

__AFL_NO_COV static void *__afl_fb_memrchr(const void *s, int c, size_t n) {
    const unsigned char *p = s;
    for (size_t i = n; i > 0; i--) {
        if (p[i - 1] == (unsigned char)c) return (void *)(p + i - 1);
    }
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
    __AFL_CMP_COUNT(__AFL_CMP_MEMCMP, result == 0);
    __afl_cmplog_bytes(a, b, n, result);
    return result;
}
__AFL_NO_COV int afl_cmp_memcmp(const void *a, const void *b, size_t n)
    __attribute__((weak, alias("memcmp")));

__AFL_NO_COV int strcmp(const char *a, const char *b) {
    __AFL_RESOLVE(real_strcmp, afl_str_cmp_fn, "strcmp", __afl_fb_strcmp);
    int result = real_strcmp(a, b);
    __AFL_CMP_COUNT(__AFL_CMP_STRCMP, result == 0);
    size_t na = __afl_fb_len(a), nb = __afl_fb_len(b), n = na < nb ? na : nb;
    if (n > 0) __afl_cmplog_bytes(a, b, n + 1, result);
    return result;
}
__AFL_NO_COV int afl_cmp_strcmp(const char *a, const char *b)
    __attribute__((weak, alias("strcmp")));

__AFL_NO_COV int strncmp(const char *a, const char *b, size_t n) {
    __AFL_RESOLVE(real_strncmp, afl_strn_cmp_fn, "strncmp", __afl_fb_strncmp);
    int result = real_strncmp(a, b, n);
    __AFL_CMP_COUNT(__AFL_CMP_STRNCMP, result == 0);
    if (n > 0) __afl_cmplog_bytes(a, b, n, result);
    return result;
}
__AFL_NO_COV int afl_cmp_strncmp(const char *a, const char *b, size_t n)
    __attribute__((weak, alias("strncmp")));

__AFL_NO_COV void *memchr(const void *s, int c, size_t n) {
    __AFL_RESOLVE(real_memchr, afl_chr_fn, "memchr", __afl_fb_memchr);
    void *result = real_memchr(s, c, n);
    __AFL_CMP_COUNT(__AFL_CMP_MEMCHR, result != NULL);
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
__AFL_NO_COV void * afl_cmp_memchr(const void *s, int c, size_t n)
    __attribute__((weak, alias("memchr")));

__AFL_NO_COV int strcasecmp(const char *a, const char *b) {
    __AFL_RESOLVE(real_strcasecmp, afl_str_cmp_fn, "strcasecmp", __afl_fb_strcasecmp);
    int result = real_strcasecmp(a, b);
    __AFL_CMP_COUNT(__AFL_CMP_STRCASECMP, result == 0);
    size_t na = __afl_fb_len(a), nb = __afl_fb_len(b), n = na < nb ? na : nb;
    if (n > 0) __afl_cmplog_bytes(a, b, n + 1, result);
    return result;
}
__AFL_NO_COV int afl_cmp_strcasecmp(const char *a, const char *b)
    __attribute__((weak, alias("strcasecmp")));

__AFL_NO_COV int strncasecmp(const char *a, const char *b, size_t n) {
    __AFL_RESOLVE(real_strncasecmp, afl_strn_cmp_fn, "strncasecmp", __afl_fb_strncasecmp);
    int result = real_strncasecmp(a, b, n);
    __AFL_CMP_COUNT(__AFL_CMP_STRNCASECMP, result == 0);
    if (n > 0) __afl_cmplog_bytes(a, b, n, result);
    return result;
}
__AFL_NO_COV int afl_cmp_strncasecmp(const char *a, const char *b, size_t n)
    __attribute__((weak, alias("strncasecmp")));

/* The NULL checks below are deliberate: these are declared nonnull, but a
 * fuzz target reaching them with NULL is a bug we want to log around, not
 * crash inside.
 *
 * Suppressing -Wnonnull-compare (which is what this block used to do) is
 * exactly the wrong remedy: it hides the diagnostic while the optimizer still
 * folds `&& n` to always-true and deletes the branch, so the guard silently
 * stops existing. Verified on gcc 13.3 -O2: with a plain `&& n` the `test`
 * against the needle register is absent from the emitted body. Neither
 * `((uintptr_t)n) != 0` nor -fno-delete-null-pointer-checks restores it --
 * the fold comes from the __nonnull attribute on glibc's declaration, not
 * from null-check deletion.
 *
 * __afl_launder_ptr() routes the value through an empty asm with a "+r"
 * constraint, so the optimizer must treat it as an opaque register value with
 * no inherited nonnull provenance. The compare survives, costs one register
 * move, and touches no memory (a `volatile` local also works but spills to
 * the stack, which is not something these interceptors should do per call).
 *
 * Because the compared value now comes out of an asm rather than directly
 * from a nonnull parameter, -Wnonnull-compare no longer fires and the pragma
 * is unnecessary -- which is the point: the warning is left armed to catch
 * any future guard that forgets to launder. */
__attribute__((always_inline))
static inline const void *__afl_launder_ptr(const void *p) {
    __asm__("" : "+r"(p));
    return p;
}

__AFL_NO_COV void *memmem(const void *h, size_t hl, const void *n, size_t nl) {
    __AFL_RESOLVE(real_memmem, afl_memmem_fn, "memmem", __afl_fb_memmem);
    void *result = real_memmem(h, hl, n, nl);
    __AFL_CMP_COUNT(__AFL_CMP_MEMMEM, result != NULL);
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
    /* h was previously unguarded here even though __afl_cmplog_bytes reads it:
     * memmem is __nonnull((1,3)), so a NULL haystack is just as reachable from
     * a buggy target as a NULL needle, and it dereferences one line later. */
    if (__afl_cmplog_fd >= 0 && __afl_launder_ptr(n) && __afl_launder_ptr(h) &&
        nl > 0 && nl <= CMPLOG_MAX_OPERAND && hl > 0) {
        size_t k = hl < nl ? hl : nl;
        __afl_cmplog_bytes(h, n, k, result ? 0 : -1);
    }
    return result;
}
__AFL_NO_COV void * afl_cmp_memmem(const void *h, size_t hl, const void *n, size_t nl)
    __attribute__((weak, alias("memmem")));

__AFL_NO_COV char *strstr(const char *h, const char *n) {
    __AFL_RESOLVE(real_strstr, afl_str_str_fn, "strstr", __afl_fb_strstr);
    char *result = real_strstr(h, n);
    __AFL_CMP_COUNT(__AFL_CMP_STRSTR, result != NULL);
    if (__afl_cmplog_fd >= 0 && __afl_launder_ptr(n) && __afl_launder_ptr(h)) {
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
__AFL_NO_COV char * afl_cmp_strstr(const char *h, const char *n)
    __attribute__((weak, alias("strstr")));

__AFL_NO_COV char *strcasestr(const char *h, const char *n) {
    __AFL_RESOLVE(real_strcasestr, afl_str_str_fn, "strcasestr", __afl_fb_strcasestr);
    char *result = real_strcasestr(h, n);
    __AFL_CMP_COUNT(__AFL_CMP_STRCASESTR, result != NULL);
    if (__afl_cmplog_fd >= 0 && __afl_launder_ptr(n) && __afl_launder_ptr(h)) {
        size_t nl = __afl_fb_len(n);
        size_t k = 0;
        while (k < nl && h[k]) k++;
        if (k > 0 && nl <= CMPLOG_MAX_OPERAND)
            __afl_cmplog_bytes(h, n, k, result ? 0 : -1);
    }
    return result;
}
__AFL_NO_COV char * afl_cmp_strcasestr(const char *h, const char *n)
    __attribute__((weak, alias("strcasestr")));

/* ── New interceptors: bcmp, widec, set-scan, memrchr ─────────────
 * Added to extend cmplog coverage beyond the original memcmp/strcmp/...
 * set of functions. Patterns:
 *   - memcmp-like: log two buffers
 *   - wchar_t functions: log n * sizeof(wchar_t) bytes
 *   - set-scan: log haystack vs set
 *   - memrchr: materialize needle like memchr */

__AFL_NO_COV int bcmp(const void *a, const void *b, size_t n) {
    __AFL_RESOLVE(real_bcmp, int (*)(const void *, const void *, size_t), "bcmp", __afl_fb_bcmp);
    int result = real_bcmp(a, b, n);
    __AFL_CMP_COUNT(__AFL_CMP_BCMP, result == 0);
    __afl_cmplog_bytes(a, b, n, result);
    return result;
}
__AFL_NO_COV int afl_cmp_bcmp(const void *a, const void *b, size_t n)
    __attribute__((weak, alias("bcmp")));

__AFL_NO_COV int wmemcmp(const wchar_t *a, const wchar_t *b, size_t n) {
    __AFL_RESOLVE(real_wmemcmp, afl_wchar_cmp_fn, "wmemcmp", __afl_fb_wmemcmp);
    int result = real_wmemcmp(a, b, n);
    __AFL_CMP_COUNT(__AFL_CMP_WMEMCMP, result == 0);
    if (__afl_cmplog_fd >= 0 && n > 0) {
        size_t k = n * sizeof(wchar_t);
        if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
        __afl_cmplog_bytes(a, b, k, result);
    }
    return result;
}
__AFL_NO_COV int afl_cmp_wmemcmp(const wchar_t *a, const wchar_t *b, size_t n)
    __attribute__((weak, alias("wmemcmp")));

__AFL_NO_COV int wcsncmp(const wchar_t *a, const wchar_t *b, size_t n) {
    __AFL_RESOLVE(real_wcsncmp, afl_wchar_cmp_fn, "wcsncmp", __afl_fb_wcsncmp);
    int result = real_wcsncmp(a, b, n);
    __AFL_CMP_COUNT(__AFL_CMP_WCSNCMP, result == 0);
    if (__afl_cmplog_fd >= 0 && n > 0) {
        size_t k = n * sizeof(wchar_t);
        if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
        if (k > 0) __afl_cmplog_bytes(a, b, k, result);
    }
    return result;
}
__AFL_NO_COV int afl_cmp_wcsncmp(const wchar_t *a, const wchar_t *b, size_t n)
    __attribute__((weak, alias("wcsncmp")));

__AFL_NO_COV int wcscmp(const wchar_t *a, const wchar_t *b) {
    __AFL_RESOLVE(real_wcscmp, afl_wchar_cmp_2arg_fn, "wcscmp", __afl_fb_wcscmp);
    int result = real_wcscmp(a, b);
    __AFL_CMP_COUNT(__AFL_CMP_WCSCMP, result == 0);
    if (__afl_cmplog_fd >= 0) {
        size_t na = __afl_fb_wcslen(a), nb = __afl_fb_wcslen(b), n = na < nb ? na : nb;
        if (n > 0) {
            size_t k = n * sizeof(wchar_t);
            if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
            __afl_cmplog_bytes(a, b, k, result);
        }
    }
    return result;
}
__AFL_NO_COV int afl_cmp_wcscmp(const wchar_t *a, const wchar_t *b)
    __attribute__((weak, alias("wcscmp")));

__AFL_NO_COV int wcscasecmp(const wchar_t *a, const wchar_t *b) {
    __AFL_RESOLVE(real_wcscasecmp, afl_wchar_cmp_2arg_fn, "wcscasecmp", __afl_fb_wcscasecmp);
    int result = real_wcscasecmp(a, b);
    __AFL_CMP_COUNT(__AFL_CMP_WCSCASECMP, result == 0);
    if (__afl_cmplog_fd >= 0) {
        size_t na = __afl_fb_wcslen(a), nb = __afl_fb_wcslen(b), n = na < nb ? na : nb;
        if (n > 0) {
            size_t k = n * sizeof(wchar_t);
            if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
            __afl_cmplog_bytes(a, b, k, result);
        }
    }
    return result;
}
__AFL_NO_COV int afl_cmp_wcscasecmp(const wchar_t *a, const wchar_t *b)
    __attribute__((weak, alias("wcscasecmp")));

__AFL_NO_COV char *strpbrk(const char *s, const char *accept) {
    __AFL_RESOLVE(real_strpbrk, afl_strpbrk_fn, "strpbrk", __afl_fb_strpbrk);
    char *result = real_strpbrk(s, accept);
    __AFL_CMP_COUNT(__AFL_CMP_STRPBRK, result != NULL);
    if (__afl_cmplog_fd >= 0 && s && accept) {
        size_t sl = __afl_fb_len(s), al = __afl_fb_len(accept);
        if (sl > 0 && al > 0) {
            size_t k = sl < al ? sl : al;
            if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
            __afl_cmplog_bytes(s, accept, k, result ? 0 : -1);
        }
    }
    return result;
}
__AFL_NO_COV char * afl_cmp_strpbrk(const char *s, const char *accept)
    __attribute__((weak, alias("strpbrk")));

__AFL_NO_COV size_t strspn(const char *s, const char *accept) {
    __AFL_RESOLVE(real_strspn, afl_strspn_fn, "strspn", __afl_fb_strspn);
    size_t result = real_strspn(s, accept);
    __AFL_CMP_COUNT(__AFL_CMP_STRSPN, result != 0);
    if (__afl_cmplog_fd >= 0 && s && accept) {
        size_t sl = __afl_fb_len(s), al = __afl_fb_len(accept);
        if (sl > 0 && al > 0) {
            size_t k = sl < al ? sl : al;
            if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
            __afl_cmplog_bytes(s, accept, k, result ? (result > 0 ? 1 : 0) : 0);
        }
    }
    return result;
}
__AFL_NO_COV size_t afl_cmp_strspn(const char *s, const char *accept)
    __attribute__((weak, alias("strspn")));

__AFL_NO_COV size_t strcspn(const char *s, const char *reject) {
    __AFL_RESOLVE(real_strcspn, afl_strcspn_fn, "strcspn", __afl_fb_strcspn);
    size_t result = real_strcspn(s, reject);
    __AFL_CMP_COUNT(__AFL_CMP_STRCSPN, result != 0);
    if (__afl_cmplog_fd >= 0 && s && reject) {
        size_t sl = __afl_fb_len(s), rl = __afl_fb_len(reject);
        if (sl > 0 && rl > 0) {
            size_t k = sl < rl ? sl : rl;
            if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
            __afl_cmplog_bytes(s, reject, k, result ? (result > 0 ? 1 : 0) : 0);
        }
    }
    return result;
}
__AFL_NO_COV size_t afl_cmp_strcspn(const char *s, const char *reject)
    __attribute__((weak, alias("strcspn")));

__AFL_NO_COV void *memrchr(const void *s, int c, size_t n) {
    __AFL_RESOLVE(real_memrchr, afl_memrchr_fn, "memrchr", __afl_fb_memrchr);
    void *result = real_memrchr(s, c, n);
    __AFL_CMP_COUNT(__AFL_CMP_MEMRCHR, result != NULL);
    if (__afl_cmplog_fd >= 0 && n > 0 && !result) {
        unsigned char needle[CMPLOG_MAX_OPERAND];
        for (size_t i = 0; i < CMPLOG_MAX_OPERAND; i++) needle[i] = (unsigned char)c;
        __afl_cmplog_bytes(s, needle, n > CMPLOG_MAX_OPERAND ? CMPLOG_MAX_OPERAND : n, -1);
    }
    return result;
}
__AFL_NO_COV void * afl_cmp_memrchr(const void *s, int c, size_t n)
    __attribute__((weak, alias("memrchr")));

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
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CMP1, a == b);
    __afl_cmplog_ints(a, b, 1, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_cmp2(uint16_t a, uint16_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CMP2, a == b);
    __afl_cmplog_ints(a, b, 2, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_cmp4(uint32_t a, uint32_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CMP4, a == b);
    __afl_cmplog_ints(a, b, 4, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_cmp8(uint64_t a, uint64_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CMP8, a == b);
    __afl_cmplog_ints(a, b, 8, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp1(uint8_t a, uint8_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CONST_CMP1, a == b);
    __afl_cmplog_ints(a, b, 1, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp2(uint16_t a, uint16_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CONST_CMP2, a == b);
    __afl_cmplog_ints(a, b, 2, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp4(uint32_t a, uint32_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CONST_CMP4, a == b);
    __afl_cmplog_ints(a, b, 4, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp8(uint64_t a, uint64_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CONST_CMP8, a == b);
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
    /* One dispatch is one comparison, whatever the case count. Counting
     * inside the loop would report a 200-case jump table as 200 compares
     * and, worse, as 199 unsatisfied ones -- the arm that was actually
     * taken drowned in the arms that never could be. Asserted here means
     * the value hit some case, i.e. the switch did not fall to default. */
    if (__afl_cmp_counts_fd >= 0) {
        int matched = 0;
        for (int64_t i = 0; i < count; i++)
            if (val == ref[2 + i]) { matched = 1; break; }
        __AFL_CMP_COUNT(__AFL_CMP_TRACE_SWITCH, matched);
    }
    for (int64_t i = 0; i < count; i++)
        __afl_cmplog_ints(val, ref[2 + i], 8, pc);
}

/* ── Lifecycle ────────────────────────────────────────────────────────
 * Called from __afl_auto_init (edge builds) or the preload-only
 * constructor below. */
__AFL_NO_COV static void __afl_cmplog_init(void) {
    /* Opened before the _CMPLOG_OUT check, and on its own fd: the counters
     * are useful on their own (a target's comparison profile costs no
     * record stream at all), and the record stream gets truncated and
     * rotated on a schedule the counts must not share. */
    const char *counts = getenv("_CMPLOG_COUNTS");
    if (counts && counts[0])
        __afl_cmp_counts_fd = open(counts, O_WRONLY | O_CREAT | O_APPEND, 0644);
    const char *path = getenv("_CMPLOG_OUT");
    if (!path || !path[0]) return;
    __afl_cmplog_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
}

__AFL_NO_COV static void __afl_cmplog_fini(void) {
    __afl_cmplog_flush();
    /* Last chance for a short-lived process: in subprocess mode this is the
     * only dump the run ever gets. */
    __afl_cmp_dump_counts();
    if (__afl_cmp_counts_fd >= 0) {
        close(__afl_cmp_counts_fd);
        __afl_cmp_counts_fd = -1;
    }
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
    /* The per-iteration sync point in direct_lite/persistent modes. Dumping
     * here (not from __afl_cmplog_flush, which also runs on buffer-full in
     * the hot path) keeps the counts channel off the fast path. */
    __afl_cmp_dump_counts();
    if (__afl_cmplog_fd >= 0) {
        if (ftruncate(__afl_cmplog_fd, 0) != 0) { /* best effort */ }
        lseek(__afl_cmplog_fd, 0, SEEK_SET);
    }
}

__AFL_NO_COV __attribute__((visibility("default")))
void __cmplog_close(void) {
    __afl_cmplog_flush();
    if (__afl_cmplog_fd >= 0) {
        close(__afl_cmplog_fd);
        __afl_cmplog_fd = -1;
    }
}

__AFL_NO_COV __attribute__((visibility("default")))
const char *__cmplog_get_path(void) { return getenv("_CMPLOG_OUT"); }

__AFL_NO_COV __attribute__((visibility("default")))
void __tracecmp_flush(void) {
    __afl_cmplog_flush();
    __afl_cmp_dump_counts();
}

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
    /* Same reasoning for the counters: a crashing execution's comparison
     * profile is the one most worth having, and the dump is write(2)-only. */
    __afl_cmp_dump_counts();
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
     * for it.  Without this guard, any ctypes.CDLL()-loaded .so that happens
     * to inherit fds 198/199 from its parent would enter the forkserver loop
     * and hang the loader waiting for a command that never comes.
     *
     * The value is compared against "1" rather than merely tested for
     * presence, so that __AFL_FORKSRV=0 disables rather than enables --
     * a bare getenv() != NULL check makes the documented "=1" spelling
     * incidental and turns every falsy value into an opt-in.
     *
     * fuzz_loader.c sets this in the forkserver child only, between fork()
     * and execl(). */
    const char *optin = getenv("__AFL_FORKSRV");
    if (!optin || strcmp(optin, "1") != 0) return;

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

#if __AFL_CMPLOG
    /* Same argument, one layer over: every child forks from the parent's
     * counter state, so anything counted before this point is re-counted
     * once per execution, forever. The offender is this function's own
     * strcmp(optin, "1") twenty lines above, which goes through the
     * interceptor like any other call and lands in the SATISFIED column --
     * the scarcer and more load-bearing of the two numbers. Measured
     * against a target making exactly one unsatisfied memcmp per run,
     * driven through the protocol below: 20 executions reported
     * memcmp (20, 0) and strcmp (20, 20), a comparison the target never
     * makes.
     *
     * Zeroed rather than dumped, for the reason the area above is detached
     * rather than saved: this is the server's own bookkeeping, not the
     * target's behaviour. Genuine comparisons from a constructor that ran
     * before us go with it, which is the same trade the detach already
     * makes for init edges -- identical on every execution, so they carry
     * no per-execution signal.
     *
     * Totals were usable without this; per-execution vectors were not,
     * since each carried a constant offset. */
    for (int i = 0; i < __AFL_CMP_SITES; i++) {
        __afl_cmp_fired[i] = 0;
        __afl_cmp_hit[i]   = 0;
    }
#endif

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
    __afl_cmp_dump_counts();
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
