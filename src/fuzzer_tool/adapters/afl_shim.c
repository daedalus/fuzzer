/*
 * Sparse 8-byte edge entry shim for in-process fuzzing.
 *
 * Replaces the traditional AFL fixed-size byte bitmap with an open-addressing
 * hash table of 8-byte entries {edge_id, count}.  Each stored edge is uniquely
 * identified by its full 32-bit edge_id (caller_ctx ^ prev_loc ^ cur_loc,
 * call-stack-sensitive by default — see __afl_get_caller_ctx() below, or
 * plain prev_loc ^ cur_loc if built with -D__AFL_CTX_SENSITIVE=0) so there
 * are no silent bucket collisions.  AFL_MAP_SIZE is the number of hash table
 * ENTRIES, not bytes -- a caller that passes a byte count sizes the table
 * eight times too large and the shim writes past the end of the segment.
 * SHM size = SHM_TABLE_OFFSET + AFL_MAP_SIZE * sizeof(struct __afl_entry),
 * i.e. 32 bytes of front region plus 8 bytes per entry (see the layout map
 * further down), plus the 16-byte distance tail in __AFL_DISTANCE_MODE.
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
 *   - Clang -fsanitize-coverage=trace-div,trace-gep callbacks, written as
 *     DIV/GEP records rather than CMP (__sanitizer_cov_trace_div{4,8},
 *     trace_gep)
 *   - __cmplog_reset() / __tracecmp_flush() / __tracecmp_reset()
 *
 * Also gated by ``-D__AFL_CMPLOG=1`` (no separate build flag) but toggled
 * at runtime by $__AFL_COMPCOV_LEVEL, COMPCOV folds byte-level comparison
 * progress directly into the edge map -- no log, no fd, no Python-side
 * drain -- so it works even when _CMPLOG_OUT is never set. See the
 * "COMPCOV" comment ahead of __afl_compcov_mark() further down for the
 * technique and the level semantics.
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
 * Debug-only (default off):
 *   __AFL_TRACE_FIRES=1    log every edge_id to $__AFL_FIRES_OUT
 *
 * Metadata layout (32 bytes at front of SHM; see SHM_TABLE_OFFSET below
 * for the authoritative map and the reasoning):
 *   offset 0:  uint32 stack_depth    (max stack depth in bytes)
 *   offset 4:  uint32 generation     (stale-entry tag, written by the fuzzer)
 *   offset 8:  uint64 path_hash      (rolling: hash = hash * 31 ^ edge_id)
 *   offset 16: uint64 edge_count     (monotonic new-slot insertion count)
 *   offset 24: uint64 dropped_edges  (saturating; edges the probe lost)
 *   offset 32+:  edge table ({edge_id, count} × map_size entries)
 *   after table:  distance tail (u64 dist_sum + u64 dist_count) in
 *                 __AFL_DISTANCE_MODE builds
 *
 * Compile target with:
 *   clang -O2 -g -shared -fPIC -include afl_shim.c -o target.so target.c -lpng -lz
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

/* ── Shim state lives in its own sections ─────────────────────────────
 *
 * The shim is -include'd into the target TU, so its writable globals share
 * the target's .data/.bss -- the span trace-loads/trace-stores features are
 * keyed on. Every shim access (reset, the inlined __afl_map_edge in harness
 * wrappers, exit-time tail writes) would then mint data-flow ids for shim
 * bookkeeping. Placing shim globals in afl_shim_{data,bss} lets
 * __afl_dataflow skip them by linker-defined bounds. Code and guard order
 * are untouched, so edge ids do not move. Reset at the end of this file.
 * gcc has no such pragma and no trace-loads either. */
#if defined(__clang__)
#pragma clang section data="afl_shim_data" bss="afl_shim_bss"
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
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <dlfcn.h>
#include <err.h>
#include <error.h>
#include <pthread.h>
#include <stdarg.h>
#include <time.h>

#if __AFL_CMPLOG
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
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

/* ── Health counters ──────────────────────────────────────────────────
 *
 * Failures the shim survives but the fuzzer cannot otherwise see. Read via
 * __afl_shim_health() (edge builds); index order is the ABI, append only.
 *   ATTACHED       1 when the edge table is attached
 *   MAP_ENTRIES    edge-table entries in effect
 *   SEG_REJECTED   SHM segments refused (bad id, size, or header)
 *   CMPLOG_DROPPED cmplog records lost (writer contention, failed write)
 *   STRAY_SIGNALS  crash signals outside __afl_guarded_call
 *   ABORTS_INTERCEPTED  abort() calls the shim's override turned into returns
 *   HANDLERS_DISPLACED  health reads that found another crash handler
 *                       installed over the shim's (re-armed on the next
 *                       guarded call)
 *   WGUARD         edge-table write guard in effect (__AFL_WGUARD_*)
 * Non-atomic: magnitudes, never accounting records. */
enum {
    __AFL_HEALTH_ATTACHED = 0,
    __AFL_HEALTH_MAP_ENTRIES,
    __AFL_HEALTH_SEG_REJECTED,
    __AFL_HEALTH_CMPLOG_DROPPED,
    __AFL_HEALTH_STRAY_SIGNALS,
    __AFL_HEALTH_ABORTS_INTERCEPTED,
    __AFL_HEALTH_HANDLERS_DISPLACED,
    __AFL_HEALTH_WGUARD,
    __AFL_HEALTH_FIELDS
};
static uint64_t __afl_health[__AFL_HEALTH_FIELDS];

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
 * AND the map has room; check the drop counter (offset 24, see
 * __afl_note_drop) rather than guessing. __AFL_CTX_BITS=0 is equivalent to __AFL_CTX_SENSITIVE=0.
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

/* Static advertisement of ONE capability of this shim: that
 * __afl_get_caller_ctx() honours FUZZER_KEEP_ASLR=1 by hashing return
 * addresses relative to the load base (see __afl_ctx_use_relative()
 * below), and is therefore exec-stable with ASLR left on.
 *
 * The Python side sets FUZZER_KEEP_ASLR for the target whenever ASLR
 * survives startup (services/fuzzer._ensure_ctx_ids_are_exec_stable), and
 * without this marker it cannot tell whether that had any effect: a target
 * built against an older shim ignores the variable and reports a different
 * edge set in every process, which looks exactly like a target with endless
 * new coverage. Before this existed the only way to find out was to execute
 * the target three times and compare edge sets.
 *
 * Deliberately a bare name, not the value-in-the-name encoding
 * __afl_ctx_bits_N uses: there is no value here, only presence. Absent
 * means "this shim predates base-relative context", which is the one thing
 * the scanner needs to distinguish, and it means that for a context-free
 * build too -- where the question is moot, since __AFL_CTX_SENSITIVE=0 ids
 * carry no address-derived term to begin with. Emitted unconditionally for
 * the same reason __afl_ctx_bits_0 is: a missing symbol must mean "old
 * shim" and nothing else. See elf.detect_ctx_relative_capable(). */
__attribute__((visibility("default"), used))
const uint32_t __afl_ctx_relative_capable = 1;

/* Edge-id scheme marker. Present means edge ids come from the hashed
 * location scheme (guard indices and hand-written __afl_map_edge ids mixed
 * through __afl_guard_mix, zero remapped instead of `|= 1`); absent means
 * the older sequential-guard / `|= 1` scheme or no shim at all. The two
 * schemes assign different ids to the same edge, so seed_edges, owner
 * counts and virgin bits persisted under one are meaningless under the
 * other -- corpus_manager.check_coverage_contract refuses that resume.
 * Presence-only, like __afl_ctx_relative_capable. See
 * elf.detect_edge_id_scheme(). */
__attribute__((visibility("default"), used))
const uint32_t __afl_edge_ids_v2 = 2;

/* Crash-handler scope marker. Present means the crash handler only jumps
 * inside __afl_guarded_call and hands every other signal back to its
 * previous owner, SIGPIPE is not hooked, and __afl_shim_health() exists.
 * Absent means an older shim: crashes outside the guard (one-shot and
 * forkserver runs) are reported as SIGSEGV, and a ctypes host dies on a
 * broken pipe. Presence-only. See elf.detect_scoped_crash_handler(). */
__attribute__((visibility("default"), used))
const uint32_t __afl_scoped_crash_handler = 1;

/* ── n-gram history depth ─────────────────────────────────────────────
 *
 * k = blocks encoded into one edge id: the current block plus its k−1
 * predecessors. k=2 IS the historical behaviour and stays byte-identical:
 * same layout, same exported __afl_prev_loc, same XOR edge ids, so existing
 * corpora and resume state remain valid. k is recorded in the resume
 * state's coverage contract and a mismatch refuses the resume outright
 * (corpus_manager.check_coverage_contract), because k>2 changes every
 * edge id. Only k>2 introduces the ring buffer and the
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

/* ── Segment layout ───────────────────────────────────────────────────
 *
 *   offset  0  uint32  stack_depth     written by this shim
 *   offset  4  uint32  generation      written by the fuzzer
 *   offset  8  uint64  path_hash       written by this shim
 *   offset 16  uint64  edge_count      written by this shim
 *   offset 24  uint64  dropped_edges   written by this shim, saturating
 *   offset 32  struct __afl_entry[__afl_map_size]
 *   ...        16-byte AFLGo distance tail (adapters/shm.py SHM_TAIL_SIZE)
 *
 * One field per address, one writer per field, no bit packing anywhere.
 * That is the whole design rule here, and it is a reaction to what the
 * previous layout cost. A single uint32 at offset 4 held three bit fields
 * with two owners -- the ctx width, a drop count, and the generation tag --
 * and every write to any of them was a read-modify-write of the other two.
 * Two bugs came directly out of that: attach masked off the fuzzer's
 * generation while publishing the ctx width, and __afl_map_reset() wrote a
 * private static over the shared tag. A third was latent in the same shape,
 * inprocess.reset_bitmap() memsetting from the segment base and destroying
 * all three at once.
 *
 * Note that two uint32 fields side by side (offsets 0 and 4) are NOT bit
 * packing. They are separate addressable objects, so a store to one is not
 * a read-modify-write of the other, and the shim and the fuzzer can own one
 * each without coordinating. Adjacency was never the problem.
 *
 * SHM_TABLE_OFFSET must equal adapters/shm.py SHM_METADATA_SIZE, and each
 * field offset its counterpart there. A disagreement is not a degradation:
 * the target writes entries at its offset and the fuzzer reads them at its
 * own, so every edge id read back is a splice of two adjacent entries and
 * the fuzzer's own header words read as edges -- a plausible-looking stream
 * of garbage rather than a visible failure. tests/test_shm_layout.py pins
 * the constants against this source, and the layout marker below lets
 * elf.detect_shm_layout() refuse a stale prebuilt target before it runs.
 *
 * The table starts at a multiple of sizeof(struct __afl_entry), which keeps
 * (addr - base) / 8 an exact entry index in both languages and keeps the
 * merged 8-byte stores the compiler emits in the wipe loop aligned. It is
 * NOT a hot-path throughput argument: measured, a table at offset 28 costs
 * +0.4% median against a 14-24% run-to-run spread, i.e. nothing. The reason
 * to keep it is that the arithmetic stays exact and a future 64-bit entry
 * field would otherwise become genuinely misaligned.
 */
#define SHM_STACK_DEPTH_OFFSET  0
#define SHM_GENERATION_OFFSET   4
#define SHM_PATH_HASH_OFFSET    8
#define SHM_EDGE_COUNT_OFFSET  16
#define SHM_DROP_OFFSET        24
#define SHM_TABLE_OFFSET       32

/* Layout generation, advertised in the symbol NAME for the same reason
 * __afl_ctx_bits_N is: the fuzzer must be able to read it from a binary it
 * has not run, and a symbol-table scan is the only mechanism that works
 * before the first execution.
 *
 *   1  table at 24; ctx width, 16-bit drop count and generation packed into
 *      one uint32 at offset 4
 *   2  an intermediate that added a dedicated uint32 drop count at 24 and
 *      moved the table to 32, keeping ctx and generation packed at offset 4
 *   3  this one: every field its own address, uint64 drop count, ctx width
 *      no longer in the segment at all
 *
 * Absence of the marker means layout 1, which is what every shim built
 * before the marker existed produced. Layout 2 was delivered for review but
 * superseded before it shipped; it is listed so that a binary built from it
 * is refused rather than misread, since its table offset matches this one
 * but its offset-4 semantics do not. */
#define __AFL_SHM_LAYOUT 3
__attribute__((visibility("default"), used))
const uint32_t __AFL_CAT(__afl_shm_layout_, __AFL_SHM_LAYOUT) = __AFL_SHM_LAYOUT;

/* Default number of hash table entries.  AFL_MAP_SIZE directly sets
 * __afl_map_size (number of entries, not bytes).  Default 8192 entries:
 * edge table = 8192 × 8 = 65536 bytes, front region = 32 bytes. */
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

/* Metadata pointers (front region, before the edge table) */
static uint32_t *__afl_stack_depth = NULL;   /* offset 0:  uint32 */
static uint32_t *__afl_gen_word    = NULL;   /* offset 4:  uint32 */
static uint64_t *__afl_path_hash   = NULL;   /* offset 8:  uint64 */
static uint64_t *__afl_edge_count  = NULL;   /* offset 16: uint64 */
static uint64_t *__afl_dropped     = NULL;   /* offset 24: uint64 */

/* ── Generation word (offset 4) ────────────────────────────────────────
 *
 * The tag that distinguishes entries written by the current execution from
 * entries left by earlier ones, so a reset does not have to memset the
 * table. Written by the fuzzer's reset_edge_map(), read by __afl_map_edge
 * on every edge fire -- the most-read field in the segment.
 *
 * One writer, which is the point. It previously shared a uint32 with the
 * ctx width and the drop count, and both of the bugs that produced came
 * from a second writer read-modify-writing the word around its own field.
 *
 * Only the low 8 bits are meaningful, and widening the field would not
 * change that: the tag is stored in each entry's `count` high byte
 * ((gen << 24) | 1), and an 8-byte entry has no room for more. The
 * wrap-at-256 table wipe in reset_edge_map() follows from the entry, not
 * from this word.                                                          */
#define __AFL_GEN_MASK 0xFFu

/* ── Dropped-edge counter (offset 24, uint64, saturating) ──────────────
 *
 * Counts edges DISCARDED because the open-addressing probe found no free
 * slot within its window. That closes a self-masking failure: when the
 * table fills, the probe loop in __afl_map_edge runs to completion and
 * returns without recording anything -- the edge is lost, silently. Every
 * occupancy figure the fuzzer computes is derived from edges it actually
 * received, so a saturated table looks UNDER-occupied from the outside, and
 * EdgeTracker.recommended_map_size() (which otherwise triggers on load
 * factor > 0.7) can never fire in precisely the situation it was written
 * for. Counting at the point of loss is the only place the information
 * exists.
 *
 * It began as 16 bits packed into the word at offset 4, and the packing
 * cost it the signal. Measured on a 1024-entry table fed 4000 guards --
 * 1,953 drops per execution -- the field pinned at 65,535 after 34
 * EXECUTIONS. Its only magnitude consumer, the stall-triggered resize, does
 * not run until --stall executions have passed with no new edge (default
 * 1,000), so on any target that saturates every value that consumer ever
 * read was the ceiling. A pinned counter also has no derivative, which
 * ruled out the per-execution question that actually matters: was THIS
 * execution's edge set truncated?
 *
 * 64 bits removes the ceiling rather than moving it, and costs nothing: on
 * a drop-saturated path where this fires on nearly every edge, u64 against
 * u32 measured +0.02% min / +0.06% median against a 1.4% spread.
 *
 * Increments are non-atomic, which is fine: a saturation signal, compared
 * against zero or used as a magnitude, never an accounting record.
 * Saturating rather than wrapping, because a wrap would read as zero drops
 * -- the exact self-masking the counter exists to prevent. On a 32-bit
 * build the increment is two stores, so a concurrent reader could observe a
 * torn value mid-carry; nothing in this tree builds -m32, but this shim is
 * compiled into third-party targets, and a torn read here is a wrong
 * magnitude in a report, not a wrong decision.
 *
 * Deliberately NOT cleared between executions: this is the cumulative count
 * for the segment, and the fuzzer derives per-execution counts by
 * differencing. Clearing it here would need the shim to know where an
 * execution begins, which on the persistent and in-process paths it does
 * not. Cleared only by ShmCoverage.reset_dropped_edges(), after a resize,
 * when drops against the old table stop being evidence about the new one. */
#define __AFL_DROP_MAX 0xFFFFFFFFFFFFFFFFull

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
    if (!__afl_dropped) return;
    uint64_t v = *__afl_dropped;
    if (v != __AFL_DROP_MAX) *__afl_dropped = v + 1;
}

/* ── Write guard ───────────────────────────────────────────────────────
 *
 * The target runs attacker-chosen input in this address space, so a wild
 * store from a memory bug can plant or erase edges in the segment. The
 * guard write-locks the shim's mapping and unlocks it only around the
 * shim's own stores:
 *
 *     target code ── store ──> segment          SIGSEGV (locked)
 *     shim hook   ── open ── store ── close     ok
 *
 *   PKEY      x86 protection key, WD bit set in PKRU. open/close are
 *             RDPKRU/WRPKRU (userspace, no syscall). PKRU is per thread:
 *             threads that existed before attach stay unlocked.
 *   MPROTECT  same brackets via mprotect, two syscalls per edge: proves
 *             every write site is bracketed on hosts without PKU.
 *   OFF       default; also when pkey_alloc is refused (no PKU).
 *
 * Opt-in: __AFL_WGUARD=pkey|mprotect, set by --shm-write-guard.
 *
 * open() returns the previous PKRU and close() restores it, so nesting
 * and signal handlers (which run with the kernel's default PKRU) restore
 * the state they found. Reads stay allowed while locked.
 *
 * Not covered: the fuzzer's own mapping of the segment in in-process
 * modes, which shares this address space and stays writable. */
enum { __AFL_WGUARD_OFF, __AFL_WGUARD_PKEY, __AFL_WGUARD_MPROTECT };

static int      __afl_wguard_mode  = __AFL_WGUARD_OFF;
static int      __afl_wguard_pkey  = -1;
static uint32_t __afl_wguard_bits  = 0;      /* AD|WD bits of our key in PKRU */
static void    *__afl_wguard_base  = NULL;   /* MPROTECT: locked region */
static size_t   __afl_wguard_len   = 0;
static uint32_t __afl_wguard_depth = 0;      /* MPROTECT: open nesting */

#if defined(__x86_64__)
__AFL_NO_COV __attribute__((always_inline))
static inline uint32_t __afl_rdpkru(void) {
    uint32_t eax, edx;
    __asm__ volatile(".byte 0x0f,0x01,0xee" : "=a"(eax), "=d"(edx) : "c"(0));
    return eax;
}

__AFL_NO_COV __attribute__((always_inline))
static inline void __afl_wrpkru(uint32_t v) {
    __asm__ volatile(".byte 0x0f,0x01,0xef" : : "a"(v), "c"(0), "d"(0) : "memory");
}
#endif

/* MPROTECT slow path. Out of line and uninstrumented: inlined into an
 * instrumented caller, a coverage callback lands between the depth bump
 * and the mprotect and writes into a still-locked mapping. */
__attribute__((noinline, cold))
__AFL_NO_COV static void __afl_wguard_mp_open(void) {
    if (__afl_wguard_depth++ == 0)
        mprotect(__afl_wguard_base, __afl_wguard_len, PROT_READ | PROT_WRITE);
}

__attribute__((noinline, cold))
__AFL_NO_COV static void __afl_wguard_mp_close(void) {
    if (--__afl_wguard_depth == 0)
        mprotect(__afl_wguard_base, __afl_wguard_len, PROT_READ);
}

/* open() token: 0 = nothing to undo, so close() tests a register instead
 * of reloading the mode. PKEY carries the saved PKRU in the low 32 bits. */
typedef uint64_t __afl_wtok_t;
#define __AFL_WTOK_PKEY     (1ull << 32)
#define __AFL_WTOK_MPROTECT (1ull << 33)

__AFL_NO_COV __attribute__((always_inline))
static inline __afl_wtok_t __afl_wguard_open(void) {
    if (__builtin_expect(__afl_wguard_mode == __AFL_WGUARD_OFF, 1)) return 0;
#if defined(__x86_64__)
    if (__afl_wguard_mode == __AFL_WGUARD_PKEY) {
        uint32_t saved = __afl_rdpkru();
        __afl_wrpkru(saved & ~__afl_wguard_bits);
        return __AFL_WTOK_PKEY | saved;
    }
#endif
    __afl_wguard_mp_open();
    return __AFL_WTOK_MPROTECT;
}

__AFL_NO_COV __attribute__((always_inline))
static inline void __afl_wguard_close(__afl_wtok_t tok) {
    if (__builtin_expect(!tok, 1)) return;
#if defined(__x86_64__)
    if (tok & __AFL_WTOK_PKEY) {
        __afl_wrpkru((uint32_t)tok);
        return;
    }
#endif
    __afl_wguard_mp_close();
}

/* Out-of-line brackets for cold instrumented callers (reset, distance
 * tail). Inlined there, the mode branches become coverage edges that
 * differ between guard modes. */
__attribute__((noinline))
__AFL_NO_COV static __afl_wtok_t __afl_wguard_open_cold(void) {
    return __afl_wguard_open();
}

__attribute__((noinline))
__AFL_NO_COV static void __afl_wguard_close_cold(__afl_wtok_t tok) {
    __afl_wguard_close(tok);
}

/* Lock [p, p+len) with protection key; 0 on success. The key is allocated
 * once and reused by a re-attach. pkey_alloc sets WD in this thread's PKRU,
 * and threads created later inherit it. */
__attribute__((noinline))
__AFL_NO_COV static int __afl_wguard_pkey_arm(void *p, size_t len) {
#if defined(__x86_64__) && defined(SYS_pkey_alloc) && defined(SYS_pkey_mprotect)
    if (__afl_wguard_pkey < 0) {
        long k = syscall(SYS_pkey_alloc, 0, PKEY_DISABLE_WRITE);
        if (k < 0) return -1;
        __afl_wguard_pkey = (int)k;
        __afl_wguard_bits = 3u << (2 * (unsigned)k);
    }
    return (int)syscall(SYS_pkey_mprotect, p, len, PROT_READ | PROT_WRITE, __afl_wguard_pkey);
#else
    (void)p; (void)len;
    return -1;
#endif
}

/* Arm the guard on a freshly attached segment. Opt-in via __AFL_WGUARD
 * (fuzzer-tool --shm-write-guard): "pkey" or "mprotect"; unset or anything
 * else leaves it off. */
__attribute__((noinline))
__AFL_NO_COV static void __afl_wguard_arm(void *p, size_t len) {
    long page = sysconf(_SC_PAGESIZE);
    if (page > 0) len = (len + (size_t)page - 1) & ~((size_t)page - 1);

    const char *want = getenv("__AFL_WGUARD");
    __afl_wguard_mode = __AFL_WGUARD_OFF;
    if (!want) return;

    if (strcmp(want, "mprotect") == 0) {
        if (mprotect(p, len, PROT_READ) != 0) return;
        __afl_wguard_base  = p;
        __afl_wguard_len   = len;
        __afl_wguard_depth = 0;
        __afl_wguard_mode  = __AFL_WGUARD_MPROTECT;
        return;
    }

    if (strcmp(want, "pkey") == 0 && __afl_wguard_pkey_arm(p, len) == 0)
        __afl_wguard_mode = __AFL_WGUARD_PKEY;
}

/* Per-iteration state */
static uint64_t  __afl_path_hash_acc = 0;       /* rolling path hash accumulator */
static uint32_t  __afl_max_stack_depth = 0;     /* max stack depth this iteration */
static uintptr_t __afl_stack_base = 0;          /* frame address of first sample this iteration */
static uint64_t  __afl_iter_edge_count = 0;     /* new-slot insertions this iteration */
static uint64_t  __afl_total_edge_count = 0;    /* cumulative, never reset across iterations */
static uint8_t   __afl_generation = 0;          /* generation counter for tag-based reset */

/* ── SHM attachment ──────────────────────────────────────────────────── */

/* Parse a SysV segment id from the environment: decimal, non-negative,
 * fits an int, nothing trailing. -1 otherwise. atoi() could not fail:
 * "123junk" attached segment 123 and "banana" attached segment 0. */
static int __afl_parse_shmid(const char *s) {
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || v < 0 || v > INT_MAX) return -1;
    return (int)v;
}

/* Set once a segment has been refused. adapters/inprocess.py retries
 * __afl_map_shm() when __afl_area is still NULL after load (for the case
 * where the environment arrived late); a refusal is final, so the retry
 * must not print and count it again. */
static int __afl_shm_refused = 0;

static void __afl_refuse_shm(void) {
    __afl_shm_refused = 1;
    __afl_health[__AFL_HEALTH_SEG_REJECTED]++;
}

__attribute__((visibility("default")))
void __afl_map_shm(void) {
    char *id = getenv("__AFL_SHM_ID");
    if (!id) return;   /* not under the fuzzer — silence is correct here */
    if (__afl_shm_refused) return;

    /* Past this point the fuzzer has explicitly asked for coverage, so a
     * failure must not be silent. It used to be: all three early returns
     * below left __afl_area NULL, the target then ran to completion and
     * exited 0 having recorded nothing, and the fuzzer read back an
     * all-zero header indistinguishable from "the child never wrote".
     * That is the whole of the "Loose thread" from the 2026-08 edge-coverage
     * analysis -- three sightings across ~50 runs, unresolvable each time
     * because neither side left a trace. Its surviving remnant is item (G)
     * in docs/handover/handover_done_2026-09-06.md §1; the
     * analysis document itself no longer exists (see the round-13 entry in
     * that file's "What was removed").
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
    int shmid = __afl_parse_shmid(id);
    if (shmid < 0) {
        char msg[128];
        int n = snprintf(msg, sizeof(msg),
                         "__afl_shim: __AFL_SHM_ID=%.32s is not a valid segment id"
                         " -- coverage disabled\n", id);
        if (n > 0) { ssize_t w = write(2, msg, (size_t)n); (void)w; }
        __afl_refuse_shm();
        return;
    }

    /* Read map size from environment.  AFL_MAP_SIZE is the number of
     * hash table entries (not bytes).  The Python side allocates SHM as
     * SHM_TABLE_OFFSET + AFL_MAP_SIZE * sizeof(struct __afl_entry) bytes,
     * plus the 16-byte distance tail.                                      */
    /* Strict parse: atoi("-5") became a 4-billion-entry map and every
     * probe wrote past the segment. Unset, "" and "0" keep the default. */
    char *size_str = getenv("AFL_MAP_SIZE");
    if (size_str && size_str[0]) {
        errno = 0;
        char *size_end = NULL;
        unsigned long long s = strtoull(size_str, &size_end, 10);
        if (size_str[0] == '-' || *size_end != '\0' || errno == ERANGE || s > UINT32_MAX) {
            char msg[128];
            int n = snprintf(msg, sizeof(msg),
                             "__afl_shim: AFL_MAP_SIZE=%.32s is not a valid entry count"
                             " -- coverage disabled\n", size_str);
            if (n > 0) { ssize_t w = write(2, msg, (size_t)n); (void)w; }
            __afl_refuse_shm();
            return;
        }
        if (s > 0)
            __afl_map_size = (uint32_t)s;
    }

    /* SHM was allocated as header bytes + table bytes */
    void *p = shmat(shmid, NULL, 0);
    if (p == (void *)-1) {
        char msg[160];
        int n = snprintf(msg, sizeof(msg),
                         "__afl_shim: shmat(%d) failed: %.64s"
                         " -- coverage disabled\n", shmid, strerror(errno));
        if (n > 0) { ssize_t w = write(2, msg, (size_t)n); (void)w; }
        __afl_refuse_shm();
        return;
    }

    /* The segment must hold the table AFL_MAP_SIZE promises. A mismatch
     * (bytes passed for entries, a foreign segment) otherwise writes past
     * the end: a SIGSEGV blamed on the input, or silent corruption.
     * IPC_STAT failing is not evidence either way, so it is skipped. */
    struct shmid_ds ds;
    size_t need = SHM_TABLE_OFFSET + (size_t)__afl_map_size * sizeof(struct __afl_entry)
                + (__AFL_DISTANCE_MODE ? 2 * sizeof(uint64_t) : 0);
    if (shmctl(shmid, IPC_STAT, &ds) == 0 && (size_t)ds.shm_segsz < need) {
        char msg[192];
        int n = snprintf(msg, sizeof(msg),
                         "__afl_shim: segment %d is %zu bytes, %u entries need %zu"
                         " -- too small, coverage disabled\n",
                         shmid, (size_t)ds.shm_segsz, __afl_map_size, need);
        if (n > 0) { ssize_t w = write(2, msg, (size_t)n); (void)w; }
        shmdt(p);
        __afl_refuse_shm();
        return;
    }

    /* Edge table starts after the front region */
    uint8_t *base = (uint8_t *)p;
    __afl_area = (struct __afl_entry *)(base + SHM_TABLE_OFFSET);

    /* One pointer per field; see the layout map above. */
    __afl_stack_depth = (uint32_t *)(base + SHM_STACK_DEPTH_OFFSET);
    __afl_gen_word    = (uint32_t *)(base + SHM_GENERATION_OFFSET);
    __afl_path_hash   = (uint64_t *)(base + SHM_PATH_HASH_OFFSET);
    __afl_edge_count  = (uint64_t *)(base + SHM_EDGE_COUNT_OFFSET);
    __afl_dropped     = (uint64_t *)(base + SHM_DROP_OFFSET);

    /* Write-lock the mapping; shim stores go through open/close. */
    size_t seg_len = need;
    if (shmctl(shmid, IPC_STAT, &ds) == 0) seg_len = (size_t)ds.shm_segsz;
    __afl_wguard_arm(p, seg_len);

    /* Nothing is published here. Attach used to write the ctx width into
     * the segment, and the mask it used to do so zeroed the fuzzer's
     * generation tag on every execution. The write is gone rather than
     * merely corrected: the segment copy of the ctx width had no reader
     * outside the test suite, and the value is already available to the
     * fuzzer before the target has ever run, from the __afl_ctx_bits_N
     * marker symbol that elf.detect_ctx_bits() reads -- which is the source
     * the sizing path actually uses, because sizing happens before the
     * first execution. A field with no reader and a second writer on a word
     * someone else owns is a hazard with no upside.
     *
     * The consequence worth stating: this shim now never writes the
     * generation word at all on the attach path, so that class of clobber
     * is structurally impossible here rather than fixed by a mask. */

#if __AFL_DISTANCE_MODE
    __afl_mapping = 1;  /* map_dist_shm is instrumented; no ctx during setup */
    __afl_map_dist_shm();
    __afl_map_node_shm();
    __afl_mapping = 0;
#endif
}

/* Copy up to n health counters (see __AFL_HEALTH_*) into out; returns the
 * number of fields this shim has, so a caller can detect a newer shim
 * with more fields than it asked for. out may be NULL when n == 0. */
static void __afl_check_crash_handlers(void);

__attribute__((visibility("default")))
uint32_t __afl_shim_health(uint64_t *out, uint32_t n) {
    __afl_check_crash_handlers();
    __afl_health[__AFL_HEALTH_ATTACHED] = __afl_area != NULL;
    __afl_health[__AFL_HEALTH_MAP_ENTRIES] = __afl_map_size;
    __afl_health[__AFL_HEALTH_WGUARD] = (uint64_t)__afl_wguard_mode;
    for (uint32_t i = 0; i < n && i < __AFL_HEALTH_FIELDS; i++)
        out[i] = __afl_health[i];
    return __AFL_HEALTH_FIELDS;
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
 * Both call sites that invoke __afl_map_loc() are real (non-inlined)
 * functions: __sanitizer_cov_trace_pc_guard() and __sanitizer_cov_trace_pc().
 * __afl_map_loc() (and its public
 * __afl_map_edge() wrapper) itself is always_inline, so it never introduces its
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

/* COMPCOV (cmplog builds) reuses the base-relative helpers below for its
 * site keys, so they are compiled in for either channel. */
#if __AFL_CTX_SENSITIVE || __AFL_CMPLOG
/* dladdr()/Dl_info, for __afl_ctx_resolve_base() below. Included here
 * (rather than assumed from the __AFL_DISTANCE_MODE include further up)
 * because CTX_SENSITIVE and DISTANCE_MODE are independently-gated default-on
 * channels -- a build with -D__AFL_DISTANCE_MODE=0 must not lose this
 * declaration. <dlfcn.h>'s own include guard makes a second #include here
 * a no-op when both channels are on. */
#include <dlfcn.h>

/* ── ASLR-invariant caller context, opt-in ────────────────────────────
 *
 * dladdr() walks the link map, so it is too slow to call on every trace_pc
 * hit -- cache resolved bases in a small direct-mapped table keyed by `ra`.
 * The set of distinct return addresses is compile-time-bounded (one per
 * call site in the binary), so a slot collision just costs one redundant
 * dladdr() call on the next hit; it is never a correctness problem, since
 * the `ra` tag check catches the mismatch and re-resolves. */
#define __AFL_CTX_CACHE_SIZE 256
static struct {
    uintptr_t ra;
    uintptr_t base;
} __afl_ctx_base_cache[__AFL_CTX_CACHE_SIZE];

__AFL_NO_COV static inline uintptr_t __afl_ctx_resolve_base(void *ra) {
    uintptr_t key = (uintptr_t)ra;
    /* Return addresses are instruction-aligned (>=2 bytes on every arch
     * this shim targets), so the low bits never disambiguate two distinct
     * call sites -- drop them before folding into a slot index. */
    uint32_t slot = (uint32_t)((key >> 2) % __AFL_CTX_CACHE_SIZE);
    if (__afl_ctx_base_cache[slot].ra == key) return __afl_ctx_base_cache[slot].base;
    Dl_info info;
    uintptr_t base = 0;
    if (dladdr(ra, &info) && info.dli_fbase) base = (uintptr_t)info.dli_fbase;
    __afl_ctx_base_cache[slot].ra = key;
    __afl_ctx_base_cache[slot].base = base;
    return base;
}

/* FUZZER_KEEP_ASLR is fixed for the process's whole lifetime -- adapters/
 * process.py's disable_aslr() reads it once, before any exec happens -- so
 * resolve it once here too rather than calling getenv() on every hit.
 * -1 = not yet resolved, 0 = raw-address mode (default: ASLR disabled,
 * so the raw return address is already exec-stable and relativizing it
 * would just add dladdr() cost for nothing), 1 = base-relative mode. */
static int __afl_ctx_relative_mode = -1;

__AFL_NO_COV static inline int __afl_ctx_use_relative(void) {
    if (__afl_ctx_relative_mode < 0) {
        const char *v = getenv("FUZZER_KEEP_ASLR");
        __afl_ctx_relative_mode = (v && v[0] == '1' && v[1] == '\0') ? 1 : 0;
    }
    return __afl_ctx_relative_mode;
}
#endif /* __AFL_CTX_SENSITIVE || __AFL_CMPLOG */

#if __AFL_CTX_SENSITIVE
/* Frame-slot load ASAN cannot see. The walk's one unvalidated hop can pass
 * the range check yet land in a stack redzone; an instrumented load then
 * aborts the whole in-process fuzzer. no_sanitize("address") on the walk
 * does not help: it is always_inline, so ASAN instruments it as part of its
 * caller. Inline asm is never instrumented. The address is inside the live
 * stack window, so the raw read cannot fault. */
__AFL_NO_COV __attribute__((always_inline))
static inline void *__afl_raw_load(void *const *p) {
    void *v;
#if defined(__x86_64__)
    __asm__("movq (%1), %0" : "=r"(v) : "r"(p));
#elif defined(__aarch64__)
    __asm__("ldr %0, [%1]" : "=r"(v) : "r"(p));
#else
    v = *p;
#endif
    return v;
}

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
    void **caller_fp = (void **)__afl_raw_load(fp);   /* saved FP of the frame above */
    uintptr_t cfp = (uintptr_t)caller_fp;
    /* The stack grows down, so a genuine older frame sits at a higher
     * address than ours and within a sane single-hop span (4 MiB covers
     * any realistic frame without risking a wild read).  Anything outside
     * that window is an unlinked/garbage frame — skip context for it. */
    if (cfp <= cur || cfp - cur > (4u << 20)) return 0;
    void *ra = __afl_raw_load(caller_fp + 1);   /* return addr into caller's caller */
    if (!ra) return 0;

    /* The claim that used to sit here -- "ASLR/PIE base differences across
     * runs don't matter, we only need identical call chains WITHIN one
     * process/session to hash identically" -- only holds under a forkserver
     * model, where every child is forked from one already-randomized parent
     * and so shares its bases. adapters/process.py's disable_aslr() docstring
     * spells out why that assumption doesn't hold here: the default execution
     * path spawns a fresh process per input, ShmCoverage._seen_edge_ids in
     * the fuzzer parent persists across all of them, and under PIE+ASLR a
     * fresh exec gets a fresh base -- so the raw address below would hash
     * differently every time even for the identical call chain, and
     * is_new_coverage() would never return False. That's why ASLR is
     * disabled globally by default; this function does not by itself make
     * caller_ctx exec-stable.
     *
     * The one case that needs ASLR left on (FUZZER_KEEP_ASLR=1 -- ASAN
     * shadow-memory range collisions, see disable_aslr()) would otherwise
     * lose caller_ctx's cross-exec meaning entirely. In that case, resolve
     * `ra` to a load-base-relative offset first: link-time layout, unlike
     * the runtime load address, survives ASLR. Same trick __AFL_DISTANCE_MODE
     * already uses for __afl_base, and PtraceCoverage.record_edge uses via
     * /proc/pid/maps -- just not previously wired into this function. */
    uintptr_t addr = (uintptr_t)ra;
    if (__afl_ctx_use_relative()) {
        uintptr_t base = __afl_ctx_resolve_base(ra);
        /* dladdr() failure (base == 0) falls back to the raw address --
         * degraded (exec-unstable) context for that one call site, not a
         * crash or a corrupted hash. */
        if (base) addr -= base;
    }

    /* Fold to 32 bits via a hash, not a truncation: return addresses (or
     * their base-relative offsets) in the same binary share high bits, so a
     * plain cast would collapse distinct call sites into the same low 32
     * bits far more than a real 64-bit space would. Fibonacci-hashing style
     * mix (splitmix64 finalizer) also spreads addresses that are only a few
     * bytes apart (adjacent call instructions — common for PLT stubs / thin
     * wrapper callers) into different buckets instead of adjacent ones. */
    uint64_t p = (uint64_t)addr;
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

/* ── Fire log (debug build only) ───────────────────────────────────────
 * -D__AFL_TRACE_FIRES=1 appends every final edge_id, one decimal per line, to
 * $__AFL_FIRES_OUT: the sequence the path hash folds, so two runs can be
 * diffed fire by fire (edge-id handover P0-1). Compiled out by default --
 * a write(2) per fire. Opened lazily: guards can fire from other
 * constructors before ours. Unset or unopenable path: logging stays off. */
#ifndef __AFL_TRACE_FIRES
#define __AFL_TRACE_FIRES 0
#endif

#if __AFL_TRACE_FIRES
#include <fcntl.h>
static int __afl_fire_fd = -2;  /* -2 not yet opened, -1 off */

__AFL_NO_COV static void __afl_log_fire(uint32_t edge_id) {
    if (__afl_fire_fd == -2) {
        const char *p = getenv("__AFL_FIRES_OUT");
        __afl_fire_fd = (p && p[0]) ? open(p, O_WRONLY | O_CREAT | O_APPEND, 0644) : -1;
    }
    if (__afl_fire_fd < 0) return;
    char b[16];
    int n = snprintf(b, sizeof b, "%u\n", edge_id);
    if (n > 0) { ssize_t w = write(__afl_fire_fd, b, (size_t)n); (void)w; }
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

/* Location width shared by guard ids and hand-written __afl_map_edge ids;
 * widened by __sanitizer_cov_trace_pc_guard_init (see "Guard numbering").
 * 16 bits is the floor for manual-only (gcc, no trace-pc-guard) builds,
 * whose wrappers carry tens of hand-picked ids. */
static uint32_t __afl_loc_mask = 0xFFFFu;

__AFL_NO_COV static inline uint32_t __afl_guard_mix(uint64_t x);

/* Edge id for cur_loc: n-gram / previous-location hash, optionally
 * XORed with the caller context. May return 0 -- caller remaps it. */
__attribute__((always_inline))
static inline uint32_t __afl_edge_hash(uint32_t cur_loc) {
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
    return __afl_get_caller_ctx() ^ h;
# else
    return h;
# endif
#else
#if __AFL_CTX_SENSITIVE
    uint32_t caller_ctx = __afl_get_caller_ctx();
    return caller_ctx ^ __afl_prev_loc ^ cur_loc;
#else
    return __afl_prev_loc ^ cur_loc;
#endif
#endif
}

/* Record edge_id in the SHM table for generation gen (see __afl_map_loc). */
__attribute__((always_inline))
static inline void __afl_probe_insert(uint32_t edge_id, uint32_t pos,
                                      uint32_t window, uint32_t gen) {
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
            return;
        }
        if (eid == edge_id) {                        /* existing edge */
            if ((__afl_area[idx].count >> 24) == gen) {
                if ((__afl_area[idx].count & 0x00FFFFFFu) < 0x00FFFFFFu)
                    __afl_area[idx].count++;
                return;
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
            return;
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
}

/* Shift cur_loc into the previous-location state for the next edge. */
__attribute__((always_inline))
static inline void __afl_push_prev(uint32_t cur_loc) {
#if __AFL_NGRAM_K > 2
    __afl_prev_locs[__afl_prev_idx] = cur_loc >> 1;
    __afl_prev_idx = (__afl_prev_idx + 1) % (__AFL_NGRAM_K - 1);
#else
    __afl_prev_loc = cur_loc >> 1;
#endif
}

/* Id-space split, active only when COMPCOV is requested.
 *
 * Bit 31 belongs to synthetic channels (COMPCOV, DATAFLOW): they set it. With
 * $__AFL_COMPCOV_LEVEL > 0, real edge ids never carry it, so a COMPCOV mark
 * can never equal a real id and a reader can tell the two apart from the id
 * alone. With COMPCOV off, __afl_map_loc leaves ids exactly as they always
 * were (keep = all ones, tag = 0), so saved corpora/state stay valid.
 *
 * Real ids are already below 2^24 with the default k=2 hashing (see
 * __AFL_GUARD_MAX_BITS), so the reservation is a no-op there. It matters for
 * __AFL_NGRAM_K > 2, whose FNV-1a edge hash is a full 32 bits (measured on
 * compcov_gates at k=3: 27 of 52 real ids had bit 31 set unmasked).
 *
 * Chosen once, at the top of __afl_auto_init, before the coverage area is
 * attached -- __afl_area is NULL until then, so no edge can be mapped under
 * the wrong scheme, and one run never mixes the two. Parsed from the same env
 * var as __afl_compcov_level but independently of it: this block precedes the
 * cmplog section. Every synthetic caller passes through __AFL_SYNTH_ID(). */
#define __AFL_SYNTH_ID_BIT 0x80000000u
#define __AFL_REAL_ID_MASK 0x7FFFFFFFu
#define __AFL_SYNTH_ID(h32) ((uint32_t)(h32) | __AFL_SYNTH_ID_BIT)

static uint32_t __afl_id_keep = 0xFFFFFFFFu; /* AND-mask applied to real ids  */
static uint32_t __afl_id_tag  = 0;           /* bits of cur_loc carried over   */

__AFL_NO_COV static void __afl_id_scheme_init(void) {
#if __AFL_CMPLOG
    const char *c = getenv("__AFL_COMPCOV_LEVEL");
    if (c && c[0] && atoi(c) > 0) {
        __afl_id_keep = __AFL_REAL_ID_MASK;
        __afl_id_tag  = __AFL_SYNTH_ID_BIT;
    }
#endif
}

/* Insert a final, non-zero edge id for the current generation. No
 * prev_loc hashing and no prev_loc push: synthetic channels that must not
 * rename the real edge after them (COMPCOV) call this directly;
 * __afl_map_loc() is this plus the edge-chain bookkeeping. */
__attribute__((always_inline))
static inline void __afl_map_id_raw(uint32_t edge_id) {
    uint32_t gen = __afl_generation;
    if (__afl_gen_word)
        gen = *__afl_gen_word & __AFL_GEN_MASK;

    uint32_t pos = edge_id % __afl_map_size;

    /* Linear probe, bounded to __AFL_PROBE_MAX slots.
     *
     * The bound converts a map_size-iteration worst case into a constant.
     * It is only correct because insertion is bounded by the same constant:
     * an edge is therefore always within __AFL_PROBE_MAX of its home slot
     * or absent, so a bounded lookup can never miss an edge a bounded
     * insert placed. Do not bound one without the other. */
    uint32_t window = __AFL_PROBE_MAX;
    if (window > __afl_map_size) window = __afl_map_size;

    __afl_probe_insert(edge_id, pos, window, gen);
}

/* __afl_map_id_raw() inside the write guard. */
__attribute__((always_inline))
static inline void __afl_map_id(uint32_t edge_id) {
    __afl_wtok_t wk = __afl_wguard_open();
    __afl_map_id_raw(edge_id);
    __afl_wguard_close(wk);
}

/* ── Stack depth ──────────────────────────────────────────────────────
 *
 * The first coverage point after a reset pins the base frame; later ones
 * measure how far below it they sit:
 *
 *     high  | base        <- first sample            depth = base - fp
 *           | ...
 *     low   | fp          <- deepest sample so far   (stack grows down)
 *
 * Frame addresses instead of -fsanitize-coverage=stack-depth: the runtimes
 * own __sancov_lowest_stack, so no build flag or link change is needed.
 * Samples at or above the base, or more than __AFL_STACK_WINDOW below it,
 * belong to another thread's stack and are ignored. Published live, like
 * path_hash, so a one-shot run that never calls __afl_map_reset still
 * reports it. Inline-8bit-counters/bool-flag builds have no per-edge
 * callback and report 0. */
#ifndef __AFL_STACK_WINDOW
#  define __AFL_STACK_WINDOW ((uintptr_t)1 << 26)  /* 64 MiB */
#endif

__attribute__((always_inline))
static inline void __afl_note_stack(void) {
    uintptr_t fp = (uintptr_t)__builtin_frame_address(0);

    if (!__afl_stack_base) {
        __afl_stack_base = fp;
        if (__afl_stack_depth) *__afl_stack_depth = 0;
        return;
    }
    if (fp >= __afl_stack_base) return;

    uintptr_t depth = __afl_stack_base - fp;
    if (depth > __AFL_STACK_WINDOW || depth <= __afl_max_stack_depth) return;

    __afl_max_stack_depth = (uint32_t)depth;
    if (__afl_stack_depth) *__afl_stack_depth = __afl_max_stack_depth;
}

__attribute__((visibility("default"), always_inline))
static inline void __afl_map_loc(uint32_t cur_loc) {
    if (!__afl_area) return;
    __afl_wtok_t wk = __afl_wguard_open();
    __afl_note_stack();

    /* COMPCOV on: clear bit 31 of the hash, then carry bit 31 over from
     * cur_loc. Real callers never set it, so their ids never carry it; the one
     * synthetic caller that routes through here (__sfuzz_state, which also
     * wants the prev_loc chain) sets it on cur_loc and keeps its tag whatever
     * the hash width. COMPCOV off: keep = ~0, tag = 0, the hash is untouched. */
    uint32_t edge_id = (__afl_edge_hash(cur_loc) & __afl_id_keep) | (cur_loc & __afl_id_tag);
    /* edge_id == 0 means "empty slot" to the probe loop below, so a valid
     * edge that hashes to 0 would be silently dropped and the slot
     * reclaimed by the next collision. Remap exactly that one value to 1.
     *
     * Not `edge_id |= 1`: that forces bit 0 on EVERY id, which erases bit 0
     * of cur_loc (and of the context tag) for all edges, so (p, 2k) and
     * (p, 2k+1) -- very often the two successors of one branch -- became
     * one id. Measured on fuzzgoat: 80 of 344 real edges merged by that
     * alone. The remap below merges only the id-0 edge with the id-1 edge. */
    if (!edge_id) edge_id = 1;
    __afl_map_id_raw(edge_id);

    /* Accumulate rolling path hash: hash = hash * 31 ^ edge_id */
    __afl_path_hash_acc = (__afl_path_hash_acc * 31) ^ edge_id;
    if (__afl_path_hash)
        *__afl_path_hash = __afl_path_hash_acc;
#if __AFL_TRACE_FIRES
    __afl_log_fire(edge_id);
#endif
    __afl_wguard_close(wk);

    __afl_push_prev(cur_loc);
}

/* ── Compiler-inserted edge coverage callbacks ────────────────────────
 * Hidden visibility: PIE builds call these via the PLT, so a libasan
 * LD_PRELOAD (the fuzzer-tool CLI preloads it for ASAN targets) would
 * interpose its own weak stubs over ours.  Hidden visibility forces
 * direct call instructions within the target, bypassing PLT resolution
 * entirely (same pattern as the abort() override below). */

/* Hand-written coverage points (the harness wrappers in targets/<name>.c call
 * this with ids like 0x1100 + depth). Those ids are small, sequential and
 * hand-picked, so fed straight in as cur_loc they alias exactly like the
 * old sequential guards did: prev >> 1 drops bit 0 of the previous id, so
 * (0x1102 -> X) and (0x1103 -> X) were one edge, and XORs of neighbouring
 * constants collide (11 of the 12 residual merges on fuzzgoat after guard
 * hashing were between wrapper ids). Mix them into the same location space
 * as guard ids. Internal callers that already hold a well-spread location
 * (guards, trace-pc keys, SGFuzz transition hashes) use __afl_map_loc.
 *
 * Deliberately outside the __AFL_DISTANCE_MODE gate: every harness wrapper
 * calls this, so under -D__AFL_DISTANCE_MODE=0 (the documented opt-out) an
 * executable failed to link and a .so linked with __afl_map_edge undefined,
 * which dlopen(RTLD_NOW) refuses and RTLD_LAZY turns into a crash on the
 * first coverage point. */
__attribute__((visibility("default"), always_inline))
static inline void __afl_map_edge(uint32_t cur_loc) {
    uint32_t v = __afl_guard_mix((uint64_t)cur_loc ^ 0x6a09e667f3bcc909ULL) & __afl_loc_mask;
    __afl_map_loc(v ? v : 1);
}

#if __AFL_DISTANCE_MODE
/* Defined further down, after the distance-table state -- forward-declared
 * here so the guard callback (which comes first in the file) can reach it.
 * __afl_probe_distance() must be called with a PC computed via
 * __builtin_return_address(0) taken directly in the CALLER's own body
 * (frame 0 there is the instrumented call site) -- never from inside
 * another function, or the wrong frame gets captured. Same rule
 * __afl_get_caller_ctx() already documents for __afl_map_edge(). */
__AFL_NO_COV static uint64_t __afl_pc_key(uintptr_t pc);
__AFL_NO_COV static void __afl_probe_distance(uint64_t key);
#endif

__attribute__((visibility("hidden")))
void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!guard || *guard == 0) return;
    __afl_map_loc(*guard);
#if __AFL_DISTANCE_MODE
    /* Guard builds get the same AFLGo distance / K-Scheduler node-bitmap
     * channel as trace-pc builds, keyed off this call site's own return
     * address -- exactly the address icfg.py's probe_key_node_table() and
     * TargetDistance.pc_distance_table() recover when they scan for calls
     * to __sanitizer_cov_trace_pc_guard instead of the bare trace_pc. */
    __afl_probe_distance(__afl_pc_key((uintptr_t)__builtin_return_address(0)));
#endif
}

/* ── SGFuzz state transitions (instrumented sources only) ─────────────
 *
 * core/state_vars.py rewrites every assignment to an enum-typed variable
 * so it also calls here with (variable id, new value). A parser keeps its
 * state in such a variable, and the assignments are the state machine's
 * transitions -- which edge coverage cannot see as transitions at all: it
 * records the code that performs one, so two runs visiting the same
 * blocks in a different order are one bitmap.
 *
 * The transition is folded into the edge map rather than into a channel
 * of its own. SGFuzz keeps an explicit State Transition Tree; an edge
 * carries the same "this pair is new" signal and arrives already wired to
 * every consumer of coverage -- scoring, scheduling, admission, the n-gram
 * ring -- with no plumbing added. What it does not carry is the tree
 * itself, so nothing can schedule *by state*; see docs/TODO.md.
 *
 * The hash mixes the previous value with the current one, so it is the
 * transition that is the coverage item, not the state: reaching DONE from
 * BODY and reaching DONE from INIT are different edges.
 *
 * No-op in an uninstrumented target -- nothing calls it. */
#define SFUZZ_MAX_VARS 256

static __thread uint64_t __sfuzz_prev[SFUZZ_MAX_VARS];

__attribute__((visibility("default")))
void __sfuzz_state(unsigned var_id, unsigned long long value) {
    unsigned slot = var_id % SFUZZ_MAX_VARS;
    uint64_t prev = __sfuzz_prev[slot];
    __sfuzz_prev[slot] = (uint64_t)value;

    /* FNV-1a over (id, previous, current). The high bit is set so the
     * result can never be 0, which __afl_map_edge reads as an empty
     * slot. */
    uint64_t h = 1469598103934665603ULL;
    h = (h ^ var_id) * 1099511628211ULL;
    h = (h ^ prev) * 1099511628211ULL;
    h = (h ^ (uint64_t)value) * 1099511628211ULL;

    __afl_map_loc(__AFL_SYNTH_ID(h >> 32));
}

/* ── Guard numbering ──────────────────────────────────────────────────
 *
 * Guards used to be numbered 1..N sequentially and used as cur_loc
 * directly. That pinned every edge_id = (prev >> 1) ^ cur below ~2N, so a
 * module with N blocks had at most N odd ids for its (typically >N) edges:
 * pigeonhole-guaranteed aliasing, and structured aliasing on top -- XOR of
 * small consecutive integers collides systematically (on fuzzgoat, 8 real
 * edges shared id 95). 344 real edges in the fuzzgoat corpus reached Python
 * as 145 ids. Classic AFL avoids this by giving each block a random cur_loc.
 *
 * Here each guard gets a deterministic hash of its index (ids must be
 * stable across execs and resumes, so never rand()), masked to a width
 * sized from the module's guard count: ceil(log2(N)) + __AFL_GUARD_SLACK_BITS,
 * capped at __AFL_GUARD_MAX_BITS. The cap keeps every id (a masked cur_loc
 * XOR a narrower prev and an 8-bit context tag) below adapters/shm.py's
 * VIRGIN_DENSE_MAX = 2^24, so the direct-indexed virgin map stays on its
 * fast path. The slack is the memory/collision trade: the Python side keeps
 * per-id dense arrays whose size tracks the largest id, so width costs
 * memory, and expected colliding pairs are ~E^2 / 2^(bits+1) for E edges.
 *
 * guard_counter is per module (hidden visibility), so the module's guard
 * count salts the hash to keep two modules from minting the same stream.
 * Modules with identical guard counts still share it -- a residual, not a
 * regression: under sequential numbering every module aliased every other. */
#ifndef __AFL_GUARD_SLACK_BITS
#define __AFL_GUARD_SLACK_BITS 10
#endif
#ifndef __AFL_GUARD_MAX_BITS
#define __AFL_GUARD_MAX_BITS 24
#endif
#if __AFL_GUARD_MAX_BITS > 24 || __AFL_GUARD_MAX_BITS < 8
#error "__AFL_GUARD_MAX_BITS must be in [8, 24] (see VIRGIN_DENSE_MAX)"
#endif

__AFL_NO_COV static inline uint32_t __afl_guard_mix(uint64_t x) {
    /* splitmix64 finalizer, same mixer __afl_get_caller_ctx uses. */
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return (uint32_t)(x ^ (x >> 31));
}

/* Id width for n blocks seen so far: ceil(log2(n)) + slack, capped. Widens
 * __afl_loc_mask to match. Shared by guards and inline counters. */
__AFL_NO_COV static uint32_t __afl_guard_width(uint64_t n) {
    unsigned bits = 1;
    while (bits < 32 && (1ULL << bits) <= n) bits++;
    bits += __AFL_GUARD_SLACK_BITS;
    if (bits > __AFL_GUARD_MAX_BITS) bits = __AFL_GUARD_MAX_BITS;
    uint32_t mask = (uint32_t)((1ULL << bits) - 1);
    if (mask > __afl_loc_mask) __afl_loc_mask = mask;
    return mask;
}

#define __AFL_GUARD_SALT 0xd1b54a32d192ed03ULL

__attribute__((visibility("hidden")))
void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    static uint32_t guard_counter;
    if (start == stop || *start) return;
    uint32_t mask = __afl_guard_width((uint64_t)(stop - start) + guard_counter);
    uint64_t salt = (uint64_t)(stop - start) * __AFL_GUARD_SALT;
    for (uint32_t *g = start; g < stop; g++) {
        /* 0 means "disabled guard" to __sanitizer_cov_trace_pc_guard. */
        uint32_t v = __afl_guard_mix(salt ^ ++guard_counter) & mask;
        *g = v ? v : 1;
    }
}

/* ── Inline counters, bool flags, pc-table ────────────────────────────
 *
 * inline-8bit-counters / inline-bool-flag bump one byte per block in place
 * and never call back, so nothing reaches the edge table until someone
 * reads the bytes. __afl_sancov_fold() does: each nonzero byte becomes one
 * block id, minted like a guard id (module-salted hash of its running
 * index, masked by __afl_guard_width), and the byte is cleared for the
 * next execution. Block ids, not edges: a byte array has no prev_loc order.
 *
 * Fold points -- Python reads the map only after one of them:
 *   __afl_guarded_call return   direct_lite / persistent loops
 *   __afl_crash_handler         crash inside that call
 *   destructor                  one-shot subprocess / forkserver child
 *
 *   module ctor ── init(start, stop) ──> region[i] = {start, stop, first, mask, salt}
 *   exec ── bytes bump ──> fold ── byte != 0 ──> __afl_map_id(mix(salt ^ (first + j)))
 *
 * pc-table maps each block to its PC for symbolizers; the edge map needs
 * none of it, so __sanitizer_cov_pcs_init only has to exist.
 *
 * Hidden visibility on all init callbacks: libasan ships weak no-op stubs
 * that would otherwise interpose over PLT calls (see guard callbacks). */
#define __AFL_SANCOV_MAX_REGIONS 64

struct __afl_sancov_region {
    uint8_t *start;
    uint8_t *stop;
    uint32_t first;  /* running index of start[0] across regions */
    uint32_t mask;
    uint64_t salt;
};

static struct __afl_sancov_region __afl_sancov_regions[__AFL_SANCOV_MAX_REGIONS];
static uint32_t __afl_sancov_nregions;

/* Bounded table: regions past the cap are left unread (one per module,
 * so 64 instrumented modules in one process before anything is lost). */
__AFL_NO_COV static void __afl_sancov_register(uint8_t *start, uint8_t *stop) {
    static uint32_t counter;
    if (start >= stop || __afl_sancov_nregions == __AFL_SANCOV_MAX_REGIONS) return;

    /* A module ctor can run twice (dlopen after static init); keep one. */
    for (uint32_t i = 0; i < __afl_sancov_nregions; i++)
        if (__afl_sancov_regions[i].start == start) return;

    uint64_t n = (uint64_t)(stop - start);
    struct __afl_sancov_region *r = &__afl_sancov_regions[__afl_sancov_nregions++];
    r->start = start;
    r->stop  = stop;
    r->first = counter;
    r->mask  = __afl_guard_width(n + counter);
    r->salt  = n * __AFL_GUARD_SALT;
    counter += (uint32_t)n;
}

__attribute__((visibility("hidden")))
void __sanitizer_cov_8bit_counters_init(uint8_t *start, uint8_t *stop) {
    __afl_sancov_register(start, stop);
}

__attribute__((visibility("hidden")))
void __sanitizer_cov_bool_flag_init(uint8_t *start, uint8_t *stop) {
    __afl_sancov_register(start, stop);
}

__attribute__((visibility("hidden")))
void __sanitizer_cov_pcs_init(const uintptr_t *start, const uintptr_t *stop) {
    (void)start;
    (void)stop;
}

__AFL_NO_COV static inline void __afl_sancov_mark(const struct __afl_sancov_region *r,
                                                   uint32_t idx) {
    uint32_t v = __afl_guard_mix(r->salt ^ (r->first + idx + 1)) & r->mask;
    __afl_map_id(v ? v : 1);
}

/* One region. Most blocks stay cold, so test 64 bytes per vector load and
 * only walk chunks that carry a hit. Measured on 1M cold counters (clang):
 * 27us at -O1 / 18us at -O2, against 83 / 66us for a u64 stride. */
typedef uint64_t __afl_v64 __attribute__((vector_size(64), aligned(1)));

__AFL_NO_COV static void __afl_sancov_fold_region(const struct __afl_sancov_region *r) {
    uint8_t *p = r->start;
    uint32_t n = (uint32_t)(r->stop - r->start);
    uint32_t i = 0;

    for (; i + 64 <= n; i += 64) {
        __afl_v64 w = *(const __afl_v64 *)(p + i);
        if (!(w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7])) continue;

        for (uint32_t j = i; j < i + 64; j++)
            if (p[j]) __afl_sancov_mark(r, j);
        /* One vector store, not memset: this runs in the crash handler,
         * and -O0 lowers memset to a libc call. */
        *(__afl_v64 *)(p + i) = (__afl_v64){0};
    }

    for (; i < n; i++) {
        if (!p[i]) continue;
        __afl_sancov_mark(r, i);
        p[i] = 0;
    }
}

/* Async-signal-safe: plain loads/stores into already-mapped memory.
 * noinline: inlined into an instrumented caller (__afl_guarded_call), the
 * body inherits the caller's counters and re-ticks them after clearing. */
__attribute__((noinline))
__AFL_NO_COV static void __afl_sancov_fold(void) {
    if (!__afl_area) return;
    for (uint32_t i = 0; i < __afl_sancov_nregions; i++)
        __afl_sancov_fold_region(&__afl_sancov_regions[i]);
}

__attribute__((destructor))
__AFL_NO_COV static void __afl_sancov_fold_exit(void) {
    __afl_sancov_fold();
}

/* ── Data-flow features (trace-loads / trace-stores) ──────────────────
 *
 * Every instrumented load/store calls here with its address. A (site,
 * offset) pair becomes one synthetic edge when the address lies in this
 * module's writable PT_LOAD span (.data/.bss): which global slot an
 * instruction touched is state edges cannot see -- table[3] and table[9]
 * run the same blocks. Stack and heap addresses move with ASLR, so they
 * are dropped; keying on them would make every run look new. Same filter
 * as Centipede's data-flow features.
 *
 * Both keys are base-relative, so ids are stable across ASLR. The span
 * is resolved in __afl_auto_init; accesses before that see lo == hi == 0
 * and return on the first compare. */
#include <link.h>

static uintptr_t __afl_data_lo;
static uintptr_t __afl_data_hi;
static uintptr_t __afl_data_base;
static uintptr_t __afl_img_lo;  /* whole module span, all PT_LOADs */
static uintptr_t __afl_img_hi;

/* Linker-defined bounds of the shim's own state (see the section pragma at
 * the top). Weak: absent under gcc, where both ranges read as empty. */
extern char __start_afl_shim_data[] __attribute__((weak, visibility("hidden")));
extern char __stop_afl_shim_data[] __attribute__((weak, visibility("hidden")));
extern char __start_afl_shim_bss[] __attribute__((weak, visibility("hidden")));
extern char __stop_afl_shim_bss[] __attribute__((weak, visibility("hidden")));

__AFL_NO_COV static inline int __afl_is_shim_state(uintptr_t a) {
    if (a - (uintptr_t)__start_afl_shim_bss <
        (uintptr_t)__stop_afl_shim_bss - (uintptr_t)__start_afl_shim_bss) return 1;
    return a - (uintptr_t)__start_afl_shim_data <
           (uintptr_t)__stop_afl_shim_data - (uintptr_t)__start_afl_shim_data;
}

/* dl_iterate_phdr callback: find the object holding `self`, record the
 * union of its writable PT_LOAD segments. Returns 1 to stop the walk. */
__AFL_NO_COV static int __afl_data_phdr(struct dl_phdr_info *info, size_t size, void *self) {
    (void)size;
    uintptr_t addr = (uintptr_t)self;
    uintptr_t lo = UINTPTR_MAX, hi = 0;
    uintptr_t img_lo = UINTPTR_MAX, img_hi = 0;
    int owns = 0;

    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
        if (ph->p_type != PT_LOAD) continue;

        uintptr_t s = info->dlpi_addr + ph->p_vaddr;
        uintptr_t e = s + ph->p_memsz;
        if (addr >= s && addr < e) owns = 1;
        if (s < img_lo) img_lo = s;
        if (e > img_hi) img_hi = e;
        if (!(ph->p_flags & PF_W)) continue;
        if (s < lo) lo = s;
        if (e > hi) hi = e;
    }
    if (!owns || hi <= lo) return 0;

    __afl_data_base = info->dlpi_addr;
    __afl_data_lo = lo;
    __afl_data_hi = hi;
    __afl_img_lo = img_lo;
    __afl_img_hi = img_hi;
    return 1;
}

__AFL_NO_COV static void __afl_map_data_range(void) {
    dl_iterate_phdr(__afl_data_phdr, &__afl_data_lo);
}

__AFL_NO_COV static inline void __afl_dataflow(void *addr, void *pc) {
    uintptr_t off = (uintptr_t)addr - __afl_data_lo;
    if (off >= __afl_data_hi - __afl_data_lo) return;  /* also lo == hi == 0 */
    if (__afl_is_shim_state((uintptr_t)addr)) return;
    if (!__afl_area) return;

    uint64_t h = 1469598103934665603ULL; /* FNV-1a, as __afl_compcov_mark */
    h = (h ^ ((uintptr_t)pc - __afl_data_base)) * 1099511628211ULL;
    h = (h ^ 0x44415441464c4f57ULL) * 1099511628211ULL; /* "DATAFLOW" salt */
    h = (h ^ off) * 1099511628211ULL;
    __afl_map_id(__AFL_SYNTH_ID(h >> 32));
}

/* The return address must be taken in the callback's own body. */
#define __AFL_DATAFLOW_CB(name)                                     \
    __attribute__((visibility("hidden"))) void name(void *addr) {   \
        __afl_dataflow(addr, __builtin_return_address(0));          \
    }

__AFL_DATAFLOW_CB(__sanitizer_cov_load1)
__AFL_DATAFLOW_CB(__sanitizer_cov_load2)
__AFL_DATAFLOW_CB(__sanitizer_cov_load4)
__AFL_DATAFLOW_CB(__sanitizer_cov_load8)
__AFL_DATAFLOW_CB(__sanitizer_cov_load16)
__AFL_DATAFLOW_CB(__sanitizer_cov_store1)
__AFL_DATAFLOW_CB(__sanitizer_cov_store2)
__AFL_DATAFLOW_CB(__sanitizer_cov_store4)
__AFL_DATAFLOW_CB(__sanitizer_cov_store8)
__AFL_DATAFLOW_CB(__sanitizer_cov_store16)

/* ── Indirect calls (indirect-calls) ─────────────────────────────────
 *
 * clang calls this with the callee address before every indirect call. Edge
 * coverage sees the callee's entry block but not the (site, callee) pair:
 * a vtable or function-pointer table dispatches the same blocks from many
 * sites, and one site reaching a new table slot is a new behaviour.
 *
 *     call site (return address) --.
 *                                  +--> hash --> synthetic id
 *     callee    (this call)     ---'
 *
 * Both keys are base-relative, so ids are stable across ASLR. A callee
 * outside this module (libc, another DSO) has an offset that moves every
 * run, so it is dropped, like stack/heap addresses in __afl_dataflow.
 * Opt-in at build time (--indir-cov / --sancov=...,indirect-calls): no
 * instrumented call, no callback, no map pressure.
 *
 * The return address must be taken in the callback's own body. */
__attribute__((visibility("hidden"))) __AFL_NO_COV
void __sanitizer_cov_trace_pc_indir(uintptr_t callee) {
    if (callee - __afl_img_lo >= __afl_img_hi - __afl_img_lo) return;  /* also lo == hi == 0 */
    if (!__afl_area) return;

    uintptr_t site = (uintptr_t)__builtin_return_address(0);
    uint64_t h = 1469598103934665603ULL; /* FNV-1a, as __afl_dataflow */
    h = (h ^ (site - __afl_data_base)) * 1099511628211ULL;
    h = (h ^ 0x494e444952454354ULL) * 1099511628211ULL; /* "INDIRECT" salt */
    h = (h ^ (callee - __afl_data_base)) * 1099511628211ULL;
    __afl_map_id(__AFL_SYNTH_ID(h >> 32));
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

/* Attach an auxiliary segment whose u32 header sizes its payload, and
 * refuse it when header + payload overruns the segment: both headers are
 * trusted as loop bounds on the hot path. Returns NULL on any failure. */
static void *__afl_attach_sized(const char *env, uint64_t entry_bytes) {
    char *id = getenv(env);
    if (!id) return NULL;
    int shmid = __afl_parse_shmid(id);
    if (shmid < 0) {
        __afl_health[__AFL_HEALTH_SEG_REJECTED]++;
        return NULL;
    }
    void *p = shmat(shmid, NULL, 0);
    if (p == (void *)-1) {
        __afl_health[__AFL_HEALTH_SEG_REJECTED]++;
        return NULL;
    }

    struct shmid_ds ds;
    uint64_t head = *(uint32_t *)p;
    uint64_t need = 4 + head * entry_bytes;
    if (head == 0 || (shmctl(shmid, IPC_STAT, &ds) == 0 && (uint64_t)ds.shm_segsz < need)) {
        shmdt(p);
        __afl_health[__AFL_HEALTH_SEG_REJECTED]++;
        return NULL;
    }
    return p;
}

static void __afl_map_node_shm(void) {
    void *p = __afl_attach_sized("__AFL_NODE_BITMAP_ID", 1);
    if (!p) return;
    uint32_t bytes = *(uint32_t *)p;
    if (bytes > (1u << 28)) return;  /* insane header: ignore */
    __afl_node_bitmap_bytes = bytes;
    __afl_node_bitmap = (uint8_t *)((uint8_t *)p + 4);
}

static void __afl_map_dist_shm(void) {
    void *p = __afl_attach_sized("__AFL_DIST_SHM_ID", sizeof(struct __afl_dist_entry));
    if (!p) return;
    __afl_dist_count = (uint32_t *)p;
    __afl_dist_table = (struct __afl_dist_entry *)((uint8_t *)p + 4);
}

/* PC -> base-relative key, resolving __afl_base lazily on first use (any
 * caller works to resolve it -- dladdr identifies the mapped object, not
 * the specific PC within it). Shared by __sanitizer_cov_trace_pc() and
 * __sanitizer_cov_trace_pc_guard(); matches the forward declaration above
 * so the guard callback, defined earlier in this file, can call it. */
__AFL_NO_COV static uint64_t __afl_pc_key(uintptr_t pc) {
    if (__afl_base == 0) {
        Dl_info info;
        if (dladdr((void *)pc, &info) && info.dli_fbase)
            __afl_base = (uintptr_t)info.dli_fbase;
        else
            __afl_base = 1;  /* dladdr failed — treat the PC as absolute */
    }
    return (uint64_t)pc - __afl_base;
}

/* Distance-table lookup + node-bitmap probe for one already-computed key.
 * Shared by __sanitizer_cov_trace_pc() (PC-based edge, trace-pc builds) and
 * __sanitizer_cov_trace_pc_guard() (guard-counter edge, trace-pc-guard
 * builds) — the AFLGo distance / K-Scheduler node-bitmap channel works
 * under either coverage flavor as long as icfg.py's probe-key scan looks
 * for calls to whichever of the two symbols the build actually calls. */
__AFL_NO_COV static void __afl_probe_distance(uint64_t key) {
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

/* Hidden visibility: same PLT-interposition rationale as the guard
 * callbacks — the CLI's libasan LD_PRELOAD must not shadow this. */
__attribute__((visibility("hidden")))
void __sanitizer_cov_trace_pc(void) {
    uintptr_t pc = (uintptr_t)__builtin_return_address(0);
    uint64_t key = __afl_pc_key(pc);

    /* Edge coverage: PC-based (prev_loc ^ cur_loc, same sparse table). */
    __afl_map_loc((uint32_t)(key >> 1));
    __afl_probe_distance(key);
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
        __afl_wtok_t wk = __afl_wguard_open_cold();
        *dist_sum = __afl_dist_sum;
        *(dist_sum + 1) = __afl_dist_hits;
        __afl_wguard_close_cold(wk);
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
        /* Advance the tag that is actually in effect, which lives in the
         * diag word: __afl_map_edge reads it from there, and the fuzzer's
         * reset_edge_map() writes it there. The private static is only the
         * fallback for a target running with no segment attached.
         *
         * This used to increment the static and write the result to the
         * word, which is correct only while the two cannot disagree. They
         * could not, for an accidental reason: __afl_map_shm() zeroed the
         * word's generation bits at attach, matching a freshly-zeroed
         * static. Now that attach preserves the fuzzer's tag, a static at 0
         * against a word at 1 writes 1 back -- no advance at all, leaving
         * the previous execution's entries readable as live. */
        uint32_t gen = __afl_generation;
        if (__afl_gen_word)
            gen = *__afl_gen_word & __AFL_GEN_MASK;
        __afl_generation = (gen + 1) & __AFL_GEN_MASK;
        __afl_wtok_t wk = __afl_wguard_open_cold();

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

        if (__afl_gen_word) {
            *__afl_gen_word = __afl_generation;
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
        __afl_wguard_close_cold(wk);
    }
#if __AFL_NGRAM_K > 2
    memset(__afl_prev_locs, 0, sizeof(__afl_prev_locs));
    __afl_prev_idx = 0;
#else
    __afl_prev_loc = 0;
#endif
    __afl_path_hash_acc = 0;
    __afl_max_stack_depth = 0;
    __afl_stack_base = 0;
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
/* Record emission off (__cmplog_pause): the collector parses one round in
 * twenty, and ffmpeg emits ~400k records per run (3.1x the target's cost).
 * Counts/sites channels keep running. */
static int    __afl_cmplog_paused = 0;

/* ── Log descriptors the target can steal ─────────────────────────────
 *
 * The shim's log fds live in the target's fd table. A target that closes
 * every descriptor (daemon-style closefrom) and opens its own file gets
 * the same number back, and the shim's records went into the target's
 * file: measured, a target's output file held our CMP lines and the log
 * held nothing. Remember what each fd was opened on and check it before
 * writing; a stolen number is dropped (never closed -- it is the
 * target's now) and the path reopened. O_CLOEXEC: programs the target
 * execs never needed them. */
struct __afl_fd_id {
    dev_t dev;
    ino_t ino;
};

static struct __afl_fd_id __afl_cmplog_fd_id;

__AFL_NO_COV static int __afl_log_open(const char *path, int extra_flags, struct __afl_fd_id *id) {
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC | extra_flags, 0644);
    struct stat st;
    if (fd >= 0 && fstat(fd, &st) == 0) {
        id->dev = st.st_dev;
        id->ino = st.st_ino;
    }
    return fd;
}

/* 1 when fd still refers to the file it was opened on. fstat(2) is
 * async-signal-safe, so the crash-handler flush may call this. */
__AFL_NO_COV static int __afl_log_fd_ours(int fd, const struct __afl_fd_id *id) {
    struct stat st;
    return fd >= 0 && fstat(fd, &st) == 0 && st.st_dev == id->dev && st.st_ino == id->ino;
}

/* Before a write: drop a stolen fd and reopen the path from *env*. */
__AFL_NO_COV static void __afl_log_keep(int *fd, struct __afl_fd_id *id, const char *env,
                                        int extra_flags) {
    if (*fd < 0 || __afl_log_fd_ours(*fd, id)) return;
    const char *path = getenv(env);
    *fd = (path && path[0]) ? __afl_log_open(path, extra_flags, id) : -1;
}

/* One owner of buf/pos at a time. Without it two threads could both pass
 * the room check, one advance pos, and the other write a whole record past
 * the buffer end. Try-lock, never wait: a contended record is dropped and
 * counted (__AFL_HEALTH_CMPLOG_DROPPED), so a signal handler or the
 * re-entrant trace-cmp path can never deadlock on it. */
static volatile int __afl_cmplog_lock = 0;
/* This thread owns the lock: a crash handler interrupting our own writer
 * may flush (pos only advances once a record is complete) and must
 * release, or siglongjmp leaves the lock held for good. */
static __thread int __afl_cmplog_held = 0;

__AFL_NO_COV static inline int __afl_cmplog_acquire(void) {
    if (!__atomic_exchange_n(&__afl_cmplog_lock, 1, __ATOMIC_ACQUIRE)) {
        __afl_cmplog_held = 1;
        return 1;
    }
    __afl_health[__AFL_HEALTH_CMPLOG_DROPPED]++;
    return 0;
}

__AFL_NO_COV static inline void __afl_cmplog_release(void) {
    __afl_cmplog_held = 0;
    __atomic_store_n(&__afl_cmplog_lock, 0, __ATOMIC_RELEASE);
}

/* ── COMPCOV: partial-match feedback straight into the edge map ────────
 *
 * CMPLOG (above) reifies a comparison's operands so redqueen-style
 * substitution can solve it in one step, at the cost of a record per
 * fire and a Python-side drain/parse pass. COMPCOV answers a narrower
 * question -- "did this execution get *closer* to satisfying a wide
 * comparison than the last one?" -- and answers it for free: it folds
 * each byte of progress directly into the same edge table CMPLOG already
 * attaches to, with no log, no fd, no drain. Same technique as AFL++'s
 * laf-intel/CompareCoverage (split-compares / libcompcov): a comparison
 * that is all-or-nothing to plain edge coverage (the branch after it is
 * one bit, taken or not) becomes visible one matching byte at a time, so
 * novelty search has a gradient to climb instead of a wall to guess past.
 * Reference: https://github.com/AFLplusplus/AFLplusplus (laf-intel,
 * libcompcov), and the QEMU-mode writeup this file's byte-walk mirrors:
 * https://andreafioraldi.github.io/articles/2019/07/20/aflpp-qemu-compcov.html
 *
 * Off by default (level 0) even in a __AFL_CMPLOG=1 build -- it rides
 * the same Layer 1/2 interception CMPLOG already pays for (no separate
 * compile-time gate, no second libc interposer, no ODR risk from two
 * memcmp()s under two flags) but is not free at runtime: it turns one
 * comparison into up to N-1 extra edge-table probe/insert calls. Opt in
 * per run with $__AFL_COMPCOV_LEVEL:
 *   0  disabled (default)
 *   1  constant/immediate comparisons only -- fires from
 *      __sanitizer_cov_trace_const_cmp*, where the compiler has already
 *      told us one operand is a compile-time constant. Cheap: this class
 *      is rare relative to total comparison volume in most targets.
 *   2  all comparisons -- also fires from the non-const trace_cmp*
 *      callbacks and from the Layer 1 libc interceptors (memcmp/strcmp/
 *      strncmp/bcmp and the wide-char and case-insensitive variants).
 *      There is no runtime-cheap way to tell a "constant" byte buffer
 *      from a mutated one the way the compiler can for an integer
 *      immediate, so libc-level comparisons only ever fire at level 2.
 *
 * Requires __AFL_EDGE (the target actually attached to an edge map);
 * under __AFL_PRELOAD_ONLY the level is still parsed but every mark call
 * is a no-op, matching that build's documented absence of edge
 * machinery -- see __afl_compcov_mark's __AFL_EDGE=0 stub below.
 *
 * A COMPCOV mark is a synthetic edge, same idea and same trade as
 * __sfuzz_state's transition hash above, but inserted via __afl_map_id():
 * it bypasses the prev_loc chain, so the real edge after a comparison
 * keeps one id however far the match got. Bit 31 is set on every mark, and
 * whenever COMPCOV is on it is masked off every real id (see the id-space
 * note at __AFL_SYNTH_ID_BIT), so a mark can never equal a real edge id; it
 * can still collide with another synthetic channel. No channel of its own, so every existing
 * coverage consumer -- scoring, scheduling, admission, novelty -- sees
 * COMPCOV progress with no plumbing added. Marking every byte of a long
 * match also means a target with wide, hot comparisons wants a bigger
 * AFL_MAP_SIZE than an edge-only run of the same target, same caveat
 * upstream documents for laf-intel/CompCov. */
static int __afl_compcov_level = 0;

/* Longest byte-buffer comparison COMPCOV walks per call, independent of
 * CMPLOG_MAX_OPERAND: this cap bounds edge-table writes per intercepted
 * call (up to one per byte of match), not bytes logged, so it is kept
 * well below CMPLOG_MAX_OPERAND to keep the extra probe/insert cost
 * bounded even at level 2 on a target with long matching prefixes. */
#define COMPCOV_MAX_OPERAND 32

/* Forward decl: defined below, ahead of __afl_cmplog_bytes, which needs it
 * for the same reason -- bounding a byte-walk to memory that is actually
 * mapped and (for ASAN builds) unpoisoned before reading past a mismatch
 * that might be the buffer's own end. */
__AFL_NO_COV static size_t __afl_readable_len(const void *p, size_t want);

/* Where a byte walk stops. The value is the element width whose all-zero
 * match ends a string: bytes past the terminator are never compared (and
 * may be uninitialised), so marking them mints input-independent noise.
 *   strncmp("AB\0QQ", "AB\0QQ", 6) -> STR marks 0..2, MEM would mark 0..5 */
enum __afl_compcov_kind {
    COMPCOV_MEM  = 0,               /* memcmp/bcmp/wmemcmp: full length */
    COMPCOV_STR  = 1,               /* strcmp/strncmp: stop after '\0' */
    COMPCOV_WSTR = sizeof(wchar_t), /* wcscmp/wcsncmp: stop after L'\0' */
};

#if __AFL_EDGE
/* Exec-stable site key: the raw return address moves with ASLR, so a
 * target run with FUZZER_KEEP_ASLR=1 keys on the load-base-relative
 * offset instead -- same rule as __afl_get_caller_ctx(). */
__AFL_NO_COV static inline uint64_t __afl_compcov_site(void *pc) {
    uintptr_t addr = (uintptr_t)pc;
    if (!__afl_ctx_use_relative()) return addr;

    uintptr_t base = __afl_ctx_resolve_base(pc);
    return base ? addr - base : addr;
}

/* Fold one byte (or one power-of-two width step) of comparison progress
 * into the edge table as a synthetic edge, keyed on the comparison site
 * (__afl_compcov_site) and how far the match got. Same FNV-1a shape as
 * __sfuzz_state's hash, salted differently so the two synthetic-edge
 * channels do not structurally alias each other. */
__AFL_NO_COV static inline void __afl_compcov_mark(uint64_t site, uint32_t tag) {
    if (!__afl_area) return;

    uint64_t h = 1469598103934665603ULL; /* FNV-1a offset basis */
    h = (h ^ site) * 1099511628211ULL;
    h = (h ^ 0x434f4d5043564356ULL) * 1099511628211ULL; /* "COMPCVCV" salt */
    h = (h ^ tag) * 1099511628211ULL;
    __afl_map_id(__AFL_SYNTH_ID(h >> 32));
}

/* Layer 2: a and b are the raw operands of an n-byte trace-cmp callback
 * (n in {1,2,4,8}). Walks byte counts 1..n-1 (never n itself -- an
 * n-byte match is full equality, which already produces a genuine branch
 * edge right after the callback returns, so marking it here would just
 * duplicate that edge under a different id) and stops at the first byte
 * that breaks the low-to-high match, mirroring afl_compcov_log_32's
 * chained-if shape. is_const selects the level gate: the trace_const_cmp*
 * callbacks pass 1 (fires from level >= 1), the plain trace_cmp*
 * callbacks pass 0 (fires from level >= 2 only). */
__AFL_NO_COV static inline void __afl_compcov_ints(uint64_t a, uint64_t b, size_t n,
                                                    void *pc, int is_const) {
    int level = __afl_compcov_level;
    if (level < (is_const ? 1 : 2)) return;
    if ((a & 0xFF) != (b & 0xFF)) return; /* no progress: skip the site lookup */

    uint64_t site = __afl_compcov_site(pc);
    for (size_t i = 1; i < n; i++) {
        uint64_t mask = (i >= 8) ? ~0ULL : ((1ULL << (i * 8)) - 1);
        if ((a & mask) != (b & mask)) return;
        __afl_compcov_mark(site, (uint32_t)i);
    }
}

/* Layer 1: walks two byte buffers from the front, marking each matching
 * position before the first mismatch (or COMPCOV_MAX_OPERAND, whichever
 * is shorter) -- the libcompcov __compcov_trace shape. Only ever gated
 * at level 2: unlike an integer compare, a libc call site gives no
 * compiler-verified signal that either buffer is a constant. Caller is
 * responsible for n already being a length safe to read from both a and
 * b (the same n it already passed to __afl_cmplog_bytes). kind says
 * whether a matched terminator ends the walk (see enum __afl_compcov_kind). */
__AFL_NO_COV static inline void __afl_compcov_bytes(const void *a, const void *b, size_t n,
                                                     void *pc, enum __afl_compcov_kind kind) {
    if (__afl_compcov_level < 2 || !a || !b || n == 0) return;
    size_t k = n > COMPCOV_MAX_OPERAND ? COMPCOV_MAX_OPERAND : n;
    /* Same readability clamp __afl_cmplog_bytes applies: a caller-supplied
     * n (strncmp's bound, in particular) is not a promise that all n bytes
     * are mapped on both sides, only that reading up to the first
     * difference or terminator is safe. */
    size_t ka = __afl_readable_len(a, k), kb = __afl_readable_len(b, k);
    k = ka < kb ? ka : kb;
    const unsigned char *pa = (const unsigned char *)a;
    const unsigned char *pb = (const unsigned char *)b;
    if (k == 0 || pa[0] != pb[0]) return; /* no progress: skip the site lookup */

    uint64_t site = __afl_compcov_site(pc);
    size_t w = (size_t)kind;
    for (size_t i = 0; i < k; i++) {
        if (pa[i] != pb[i]) return;
        __afl_compcov_mark(site, (uint32_t)i);

        /* Matched a whole element: stop if it was the terminator. */
        if (w == 0 || (i + 1) % w != 0) continue;
        size_t z = 0;
        while (z < w && pa[i + 1 - w + z] == 0) z++;
        if (z == w) return;
    }
}
#else /* !__AFL_EDGE: __AFL_PRELOAD_ONLY has no edge map to mark into */
__AFL_NO_COV static inline void __afl_compcov_ints(uint64_t a, uint64_t b, size_t n,
                                                    void *pc, int is_const) {
    (void)a; (void)b; (void)n; (void)pc; (void)is_const;
}
__AFL_NO_COV static inline void __afl_compcov_bytes(const void *a, const void *b, size_t n,
                                                     void *pc, enum __afl_compcov_kind kind) {
    (void)a; (void)b; (void)n; (void)pc; (void)kind;
}
#endif /* __AFL_EDGE */

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
static struct __afl_fd_id __afl_cmp_counts_fd_id;

/* ── Per-SITE comparison counters ($_CMPLOG_SITE_COUNTS) ──────────────
 *
 * The per-callback counters above answer "which family is the wall" and
 * can never answer "which comparison". Two memcmp call sites share one
 * bucket, so a target with one satisfied memcmp per execution and a
 * million unsatisfied ones reports a 10^-6 assert rate rather than the two
 * separate facts it actually is.
 *
 * Keying by program counter separates them. The call site comes from
 * __builtin_return_address(0) evaluated inside the interceptor, which is
 * the instruction after the call -- so the counting macro picks it up
 * without a single interceptor having to pass it, and layer 2 gets the
 * same site identity it already puts in its records.
 *
 * Absolute addresses rather than module offsets, matching the pc field
 * layer-2 records already carry. That is only sound because the fuzzer
 * disables ASLR for the target (personality(ADDR_NO_RANDOMIZE), set in
 * adapters/process.py and inherited across fork and exec); without it the
 * same site would occupy a fresh slot on every execution and the table
 * would fill with one-hit entries.
 *
 * Open addressing with linear probing and a hard probe bound. Never grows
 * and never evicts: a full table stops admitting new sites and counts the
 * refusals, which is the failure mode that can be reported rather than the
 * one that silently reallocates on the hot path. Entries are zeroed by a
 * dump but keep their keys, so a site pays its insertion cost once.
 *
 * On its own env var and its own fd, deliberately separate from
 * $_CMPLOG_COUNTS: this is a hash and a probe per comparison, against two
 * array increments for the per-callback counters, and memcmp is hot enough
 * in most targets that the difference is not something to opt everyone
 * into. */
#define __AFL_CMP_SITE_SLOTS 4096u   /* power of two: mask instead of modulo */
#define __AFL_CMP_SITE_PROBES 8      /* give up rather than walk the table */

struct __afl_cmp_site {
    uint64_t pc;        /* 0 = empty slot; a real return address is never 0 */
    uint64_t fired;
    uint64_t hit;
    uint32_t id;        /* which callback, index into __afl_cmp_names */
};

static struct __afl_cmp_site __afl_cmp_sites[__AFL_CMP_SITE_SLOTS];
static uint64_t __afl_cmp_site_dropped;   /* insertions the table refused */
static int      __afl_cmp_sites_fd = -1;
static struct __afl_fd_id __afl_cmp_sites_fd_id;

/* splitmix64's finalizer. The low bits of a return address are nearly
 * constant across sites in one function, so the index has to come from
 * mixed bits or every site in a hot loop lands in the same probe run. */
__AFL_NO_COV static inline uint64_t __afl_cmp_site_hash(uint64_t pc, uint32_t id) {
    uint64_t x = pc ^ ((uint64_t)id << 56);
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__AFL_NO_COV static void __afl_cmp_site_count(uint32_t id, int satisfied, void *pc_ptr) {
    uint64_t pc = (uint64_t)(uintptr_t)pc_ptr;
    if (pc == 0) return;   /* no caller frame: nothing to key on */
    uint64_t idx = __afl_cmp_site_hash(pc, id) & (__AFL_CMP_SITE_SLOTS - 1);
    for (int probe = 0; probe < __AFL_CMP_SITE_PROBES; probe++) {
        struct __afl_cmp_site *e = &__afl_cmp_sites[(idx + (uint64_t)probe)
                                                    & (__AFL_CMP_SITE_SLOTS - 1)];
        if (e->pc == 0) {
            e->pc = pc;
            e->id = id;
        } else if (e->pc != pc || e->id != id) {
            continue;
        }
        e->fired++;
        if (satisfied) e->hit++;
        return;
    }
    __afl_cmp_site_dropped++;
}

/* The fd doubles as the enable flag: no output file, no counting. */
#define __AFL_CMP_COUNT(id, satisfied)                                   \
    do {                                                                 \
        if (__afl_cmp_counts_fd >= 0) {                                  \
            __afl_cmp_fired[(id)]++;                                     \
            if (satisfied) __afl_cmp_hit[(id)]++;                        \
        }                                                                \
        if (__afl_cmp_sites_fd >= 0) {                                   \
            /* Expanded inside the interceptor, so frame 0's return       \
             * address is the call site in the target. Every interceptor  \
             * already invokes this macro; none needed editing. */        \
            __afl_cmp_site_count((id), (satisfied),                      \
                                 __builtin_return_address(0));           \
        }                                                                \
    } while (0)

/* Caller holds __afl_cmplog_lock. */
__AFL_NO_COV static void __afl_cmplog_flush_locked(void) {
    if (__afl_cmplog_pos == 0) {
        return;
    }
    /* Lazy reopen: if Python rotated the log and closed the fd, reopen
     * from the current _CMPLOG_OUT so the next write doesn't silently
     * drop cmplog records.
     *
     * O_NONBLOCK here is what makes _CMPLOG_OUT safe to point at a FIFO:
     * a blocking open(2) of a FIFO for write-only stalls until a reader
     * shows up, and this call can run from __afl_auto_init before the
     * forkserver hello handshake. With O_NONBLOCK, opening a FIFO with no
     * reader yet fails immediately (ENXIO) instead of hanging, and this
     * lazy-reopen path already retries on every flush -- so the reader
     * (the Python-side FIFO drain thread) just needs to exist by the time
     * the *next* flush fires, not before this one. No-op for a regular
     * file: O_NONBLOCK only changes open(2)/write(2) semantics for FIFOs
     * and some device nodes. */
    __afl_log_keep(&__afl_cmplog_fd, &__afl_cmplog_fd_id, "_CMPLOG_OUT", O_NONBLOCK);
    if (__afl_cmplog_fd < 0) {
        const char *path = getenv("_CMPLOG_OUT");
        if (path && path[0]) {
            __afl_cmplog_fd = __afl_log_open(path, O_NONBLOCK, &__afl_cmplog_fd_id);
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
        /* w <= 0 covers EAGAIN (pipe full, nothing draining it right now)
         * the same way it already covered EINTR/ENOSPC: drop the rest of
         * this batch and move on, never block or spin. On a regular file
         * this is unreachable in practice; on a FIFO it is the expected
         * steady-state path whenever the collector falls behind. */
        if (w <= 0) {
            __afl_health[__AFL_HEALTH_CMPLOG_DROPPED]++;
            break;
        }
        off += (size_t)w;
    }
    __afl_cmplog_pos = 0;
}

/* Skipped, not waited for, while another thread owns the buffer. */
__AFL_NO_COV static void __afl_cmplog_flush(void) {
    if (__afl_cmplog_held) {   /* signal landed inside our own writer */
        __afl_cmplog_flush_locked();
        return;
    }
    if (!__afl_cmplog_acquire()) return;
    __afl_cmplog_flush_locked();
    __afl_cmplog_release();
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
    __afl_log_keep(&__afl_cmp_counts_fd, &__afl_cmp_counts_fd_id, "_CMPLOG_COUNTS", 0);
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

__AFL_NO_COV static void __afl_cmp_write_all(int fd, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, buf + off, len - off);
        if (w <= 0) return;   /* EINTR/ENOSPC: drop the rest, never spin */
        off += (size_t)w;
    }
}

/* CNS <callback> <pc-hex> <fired> <asserted>, deltas, same contract as CNT.
 * Keys survive the dump -- only the counters are zeroed -- so a site pays
 * its insertion probe once for the life of the process rather than once
 * per sync point. */
__AFL_NO_COV static void __afl_cmp_dump_sites(void) {
    __afl_log_keep(&__afl_cmp_sites_fd, &__afl_cmp_sites_fd_id, "_CMPLOG_SITE_COUNTS", 0);
    if (__afl_cmp_sites_fd < 0) return;
    char buf[4096];
    char *p = buf;
    for (unsigned i = 0; i < __AFL_CMP_SITE_SLOTS; i++) {
        struct __afl_cmp_site *e = &__afl_cmp_sites[i];
        if (e->pc == 0 || e->fired == 0) continue;
        if ((size_t)(p - buf) + 96 > sizeof(buf)) {
            /* Flush and keep going: the table is orders of magnitude
             * larger than any sane stack buffer, and dropping the tail
             * would systematically lose whichever sites hash high. */
            __afl_cmp_write_all(__afl_cmp_sites_fd, buf, (size_t)(p - buf));
            p = buf;
        }
        const char *nm = e->id < __AFL_CMP_SITES ? __afl_cmp_names[e->id] : "?";
        *p++ = 'C'; *p++ = 'N'; *p++ = 'S'; *p++ = ' ';
        while (*nm) *p++ = *nm++;
        *p++ = ' ';
        p = __afl_put_hex64(p, e->pc);
        *p++ = ' ';
        p = __afl_put_i64(p, (int64_t)e->fired);
        *p++ = ' ';
        p = __afl_put_i64(p, (int64_t)e->hit);
        *p++ = '\n';
        e->fired = 0;
        e->hit   = 0;
    }
    if (__afl_cmp_site_dropped) {
        if ((size_t)(p - buf) + 64 > sizeof(buf)) {
            __afl_cmp_write_all(__afl_cmp_sites_fd, buf, (size_t)(p - buf));
            p = buf;
        }
        *p++ = 'C'; *p++ = 'N'; *p++ = 'D'; *p++ = ' ';
        p = __afl_put_i64(p, (int64_t)__afl_cmp_site_dropped);
        *p++ = '\n';
        __afl_cmp_site_dropped = 0;
    }
    if (p != buf) __afl_cmp_write_all(__afl_cmp_sites_fd, buf, (size_t)(p - buf));
}

/* ── Operand readability ──────────────────────────────────────────────
 * The length an interceptor hands us is a semantic bound -- how many bytes
 * the real call was allowed to look at -- not a readable one. strncmp stops
 * at the first NUL or mismatch, so strncmp(p, "http:", 5) on a 2-byte p that
 * differs at byte 1 is legal C, and logging 5 bytes of p reads 3 past the
 * end. FFmpeg's url_find_protocol does exactly that against the protocol
 * table; a campaign against ffmpeg_read surfaced it as an ASAN
 * heap-buffer-overflow whose top frames were __afl_put_hexbytes and
 * __afl_cmplog_bytes -- our own instrumentation crashing, reported as a
 * crash in the target.
 *
 * Bounding it inside each interceptor is what the file has been doing
 * (strcmp measures with __afl_fb_len; the memmem site carries a comment
 * about the same class of read), and strncmp and strncasecmp are the two
 * that were missed out of 18 call sites. So the bound belongs here, at the
 * one place that dereferences the operands, where a new interceptor cannot
 * forget it.
 *
 * Two layers, because they cover different builds:
 *   - Page clamp, always. A read that stays inside the page holding p cannot
 *     fault: the caller just read from that page. This is the only guard a
 *     non-ASAN target gets, and it is the one that matters there -- a fault
 *     in the nosan .so is an in-process SIGSEGV with our frame on top and no
 *     redzone diagnostic to say whose fault it was.
 *   - __asan_region_is_poisoned when it resolves. Weak symbol: present in an
 *     ASAN build, NULL otherwise. Gives the exact object bound, not the page
 *     bound.
 *
 * Deliberately not a truncation to the first mismatch. That is provably safe
 * and destroys the point of the record: with buf all 'A',
 * strncmp(buf, "PREFIX_", 7) mismatches at byte 0, so the semantic bound is
 * one byte and the literal never reaches the pool -- the token redqueen
 * exists to inject. Clamping by readability keeps the whole operand whenever
 * it is legal to read, and shortens only when it is not. */
extern const void *__asan_region_is_poisoned(const void *p, size_t n)
    __attribute__((weak));

__AFL_NO_COV static size_t __afl_readable_len(const void *p, size_t want) {
    static size_t page_size = 0;
    if (!p || want == 0) return 0;
    if (page_size == 0) {
        long ps = sysconf(_SC_PAGESIZE);
        page_size = ps > 0 ? (size_t)ps : 4096;
    }
    uintptr_t addr = (uintptr_t)p;
    size_t to_page_end = page_size - (size_t)(addr & (uintptr_t)(page_size - 1));
    if (want > to_page_end) want = to_page_end;
    if (__asan_region_is_poisoned) {
        const void *bad = __asan_region_is_poisoned(p, want);
        if (bad) want = (size_t)((const char *)bad - (const char *)p);
    }
    return want;
}

/* ── Per-drain record dedup ───────────────────────────────────────────
 * 79% of ffmpeg's records repeat one already written since the last drain
 * (format probes and av_opt_set_defaults2 in loops); the collector dedups
 * each drain anyway, so a repeat only costs formatting, write(2) and room
 * under the 10k-line read cap. Each record's fingerprint goes into an
 * open-addressed table and a hit is dropped before formatting. The
 * generation stamp clears the table in O(1) at __cmplog_reset (the drain
 * boundary) and in a fork child. A full probe run fails open: a repeat
 * costs bytes, a false drop would cost a record. Caller holds the lock. */
#define CMPLOG_DEDUP_SLOTS  (1u << 17)
#define CMPLOG_DEDUP_PROBES 8
#define CMPLOG_FNV_OFFSET   0xcbf29ce484222325ULL
#define CMPLOG_FNV_PRIME    0x100000001b3ULL

struct __afl_cmplog_seen_slot {
    uint64_t fp;
    uint32_t gen;
};
static struct __afl_cmplog_seen_slot __afl_cmplog_seen_tab[CMPLOG_DEDUP_SLOTS];
static uint32_t __afl_cmplog_gen = 1;

/* splitmix64 finalizer. */
__AFL_NO_COV static inline uint64_t __afl_rec_mix(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__AFL_NO_COV static void __afl_cmplog_new_scope(void) {
    if (++__afl_cmplog_gen != 0) return;
    /* 2^32 drains: wrapped stamps would read as live, so wipe. */
    memset(__afl_cmplog_seen_tab, 0, sizeof __afl_cmplog_seen_tab);
    __afl_cmplog_gen = 1;
}

/* 1 when *fp* was already written in this scope; records it otherwise. */
__AFL_NO_COV static int __afl_cmplog_seen(uint64_t fp) {
    uint32_t idx = (uint32_t)fp & (CMPLOG_DEDUP_SLOTS - 1);
    for (int probe = 0; probe < CMPLOG_DEDUP_PROBES; probe++) {
        struct __afl_cmplog_seen_slot *e =
            &__afl_cmplog_seen_tab[(idx + (uint32_t)probe) & (CMPLOG_DEDUP_SLOTS - 1)];
        if (e->gen != __afl_cmplog_gen) {
            e->gen = __afl_cmplog_gen;
            e->fp = fp;
            return 0;
        }
        if (e->fp == fp) return 1;
    }
    return 0;
}

/* Fingerprint of a layer-1 record: every field its line prints. */
__AFL_NO_COV static uint64_t __afl_bytes_fp(const unsigned char *a, const unsigned char *b,
                                            size_t k, size_t n, int result) {
    uint64_t h = CMPLOG_FNV_OFFSET;
    for (size_t i = 0; i < k; i++) h = (h ^ a[i]) * CMPLOG_FNV_PRIME;
    for (size_t i = 0; i < k; i++) h = (h ^ b[i]) * CMPLOG_FNV_PRIME;
    h = __afl_rec_mix(h ^ (uint64_t)n);
    return __afl_rec_mix(h ^ (uint64_t)(int64_t)result ^ ((uint64_t)k << 56));
}

/* ── Layer 1 record: two byte buffers ─────────────────────────────────
 * result == 0 is dropped: an already-satisfied comparison is exactly the
 * "looks unsolved but is solved" pollution the pair pool must not carry. */
__AFL_NO_COV static void __afl_cmplog_bytes(const void *a, const void *b, size_t n, int result) {
    if (__afl_cmplog_fd < 0 || __afl_cmplog_paused || !a || !b || n == 0 || result == 0) return;
    size_t k = n > CMPLOG_MAX_OPERAND ? CMPLOG_MAX_OPERAND : n;
    /* Both operands are hexdumped at the same width, so the record can only
     * be as wide as the shorter readable side. */
    size_t ka = __afl_readable_len(a, k), kb = __afl_readable_len(b, k);
    k = ka < kb ? ka : kb;
    if (k == 0) return;
    uint64_t fp = __afl_bytes_fp((const unsigned char *)a, (const unsigned char *)b, k, n, result);
    if (!__afl_cmplog_acquire()) return;
    if (__afl_cmplog_seen(fp)) {
        __afl_cmplog_release();
        return;
    }
    if (__afl_cmplog_pos + CMPLOG_MAX_RECORD > CMPLOG_BUFFER_SIZE)
        __afl_cmplog_flush_locked();
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
    __afl_cmplog_release();
}

/* ── Layer 2 record: two integers plus the comparison site ────────────
 * pc is __builtin_return_address(0) from the callback: the instruction
 * after the call, i.e. the comparison site itself once inlined. */

__AFL_NO_COV static inline void __afl_cmplog_ints(uint64_t a, uint64_t b, size_t n, void *pc) {
    if (__afl_cmplog_fd < 0 || __afl_cmplog_paused) return;
    /* The lock is also the backstop for toolchains where __AFL_NO_COV
     * expands to nothing: an instrumented record writer re-enters through
     * its own trace-cmp callbacks, finds the lock held, and returns
     * instead of recursing until the stack is gone. */
    if (!__afl_cmplog_acquire()) return;
    /* a and b fix the printed sign field; 'I' keeps it apart from operands. */
    uint64_t fp = __afl_rec_mix(__afl_rec_mix(a ^ 'I') ^ b);
    if (__afl_cmplog_seen(__afl_rec_mix(fp ^ ((uint64_t)n << 56) ^ (uint64_t)(uintptr_t)pc))) {
        __afl_cmplog_release();
        return;
    }
    if (__afl_cmplog_pos + CMPLOG_MAX_RECORD > CMPLOG_BUFFER_SIZE)
        __afl_cmplog_flush_locked();
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
    __afl_cmplog_release();
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
    __afl_compcov_bytes(a, b, n, __builtin_return_address(0), COMPCOV_MEM);
    return result;
}
__AFL_NO_COV int afl_cmp_memcmp(const void *a, const void *b, size_t n)
    __attribute__((weak, alias("memcmp")));

__AFL_NO_COV int strcmp(const char *a, const char *b) {
    __AFL_RESOLVE(real_strcmp, afl_str_cmp_fn, "strcmp", __afl_fb_strcmp);
    int result = real_strcmp(a, b);
    __AFL_CMP_COUNT(__AFL_CMP_STRCMP, result == 0);
    if (__afl_cmplog_fd < 0 && __afl_compcov_level < 2) return result;
    size_t na = __afl_fb_len(a), nb = __afl_fb_len(b), n = na < nb ? na : nb;
    if (n > 0) {
        __afl_cmplog_bytes(a, b, n + 1, result);
        __afl_compcov_bytes(a, b, n + 1, __builtin_return_address(0), COMPCOV_STR);
    }
    return result;
}
__AFL_NO_COV int afl_cmp_strcmp(const char *a, const char *b)
    __attribute__((weak, alias("strcmp")));

__AFL_NO_COV int strncmp(const char *a, const char *b, size_t n) {
    __AFL_RESOLVE(real_strncmp, afl_strn_cmp_fn, "strncmp", __afl_fb_strncmp);
    int result = real_strncmp(a, b, n);
    __AFL_CMP_COUNT(__AFL_CMP_STRNCMP, result == 0);
    if (n > 0) {
        __afl_cmplog_bytes(a, b, n, result);
        __afl_compcov_bytes(a, b, n, __builtin_return_address(0), COMPCOV_STR);
    }
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
    if (__afl_cmplog_fd < 0) return result;
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
    __afl_compcov_bytes(a, b, n, __builtin_return_address(0), COMPCOV_MEM);
    return result;
}
__AFL_NO_COV int afl_cmp_bcmp(const void *a, const void *b, size_t n)
    __attribute__((weak, alias("bcmp")));

__AFL_NO_COV int wmemcmp(const wchar_t *a, const wchar_t *b, size_t n) {
    __AFL_RESOLVE(real_wmemcmp, afl_wchar_cmp_fn, "wmemcmp", __afl_fb_wmemcmp);
    int result = real_wmemcmp(a, b, n);
    __AFL_CMP_COUNT(__AFL_CMP_WMEMCMP, result == 0);
    /* Gated on cmplog-log-active OR compcov-active: compcov has no fd of
     * its own, so a run with only $__AFL_COMPCOV_LEVEL set (no
     * _CMPLOG_OUT) must not skip this block the way it did before. */
    if ((__afl_cmplog_fd >= 0 || __afl_compcov_level >= 2) && n > 0) {
        size_t k = n * sizeof(wchar_t);
        if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
        __afl_cmplog_bytes(a, b, k, result);
        __afl_compcov_bytes(a, b, k, __builtin_return_address(0), COMPCOV_MEM);
    }
    return result;
}
__AFL_NO_COV int afl_cmp_wmemcmp(const wchar_t *a, const wchar_t *b, size_t n)
    __attribute__((weak, alias("wmemcmp")));

__AFL_NO_COV int wcsncmp(const wchar_t *a, const wchar_t *b, size_t n) {
    __AFL_RESOLVE(real_wcsncmp, afl_wchar_cmp_fn, "wcsncmp", __afl_fb_wcsncmp);
    int result = real_wcsncmp(a, b, n);
    __AFL_CMP_COUNT(__AFL_CMP_WCSNCMP, result == 0);
    if ((__afl_cmplog_fd >= 0 || __afl_compcov_level >= 2) && n > 0) {
        size_t k = n * sizeof(wchar_t);
        if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
        if (k > 0) {
            __afl_cmplog_bytes(a, b, k, result);
            __afl_compcov_bytes(a, b, k, __builtin_return_address(0), COMPCOV_WSTR);
        }
    }
    return result;
}
__AFL_NO_COV int afl_cmp_wcsncmp(const wchar_t *a, const wchar_t *b, size_t n)
    __attribute__((weak, alias("wcsncmp")));

__AFL_NO_COV int wcscmp(const wchar_t *a, const wchar_t *b) {
    __AFL_RESOLVE(real_wcscmp, afl_wchar_cmp_2arg_fn, "wcscmp", __afl_fb_wcscmp);
    int result = real_wcscmp(a, b);
    __AFL_CMP_COUNT(__AFL_CMP_WCSCMP, result == 0);
    if (__afl_cmplog_fd >= 0 || __afl_compcov_level >= 2) {
        size_t na = __afl_fb_wcslen(a), nb = __afl_fb_wcslen(b), n = na < nb ? na : nb;
        if (n > 0) {
            size_t k = n * sizeof(wchar_t);
            if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
            __afl_cmplog_bytes(a, b, k, result);
            __afl_compcov_bytes(a, b, k, __builtin_return_address(0), COMPCOV_WSTR);
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
    if (__afl_cmplog_fd >= 0 && __afl_launder_ptr(s) && __afl_launder_ptr(accept)) {
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
    if (__afl_cmplog_fd >= 0 && __afl_launder_ptr(s) && __afl_launder_ptr(accept)) {
        size_t sl = __afl_fb_len(s), al = __afl_fb_len(accept);
        if (sl > 0 && al > 0) {
            size_t k = sl < al ? sl : al;
            if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
            __afl_cmplog_bytes(s, accept, k, result ? 0 : -1);  /* drop solved */
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
    if (__afl_cmplog_fd >= 0 && __afl_launder_ptr(s) && __afl_launder_ptr(reject)) {
        size_t sl = __afl_fb_len(s), rl = __afl_fb_len(reject);
        if (sl > 0 && rl > 0) {
            size_t k = sl < rl ? sl : rl;
            if (k > CMPLOG_MAX_OPERAND) k = CMPLOG_MAX_OPERAND;
            __afl_cmplog_bytes(s, reject, k, result ? 0 : -1);  /* drop solved */
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
        /* memrchr scans from the end, so the bytes it compared first are
         * the tail; logging the head anchored the pair on bytes it may
         * never have reached. */
        unsigned char needle[CMPLOG_MAX_OPERAND];
        for (size_t i = 0; i < CMPLOG_MAX_OPERAND; i++) needle[i] = (unsigned char)c;
        size_t k = n > CMPLOG_MAX_OPERAND ? CMPLOG_MAX_OPERAND : n;
        __afl_cmplog_bytes((const unsigned char *)s + (n - k), needle, k, -1);
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
    __afl_compcov_ints(a, b, 2, __builtin_return_address(0), 0);
}
__AFL_CMP_VIS void __sanitizer_cov_trace_cmp4(uint32_t a, uint32_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CMP4, a == b);
    __afl_cmplog_ints(a, b, 4, __builtin_return_address(0));
    __afl_compcov_ints(a, b, 4, __builtin_return_address(0), 0);
}
__AFL_CMP_VIS void __sanitizer_cov_trace_cmp8(uint64_t a, uint64_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CMP8, a == b);
    __afl_cmplog_ints(a, b, 8, __builtin_return_address(0));
    __afl_compcov_ints(a, b, 8, __builtin_return_address(0), 0);
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp1(uint8_t a, uint8_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CONST_CMP1, a == b);
    __afl_cmplog_ints(a, b, 1, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp2(uint16_t a, uint16_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CONST_CMP2, a == b);
    __afl_cmplog_ints(a, b, 2, __builtin_return_address(0));
    __afl_compcov_ints(a, b, 2, __builtin_return_address(0), 1);
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp4(uint32_t a, uint32_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CONST_CMP4, a == b);
    __afl_cmplog_ints(a, b, 4, __builtin_return_address(0));
    __afl_compcov_ints(a, b, 4, __builtin_return_address(0), 1);
}
__AFL_CMP_VIS void __sanitizer_cov_trace_const_cmp8(uint64_t a, uint64_t b) {
    __AFL_CMP_COUNT(__AFL_CMP_TRACE_CONST_CMP8, a == b);
    __afl_cmplog_ints(a, b, 8, __builtin_return_address(0));
    __afl_compcov_ints(a, b, 8, __builtin_return_address(0), 1);
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
    /* ref[1] is the case width in bits. Logging a 1-byte switch at 8
     * bytes pads the operand with zeros the input never contains. */
    size_t width = (size_t)(ref[1] / 8);
    if (width != 1 && width != 2 && width != 4) width = 8;
    for (int64_t i = 0; i < count; i++)
        __afl_cmplog_ints(val, ref[2 + i], width, pc);
}

/* ── Layer 3: single operands (-fsanitize-coverage=trace-div,trace-gep) ─
 *
 * trace-div hands us the runtime divisor of every non-constant division;
 * trace-gep the runtime index of every GEP. Both are input-derived far more
 * often than they are constant, and neither is visible to trace-cmp -- a
 * division is not a comparison and an array index is only compared against
 * the bound when the target bothers to check it.
 *
 * They get their own record kinds rather than a CMP with an invented
 * opponent: the fuzzer counts CMP records to find comparison walls, and a
 * divisor has nothing to be satisfied against. core/cmplog.py pairs each
 * observed operand with the values that make it interesting (0/1 for a
 * divisor, 0/all-ones for an index).
 *
 *   DIV <hex value> <width> 0x<pc>
 *   GEP <hex index> <width> 0x<pc>
 *
 * The floor is the throttle. Every loop counter is a GEP index, so logging
 * them all would cost more stream than the pairs are worth; below
 * CMPLOG_OPERAND_MIN plain havoc reaches the value anyway. Keep it in step
 * with track_parser.OPERAND_MIN_VALUE, which drops the same records again
 * on the reading side. */
#define CMPLOG_OPERAND_MIN 256

__AFL_NO_COV static inline void __afl_cmplog_operand(const char *kind, uint64_t v,
                                                     size_t n, void *pc) {
    if (__afl_cmplog_fd < 0 || __afl_cmplog_paused || v < CMPLOG_OPERAND_MIN) return;
    if (!__afl_cmplog_acquire()) return;
    uint64_t fp = __afl_rec_mix(v ^ (uint64_t)(unsigned char)kind[0]);
    if (__afl_cmplog_seen(__afl_rec_mix(fp ^ ((uint64_t)n << 56) ^ (uint64_t)(uintptr_t)pc))) {
        __afl_cmplog_release();
        return;
    }

    if (__afl_cmplog_pos + CMPLOG_MAX_RECORD > CMPLOG_BUFFER_SIZE)
        __afl_cmplog_flush_locked();

    unsigned char vb[8];
    for (size_t i = 0; i < n; i++) vb[i] = (unsigned char)(v >> (i * 8));

    char *p = __afl_cmplog_buf + __afl_cmplog_pos;
    *p++ = kind[0]; *p++ = kind[1]; *p++ = kind[2]; *p++ = ' ';
    p = __afl_put_hexbytes(p, vb, n);
    *p++ = ' ';
    p = __afl_put_i64(p, (int64_t)n);
    *p++ = ' ';
    p = __afl_put_hex64(p, (uint64_t)(uintptr_t)pc);
    *p++ = '\n';
    __afl_cmplog_pos = (size_t)(p - __afl_cmplog_buf);

    __afl_cmplog_release();
}

__AFL_CMP_VIS void __sanitizer_cov_trace_div4(uint32_t val) {
    __afl_cmplog_operand("DIV", val, 4, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_div8(uint64_t val) {
    __afl_cmplog_operand("DIV", val, 8, __builtin_return_address(0));
}
__AFL_CMP_VIS void __sanitizer_cov_trace_gep(uintptr_t idx) {
    __afl_cmplog_operand("GEP", (uint64_t)idx, sizeof(uintptr_t),
                         __builtin_return_address(0));
}

/* Zero the per-callback and per-site counters, keeping site keys (a site
 * the parent already inserted costs a child no probe). Shared by the fork
 * hook below and the forkserver. */
__AFL_NO_COV static void __afl_cmp_counts_zero(void) {
    for (int i = 0; i < __AFL_CMP_SITES; i++) {
        __afl_cmp_fired[i] = 0;
        __afl_cmp_hit[i]   = 0;
    }
    for (unsigned i = 0; i < __AFL_CMP_SITE_SLOTS; i++) {
        __afl_cmp_sites[i].fired = 0;
        __afl_cmp_sites[i].hit   = 0;
    }
    __afl_cmp_site_dropped = 0;
}

/* fork() child: the record buffer, its lock and the counters are the
 * parent's. A lock held by another parent thread has no owner here, so
 * every child record was dropped; records the parent had buffered were
 * flushed twice, once per process. Start the child clean. */
__AFL_NO_COV static void __afl_cmplog_atfork_child(void) {
    __afl_cmplog_pos = 0;
    __afl_cmplog_held = 0;
    __atomic_store_n(&__afl_cmplog_lock, 0, __ATOMIC_RELEASE);
    __afl_cmp_counts_zero();
    __afl_cmplog_new_scope();
}

/* ── Lifecycle ────────────────────────────────────────────────────────
 * Called from __afl_auto_init (edge builds) or the preload-only
 * constructor below. */
__AFL_NO_COV static void __afl_cmplog_init(void) {
    pthread_atfork(NULL, NULL, __afl_cmplog_atfork_child);
    /* COMPCOV level: parsed independently of _CMPLOG_OUT below -- a run
     * that wants only the edge-map partial-match signal, and none of the
     * record/counts/sites log machinery, sets this and nothing else. */
    const char *compcov = getenv("__AFL_COMPCOV_LEVEL");
    if (compcov && compcov[0]) {
        int v = atoi(compcov);
        __afl_compcov_level = v < 0 ? 0 : (v > 2 ? 2 : v);
    }
    /* Opened before the _CMPLOG_OUT check, and on its own fd: the counters
     * are useful on their own (a target's comparison profile costs no
     * record stream at all), and the record stream gets truncated and
     * rotated on a schedule the counts must not share. */
    const char *counts = getenv("_CMPLOG_COUNTS");
    if (counts && counts[0])
        __afl_cmp_counts_fd = __afl_log_open(counts, 0, &__afl_cmp_counts_fd_id);
    /* Separate switch, not a mode of the one above: per-site counting is a
     * hash and a probe per comparison against two array increments, and
     * memcmp is hot enough in most targets that it is not something to opt
     * everyone into. */
    const char *sites = getenv("_CMPLOG_SITE_COUNTS");
    if (sites && sites[0])
        __afl_cmp_sites_fd = __afl_log_open(sites, 0, &__afl_cmp_sites_fd_id);
    const char *path = getenv("_CMPLOG_OUT");
    if (!path || !path[0]) return;
    /* O_NONBLOCK: see the matching comment in __afl_cmplog_flush. Without
     * it, a FIFO sink with no reader open yet would hang the target here,
     * before the forkserver hello -- the loader would then be waiting on
     * a hello that never arrives, with nothing in the logs to explain
     * why. */
    __afl_cmplog_fd = __afl_log_open(path, O_NONBLOCK, &__afl_cmplog_fd_id);
}

__AFL_NO_COV static void __afl_cmplog_fini(void) {
    __afl_cmplog_flush();
    /* Last chance for a short-lived process: in subprocess mode this is the
     * only dump the run ever gets. */
    __afl_cmp_dump_counts();
    __afl_cmp_dump_sites();
    if (__afl_cmp_counts_fd >= 0) {
        close(__afl_cmp_counts_fd);
        __afl_cmp_counts_fd = -1;
    }
    if (__afl_cmp_sites_fd >= 0) {
        close(__afl_cmp_sites_fd);
        __afl_cmp_sites_fd = -1;
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
    __afl_cmplog_new_scope();
    /* The per-iteration sync point in direct_lite/persistent modes. Dumping
     * here (not from __afl_cmplog_flush, which also runs on buffer-full in
     * the hot path) keeps the counts channel off the fast path. */
    __afl_cmp_dump_counts();
    __afl_cmp_dump_sites();
    if (__afl_cmplog_fd >= 0) {
        /* ftruncate/lseek are meaningless on a FIFO (ESPIPE) -- the
         * truncate-to-resync protocol direct_lite relies on only applies
         * to a seekable, size-bounded sink. fstat guards the syscalls
         * instead of just letting them fail: on a pipe there is nothing
         * to reset because there is nothing sitting on disk to begin
         * with -- the Python-side drain thread already owns and empties
         * it continuously, so "reset" is correctly a no-op here rather
         * than a silently-ignored error. */
        struct stat __cmplog_st;
        if (fstat(__afl_cmplog_fd, &__cmplog_st) == 0 && S_ISREG(__cmplog_st.st_mode)) {
            if (ftruncate(__afl_cmplog_fd, 0) != 0) { /* best effort */ }
            lseek(__afl_cmplog_fd, 0, SEEK_SET);
        }
    }
}

/* paused != 0: drop CMP/DIV/GEP records until called again with 0. */
__AFL_NO_COV __attribute__((visibility("default")))
void __cmplog_pause(int paused) { __afl_cmplog_paused = paused != 0; }

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
    __afl_cmp_dump_sites();
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

static struct sigaction __afl_old_handlers[8];
/* No SIGPIPE: its default (and Python's SIG_IGN) is not a crash, and a
 * host interpreter writing to a closed pipe must get EPIPE, not a jump. */
static int __afl_guard_signals[] = {
    SIGSEGV, SIGABRT, SIGFPE, SIGBUS, SIGILL, SIGSYS,
};
#define __afl_NUM_GUARD_SIGNALS \
    (int)(sizeof(__afl_guard_signals) / sizeof(__afl_guard_signals[0]))

/* The active guard frame of this thread: the jump buffer of the innermost
 * __afl_guarded_call on its stack, or NULL when nothing is guarded.
 *
 * Only valid while that call is on the stack. Jumping anywhere else --
 * one-shot and forkserver children, which never call it, or a host
 * process after the call returned -- resumes a dead frame: SIGFPE
 * surfaced as SIGSEGV, ASAN's SEGV report was lost, and a ctypes host died
 * on its own broken pipe.
 *
 * Per thread: a fault on a worker thread the entry spawned must not jump
 * onto the guarded thread's stack (the host then ran on the worker's OS
 * thread and hung at exit). Per frame: a nested guarded call saves the
 * outer frame and restores it, so a crash after the inner call returned
 * still lands in the outer one (a single global buffer killed the host).
 * Cleared in a fork() child, whose copy of the guarded frame belongs to a
 * copy of the host. */
static __thread sigjmp_buf *volatile __afl_guard_jmp = NULL;
static volatile sig_atomic_t __afl_handlers_live = 0;

/* siglongjmp values: a signal number, or one of these escape codes. */
#define __AFL_JMP_EXIT     0x100   /* | (status & 0xFF): exit()/_exit() in the guard */
#define __AFL_JMP_TIMEOUT  0x200   /* the guard's timer expired */

/* What __afl_guarded_call_timeout returns when the budget runs out. -1 is
 * free: guard signals are >= SIGILL (4), so -sig is never -1, and it is
 * the fuzzer's cross-backend timeout sentinel. */
#define __AFL_GUARD_TIMEOUT_RC (-1)

/* Only a hardware fault re-executes into the same fault. Returning from
 * anything else loses the signal: a seccomp SIGSYS (si_code SYS_SECCOMP,
 * > 0) resumed past the trapped syscall and the violation vanished. */
static int __afl_is_hw_fault(int sig, const siginfo_t *si) {
    if (!si || si->si_code <= 0) return 0;
    return sig == SIGSEGV || sig == SIGBUS || sig == SIGFPE || sig == SIGILL;
}

/* Alternate signal stack for threads that call __afl_guarded_call. A stack
 * overflow leaves no room to run the handler on the faulting stack, so the
 * kernel killed the whole process -- the fuzzer itself in direct mode.
 * One mapping per guarded thread, never freed (bounded by thread count);
 * an existing alternate stack (ASAN's, the host's) is kept. */
#define __AFL_ALTSTACK_SIZE (64 * 1024)
static __thread int __afl_altstack_ready = 0;

static void __afl_install_altstack(void) {
    __afl_altstack_ready = 1;

    stack_t cur;
    if (sigaltstack(NULL, &cur) == 0 && !(cur.ss_flags & SS_DISABLE)) return;

    void *mem = mmap(NULL, __AFL_ALTSTACK_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) return;
    stack_t ss;
    ss.ss_sp = mem;
    ss.ss_size = __AFL_ALTSTACK_SIZE;
    ss.ss_flags = 0;
    if (sigaltstack(&ss, NULL) != 0) munmap(mem, __AFL_ALTSTACK_SIZE);
}

static void __afl_restore_crash_handlers(void) {
    for (int i = 0; i < __afl_NUM_GUARD_SIGNALS; i++)
        sigaction(__afl_guard_signals[i], &__afl_old_handlers[i], NULL);
    __afl_handlers_live = 0;
}

static void __afl_crash_handler(int sig, siginfo_t *si, void *uc) {
    (void)uc;
#if __AFL_CMPLOG
    /* Flush before escaping. cmplog_shim.c installed a second handler for
     * this and restored the previous disposition from inside it, so the
     * comparison buffer was flushed on the FIRST crash only -- every later
     * crash in a persistent/direct_lite loop lost up to 256KB of records.
     * Folding it in here runs it on every crash, and __afl_cmplog_flush is
     * write(2)-based precisely so it is legal at this point. */
    __afl_cmplog_flush();
    if (__afl_cmplog_held) __afl_cmplog_release();
    /* Same reasoning for the counters: a crashing execution's comparison
     * profile is the one most worth having, and the dump is write(2)-only. */
    __afl_cmp_dump_counts();
    __afl_cmp_dump_sites();
#endif
    __afl_sancov_fold();
    sigjmp_buf *jb = __afl_guard_jmp;
    if (jb)
        siglongjmp(*jb, sig);

    /* Not ours to recover: hand the signal back to whoever owned it.
     * A hardware fault re-executes and faults again with its real si_addr;
     * anything else (kill, abort, seccomp) is re-raised and delivered once
     * this handler returns. __afl_guarded_call re-arms later. */
    __afl_health[__AFL_HEALTH_STRAY_SIGNALS]++;
    __afl_restore_crash_handlers();
    if (__afl_is_hw_fault(sig, si))
        return;
    raise(sig);
}

static void __afl_install_crash_handlers(void) {
    struct sigaction sa;
    sa.sa_sigaction = __afl_crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    for (int i = 0; i < __afl_NUM_GUARD_SIGNALS; i++)
        sigaction(__afl_guard_signals[i], &sa, &__afl_old_handlers[i]);
    __afl_handlers_live = 1;
}

/* Called from __afl_shim_health(): a host that installs its own handler
 * after load (Python's signal module, faulthandler) leaves the guard
 * unable to recover -- the host's handler returns, the fault re-executes,
 * forever. Count it and let the next guarded call take the signals back;
 * the host's handler becomes the "previous owner" stray signals go to. */
static void __afl_check_crash_handlers(void) {
    if (!__afl_handlers_live) return;
    struct sigaction cur;
    for (int i = 0; i < __afl_NUM_GUARD_SIGNALS; i++) {
        if (sigaction(__afl_guard_signals[i], NULL, &cur) != 0) continue;
        if ((cur.sa_flags & SA_SIGINFO) && cur.sa_sigaction == __afl_crash_handler) continue;
        __afl_health[__AFL_HEALTH_HANDLERS_DISPLACED]++;
        __afl_handlers_live = 0;
        return;
    }
}

/* ── Guard timer ──────────────────────────────────────────────────────
 *
 * A C infinite loop never returns to the host, so the host's own SIGALRM
 * handler (a Python one only sets a flag between bytecodes) cannot end it:
 * direct mode hung forever. A per-thread POSIX timer delivers a real-time
 * signal to the guarded thread only; its handler leaves through the guard.
 *
 * Arming and disarming a one-shot timer per call cost two syscalls,
 * measured +455 ns per call -- about 12% of the cheapest direct-mode
 * execution. Instead the timer ticks periodically at a quarter of the
 * budget and is only reprogrammed when the budget changes; every timed
 * call bumps an epoch, and a tick ends the call once the same epoch has
 * spanned __AFL_TIMEOUT_TICKS more ticks:
 *
 *     call starts      tick 1        tick 2  ...  tick 5
 *     |----------------|- - - -|- - - -|- - - -|- - -X   timeout
 *     ^ epoch e         saw e, 1      2              > 4 -> jump
 *
 * so a hang is cut between the budget and 1.25x it, with no syscall in the
 * steady state. The period is floored at 1 ms (finer budgets round up).
 * Ticks outside a timed frame are ignored. Cost: the guarded thread takes
 * one signal per period while the timer runs; SA_RESTART restarts most
 * syscalls, but sleep()/poll() inside a timed entry can return early with
 * EINTR. One timer per guarded thread, never deleted (bounded). */
#define __AFL_TIMEOUT_SIG        (SIGRTMIN + 5)
#define __AFL_TIMEOUT_TICKS      4
#define __AFL_TIMEOUT_MIN_TICK_US 1000u

static __thread int __afl_timer_state = 0;       /* 0 unset, 1 ready, -1 unavailable */
static __thread timer_t __afl_timer;
static __thread uint64_t __afl_timer_budget_us = 0;  /* budget the period was set for */
static __thread volatile sig_atomic_t __afl_guard_timed = 0;  /* active frame is timed */
static __thread volatile uint64_t __afl_guard_epoch = 0;      /* bumped per timed call */
static __thread uint64_t __afl_tick_epoch = 0;
static __thread uint32_t __afl_tick_count = 0;
static int __afl_timeout_handler_live = 0;

static void __afl_timeout_handler(int sig, siginfo_t *si, void *uc) {
    (void)sig; (void)uc;
    sigjmp_buf *jb = __afl_guard_jmp;
    if (!jb || !__afl_guard_timed || !si || si->si_code != SI_TIMER) return;

    if (__afl_tick_epoch != __afl_guard_epoch) {
        __afl_tick_epoch = __afl_guard_epoch;
        __afl_tick_count = 0;
    }
    if (++__afl_tick_count > __AFL_TIMEOUT_TICKS)
        siglongjmp(*jb, __AFL_JMP_TIMEOUT);
}

static int __afl_timer_ready(void) {
    if (__afl_timer_state) return __afl_timer_state > 0;
    __afl_timer_state = -1;

    if (!__afl_timeout_handler_live) {
        struct sigaction sa;
        sa.sa_sigaction = __afl_timeout_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
        if (sigaction(__AFL_TIMEOUT_SIG, &sa, NULL) != 0) return 0;
        __afl_timeout_handler_live = 1;
    }

    struct sigevent sev;
    memset(&sev, 0, sizeof(sev));
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo = __AFL_TIMEOUT_SIG;
    sev._sigev_un._tid = (pid_t)syscall(SYS_gettid);
    if (timer_create(CLOCK_MONOTONIC, &sev, &__afl_timer) != 0) return 0;
    __afl_timer_state = 1;
    return 1;
}

/* Reprogram the periodic tick for a new budget; a no-op when unchanged. */
static void __afl_timer_budget(uint64_t budget_us) {
    if (budget_us == __afl_timer_budget_us) return;
    __afl_timer_budget_us = budget_us;

    uint64_t tick = budget_us / __AFL_TIMEOUT_TICKS;
    if (tick < __AFL_TIMEOUT_MIN_TICK_US) tick = __AFL_TIMEOUT_MIN_TICK_US;
    struct itimerspec its;
    its.it_value.tv_sec = (time_t)(tick / 1000000u);
    its.it_value.tv_nsec = (long)(tick % 1000000u) * 1000;
    its.it_interval = its.it_value;
    timer_settime(__afl_timer, 0, &its, NULL);
}

/* ── Guarded call wrapper ─────────────────────────────────────────────
 *
 * InProcessRunner's direct_lite mode calls this instead of the fuzz entry.
 * It sets up a sigsetjmp frame, calls the entry, and returns normally;
 * a crash, exit or timeout inside the entry siglongjmps back here.
 *
 * Returns the entry's own return value, or:
 *   -sig     a crash signal on the guarded thread (-6 SIGABRT, -11 SIGSEGV)
 *   status   exit(status) / _exit(status) inside the entry (see below)
 *   -1       the timeout expired (__afl_guarded_call_timeout only)
 * Matches the subprocess convention (run_target_stdin returns -sig, an
 * exit status, or -1 for a timeout). */
static int __afl_guard_run(int (*entry)(const uint8_t *, size_t),
                           const uint8_t *data, size_t size, uint64_t timeout_us) {
    /* A stray signal handed the dispositions back, or a health read found
     * them displaced; take them again. */
    if (!__afl_handlers_live)
        __afl_install_crash_handlers();
    if (!__afl_altstack_ready)
        __afl_install_altstack();
    int timed = timeout_us && __afl_timer_ready();
    if (timed) __afl_timer_budget(timeout_us);

    sigjmp_buf jb;
    sigjmp_buf *prev = __afl_guard_jmp;
    sig_atomic_t prev_timed = __afl_guard_timed;
    int code = sigsetjmp(jb, 1);
    if (code == 0) {
        /* A frame nested in a timed one stays under the outer budget: a
         * plain inner call used to switch timing off and its hang survived.
         * The budget's clock starts at the outermost timed frame. */
        if (timed && !prev_timed) __afl_guard_epoch++;
        __afl_guard_timed = timed || prev_timed;
        __afl_guard_jmp = &jb;
        int rc = entry(data, size);
        __afl_guard_jmp = prev;
        __afl_guard_timed = prev_timed;
        __afl_sancov_fold();
        return rc;
    }

    __afl_guard_jmp = prev;
    __afl_guard_timed = prev_timed;
    __afl_sancov_fold();
    if (code == __AFL_JMP_TIMEOUT) return __AFL_GUARD_TIMEOUT_RC;
    if (code & __AFL_JMP_EXIT) return code & 0xFF;
    return -code;
}

__attribute__((visibility("default")))
int __afl_guarded_call(int (*entry)(const uint8_t *, size_t),
                       const uint8_t *data, size_t size) {
    return __afl_guard_run(entry, data, size, 0);
}

/* As __afl_guarded_call, but ends a hang after timeout_us microseconds
 * (0 = no limit) and returns -1. The target is abandoned mid-execution,
 * exactly as on a crash: whatever it held (locks, half-built heap state)
 * stays as it was. */
__attribute__((visibility("default")))
int __afl_guarded_call_timeout(int (*entry)(const uint8_t *, size_t),
                               const uint8_t *data, size_t size, uint64_t timeout_us) {
    return __afl_guard_run(entry, data, size, timeout_us);
}

/* fork() inside a guarded entry: the child's copy of the guard frame
 * belongs to a copy of the host. Left armed, a crash in the child jumped
 * there and the child went on running the fuzzer. */
static void __afl_guard_atfork_child(void) {
    __afl_guard_jmp = NULL;
    __afl_guard_timed = 0;
    __afl_timer_state = 0;      /* POSIX timers are not inherited */
    __afl_timer_budget_us = 0;
}

/* ── exit() / _exit() inside a guarded call ──────────────────────────
 *
 * A target that exits on bad input (libjpeg's default error_exit, CLI-style
 * libraries) ended the host process in direct mode -- the fuzzer itself.
 * Inside a guard they now leave through it and return the status, as a
 * subprocess's exit status would. Outside a guard they are the real calls,
 * so one-shot targets and forked children exit exactly as before.
 *
 * Hidden visibility binds every call in this module (the wrapper and any
 * library objects linked into it) without interposing on the host, the
 * same pattern as the trace-pc-guard callbacks. Other shared libraries
 * keep libc's exit. */
static void __afl_guard_escape(int status) {
    sigjmp_buf *jb = __afl_guard_jmp;
    if (jb)
        siglongjmp(*jb, __AFL_JMP_EXIT | (status & 0xFF));
}

__attribute__((visibility("hidden"), noreturn))
void _exit(int status) {
    __afl_guard_escape(status);
    syscall(SYS_exit_group, status);
    __builtin_unreachable();
}

__attribute__((visibility("hidden"), noreturn))
void _Exit(int status) {
    _exit(status);
}

/* libc's own error-and-exit helpers reach exit() through an internal
 * alias, past the hidden definition below, so errx(4, ...) inside a guard
 * still ended the host. Reimplemented on top of it: the message goes to
 * stderr as libc would print it, then exit() leaves through the guard.
 * gnulib error() (grep) and BSD err() are the common callers. */
__attribute__((visibility("hidden"), noreturn))
void exit(int status);

__attribute__((visibility("hidden"), noreturn))
void verr(int status, const char *fmt, va_list ap) {
    vwarn(fmt, ap);
    exit(status);
}

__attribute__((visibility("hidden"), noreturn))
void verrx(int status, const char *fmt, va_list ap) {
    vwarnx(fmt, ap);
    exit(status);
}

__attribute__((visibility("hidden"), noreturn))
void err(int status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    verr(status, fmt, ap);
}

__attribute__((visibility("hidden"), noreturn))
void errx(int status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    verrx(status, fmt, ap);
}

/* glibc error(): "prog: message[: strerror(errnum)]\n" after flushing
 * stdout; exits only for a nonzero status. */
static void __afl_verror(int status, int errnum, const char *where, unsigned line,
                         const char *fmt, va_list ap) {
    fflush(stdout);
    fprintf(stderr, "%s:", program_invocation_name);
    if (where) fprintf(stderr, "%s:%u:", where, line);
    fputc(' ', stderr);
    vfprintf(stderr, fmt, ap);
    if (errnum) fprintf(stderr, ": %s", strerror(errnum));
    fputc('\n', stderr);
    if (status) exit(status);
}

__attribute__((visibility("hidden")))
void error(int status, int errnum, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    __afl_verror(status, errnum, NULL, 0, fmt, ap);
    va_end(ap);
}

__attribute__((visibility("hidden")))
void error_at_line(int status, int errnum, const char *file, unsigned line,
                   const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    __afl_verror(status, errnum, file, line, fmt, ap);
    va_end(ap);
}

__attribute__((visibility("hidden"), noreturn))
void exit(int status) {
    __afl_guard_escape(status);
    void (*real_exit)(int) = (void (*)(int))dlsym(RTLD_NEXT, "exit");
    if (real_exit)
        real_exit(status);
    _exit(status);   /* no libc exit reachable: at least end the process */
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
    __afl_health[__AFL_HEALTH_ABORTS_INTERCEPTED]++;
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
     * since each carried a constant offset.
     *
     * The site table inherits identically -- the opt-in strcmp above has a
     * call site like any other, so without this it becomes a permanent
     * phantom entry reporting one satisfied comparison per execution, at a
     * PC inside the shim. Counters only: the keys are worth keeping. */
    __afl_cmp_counts_zero();
#endif

    /* An ignored SIGCHLD auto-reaps children, so waitpid() below fails
     * with ECHILD and the status is lost. Reap here with the default
     * disposition; each child gets the target's own setting back. */
    struct sigaction chld_dfl, chld_old;
    chld_dfl.sa_handler = SIG_DFL;
    sigemptyset(&chld_dfl.sa_mask);
    chld_dfl.sa_flags = 0;
    sigaction(SIGCHLD, &chld_dfl, &chld_old);

    while (1) {
        char cmd[4];
        if (read(AFL_FORKSRV_FD, cmd, 4) != 4) _exit(0);

        pid_t child = fork();
        if (child < 0) _exit(1);

        if (child == 0) {
            /* Child: lead its own process group, so the loader's timeout
             * kill reaches whatever the target forks (fuzz_loader.c
             * kill_tree); restore recording, drop the control pipe, and
             * fall through into main(). */
            setpgid(0, 0);
            __afl_area = saved_area;
            sigaction(SIGCHLD, &chld_old, NULL);
            close(AFL_FORKSRV_FD);
            close(AFL_FORKSRV_FD + 1);
            return;
        }

        /* Both sides, before the pid is published: the loader may kill
         * the group as soon as it reads it. */
        setpgid(child, child);
        if (write(AFL_FORKSRV_FD + 1, &child, 4) != 4) _exit(0);

        int status = 0;
        while (waitpid(child, &status, 0) < 0) {
            /* Retry EINTR only — a stray signal must not be read as a
             * crash. Anything else would spin forever; end the server so
             * the loader sees EOF instead of a hang. */
            if (errno != EINTR) _exit(1);
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
    __afl_id_scheme_init(); /* before __afl_map_shm: see the id-space note */
    __afl_mapping = 1;
    __afl_map_shm();
    __afl_map_data_range();
#if __AFL_CMPLOG
    /* One constructor, one attachment. cmplog_shim.c had its own, which
     * re-entered __afl_map_shm and shmat'd the segment a second time --
     * measured 2 attachments per exec against 1 for the shim alone. */
    __afl_cmplog_init();
#endif
    __afl_install_crash_handlers();
    pthread_atfork(NULL, NULL, __afl_guard_atfork_child);
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
    __afl_cmp_dump_sites();
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

#if defined(__clang__)
#pragma clang section data="" bss=""
#endif
