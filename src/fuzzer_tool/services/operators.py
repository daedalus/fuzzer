"""Mutation operator dispatch and execution.

Extracted from Fuzzer class (~lines 1116-1845). Contains:
- All _op_* handler methods
- build_dispatch() — maps operator names to handlers (via REGISTRY)
- _havoc_mutate() / _apply_single_mutation() — random compound mutations
- build_ops() — builds list of available operators (via REGISTRY)
- _select_op() — selects operator via scheduling strategy
- _select_position() — selects byte position for mutation
- mutate() — main mutation orchestrator

Operator names, categories, and availability conditions are the single
source of truth in ``fuzzer_tool.core.operator_registry.REGISTRY``; this
module only supplies the ``_op_*`` handlers that the registry dispatches to.
"""

import bisect
import logging
import math
import os
import struct
import time
from array import array

import xxhash

from fuzzer_tool.core.cond_stmt import CondState, CondStmt
from fuzzer_tool.core.crc32 import crc32
from fuzzer_tool.core.live_bit_mask import LiveBitMaskEstimator
from fuzzer_tool.core.mutations import (
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
    INTERESTING_UNSIGNED_8,
    INTERESTING_UNSIGNED_16,
    INTERESTING_UNSIGNED_32,
    MAGIC_TABLE,
    SPECIAL_STRINGS,
    ascii_num_replace,
    choose_len,
    could_be_arith,
    could_be_bitflip,
    could_be_interest,
    radamsa_mutate_num,
    splice,
    splice_common_prefix,
    splice_diff_located,
)
from fuzzer_tool.core.operator_registry import REGISTRY, format_gate_matches
from fuzzer_tool.core.skipdet import MAX_DET_MUTATIONS, trace_mini_from_edges

log = logging.getLogger(__name__)

# ── invariant_break corpus scan ──────────────────────────────────────────
# Minimum corpus size before an "every input agrees here" offset is treated
# as a real invariant rather than a coincidence. Mirrors the availability
# predicate in operator_registry so a direct dispatch call is guarded too.
_INVARIANT_MIN_SAMPLES = 16
# Newest N seeds only. The scan is O(samples x length) and the invariant set
# converges quickly; scanning a 50k-entry corpus on every recompute would
# cost far more than the operator can repay.
_INVARIANT_SAMPLE_CAP = 128

# ── elite_fuse corpus scan ────────────────────────────────────────────────
# Need at least two seeds to pick two distinct parents from.
_ELITE_FUSE_MIN_CORPUS = 2
# Size of the "most coverage" pool that parents are drawn from. Keeping this
# small (rather than always taking the single top-2) means the op doesn't
# degenerate into repeatedly fusing the same pair once one seed pulls far
# ahead on edge count -- there is still randomness in which two elites meet.
_ELITE_FUSE_POOL_SIZE = 8

# ── region-profile position weighting ────────────────────────────────────
# Distinct seeds to keep profiles for. The corpus is far larger than this,
# but seed selection is heavily skewed toward a working set, so a small
# cache with a clear-on-full policy keeps the hit rate high without tracking
# recency. Cleared wholesale rather than evicted one at a time: the profiles
# are cheap to rebuild and an LRU would cost more bookkeeping than it saves.
_REGION_CACHE_MAX = 64
# profile_buffer() skips windows below 512 bytes, so anything shorter has no
# profile to weight by.
_REGION_MIN_LEN = 512

# ── region liveness (item 4, handover_skittercreek_tailslayer_port.md) ───
# Coverage-bit width the per-region LiveBitMaskEstimator folds edge ids
# into (edge_id % _LIVENESS_MAP_BITS). This is deliberately independent of
# the live SHM map_size: it only needs to be wide enough that two distinct
# edges rarely alias into the same bit within one region's observation
# window, not wide enough to be collision-free -- the estimator is already
# a one-sided, monotone-growing signal, so an occasional alias only costs
# a slightly-too-eager "not dead yet", never a false dead verdict.
_LIVENESS_MAP_BITS = 65536
# Consecutive coverage-diff observations with no new bit revealed before a
# region is considered converged-dead. Matches LiveBitMaskEstimator's own
# default and MASK_SWITCH_AFTER in the ported source.
#
# Threshold stability is established: four real campaigns
# (`zlib_read`, `png_read` pre- and post-fix, `jpeg_read`; see
# docs/sweeps/) converged to the same live mask at every
# `switch_after` in {50, 100, 200, 400, 800}, with zero false-dead
# verdicts.
#
# The FALSE-NEGATIVE rate is NOT established, and on the current target
# matrix it is not measurable. All four campaigns produced zero
# genuinely-dead regions to test against, for reasons that are
# structural rather than sampling accidents: compressed data has no
# padding, and any CRC-covered format rules out coverage-dead bytes
# outright, since mutating any byte flips the CRC-check edge regardless
# of semantic relevance. So this value and _LIVENESS_DEAD_WEIGHT below
# are conservative guesses that have never been calibrated against a
# true-dead region. Measuring them needs a target with neither a
# whole-file nor a per-chunk checksum; if one is added to the matrix,
# rerun the sweep before treating either number as tuned.
# See handover_skittercreek_tailslayer_port.md, Sequencing step 6.
_LIVENESS_SWITCH_AFTER = 200
# Multiplicative down-weight applied to a region's mutation-site weight
# once its liveness estimator has converged with an empty mask (i.e.
# "never once moved coverage across >= _LIVENESS_SWITCH_AFTER consecutive
# mutations touching it"). Deliberately not 0.0: convergence is strong
# evidence, not proof, and a hard-zero would make a misclassified region
# permanently unreachable by this weighting path. See the false-negative
# caveat on _LIVENESS_SWITCH_AFTER above: with the true-dead rate
# unmeasured, "not 0.0" is what keeps a wrong verdict recoverable.
_LIVENESS_DEAD_WEIGHT = 0.1

# ── havoc sub-mutation weighting ─────────────────────────────────────────
# _apply_single_mutation() dispatches to one of 11 inline branches. That
# choice was `r[0] % 11` -- flat odds, no feedback -- while every top-level
# operator above it is scheduled by a bandit fed real per-operator success
# rates. Havoc is reachable from every mutation round and applies 2-8
# sub-mutations per call (8-16 under stall recovery), so a mis-split inside
# it is a continuous tax rather than an occasional one.
#
# These are branches, not dispatchable operators: they stay out of REGISTRY
# (Hard Rule 12 governs op mutators, which need availability predicates and
# a handler) and out of the select_op()/record() scheduler interface, whose
# per-call cost the havoc hot path cannot absorb. The weighting here is
# deliberately the cheapest thing that consumes the same signal: O(1)
# integer counts, an O(256) table rebuild every _HAVOC_TABLE_REFRESH draws,
# and a single list index per draw.
HAVOC_SUB_OPS = (
    "bit_flip",
    "byte_set",
    "byte_swap",
    "insert_byte",
    "delete_block",
    "crc32_repair",
    "swap_regions",
    "endian_swap",
    "byte_insert",
    "random_byte",
    "shuffle_range",
)
_HAVOC_N = len(HAVOC_SUB_OPS)
# Sampling is a precomputed inverse-CDF table: 256 slots, each holding a
# branch index, indexed by the low byte of the draw. Measured against the
# alternatives at 2M draws (see tools/bench_havoc_subop.py): uniform
# `r[0] % 11` 89ns, bisect over an 11-float CDF 313ns, this table 202ns --
# so the table halves the cost of the feature versus the obvious bisect.
# 256 slots quantize probabilities to 0.39%, well under the explore floor.
_HAVOC_TABLE_SLOTS = 256
# Mutation rounds between table rebuilds. The rebuild is triggered from
# mutate() and from credit_havoc_subops(), both of which run once per
# execution -- never from _apply_single_mutation, which runs 2-16 times per
# havoc selection. Keeping the refresh counter out of the inner loop saves
# two attribute operations per sub-mutation; the distribution moves on the
# scale of thousands of executions, so per-round granularity is ample.
_HAVOC_TABLE_REFRESH = 256
# Uniform mixing weight (EXP3-style). Guarantees every sub-mutation keeps
# at least _HAVOC_EXPLORE/_HAVOC_N ~= 1.4% of draws -- 3 of 256 table slots
# -- so a branch whose guard fails on small inputs (byte_swap,
# delete_block, shuffle_range) can recover once inputs grow instead of
# being starved permanently.
_HAVOC_EXPLORE = 0.15
# Halve all counts once any branch exceeds this many trials: keeps the
# ratios responsive to a target whose reachable behaviour shifts mid-run,
# and bounds float growth over billion-exec campaigns.
_HAVOC_DECAY_AT = 100_000.0
# Precomputed 1 << i, avoiding a shift per draw in the credit bitmask.
_HAVOC_BITS = tuple(1 << i for i in range(_HAVOC_N))

# Precomputed colorization lookup table: byte -> different value from same class.
# Avoids per-call table construction in _op_colorization().
_COLORIZE_TBL = bytearray(256)
for _b in range(256):
    if 0x30 <= _b <= 0x39:
        _COLORIZE_TBL[_b] = 0x30 + (_b * 7 + 3) % 10
    elif 0x41 <= _b <= 0x5A:
        _COLORIZE_TBL[_b] = 0x41 + (_b * 13 + 5) % 26
    elif 0x61 <= _b <= 0x7A:
        _COLORIZE_TBL[_b] = 0x61 + (_b * 17 + 7) % 26
    elif _b in (0x20, 0x09, 0x0A, 0x0D):
        _COLORIZE_TBL[_b] = (0x20, 0x09, 0x0A, 0x0D)[_b % 4]
    else:
        _COLORIZE_TBL[_b] = 0x21 + (_b * 31 + 11) % 94
_COLORIZE_TBL = bytes(_COLORIZE_TBL)

# ── Contextual bandit (LinUCB) feature schema ────────────────────────────
# Coarse format buckets for the context one-hot. Deliberately coarser than
# _FORMAT_SNIFFERS in operator_registry.py (which gates ~15 format-specific
# operators individually) -- the bandit only needs "what kind of structure
# is this" to condition its ranking, not which exact mutator applies.
_CONTEXT_FORMAT_CATEGORIES = (
    "png",
    "jpeg",
    "compressed",  # gzip/zlib
    "image_other",  # gif/bmp/riff family (webp/webm)
    "archive",  # zip/isobmff
    "elf",
    "other",
)
# 6 scalar features (log-size, entropy, edge-coverage frac, lineage depth,
# cmplog-pairs-exist, corpus-size percentile) + format one-hot + 1 per-arm
# operator-cost feature appended in _context_vector().
CONTEXT_DIM = 6 + len(_CONTEXT_FORMAT_CATEGORIES) + 1

# ── deterministic stage ───────────────────────────────────────────────────
# AFL's classic deterministic schedule, walked systematically across a seed
# rather than sampled at a random position by the bandit. This is the
# machinery core/skipdet.py names and gates (SkipDetector.should_det_fuzz)
# but that a first version of this feature only partly drove: it dispatched
# _op_bit_flip once per byte, and that handler picks a single random bit --
# so "bitflip 1/1" only ever touched 1/8th of the bits a real AFL
# deterministic pass covers. The generator below is the actual systematic
# walk: every bit, then every byte, then every arithmetic delta, then every
# interesting value, in order.


def _deterministic_mutation_stream(data: bytes, max_mutations: int = MAX_DET_MUTATIONS):
    """Yield mutants from AFL's classic deterministic schedule, in order.

    Walks bitflip 1/1, byte flip 8/8, 8-bit arithmetic, and 8-bit
    interesting-value substitution across every position in *data* in turn.
    Each mutant differs from *data* at exactly one position (or one bit),
    which is what lets a deterministic stage attribute a coverage gain to a
    specific offset -- unlike havoc, which changes several positions in one
    round and can only credit the mutation as a whole.

    Deliberately narrower than AFL++'s full schedule (no 2/1, 4/1, 16-bit or
    32-bit walks): those add coverage at steeply diminishing returns per
    exec once 1/1 and 8/8 have run. Extending the schedule is a matter of
    adding more passes below; the gating and queueing around it doesn't
    change.

    Args:
        data: The seed to generate a deterministic schedule for. Not
            mutated -- each yielded mutant is a fresh bytearray.
        max_mutations: Hard cap on total mutants yielded, so one huge seed
            can't turn a single deterministic pass into an unbounded stall.
            Bitflip 1/1 alone costs 8*len(data) mutants; the cap simply
            truncates the schedule when it's exceeded, same as AFL++'s own
            time-boxing of deterministic stages on large inputs.

    Yields:
        bytes mutants, each one mutation away from *data*.
    """
    from fuzzer_tool.core.mutations import ARITHMETIC_DELTAS

    length = len(data)
    if length == 0:
        return

    n = 0

    # bitflip 1/1: flip every bit in turn.
    for byte_idx in range(length):
        orig = data[byte_idx]
        for bit in range(8):
            if n >= max_mutations:
                return
            mutant = bytearray(data)
            mutant[byte_idx] = orig ^ (1 << bit)
            yield bytes(mutant)
            n += 1

    # byte flip 8/8: XOR every byte with 0xFF in turn.
    for byte_idx in range(length):
        if n >= max_mutations:
            return
        mutant = bytearray(data)
        mutant[byte_idx] ^= 0xFF
        yield bytes(mutant)
        n += 1

    # arithmetic 8-bit: add/subtract each delta at every byte position.
    for byte_idx in range(length):
        orig = data[byte_idx]
        for delta in ARITHMETIC_DELTAS:
            if n >= max_mutations:
                return
            mutant = bytearray(data)
            mutant[byte_idx] = (orig + delta) & 0xFF
            yield bytes(mutant)
            n += 1
            if n >= max_mutations:
                return
            mutant = bytearray(data)
            mutant[byte_idx] = (orig - delta) & 0xFF
            yield bytes(mutant)
            n += 1

    # interesting values 8-bit: substitute each known-interesting byte.
    for byte_idx in range(length):
        for val in INTERESTING_UNSIGNED_8:
            if n >= max_mutations:
                return
            mutant = bytearray(data)
            mutant[byte_idx] = val & 0xFF
            yield bytes(mutant)
            n += 1


class OperatorEngine:
    """Manages mutation operator selection and execution.

    Holds a reference to the Fuzzer instance for accessing shared state
    (dictionary, markov, mc, grammar, corpus, seed_meta, etc.).
    """

    def __init__(self, fuzzer):
        self.f = fuzzer
        # Cache for _op_redqueen_xform: sorted cmplog pairs + version counter.
        # Rebuilt only when the pair list grows or is recreated (collect_tokens),
        # avoiding O(N log N) sort on every invocation (~2,500 sorts saved per run).
        self._redqueen_sorted_pairs: list | None = None
        self._redqueen_pair_lengths: array | None = None
        self._redqueen_sorted_version: int = 0
        # Cache for _op_invariant_break: CorpusInvariants plus the corpus
        # size it was measured at, so it is rebuilt on growth and not per
        # mutation (see corpus_invariants()).
        self._invariants = None
        self._invariants_corpus_len: int = -1
        # Cache for _op_elite_fuse: the top-N corpus seeds by coverage_edges,
        # plus the corpus size it was ranked at. Rebuilt on growth only, same
        # rationale as _invariants -- ranking the whole corpus by seed_meta
        # lookup on every mutation would cost more than the operator repays.
        self._elite_pool: list[bytes] | None = None
        self._elite_pool_corpus_len: int = -1
        # Cache for region-profile position weighting, keyed by seed content
        # hash. profile_buffer() runs a whole statistical battery per 4 KiB
        # window (~1 ms), so it must be paid once per seed, not once per
        # mutation -- see region_weights().
        self._region_cache: dict[int, tuple | None] = {}
        # Per-region liveness estimators, keyed by the same seed content
        # hash as _region_cache, one LiveBitMaskEstimator per region index
        # (parallel array to the bounds/cumulative lists in that cache
        # entry). Lazily populated by record_coverage_diff() as mutation
        # exec results come in -- most seeds never get an entry here if
        # the caller never reports a diff for them. Same clear-on-full
        # policy and cap as _region_cache; the two caches are evicted
        # together in region_weights() so they never disagree about which
        # seeds are tracked.
        self._region_liveness: dict[int, list[LiveBitMaskEstimator | None]] = {}
        # Havoc sub-mutation credit. Laplace-smoothed at 1 hit / 2 trials so
        # every branch starts at a 0.5 ratio -- identical weights, so the
        # first table is uniform and the adaptive path only diverges from
        # `r[0] % 11` once there is evidence to diverge on. Plain int lists,
        # not array("d"): 11 elements make the memory argument moot, and the
        # measured per-draw cost is 202ns against 494ns for array("d").
        self._havoc_hits = [1] * _HAVOC_N
        self._havoc_trials = [2] * _HAVOC_N
        self._havoc_table = [0] * _HAVOC_TABLE_SLOTS
        self._havoc_rounds_since_rebuild = 0
        self._rebuild_havoc_table()
        # Deterministic-stage queue, one entry per seed currently mid-stage.
        # A generator rather than a materialized list: bitflip 1/1 alone is
        # 8*len(seed) mutations, and most seeds never reach the gate (see
        # maybe_deterministic_mutation), so building the full schedule
        # upfront for every seed would waste memory that's never read.
        self._det_queues: dict = {}

    # ── Operator handlers ──────────────────────────────────────────────
    # Each handler: (buf, byte_idx, data) -> None (in-place) or bytes (replace buf)

    def _op_bit_flip(self, buf, byte_idx, _data):
        rng = self.f._rand_pool
        if buf:
            buf[byte_idx] ^= 1 << rng.randint(0, 7)

    def _op_bit_offset_flip(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if not buf:
            return
        total_bits = len(buf) * 8
        bit_offset = rng.randint(0, total_bits - 1)
        byte_idx = bit_offset >> 3
        bit_idx = bit_offset & 7
        buf[byte_idx] ^= 1 << bit_idx

    def _op_bit_offset_span(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if not buf:
            return
        total_bits = len(buf) * 8
        span_width = rng.weighted_choice(
            [1, 2, 3, 4, 5, 6, 7, 8], weights=[10, 15, 20, 20, 15, 10, 5, 5]
        )
        start_offset = rng.randint(0, max(0, total_bits - span_width))
        for i in range(span_width):
            bit_offset = start_offset + i
            if bit_offset >= total_bits:
                break
            byte_idx = bit_offset >> 3
            bit_idx = bit_offset & 7
            buf[byte_idx] ^= 1 << bit_idx

    def _op_simd_boundary(self, buf, _byte_idx, _data):
        """Resize buffer to SIMD boundary lengths (AVX2: 32, SSE2: 16)."""
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import SIMD_BOUNDARIES

        if not buf:
            buf.extend(rng.randint(0, 255) for _ in range(rng.choice(SIMD_BOUNDARIES)))
            return
        target_len = rng.choice(SIMD_BOUNDARIES)
        current_len = len(buf)
        if target_len == current_len:
            return
        elif target_len < current_len:
            del buf[target_len:]
        else:
            grow = min(target_len - current_len, self.f.max_len - current_len)
            if grow > 0:
                buf.extend(bytearray(grow))  # zero-filled, fast

    def _op_regex_bomb(self, buf, _byte_idx, _data):
        """Replace input with a known regex backtracking bomb pattern.

        Mutates in place and returns None, so it never reaches mutate()'s
        post-operator f.max_len clamp. The old code extended the buffer to
        the pattern length unconditionally, which meant a max_len below the
        longest bomb (13 bytes) was simply ignored -- and the extension is
        a floor, not a cap, so a run configured for tiny inputs got
        13-byte ones instead.

        Patterns that do not fit are filtered out rather than truncated: a
        truncated backtracking bomb is not a bomb, and silently emitting
        one would make this operator look like it fired when it had not.
        """
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import REGEX_BOMBS

        max_len = self.f.max_len
        usable = [p for p in REGEX_BOMBS if not max_len or len(p.encode()) <= max_len]
        if not usable:
            return
        pattern = rng.choice(usable).encode()
        if len(buf) < len(pattern):
            buf.extend(b"\x00" * (len(pattern) - len(buf)))
        # Insert bomb at random position
        pos = rng.randint(0, max(0, len(buf) - len(pattern)))
        buf[pos : pos + len(pattern)] = pattern

    def _op_clone_fixed(self, buf, _byte_idx, _data):
        """Insert a block of repeated constant bytes (AFL++ clone_fixed)."""
        rng = self.f._rand_pool
        if not buf or len(buf) >= self.f.max_len:
            return
        fill_byte = rng.choice([buf[rng.randint(0, len(buf) - 1)], 0, 0xFF])
        block_size = rng.randint(1, min(32, self.f.max_len - len(buf)))
        ins_pos = rng.randint(0, len(buf))
        buf[ins_pos:ins_pos] = bytes([fill_byte] * block_size)

    def _op_overwrite_copy(self, buf, _byte_idx, _data):
        """Overwrite a region with bytes from another position (AFL++ overwrite_copy)."""
        rng = self.f._rand_pool
        if len(buf) < 2:
            return
        src_len = rng.randint(1, min(16, len(buf)))
        src_pos = rng.randint(0, len(buf) - src_len)
        dst_pos = rng.randint(0, max(0, len(buf) - src_len))
        if src_pos != dst_pos:
            buf[dst_pos : dst_pos + src_len] = buf[src_pos : src_pos + src_len]

    def _op_overwrite_fixed(self, buf, _byte_idx, _data):
        """Overwrite a region with repeated constant bytes (AFL++ overwrite_fixed)."""
        rng = self.f._rand_pool
        if len(buf) < 2:
            return
        fill_byte = rng.choice([buf[rng.randint(0, len(buf) - 1)], 0, 0xFF])
        block_len = rng.randint(1, min(16, len(buf)))
        dst_pos = rng.randint(0, len(buf) - block_len)
        buf[dst_pos : dst_pos + block_len] = bytes([fill_byte] * block_len)

    def _op_redqueen_xform(self, buf, byte_idx, _data):
        """RedQueen transforms: solve comparisons via encoding transforms.

        Uses the encoding engine from ``rq_encodings.py`` (ported from Redqueen
        NDSS 2019) to find and replace comparison operands using a variety of
        encoding strategies: plain substitution, zero/sign extension, ASCII
        number representation, C-string termination, split 64-bit words, etc.

        Falls back to single-byte transforms (XOR, arithmetic, boundary) when
        the encoding engine yields no applicable mutations for the current pair.
        """
        f = self.f
        rng = f._rand_pool
        if not hasattr(f, "_cmplog") or not f._cmplog or not f._cmplog.pairs:
            return
        if not buf or len(buf) < 2:
            return
        from fuzzer_tool.core.rq_encodings import generate_mutations  # noqa: PLC0415

        # Sample up to 3 pairs from the cmplog pool, prefering shorter pairs
        # (they're more likely to be found in the buffer).
        # Use cached sorted pair list, resorting only when pairs change,
        # to avoid O(N log N) sort on every invocation.
        cmplog_pairs = f._cmplog.pairs
        _version = id(cmplog_pairs) + len(cmplog_pairs)
        if not self._redqueen_sorted_pairs or _version != self._redqueen_sorted_version:
            _temp = [(len(p[0]), p) for p in cmplog_pairs]
            _temp.sort(key=lambda x: x[0])
            self._redqueen_sorted_pairs = [p for n, p in _temp if n >= 2]
            self._redqueen_pair_lengths = array("I", (n for n, _p in _temp if n >= 2))
            self._redqueen_sorted_version = _version
        # Pairs with 2 <= len(op_a) <= len(buf) form a prefix of the
        # length-sorted list — find the cutoff with a bisect instead of
        # rescanning every pair on every invocation.
        cutoff = bisect.bisect_right(self._redqueen_pair_lengths, len(buf))
        pairs = self._redqueen_sorted_pairs[:cutoff]
        if not pairs:
            return
        _sample_idx = rng.sample(len(pairs), min(3, len(pairs)))
        sample = [pairs[i] for i in _sample_idx]
        input_bytes = bytes(buf)

        for op_a, op_b in sample:
            cmp_size = 512 if len(op_a) > 8 or len(op_b) > 8 else max(len(op_a), len(op_b)) * 8
            cmp_type = "STR" if len(op_a) > 8 else "CMP"

            mutations = generate_mutations(
                op_a,
                op_b,
                cmp_size,
                cmp_type,
                input_bytes,
                hammer=True,
                is_hash=f._cmplog.is_hash_candidate
                if hasattr(f._cmplog, "is_hash_candidate")
                else None,
            )
            if mutations:
                offsets, replacements, enc = rng.choice(mutations)
                for i, off in enumerate(offsets):
                    if i < len(replacements):
                        chunk = replacements[i]
                        end = off + len(chunk)
                        if end <= len(buf):
                            buf[off:end] = chunk
                return

        # Fallback: single-byte transforms on a random pair
        if not pairs:
            return
        op_a, _ = rng.choice(pairs)
        if len(op_a) <= len(buf):
            pos = 0
            candidates = []
            buf_bytes = bytes(buf)
            while pos <= len(buf_bytes) - len(op_a):
                idx = buf_bytes.find(op_a, pos)
                if idx == -1:
                    break
                candidates.append(idx)
                pos = idx + 1
                if len(candidates) >= 5:
                    break
            if candidates:
                offset = rng.choice(candidates)
                vb = int.from_bytes(op_a, "little") & 0xFF
                transform = rng.choice(
                    ["xor", "arithmetic", "boundary", "hex", "toupper", "tolower"]
                )
                if transform == "xor":
                    const = rng.randint(1, 255)
                    buf[offset] = (vb ^ const) & 0xFF
                elif transform == "arithmetic":
                    delta = rng.randint(-128, 127)
                    buf[offset] = (vb - delta) & 0xFF
                elif transform == "boundary":
                    buf[offset] = (vb - 1) & 0xFF
                elif transform == "hex":
                    hex_chars = b"0123456789abcdef"
                    buf[offset] = hex_chars[vb % 16]
                elif transform == "toupper":
                    if ord("a") <= buf[offset] <= ord("z"):
                        buf[offset] = buf[offset] - 0x20
                    else:
                        buf[offset] = ord("A") + (vb % 26)
                elif transform == "tolower":
                    if ord("A") <= buf[offset] <= ord("Z"):
                        buf[offset] = buf[offset] + 0x20
                    else:
                        buf[offset] = ord("a") + (vb % 26)

    # ── Fuse mutations (from Radamsa) ──────────────────────────────

    def _op_fuse_this(self, buf, _byte_idx, _data):
        """Jump between two similar positions within the same input.

        Radamsa sed-fuse-this: picks two positions in the buffer,
        finds a short common prefix at each, and swaps the tails.
        Creates structural hybrids without format awareness.

        Mutates ``buf`` in place and returns None, which means it does
        NOT go through mutate()'s post-operator f.max_len clamp (that
        clamp only runs in the `result is not None` branch). Each
        application can grow the buffer to nearly 2x its input size
        (len(result) = 2*len(buf) - a_end, and a_end >= 0), so repeated
        selection compounds: a 4KB seed can silently reach multi-MB size
        over successive mutations/generations with no cap. Downstream
        consumers that scale allocations with input length (e.g. QEA's
        amplitude arrays, 8 float64s per byte) turn that into an OOM.
        Clamp explicitly here rather than relying on the caller.
        """
        f = self.f
        rng = f._rand_pool
        max_len = getattr(f, "max_len", 65536)
        if len(buf) < 4:
            return
        p1 = rng.randint(0, len(buf) - 2)
        p2 = rng.randint(0, len(buf) - 2)
        if p1 == p2:
            p2 = (p2 + 1) % max(1, len(buf) - 1)
        max_pre = min(16, len(buf) - max(p1, p2))
        pre_len = 0
        for pre_len in range(max_pre):
            if buf[p1 + pre_len] != buf[p2 + pre_len]:
                break
        else:
            pre_len = max_pre
        if pre_len < 1:
            tail1 = bytes(buf[p1:])
            tail2 = bytes(buf[p2:])
            combined = bytearray(buf[:p1]) + tail2 + buf[p1:p2] + tail1
            if (
                len(combined) <= 2 * len(buf)
                and len(combined) >= len(buf) // 2
                and len(combined) <= max_len
            ):
                buf[:] = combined
            return
        a_end = p1 + pre_len
        b_end = p2 + pre_len
        tail_a = bytes(buf[a_end:])
        tail_b = bytes(buf[b_end:])
        result = bytearray(buf[:a_end]) + tail_b + buf[a_end:b_end] + tail_a
        if len(result) <= 2 * len(buf) and len(result) <= max_len:
            buf[:] = result

    def _op_fuse_next(self, buf, _byte_idx, data):
        """Fuse current input with another corpus entry.

        Radamsa sed-fuse-next: fuses prefix of the current block
        with suffix of a random corpus entry at a shared position.
        """
        rng = self.f._rand_pool
        f = self.f
        corpus = getattr(f, "corpus", [])
        if not corpus or len(buf) < 3:
            return
        other = bytes(rng.choice(corpus))
        if other is data or other == buf:
            others = [c for c in corpus if c is not data]
            if not others:
                return
            other = bytes(rng.choice(others))
        if len(other) < 3:
            return
        split_point = min(len(buf), len(other)) // 2
        split_point = rng.randint(1, max(1, split_point))
        result = bytearray(buf[:split_point]) + other[split_point:]
        if len(result) <= f.max_len:
            buf[:] = result[: f.max_len]

    def _op_fuse_old(self, buf, _byte_idx, _data):
        """Fuse current input with a remembered block from earlier fuzz runs.

        Radamsa sed-fuse-old: maintains a ring buffer of previously
        mutated data and fuses fragments from it into the current input.
        """
        rng = self.f._rand_pool
        if len(buf) < 3:
            return
        if not hasattr(self, "_fuse_memory"):
            self._fuse_memory = []
            self._fuse_mem_max = 32
        mem = self._fuse_memory
        mem.append(bytes(buf))
        if len(mem) > self._fuse_mem_max:
            mem.pop(0)
        if len(mem) < 2:
            return
        old = rng.choice(mem[:-1])
        if len(old) < 3:
            return
        p_cur = rng.randint(1, len(buf) - 1)
        p_old = rng.randint(1, len(old) - 1)
        tail_old = bytes(old[p_old:])
        result = bytearray(buf[:p_cur]) + tail_old
        if len(result) <= getattr(self.f, "max_len", 65536):
            buf[:] = result

    def _op_tree_mutate(self, buf, _byte_idx, _data):
        """Lightweight delimiter-based tree mutation.

        Uses heuristic delimiter parsing (``tree_mutator.py``, ported from
        Radamsa ``sed-tree-*``) to parse the input into a tree using
        ``() {} [] \"\" ''``, then applies one of: delete, duplicate,
        swap, or stutter a subtree node.  Flattens back to bytes.

        No grammar file is needed — structure is detected from delimiters.
        """
        from fuzzer_tool.core.tree_mutator import lightweight_tree_mutate  # noqa: PLC0415

        result = lightweight_tree_mutate(bytes(buf), max_len=self.f.max_len, rng=self.f._rand_pool)
        if result != bytes(buf):
            buf[:] = result[: len(buf)]

    # ── UTF-8 confusion mutations (from Radamsa) ────────────────────

    def _op_utf8_widen(self, buf, _byte_idx, _data):
        """Widen 7-bit ASCII bytes into overlong UTF-8 sequences.

        Radamsa ``sed-utf8-widen``: replaces a random ASCII byte with an
        overlong UTF-8 encoding (2-byte sequence).  Exercises UTF-8
        length-checking bugs in parsers.

        Grows the buffer by exactly one byte, in place, returning None --
        so like ``_op_fuse_this`` it never reaches mutate()'s post-operator
        f.max_len clamp, which only runs in the ``result is not None``
        branch. One byte per application looks harmless and is not: the
        operator can be selected repeatedly across generations, and there
        is no other cap on this path. The guard below follows the
        convention ``_op_clone_fixed`` already uses -- decline rather than
        truncate, since truncating would corrupt the two-byte sequence
        this operator exists to produce.
        """
        rng = self.f._rand_pool
        if not buf:
            return
        max_len = self.f.max_len
        if max_len and len(buf) + 1 > max_len:
            return
        # Find a 7-bit ASCII byte (0x00-0x7F)
        candidates = [i for i, b in enumerate(buf) if b < 0x80]
        if not candidates:
            return
        idx = rng.choice(candidates)
        b = buf[idx]
        # Overlong 2-byte encoding: 110xxxxx 10xxxxxx
        buf[idx : idx + 1] = bytes([0xC0 | (b >> 6), 0x80 | (b & 0x3F)])

    def _op_utf8_insert(self, buf, _byte_idx, _data):
        """Insert problematic Unicode byte sequences.

        Radamsa ``sed-utf8-insert``: inserts curated byte sequences that
        have caused security issues: BOMs, right-to-left override,
        zero-width joiners, illegal surrogates, NFKC expansion bombs, etc.
        """
        rng = self.f._rand_pool
        if not buf or len(buf) >= getattr(self.f, "max_len", 65536):
            return
        from fuzzer_tool.core.mutations import _FUNNY_UNICODE  # noqa: PLC0415

        seq = rng.choice(_FUNNY_UNICODE)
        pos = rng.randint(0, len(buf))
        buf[pos:pos] = seq

    # ── Line-level mutations (from Radamsa) ─────────────────────────

    def _op_line_mutate(self, buf, _byte_idx, _data):
        """Line-level mutation: split on newlines, mutate the line list, rejoin.

        Radamsa ``sed-line-*`` operators: one of:
        - Delete a random line
        - Duplicate a random line
        - Swap two adjacent lines
        - Permute lines
        - Repeat a line
        - Insert a line from elsewhere in the buffer
        """
        rng = self.f._rand_pool
        if not buf or len(buf) < 4:
            return
        # Split on newline
        parts = buf.split(b"\n")
        if len(parts) < 2:
            return
        mutate = rng.choice(["del", "dup", "swap", "perm", "repeat", "clone"])
        if mutate == "del":
            idx = rng.randint(0, len(parts) - 1)
            del parts[idx]
        elif mutate == "dup":
            idx = rng.randint(0, len(parts) - 1)
            parts.insert(idx + 1, parts[idx])
        elif mutate == "swap" and len(parts) >= 2:
            idx = rng.randint(0, len(parts) - 2)
            parts[idx], parts[idx + 1] = parts[idx + 1], parts[idx]
        elif mutate == "perm" and len(parts) >= 3:
            # Shuffle a subset of lines
            start = rng.randint(0, len(parts) - 3)
            end = min(start + rng.randint(2, 6), len(parts))
            segment = parts[start:end]
            rng.shuffle(segment)
            parts[start:end] = segment
        elif mutate == "repeat" and len(parts) >= 1:
            idx = rng.randint(0, len(parts) - 1)
            n = rng.randint(1, 32)
            for _ in range(n):
                parts.insert(idx, parts[idx])
        elif mutate == "clone" and len(parts) >= 2:
            src = rng.randint(0, len(parts) - 1)
            dst = rng.randint(0, len(parts) - 1)
            parts.insert(dst, parts[src])
        result = b"\n".join(parts)
        if len(result) <= getattr(self.f, "max_len", 65536) and result != bytes(buf):
            buf[:] = result

    def _op_colorization(self, buf, _byte_idx, _data):
        """Taint-aware byte randomization preserving character classes.

        AFL++ colorization: replace each byte with a random value from
        the same character class. Identifies comparison-relevant bytes.

        When cmplog data is available, uses the ``CmplogColorizer`` to
        prefer mutating bytes that appear in comparison operands, which
        increases the chance of cracking comparisons.
        """
        rng = self.f._rand_pool
        if not buf:
            return
        tbl = _COLORIZE_TBL
        # Try to use cmplog-derived colorization mask for targeted selection
        f = self.f
        cmplog_pairs = getattr(f._cmplog, "pairs", None) if hasattr(f, "_cmplog") else None
        if cmplog_pairs:
            from fuzzer_tool.core.colorizer import CmplogColorizer  # noqa: PLC0415

            # Cache color mask per (input hash, cmplog pairs id)
            buf_bytes = bytes(buf)
            buf_hash = hash(buf_bytes)
            pairs_id = id(cmplog_pairs)
            cache = getattr(self, "_colorize_cache", None)
            if cache is None:
                self._colorize_cache = {}
                cache = self._colorize_cache
            cached = cache.get((buf_hash, pairs_id))
            if cached is None:
                colorizer = CmplogColorizer()
                mask = colorizer.colorize_from_cmplog(buf_bytes, cmplog_pairs)
                colorable = [i for i, m in enumerate(mask[: len(buf)]) if m == 0xFF]
                cached = colorable
                # Bounded cache: evict when > 256 entries
                if len(cache) > 256:
                    cache.clear()
                cache[(buf_hash, pairs_id)] = cached
            colorable = cached
            if colorable:
                n_mutate = max(1, min(len(colorable), len(buf) // rng.randint(2, 10)))
                indices = rng.choice_list(colorable, n_mutate)
                for idx in indices:
                    buf[idx] = tbl[buf[idx]]
                return
        # Fallback: random selection from entire buffer
        n_mutate = max(1, len(buf) // rng.randint(2, 10))
        indices = [rng.randint(0, len(buf) - 1) for _ in range(n_mutate)]
        for idx in indices:
            buf[idx] = tbl[buf[idx]]

    def _op_skipdet_probe(self, buf, _byte_idx, _data):
        """Block-flip probe to identify inert byte regions.

        AFL++ SkipDet: flip a block of bytes and check if the path changes.
        If not, those bytes are inert. This operator flips a random block
        to help discover which regions affect execution paths.
        """
        rng = self.f._rand_pool
        if len(buf) < 8:
            return
        # Pick a random block (10-25% of buffer) and flip all bytes
        block_size = rng.randint(len(buf) // 10, len(buf) // 4)
        block_size = max(2, min(block_size, len(buf)))
        start = rng.randint(0, len(buf) - block_size)
        # Bulk XOR via memoryview (avoids per-byte Python loop)
        mv = memoryview(buf)[start : start + block_size]
        for i in range(block_size):
            mv[i] ^= 0xFF

    def _op_auto_extras(self, buf, _byte_idx, _data):
        """Collect and inject sequences of "interesting" bytes.

        AFL++ auto-extras: during deterministic stages, consecutive bytes
        that affect the execution path are collected as magic values.
        This operator injects previously collected extras or generates
        new sequences of consecutive interesting bytes.
        """
        rng = self.f._rand_pool
        if not buf:
            return
        f = self.f
        # Use dictionary tokens as "auto-extras" if available
        if f.dictionary:
            tid = (
                f._dict_scratch[f._dict_scratch_idx]
                if f._dict_scratch_idx < len(f._dict_scratch)
                else 0
            )
            f._dict_scratch_idx += 1
            token = f.dictionary[tid]
            if isinstance(token, str):
                token = token.encode()
            if len(token) <= len(buf):
                pos = rng.randint(0, len(buf) - len(token))
                buf[pos : pos + len(token)] = token
            return
        # Generate a short sequence of consecutive interesting bytes
        seq_len = rng.randint(2, 8)
        seq = bytes(
            rng.choice([0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0xFF])
            for _ in range(seq_len)
        )
        if len(buf) >= seq_len:
            pos = rng.randint(0, len(buf) - seq_len)
            buf[pos : pos + seq_len] = seq

    def _op_byte_flip(self, buf, byte_idx, _data):
        if buf:
            buf[byte_idx] ^= 0xFF

    def _op_interesting_8(self, buf, byte_idx, _data):
        rng = self.f._rand_pool
        if buf:
            if self.f._crash_mi and self.f._crash_mi.total_execs >= 50 and rng.random() < 0.3:
                crash_vals = self.f._crash_mi.top_values(byte_idx, k=5)
                if crash_vals:
                    buf[byte_idx] = rng.choice(crash_vals) & 0xFF
                    return
            vals = INTERESTING_UNSIGNED_8 if rng.random() < 0.5 else INTERESTING_8
            buf[byte_idx] = rng.choice(vals) & 0xFF

    def _op_interesting_16(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) >= 2:
            idx = rng.randint(0, len(buf) - 2)
            if self.f._crash_mi and self.f._crash_mi.total_execs >= 50 and rng.random() < 0.3:
                crash_vals = self.f._crash_mi.top_values(idx, k=5)
                if crash_vals:
                    v = rng.choice(crash_vals)
                    fmt = "<H" if v > 32767 or v < -32768 else "<h"
                    struct.pack_into(fmt, buf, idx, v)
                    return
            use_unsigned = rng.random() < 0.5
            vals = INTERESTING_UNSIGNED_16 if use_unsigned else INTERESTING_16
            v = rng.choice(vals)
            fmt = "<H" if v > 32767 or v < -32768 else "<h"
            struct.pack_into(fmt, buf, idx, v)

    def _op_interesting_32(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) >= 4:
            idx = rng.randint(0, len(buf) - 4)
            if self.f._crash_mi and self.f._crash_mi.total_execs >= 50 and rng.random() < 0.3:
                crash_vals = self.f._crash_mi.top_values(idx, k=5)
                if crash_vals:
                    v = rng.choice(crash_vals)
                    fmt = "<I" if v > 2147483647 or v < -2147483648 else "<i"
                    struct.pack_into(fmt, buf, idx, v)
                    return
            use_unsigned = rng.random() < 0.5
            vals = INTERESTING_UNSIGNED_32 if use_unsigned else INTERESTING_32
            v = rng.choice(vals)
            fmt = "<I" if v > 2147483647 or v < -2147483648 else "<i"
            struct.pack_into(fmt, buf, idx, v)

    def _op_arithmetic(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import ARITHMETIC_DELTAS

        width = rng.choice([1, 2, 4, 8])
        if len(buf) >= width:
            max_start = len(buf) - width
            idx = (rng.randint(0, max_start) // width) * width
            delta = rng.choice(ARITHMETIC_DELTAS)
            if rng.random() < 0.5:
                delta = -delta
            if width == 1:
                buf[idx] = (buf[idx] + delta) & 0xFF
            elif width == 2:
                le = rng.random() < 0.5
                if le:
                    val = (struct.unpack_from("<H", buf, idx)[0] + delta) & 0xFFFF
                    struct.pack_into("<H", buf, idx, val)
                else:
                    val = (struct.unpack_from(">H", buf, idx)[0] + delta) & 0xFFFF
                    struct.pack_into(">H", buf, idx, val)
            elif width == 4:
                le = rng.random() < 0.5
                if le:
                    val = (struct.unpack_from("<I", buf, idx)[0] + delta) & 0xFFFFFFFF
                    struct.pack_into("<I", buf, idx, val)
                else:
                    val = (struct.unpack_from(">I", buf, idx)[0] + delta) & 0xFFFFFFFF
                    struct.pack_into(">I", buf, idx, val)
            elif width == 8:
                le = rng.random() < 0.5
                if le:
                    val = (struct.unpack_from("<Q", buf, idx)[0] + delta) & 0xFFFFFFFFFFFFFFFF
                    struct.pack_into("<Q", buf, idx, val)
                else:
                    val = (struct.unpack_from(">Q", buf, idx)[0] + delta) & 0xFFFFFFFFFFFFFFFF
                    struct.pack_into(">Q", buf, idx, val)

    def _op_random_bytes(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if buf:
            buf[rng.randint(0, len(buf) - 1)] = rng.randint(0, 255)

    def _op_block_insert(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) < self.f.max_len:
            idx = rng.randint(0, len(buf))
            max_size = min(64, self.f.max_len - len(buf))
            if max_size >= 1:
                size = choose_len(max_size, rng=rng)
                buf[idx:idx] = bytes(rng.randint(0, 255) for _ in range(size))

    def _op_block_delete(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) > 1:
            idx = rng.randint(0, len(buf) - 1)
            max_size = min(64, len(buf) - idx, len(buf) - 1)
            if max_size >= 1:
                size = choose_len(max_size, rng=rng)
                del buf[idx : idx + size]

    def _op_block_duplicate(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) < 2 or len(buf) >= self.f.max_len:
            return
        idx = rng.randint(0, len(buf) - 1)
        max_size = min(64, len(buf) - idx)
        if max_size >= 1:
            size = choose_len(max_size, rng=rng)
            block = buf[idx : idx + size]
            ins = rng.randint(0, len(buf))
            buf[ins:ins] = block

    def _op_dict_insert(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        f = self.f
        if f.dictionary and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            if len(buf) + len(token) <= f.max_len:
                buf[rng.randint(0, len(buf)) : 0] = token

    def _op_dict_replace(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        f = self.f
        if f.dictionary and buf and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            idx = rng.randint(0, len(buf) - 1)
            end = min(idx + len(token), len(buf))
            buf[idx:end] = token[: end - idx]

    def _op_dict_overwrite(self, buf, _byte_idx, _data):
        f = self.f
        if f.dictionary and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            return bytearray(token[: f.max_len])

    def _op_dict_prepend(self, buf, _byte_idx, _data):
        f = self.f
        if f.dictionary and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            if len(buf) + len(token) <= f.max_len:
                return bytearray(token) + buf

    def _op_dict_append(self, buf, _byte_idx, _data):
        f = self.f
        if f.dictionary and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            if len(buf) + len(token) <= f.max_len:
                buf.extend(token)

    def _op_corpus_literal_insert(self, buf, _byte_idx, _data):
        """Insert or overwrite with a literal learned from the corpus.

        Ported from go-fuzz cases 18/19: extracts integer and string
        literals from prior corpus inputs, then either inserts one at a
        random position or overwrites a random range with one.
        """
        if not buf:
            return None
        f = self.f
        if (
            not hasattr(f, "_corpus_literals")
            or f._corpus_literals is None
            or getattr(f, "_corpus_literals_len", 0) != len(f.corpus)
        ):
            from fuzzer_tool.core.mutations import extract_corpus_literals

            f._corpus_literals = extract_corpus_literals(list(f.corpus))
            f._corpus_literals_len = len(f.corpus)
        int_lits, str_lits = f._corpus_literals
        if not int_lits and not str_lits:
            return None
        rng = f._rand_pool
        if rng.random() < 0.5 and int_lits:
            lit = rng.choice(int_lits)
        elif str_lits:
            lit = rng.choice(str_lits)
        else:
            lit = rng.choice(int_lits)
        if len(lit) >= len(buf):
            return None
        if rng.random() < 0.5:
            pos = rng.randint(0, len(buf) - len(lit))
            buf[pos : pos + len(lit)] = lit
        else:
            pos = rng.randint(0, len(buf))
            buf[pos:pos] = lit
            if len(buf) > f.max_len:
                del buf[f.max_len :]

    def _op_checksum_repair(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool

        if buf and len(buf) >= 4:
            pos = rng.randint(0, max(0, len(buf) - 4))
            buf[pos : pos + 4] = crc32(bytes(buf[:pos])).to_bytes(4, "big")

    def _op_crc_learn(self, buf, _byte_idx, _data):
        """Patch checksum fields using a recovered checksum model.

        The format-aware patchers stay on the GF(2) polynomial: PNG chunk
        CRCs and ZIP CRCs are CRC-32 *by specification*, so a recovered
        integer model (typically the Adler-32 living one layer below, inside
        the IDAT zlib stream) must never be written into those fields. An
        integer model drives only the generic trailing-field patch, and an
        XOR-bitmask model likewise -- the three families are disjoint and
        substituting one for another silently corrupts the field.

        All three must be handled here, because availability gates on
        ``ChecksumLearner.ensure_model()``, which is true when *any* of the
        three has a verified model. A family with no branch below is not
        merely unsupported: the operator is then offered on every selection,
        returns the buffer untouched, and is still selected, timed and
        credited by the schedulers -- a pure no-op of exactly the kind
        ``tests/test_regression_no_op_mutations.py`` exists to catch, and one
        that guard cannot see because it never builds a learner.
        """
        if not buf or len(buf) < 4:
            return
        learner = getattr(self.f, "checksum_learner", None)
        if not learner:
            return
        rng = self.f._rand_pool

        if learner.ensure_poly() is not None:
            # Try format-aware patching first
            patched = self._try_format_crc_patch(buf, learner, rng)
            if patched:
                buf[:] = patched
                return

            # Fallback: patch the last 4 bytes (common checksum placement)
            checksum = learner.compute_checksum(bytes(buf[:-4]))
            buf[-4:] = checksum.to_bytes(4, "big")
            return

        model = learner.ensure_int_model()
        if model is not None:
            # Width comes from the model: a Fletcher-16 field is 2 bytes, and
            # zero-padding it into 4 would clobber two bytes of real data.
            nbytes = model.nbytes
            if len(buf) <= nbytes:
                return
            checksum = learner.compute_int_checksum(bytes(buf[:-nbytes]))
            if checksum is None:
                return
            buf[-nbytes:] = checksum.to_bytes(nbytes, "big")
            return

        # XOR-bitmask family (XOR-of-selected-input-bits). Recovered last, and
        # only once both GF(2) and integer recovery have failed to verify, so
        # reaching here means it is the only model there is.
        xmodel = learner.ensure_xor_model()
        if xmodel is None:
            return
        nbytes = xmodel.nbytes
        if nbytes <= 0 or len(buf) <= nbytes:
            return
        checksum = learner.compute_xor_checksum(bytes(buf[:-nbytes]))
        if checksum is None:
            return
        buf[-nbytes:] = checksum.to_bytes(nbytes, "big")

    # --- helpers for _op_crc_learn ---------------------------------------

    def _try_format_crc_patch(self, buf, learner, rng):
        """Attempt format-aware CRC patching; return patched bytes or None."""
        data = bytes(buf)
        # PNG
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return self._patch_png_crc(data, learner)
        # ZIP
        if len(data) >= 30 and data[:4] == b"PK\x03\x04":
            return self._patch_zip_crc(data, learner, rng)
        return None

    def _patch_png_crc(self, data, learner):
        """Recompute all PNG chunk CRCs using the recovered polynomial."""
        if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        out = bytearray(data[:8])
        pos = 8
        while pos + 12 <= len(data):
            length = struct.unpack_from(">I", data, pos)[0]
            chunk_type = data[pos + 4 : pos + 8]
            chunk_data = data[pos + 8 : pos + 8 + length]
            crc = learner.compute_checksum(chunk_type + chunk_data)
            out += struct.pack(">I", length)
            out += chunk_type
            out += chunk_data
            out += struct.pack(">I", crc)
            pos += 12 + length
            if chunk_type == b"IEND":
                break
        # Append any trailing data
        if pos < len(data):
            out += data[pos:]
        return bytes(out)

    def _patch_zip_crc(self, data, learner, rng):
        """Recompute CRCs in ZIP local file headers using the recovered polynomial."""
        out = bytearray()
        pos = 0
        while pos + 30 <= len(data):
            sig = struct.unpack_from("<I", data, pos)[0]
            if sig != 0x04034B50:
                out += data[pos:]
                break
            fname_len = struct.unpack_from("<H", data, pos + 26)[0]
            extra_len = struct.unpack_from("<H", data, pos + 28)[0]
            comp_size = struct.unpack_from("<I", data, pos + 18)[0]
            # Recompute CRC from filename (heuristic: filename is the data)
            fname = data[pos + 30 : pos + 30 + fname_len]
            crc = learner.compute_checksum(fname)
            header = bytearray(data[pos : pos + 30])
            struct.pack_into("<I", header, 14, crc & 0xFFFFFFFF)
            out += header
            out += data[pos + 30 : pos + 30 + fname_len + extra_len + comp_size]
            pos += 30 + fname_len + extra_len + comp_size
        return bytes(out)

    def _op_token_dup(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        f = self.f
        if f.dictionary and buf and f._dict_scratch_idx < len(f._dict_scratch):
            token = f.dictionary[f._dict_scratch[f._dict_scratch_idx]]
            f._dict_scratch_idx += 1
            if len(buf) + len(token) <= f.max_len:
                buf[rng.randint(0, len(buf)) : 0] = token

    def _op_markov_bytes(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if buf:
            idx = rng.randint(0, len(buf) - 1)
            ctx = (
                bytes(buf[max(0, idx - self.f.markov.order) : idx]) if self.f.markov.order else b""
            )
            buf[idx] = self.f.markov.sample_byte(ctx)

    def _op_cem_bytes(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if self.f.mc and self.f.mc.cem_fitted:
            if buf:
                buf[rng.randint(0, len(buf) - 1)] = self.f.mc.cem_byte(rng.randint(0, len(buf) - 1))
            else:
                return bytearray(self.f.mc.cem_sample(rng.randint(1, min(32, self.f.max_len))))

    def _op_splice(self, buf, _byte_idx, data):
        rng = self.f._rand_pool
        if len(self.f.corpus) >= 2:
            a = rng.choice(self.f.corpus)
            b = rng.choice(self.f.corpus)
            if a is not data and b is not data:
                return bytearray(splice(a, b)[: self.f.max_len])
            others = [c for c in self.f.corpus if c is not data]
            if others:
                return bytearray(splice(bytes(buf), rng.choice(others))[: self.f.max_len])

    def _op_splice_diff_located(self, buf, _byte_idx, data):
        rng = self.f._rand_pool
        if len(self.f.corpus) >= 2:
            a = rng.choice(self.f.corpus)
            b = rng.choice(self.f.corpus)
            if a is not data and b is not data:
                return bytearray(splice_diff_located(a, b, rng=rng)[: self.f.max_len])
            others = [c for c in self.f.corpus if c is not data]
            if others:
                return bytearray(
                    splice_diff_located(bytes(buf), rng.choice(others), rng=rng)[: self.f.max_len]
                )

    def _op_splice_common_prefix(self, buf, _byte_idx, data):
        """Splice donor into base, aligned by common prefix/suffix.

        Ported from go-fuzz case 16.  Finds common prefix/suffix between
        two inputs and only splices the differing middle, skipping when
        the differing region is < 4 bytes.
        """
        rng = self.f._rand_pool
        if len(self.f.corpus) < 2:
            return None
        base = bytes(buf) if buf else b""
        others = [c for c in self.f.corpus if c is not data]
        if not others:
            return None
        donor = rng.choice(others)
        result = splice_common_prefix(base, donor, rng=rng)
        return bytearray(result[: self.f.max_len])

    def _op_insert_range_from_other(self, buf, _byte_idx, data):
        """Insert a sub-range from another corpus entry into the current buffer.

        Ported from go-fuzz case 17.  Picks a random corpus input, extracts a
        short range from it, and splices it into a random position in the
        current buffer.
        """
        rng = self.f._rand_pool
        if len(buf) < 4 or len(self.f.corpus) < 2:
            return None
        others = [c for c in self.f.corpus if c is not data]
        if not others:
            return None
        donor = rng.choice(others)
        if len(donor) < 4:
            return None
        pos0 = rng.randint(0, len(buf))
        pos1 = rng.randint(0, len(donor) - 2)
        max_n = min(len(donor) - pos1 - 2, self.f.max_len - len(buf))
        if max_n < 2:
            return None
        n = choose_len(max_n, rng=rng) + 2
        buf[pos0:pos0] = donor[pos1 : pos1 + n]
        return None

    def _op_radamsa_num(self, buf, _byte_idx, data):
        """Radamsa-style number mutation on a random byte."""
        if not buf:
            return None
        rng = self.f._rand_pool
        pos = rng.randint(0, len(buf) - 1)
        val = radamsa_mutate_num(buf[pos], rng=rng)
        buf[pos] = val & 0xFF
        return buf

    def _op_crossover(self, buf, _byte_idx, data):
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import crossover

        if len(self.f.corpus) >= 2 and buf:
            a = rng.choice(self.f.corpus)
            b = rng.choice(self.f.corpus)
            if a is not data and b is not data:
                return bytearray(crossover(a, b, rng=rng)[: self.f.max_len])
            others = [c for c in self.f.corpus if c is not data]
            if others:
                return bytearray(
                    crossover(bytes(buf), rng.choice(others), rng=rng)[: self.f.max_len]
                )

    def _op_type_replace(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import type_replace

        if buf:
            return bytearray(type_replace(bytes(buf))[: self.f.max_len])

    def _op_ascii_num(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import ascii_num_replace

        if buf:
            return bytearray(ascii_num_replace(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_digit_replace(self, buf, _byte_idx, _data):
        """Replace a single ASCII digit with another random digit.

        Ported from go-fuzz case 14.
        """
        rng = self.f._rand_pool
        digits = [i for i, b in enumerate(buf) if 0x30 <= b <= 0x39]
        if not digits:
            return None
        pos = rng.choice(digits)
        was = buf[pos]
        now = was
        while now == was:
            now = rng.randint(0, 9) + 0x30
        buf[pos] = now
        return None

    def _op_byte_shuffle(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_shuffle

        if buf and len(buf) > 1:
            return bytearray(byte_shuffle(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_byte_delete(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_delete

        if buf and len(buf) > 1:
            return bytearray(byte_delete(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_byte_insert(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_insert

        if buf and len(buf) < self.f.max_len:
            return bytearray(
                byte_insert(bytes(buf), self.f.max_len, rng=self.f._rand_pool)[: self.f.max_len]
            )

    def _op_insert_ascii_num(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import insert_ascii_num

        if buf and len(buf) < self.f.max_len:
            return bytearray(
                insert_ascii_num(bytes(buf), self.f.max_len, rng=self.f._rand_pool)[
                    : self.f.max_len
                ]
            )

    def _op_transpose_16(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 2:
            return bytearray(
                transpose_bytes(bytes(buf), 2, rng=self.f._rand_pool)[: self.f.max_len]
            )

    def _op_transpose_32(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 4:
            return bytearray(
                transpose_bytes(bytes(buf), 4, rng=self.f._rand_pool)[: self.f.max_len]
            )

    def _op_transpose_64(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 8:
            return bytearray(
                transpose_bytes(bytes(buf), 8, rng=self.f._rand_pool)[: self.f.max_len]
            )

    def _op_bit_transpose_8(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if buf:
            return bytearray(bit_transpose(bytes(buf), 1, rng=self.f._rand_pool)[: self.f.max_len])

    def _op_bit_transpose_16(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 2:
            return bytearray(bit_transpose(bytes(buf), 2, rng=self.f._rand_pool)[: self.f.max_len])

    def _op_bit_transpose_32(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 4:
            return bytearray(bit_transpose(bytes(buf), 4, rng=self.f._rand_pool)[: self.f.max_len])

    def _op_bit_transpose_64(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 8:
            return bytearray(bit_transpose(bytes(buf), 8, rng=self.f._rand_pool)[: self.f.max_len])

    # These three pick their own word width from what the buffer can hold, so
    # unlike the bit_transpose family they need no per-width length guard --
    # a 1-byte buffer is a legitimate 8-bit window.
    def _op_bit_rotate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_rotate

        if buf:
            return bytearray(bit_rotate(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_bit_shift(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_shift

        if buf:
            return bytearray(bit_shift(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_span_invert(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import span_invert

        if buf:
            return bytearray(span_invert(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_bit_repack(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_repack

        # Takes max_len rather than relying on the trailing clamp: repacking
        # changes length by ~dst_w/src_w, and truncating a repacked stream
        # mid-element would silently turn this into a truncator.
        if len(buf) >= 2:
            return bytearray(
                bit_repack(bytes(buf), rng=self.f._rand_pool, max_len=self.f.max_len)[
                    : self.f.max_len
                ]
            )

    def _op_length_grow(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if buf and len(buf) < self.f.max_len:
            size = rng.randint(1, min(64, self.f.max_len - len(buf)))
            if size > 0:
                buf.extend(rng.randint(0, 255) for _ in range(size))

    def _op_length_shrink(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) > 2:
            del buf[rng.randint(1, len(buf) - 1) :]

    def _op_repeat_clone(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if buf and len(buf) < self.f.max_len:
            idx = rng.randint(0, len(buf) - 1)
            size = rng.randint(1, min(16, len(buf) - idx))
            block = buf[idx : idx + size]
            ins = idx + size
            if ins <= len(buf) and len(buf) + len(block) <= self.f.max_len:
                buf[ins:ins] = block

    def _op_truncate(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) > 2:
            del buf[rng.randint(2, len(buf)) :]

    def _op_length_boundary(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import LENGTH_BOUNDARIES

        if not buf:
            buf.extend(rng.randint(0, 255) for _ in range(rng.randint(1, 32)))
            return
        # 30% chance: bias toward lengths that historically discovered edges
        if hasattr(self.f, "_length_tracker") and self.f._length_tracker and rng.random() < 0.3:
            recs = self.f._length_tracker.recommended_lengths(k=5)
            if recs:
                target_len = rng.choice(recs)
            else:
                target_len = rng.weighted_choice(
                    LENGTH_BOUNDARIES,
                    weights=[10, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 6, 6, 4, 4, 3, 3, 2, 2, 1],
                )
        else:
            target_len = rng.weighted_choice(
                LENGTH_BOUNDARIES,
                weights=[10, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 6, 6, 4, 4, 3, 3, 2, 2, 1],
            )
        current_len = len(buf)
        if target_len == current_len:
            return
        elif target_len < current_len:
            del buf[target_len:]
        else:
            grow = min(target_len - current_len, self.f.max_len - current_len)
            if grow > 0:
                buf.extend(bytearray(grow))  # zero-filled, fast

    def _op_swap_regions(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) >= 4:
            i = rng.randint(0, len(buf) - 3)
            j = rng.randint(i + 2, len(buf) - 1)
            # size must also respect len(buf) - j, or buf[j : j + size] comes
            # back shorter than `size` and the slice assignments below change
            # buf's length instead of swapping two equal-sized regions (this
            # op mutates in place and returns None, so nothing downstream
            # re-clamps it to max_len).
            size = rng.randint(1, min(j - i, 16, len(buf) - j))
            a, b = buf[i : i + size], buf[j : j + size]
            buf[i : i + size] = b
            buf[j : j + size] = a

    def _op_swap_bytes(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if len(buf) >= 2:
            i, j = rng.sample(len(buf), 2)
            buf[i], buf[j] = buf[j], buf[i]

    def _op_endianness_swap(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if buf:
            width = rng.choice([2, 4, 8])
            if len(buf) >= width:
                idx = rng.randint(0, len(buf) - width)
                val = int.from_bytes(buf[idx : idx + width], "little")
                buf[idx : idx + width] = val.to_bytes(width, "big")

    def _op_insert_repeated_bytes(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        if not buf or len(buf) >= self.f.max_len:
            return
        fill_byte = rng.randint(0, 255)
        block_size = rng.randint(1, min(32, self.f.max_len - len(buf)))
        ins_pos = rng.randint(0, len(buf))
        buf[ins_pos:ins_pos] = bytes([fill_byte] * block_size)

    def _op_sort_bytes(self, buf, _byte_idx, _data):
        if buf and len(buf) > 1:
            start = self.f._rand_pool.randint(0, len(buf) - 1)
            end = min(start + self.f._rand_pool.randint(2, len(buf) - start + 1), len(buf))
            buf[start:end] = sorted(buf[start:end])

    def _op_leb128_encode(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import leb128_encode

        if buf:
            result = leb128_encode(bytes(buf), rng=self.f._rand_pool, max_len=self.f.max_len)
            if result != bytes(buf):
                return bytearray(result[: self.f.max_len])

    def _op_tlv_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.tlv_mutate import tlv_mutate

        if buf:
            return bytearray(tlv_mutate(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_token_shuffle(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.token_shuffle import token_shuffle

        if buf and len(buf) >= 4:
            return bytearray(token_shuffle(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_gradient_cmp(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.gradient_cmp import gradient_cmp

        if buf and self.f._cmplog and self.f._cmplog.pairs:
            return bytearray(
                gradient_cmp(bytes(buf), self.f._cmplog.pairs, rng=self.f._rand_pool)[
                    : self.f.max_len
                ]
            )

    def _op_gradient_descent(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.gradient_descent import gradient_descent

        if not (buf and self.f._cmplog and self.f._cmplog.pairs):
            return
        pair = self.f._rand_pool.choice(self.f._cmplog.pairs)
        result = gradient_descent(bytes(buf), pair, max_len=self.f.max_len, rng=self.f._rand_pool)
        if result and result != bytes(buf):
            return bytearray(result[: self.f.max_len])

    def _op_magic_byte_search(self, buf, _byte_idx, _data):
        """Plant a cmplog operand verbatim at a candidate site (Angora MB)."""
        from fuzzer_tool.core.mb_cbh import magic_byte_search

        if not (buf and self.f._cmplog and self.f._cmplog.pairs):
            return
        pair = self.f._rand_pool.choice(self.f._cmplog.pairs)
        result = magic_byte_search(bytes(buf), pair, self.f._rand_pool, max_len=self.f.max_len)
        if result and result != bytes(buf):
            return bytearray(result[: self.f.max_len])

    def _op_climb_hill(self, buf, _byte_idx, _data):
        """Stochastic hill-climb toward a cmplog operand (Angora CBH)."""
        from fuzzer_tool.core.mb_cbh import climb_hill

        if not (buf and self.f._cmplog and self.f._cmplog.pairs):
            return
        pair = self.f._rand_pool.choice(self.f._cmplog.pairs)
        result = climb_hill(bytes(buf), pair, self.f._rand_pool, max_len=self.f.max_len)
        if result and result != bytes(buf):
            return bytearray(result[: self.f.max_len])

    def _op_condstmt_solve(self, buf, _byte_idx, _data):
        """Solve one unsolved comparison branch via CondStmt substitution.

        Lazily builds a ``CondStmt`` list from the current cmplog pairs,
        then picks an unsolved branch and tries a direct operand swap at
        the first matching offset.  On success the branch state advances
        to ``SOLVED``; on failure it advances to ``UNSOLVABLE`` or
        ``TIMEOUT`` so the solver does not waste budget on it again.
        Falls back to havoc when no applicable branch exists.
        """
        f = self.f
        rng = f._rand_pool
        if not (buf and hasattr(f, "_cmplog") and f._cmplog and f._cmplog.pairs):
            return self._op_havoc(buf, _byte_idx, _data)

        conds = self._get_cond_stmts()
        if not conds:
            return self._op_havoc(buf, _byte_idx, _data)

        # Prefer unsolved branches; fall back to any branch when all are
        # solved/unsolvable/timeout so the operator still produces a useful
        # operand-substitution mutation.
        unsolved = [c for c in conds if c.state is CondState.UNSOLVED]
        target = rng.choice(unsolved) if unsolved else rng.choice(conds)

        data = bytes(buf)
        target_value = target.base.op_b if rng.random() < 0.5 else target.base.op_a
        source_value = target.base.op_a if target_value is target.base.op_b else target.base.op_b
        width = target.base.width

        # Find the first offset where the source operand appears.
        idx = data.find(source_value[:width])
        if idx != -1 and idx + width <= len(buf):
            buf[idx : idx + width] = target_value[:width]
            target.mark_solved()
            return buf

        # Fallback: insert the target operand at a random position.
        if len(buf) + width <= f.max_len:
            pos = rng.randint(0, len(buf))
            buf[pos:pos] = target_value[:width]
            target.mark_solved()
            return buf

        target.mark_unsolvable()
        return self._op_havoc(buf, _byte_idx, _data)

    def _get_cond_stmts(self) -> list[CondStmt]:
        """Lazily build and cache the CondStmt list from cmplog pairs."""
        f = self.f
        cached = getattr(f, "_cond_stmts", None)
        if cached is not None:
            # Invalidate when the pair list changes.
            pair_id = id(f._cmplog.pairs) if f._cmplog and f._cmplog.pairs else -1
            if getattr(f, "_cond_stmts_pair_id", None) == pair_id:
                return cached
        from fuzzer_tool.core.cond_stmt import (
            conds_from_cmplog_pairs,
        )

        pair_meta = (
            getattr(f._cmplog, "_pair_cmp", {}) if hasattr(f, "_cmplog") and f._cmplog else {}
        )
        pair_pc = getattr(f._cmplog, "_pair_pc", {}) if hasattr(f, "_cmplog") and f._cmplog else {}
        cached = conds_from_cmplog_pairs(
            f._cmplog.pairs if f._cmplog and f._cmplog.pairs else [],
            pair_meta=pair_meta,
            pair_pc=pair_pc,
        )
        f._cond_stmts = cached
        f._cond_stmts_pair_id = id(f._cmplog.pairs) if f._cmplog and f._cmplog.pairs else -1
        return cached

    def _op_special_strings(self, buf, _byte_idx, _data):
        """Insert a security-sensitive string at a random position.

        Ported from honggfuzz mangle_SpecialStrings: 44 curated strings
        covering SQL injection, XSS, path traversal, format strings,
        command injection, JSON edge cases, and control characters.
        """
        rng = self.f._rand_pool
        if not buf or len(buf) >= self.f.max_len:
            return
        s = rng.choice(SPECIAL_STRINGS)
        pos = rng.randint(0, len(buf))
        buf[pos:pos] = s

    def _op_magic_values(self, buf, _byte_idx, _data):
        """Insert a magic/boundary value from the honggfuzz table.

        Ported from honggfuzz mangle_Magic: 229 hardcoded boundary values
        covering 1/2/4/8-byte widths in both LE and BE endianness.
        """
        rng = self.f._rand_pool
        if not buf:
            return
        width, packed = rng.choice(MAGIC_TABLE)
        if len(buf) + width <= self.f.max_len:
            pos = rng.randint(0, len(buf))
            buf[pos:pos] = packed
        elif len(buf) >= width:
            pos = rng.randint(0, len(buf) - width)
            buf[pos : pos + width] = packed

    def _op_ascii_num_arithmetic(self, buf, _byte_idx, _data):
        """Mutate an existing ASCII number in-place using arithmetic.

        Ported from honggfuzz mangle_ASCIINumChange: finds a digit sequence
        and applies +1, -1, *2, /2, NOT, random replace, +random, or -random.
        """
        from fuzzer_tool.core.mutations import ascii_num_arithmetic

        if buf and len(buf) >= 1:
            result = ascii_num_arithmetic(bytes(buf), rng=self.f._rand_pool)
            if result is not None:
                return bytearray(result[: self.f.max_len])

    def _op_ascii_num_replace(self, buf, _byte_idx, _data):
        """Replace a whole multi-digit ASCII number with a random value.

        Ported from go-fuzz case 15: finds runs of digits (optionally
        prefixed with '-') of length >= 2 and replaces the whole token
        with a random value drawn from small ints, big ints, big-int
        squares, or negative big ints.
        """
        if buf and len(buf) >= 2:
            result = ascii_num_replace(bytes(buf), rng=self.f._rand_pool)
            return bytearray(result[: self.f.max_len])

    def _op_chunk_shuffle(self, buf, _byte_idx, data):
        """Shuffle fixed-size chunks, preserving chunk boundaries.

        Ported from honggfuzz mangle_ChunkShuffle: divides input into 1-4 byte
        chunks and swaps random pairs. Important for width-sensitive binary formats.
        When the parent seed has an inferred record stride, whole stride-sized
        records are shuffled instead — used with probability 0.5 so the operator
        does not over-latch on a single inferred stride.
        """
        from fuzzer_tool.core.mutations import chunk_shuffle

        if buf and len(buf) >= 8:
            rng = self.f._rand_pool
            parent_meta = self.f.seed_meta.get(data)
            stride = None
            if parent_meta and rng.random() < 0.5:
                stride = parent_meta.get("record_stride")
            return bytearray(chunk_shuffle(bytes(buf), rng=rng, stride=stride)[: self.f.max_len])

    def _op_block_shuffle_variable(self, buf, _byte_idx, _data):
        """Shuffle variable-width blocks using order-statistics spacings trick.

        Divides the input into 2-5 random-width blocks using the normalized-
        Exponential spacing trick (order_statistics.py Part 3). Unlike
        chunk_shuffle (fixed-width chunks), this produces variable-width
        blocks that can rearrange structural elements at any granularity.
        """
        from fuzzer_tool.core.mutations import block_shuffle_variable

        if buf and len(buf) >= 8:
            return bytearray(
                block_shuffle_variable(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len]
            )

    def _op_dict_compound(self, buf, _byte_idx, _data):
        """Insert two dictionary tokens concatenated with a random separator.

        Ported from honggfuzz mangle_DictionaryInsert: generates compound
        tokens like ``key=value`` or ``param1&param2`` by joining two
        dictionary entries with a random separator.
        """
        rng = self.f._rand_pool
        f = self.f
        if not f.dictionary or len(f.dictionary) < 2:
            return
        if not buf or len(buf) >= f.max_len:
            return
        from fuzzer_tool.core.mutations import DICT_COMPOUND_SEPARATORS

        t1 = rng.choice(f.dictionary)
        t2 = rng.choice(f.dictionary)
        sep = rng.choice(DICT_COMPOUND_SEPARATORS)
        if isinstance(t1, str):
            t1 = t1.encode()
        if isinstance(t2, str):
            t2 = t2.encode()
        compound = t1 + sep + t2
        if len(buf) + len(compound) <= f.max_len:
            pos = rng.randint(0, len(buf))
            buf[pos:pos] = compound

    def _op_punctuation_insert(self, buf, _byte_idx, _data):
        """Insert 1-4 random punctuation characters at a random position.

        Ported from honggfuzz mangle_Punctuation: useful for breaking
        escaping and structure in text-based protocols.
        """
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations import PUNCTUATION_CHARS

        if not buf or len(buf) >= self.f.max_len:
            return
        n = rng.randint(1, min(4, self.f.max_len - len(buf)))
        chars = bytes(rng.choice(PUNCTUATION_CHARS) for _ in range(n))
        pos = rng.randint(0, len(buf))
        buf[pos:pos] = chars

    def _op_grammar_mutate(self, buf, _byte_idx, _data):
        if self.f.grammar:
            return bytearray(
                self.f.grammar.mutate(bytes(buf), max_len=self.f.max_len, rng=self.f._rand_pool)[
                    : self.f.max_len
                ]
            )

    def _op_grammar_tree_mutate(self, buf, _byte_idx, data):
        if self.f.grammar:
            from fuzzer_tool.core.grammar import SubtreePopulation, TreeMutator

            f = self.f
            rng = f._rand_pool
            if not hasattr(f, "_tree_mutator"):
                f._tree_mutator = TreeMutator(f.grammar)
                f._subtree_population = SubtreePopulation()
                f._subtree_pop_next_idx = 0
            parent_meta = f.seed_meta.get(data)
            stride = parent_meta.get("record_stride") if parent_meta else None
            tree = f._tree_mutator.parse(bytes(buf), chunk_size=stride)

            # Incrementally harvest newly-added corpus entries into the
            # shared subtree population (subtree-population crossover, see
            # docs/web_research_port_candidates_2026-08.md #8) instead of
            # reparsing the whole corpus on every call.
            corpus = getattr(f, "corpus", None) or []
            next_idx = f._subtree_pop_next_idx
            if next_idx > len(corpus):
                next_idx = 0  # corpus was replaced/shrunk — restart harvesting
            for seed in corpus[next_idx:]:
                donor_tree = f._tree_mutator.parse(bytes(seed))
                f._subtree_population.add(donor_tree, rng=rng)
            f._subtree_pop_next_idx = len(corpus)

            return bytearray(
                f._tree_mutator.mutate_tree(
                    tree, max_len=f.max_len, rng=rng, population=f._subtree_population
                )[: f.max_len]
            )

    def _op_versifier_generate(self, buf, _byte_idx, _data):
        """Generate text-like input by learning structure from the corpus.

        Ported from go-fuzz versifier: tokenizes corpus inputs into
        whitespace/alphanum/numeric/control/bracket/kv/list/line nodes,
        then generates structurally similar text from the learned grammar.
        """
        if not buf or not hasattr(self.f, "corpus") or not self.f.corpus:
            return None
        f = self.f
        rng = f._rand_pool
        verse = getattr(f, "_versifier_verse", None)
        if verse is None or getattr(f, "_versifier_corpus_len", 0) != len(f.corpus):
            from fuzzer_tool.core.mutations.generic import _build_verse

            corpus = f.corpus
            verse = None
            for raw in reversed(corpus):
                candidate = _build_verse(raw, rng)
                if candidate is not None:
                    verse = candidate
                    break
            f._versifier_verse = verse
            f._versifier_corpus_len = len(corpus)
        if verse is None:
            return None
        result = verse.Rhyme()
        if not result:
            return None
        return bytearray(result[: f.max_len])

    def _op_png_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.png import PngChunkMutator, parse_png_chunks

        if not hasattr(self.f, "_png_mutator"):
            self.f._png_mutator = PngChunkMutator()
        # Set WFC mode from fuzzer state
        self.f._png_mutator.use_wfc = getattr(self.f, "_wfc_enabled", False)
        rng = self.f._rand_pool
        if parse_png_chunks(bytes(buf)):
            mutated = self.f._png_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._png_mutator._generate_random_png(self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_jpeg_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.jpeg import JpegMutator, parse_jpeg_markers

        if not hasattr(self.f, "_jpeg_mutator"):
            self.f._jpeg_mutator = JpegMutator()
        self.f._jpeg_mutator.use_wfc = getattr(self.f, "_wfc_enabled", False)
        rng = self.f._rand_pool
        if parse_jpeg_markers(bytes(buf)):
            mutated = self.f._jpeg_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._jpeg_mutator._generate_random_jpeg(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_jpeg_crc_fix(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations.jpeg import (
            STANDALONE_MARKERS,
            parse_jpeg_markers,
            serialize_jpeg_markers,
        )

        if buf:
            markers = parse_jpeg_markers(bytes(buf))
            if markers and len(markers) > 2:
                candidates = [
                    i
                    for i, m in enumerate(markers)
                    if m.marker not in STANDALONE_MARKERS and len(m.data) > 0
                ]
                if candidates:
                    idx = rng.choice(candidates)
                    marker = markers[idx]
                    data = bytearray(marker.data)
                    for _ in range(rng.randint(1, min(4, len(data)))):
                        data[rng.randint(0, len(data) - 1)] ^= 1 << rng.randint(0, 7)
                    marker.data = bytes(data)
                    return bytearray(serialize_jpeg_markers(markers)[: self.f.max_len])

    def _op_gzip_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.gzip import GzipMutator, parse_gzip

        if not hasattr(self.f, "_gzip_mutator"):
            self.f._gzip_mutator = GzipMutator()
        rng = self.f._rand_pool
        if parse_gzip(bytes(buf)):
            mutated = self.f._gzip_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._gzip_mutator._generate_random_gzip(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_bmp_chunk_mutate(self, buf, _byte_idx, data):
        from fuzzer_tool.core.mutations.bmp import BmpMutator, parse_bmp

        if not hasattr(self.f, "_bmp_mutator"):
            self.f._bmp_mutator = BmpMutator()
        self.f._bmp_mutator.use_wfc = getattr(self.f, "_wfc_enabled", False)
        parent_meta = self.f.seed_meta.get(data)
        stride = parent_meta.get("record_stride") if parent_meta else None
        self.f._bmp_mutator.tile_bytes = stride
        rng = self.f._rand_pool
        if parse_bmp(bytes(buf)):
            mutated = self.f._bmp_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._bmp_mutator._generate_random_bmp(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_zlib_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.zlib import ZlibMutator, parse_zlib

        if not hasattr(self.f, "_zlib_mutator"):
            self.f._zlib_mutator = ZlibMutator()
        rng = self.f._rand_pool
        if parse_zlib(bytes(buf)):
            mutated = self.f._zlib_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._zlib_mutator._generate_random_zlib(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_format_lock(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.magic_lock import format_lock_havoc

        rng = self.f._rand_pool
        if buf:
            result = format_lock_havoc(bytes(buf), self.f.max_len, rng=rng)
            if result is not None:
                return bytearray(result[: self.f.max_len])

    def _op_pgs_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.pgs import PgsMutator, parse_pgs_segments

        if not hasattr(self.f, "_pgs_mutator"):
            self.f._pgs_mutator = PgsMutator()
        rng = self.f._rand_pool
        if parse_pgs_segments(bytes(buf)):
            mutated = self.f._pgs_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._pgs_mutator._generate_random_pgs(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_isobmff_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.isobmff import IsobmffMutator, parse_boxes

        if not hasattr(self.f, "_isobmff_mutator"):
            self.f._isobmff_mutator = IsobmffMutator()
        rng = self.f._rand_pool
        if parse_boxes(bytes(buf)):
            mutated = self.f._isobmff_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._isobmff_mutator._generate_random_isobmff(
                max_len=self.f.max_len, rng=rng
            )
        return bytearray(mutated[: self.f.max_len])

    def _op_nal_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.nal import NalMutator, parse_nal_units

        if not hasattr(self.f, "_nal_mutator"):
            self.f._nal_mutator = NalMutator()
        rng = self.f._rand_pool
        if parse_nal_units(bytes(buf)):
            mutated = self.f._nal_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._nal_mutator._generate_random_nal_stream(
                max_len=self.f.max_len, rng=rng
            )
        return bytearray(mutated[: self.f.max_len])

    def _op_mpegts_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.mpegts import MpegtsMutator, parse_ts_packets

        if not hasattr(self.f, "_mpegts_mutator"):
            self.f._mpegts_mutator = MpegtsMutator()
        rng = self.f._rand_pool
        if parse_ts_packets(bytes(buf)):
            mutated = self.f._mpegts_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._mpegts_mutator._generate_random_ts(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_adts_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.adts import AdtsMutator, parse_adts_frames

        if not hasattr(self.f, "_adts_mutator"):
            self.f._adts_mutator = AdtsMutator()
        rng = self.f._rand_pool
        if parse_adts_frames(bytes(buf)):
            mutated = self.f._adts_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._adts_mutator._generate_random_adts(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_mp3_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.mp3 import Mp3Mutator, parse_mp3_frames

        if not hasattr(self.f, "_mp3_mutator"):
            self.f._mp3_mutator = Mp3Mutator()
        rng = self.f._rand_pool
        if parse_mp3_frames(bytes(buf)):
            mutated = self.f._mp3_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._mp3_mutator._generate_random_mp3(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_ogg_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.ogg import OggMutator, parse_ogg_pages

        if not hasattr(self.f, "_ogg_mutator"):
            self.f._ogg_mutator = OggMutator()
        rng = self.f._rand_pool
        if parse_ogg_pages(bytes(buf)):
            mutated = self.f._ogg_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._ogg_mutator._generate_random_ogg(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_flv_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.flv import FlvMutator, parse_flv

        if not hasattr(self.f, "_flv_mutator"):
            self.f._flv_mutator = FlvMutator()
        rng = self.f._rand_pool
        if parse_flv(bytes(buf)):
            mutated = self.f._flv_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._flv_mutator._generate_random_flv(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_asf_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.asf import AsfMutator, parse_asf_objects

        if not hasattr(self.f, "_asf_mutator"):
            self.f._asf_mutator = AsfMutator()
        rng = self.f._rand_pool
        if parse_asf_objects(bytes(buf)):
            mutated = self.f._asf_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._asf_mutator._generate_random_asf(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_riff_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.riff import RiffMutator, parse_riff_chunks

        if not hasattr(self.f, "_riff_mutator"):
            self.f._riff_mutator = RiffMutator()
        rng = self.f._rand_pool
        if parse_riff_chunks(bytes(buf)):
            mutated = self.f._riff_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._riff_mutator._generate_random_riff(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_avif_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.avif import AvifMutator, parse_avif

        if not hasattr(self.f, "_avif_mutator"):
            self.f._avif_mutator = AvifMutator()
        rng = self.f._rand_pool
        if parse_avif(bytes(buf)):
            mutated = self.f._avif_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._avif_mutator._generate_random_avif(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_sqlite_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator, parse_sqlite

        if not hasattr(self.f, "_sqlite_mutator"):
            self.f._sqlite_mutator = SqliteMutator()
        rng = self.f._rand_pool
        if parse_sqlite(bytes(buf)):
            mutated = self.f._sqlite_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._sqlite_mutator._generate_random_sqlite(
                max_len=self.f.max_len, rng=rng
            )
        return bytearray(mutated[: self.f.max_len])

    def _op_protobuf_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.protobuf import ProtobufMutator, parse_protobuf

        if not hasattr(self.f, "_protobuf_mutator"):
            self.f._protobuf_mutator = ProtobufMutator()
        rng = self.f._rand_pool
        if parse_protobuf(bytes(buf)):
            mutated = self.f._protobuf_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._protobuf_mutator._generate_random_protobuf(
                max_len=self.f.max_len, rng=rng
            )
        return bytearray(mutated[: self.f.max_len])

    def _op_gif_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.gif import GifMutator, parse_gif

        if not hasattr(self.f, "_gif_mutator"):
            self.f._gif_mutator = GifMutator()
        rng = self.f._rand_pool
        if parse_gif(bytes(buf)):
            mutated = self.f._gif_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._gif_mutator._generate_random_gif(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_webp_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.webp import WebpMutator, parse_webp

        if not hasattr(self.f, "_webp_mutator"):
            self.f._webp_mutator = WebpMutator()
        rng = self.f._rand_pool
        if parse_webp(bytes(buf)):
            mutated = self.f._webp_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._webp_mutator._generate_random_webp(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_webm_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.webm import WebmMutator, parse_webm

        if not hasattr(self.f, "_webm_mutator"):
            self.f._webm_mutator = WebmMutator()
        rng = self.f._rand_pool
        if parse_webm(bytes(buf)):
            mutated = self.f._webm_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._webm_mutator._generate_random_webm(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_zip_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.zip import ZipMutator, parse_zip

        if not hasattr(self.f, "_zip_mutator"):
            self.f._zip_mutator = ZipMutator()
        rng = self.f._rand_pool
        if parse_zip(bytes(buf)):
            mutated = self.f._zip_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._zip_mutator._generate_random_zip(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_x86_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.x86 import X86Mutator, _decode_insns

        if not hasattr(self.f, "_x86_mutator"):
            self.f._x86_mutator = X86Mutator()
        rng = self.f._rand_pool
        if _decode_insns(bytes(buf)):
            mutated = self.f._x86_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._x86_mutator._generate_random_x86(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_arm_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.arm import ArmMutator, parse_arm

        if not hasattr(self.f, "_arm_mutator"):
            self.f._arm_mutator = ArmMutator()
        rng = self.f._rand_pool
        if parse_arm(bytes(buf)):
            mutated = self.f._arm_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        else:
            mutated = self.f._arm_mutator._generate_random_arm(max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_tlv_nest_mutate(self, buf, byte_idx, data):
        """Mutate a nested TLV value and fix every enclosing length.

        ``tlv_mutate`` writes boundary constants into candidate length
        fields, which breaks the frame and stops the parser at the outermost
        container. This edits a value and re-derives the length of every
        frame enclosing it, so the input stays well-formed several levels
        down and the mutated leaf is actually reached.
        """
        from fuzzer_tool.core.structural_constraints import (
            parse_tlv,
            resize_tlv_value,
        )

        rng = self.f._rand_pool
        raw = bytes(buf)
        for tag_w, len_w, big in ((1, 2, True), (2, 2, True), (1, 4, True), (1, 2, False)):
            roots = parse_tlv(raw, tag_width=tag_w, length_width=len_w, big_endian=big)
            if not roots:
                continue
            nodes = [n for root in roots for n in root.walk()]
            if not nodes:
                continue
            target = nodes[rng.randint(0, len(nodes) - 1)]
            old = raw[target.value_start : target.value_end]
            if target.children and old:
                # Container: perturb bytes without changing the frame size.
                payload = bytearray(old)
                payload[rng.randint(0, len(payload) - 1)] ^= 1 << rng.randint(0, 7)
                payload = bytes(payload)
            else:
                delta = rng.choice((-8, -1, 0, 1, 8, 64))
                size = max(0, min(len(old) + delta, self.f.max_len // 2))
                payload = (old + bytes(rng.randint(0, 255) for _ in range(size)))[:size]
            out = resize_tlv_value(
                raw, target, payload, tag_width=tag_w, length_width=len_w, big_endian=big
            )
            if out:
                return bytearray(out[: self.f.max_len])
        return self._op_havoc(buf, byte_idx, data)

    def _der_mutate(self, method: str, buf, byte_idx, data):
        """Shared driver for the BER/DER operators (png-handler pattern)."""
        from fuzzer_tool.core.mutations.der import DerMutator, parse_der

        if not hasattr(self.f, "_der_mutator"):
            self.f._der_mutator = DerMutator()
        rng = self.f._rand_pool
        raw = bytes(buf)
        if parse_der(raw) is None:
            # Not DER yet (bootstrap): grow a random DER-shaped input so the
            # magic-byte sniffer can latch the format on a garbage corpus.
            mutated = self.f._der_mutator._generate_random_der(self.f.max_len, rng=rng)
        else:
            mutated = getattr(self.f._der_mutator, method)(raw, max_len=self.f.max_len, rng=rng)
        if mutated is None:
            return self._op_havoc(buf, byte_idx, data)
        return bytearray(mutated[: self.f.max_len])

    def _op_der_len_mutate(self, buf, byte_idx, data):
        """Mutate a BER/DER TLV length field (form flips, shrink/grow, indefinite)."""
        return self._der_mutate("mutate_length", buf, byte_idx, data)

    def _op_der_tag_mutate(self, buf, byte_idx, data):
        """Mutate a BER/DER TLV tag byte (class, constructed, number)."""
        return self._der_mutate("mutate_tag", buf, byte_idx, data)

    def _op_der_tlv_reorder(self, buf, byte_idx, data):
        """Reorder/duplicate/remove siblings inside a constructed value."""
        return self._der_mutate("reorder_children", buf, byte_idx, data)

    def _op_der_tlv_insert(self, buf, byte_idx, data):
        """Insert a fresh or truncated TLV into a constructed value."""
        return self._der_mutate("insert_tlv", buf, byte_idx, data)

    def _op_length_offset_goal(self, buf, byte_idx, data):
        """Write a solved offset/size pair into a candidate length field.

        The existing TLV and ELF operators write boundary constants and hit
        the wraparound branch only by luck. These pairs satisfy the arithmetic
        condition by construction — a sum that wraps while the offset alone
        still passes a naive bounds check, for instance.
        """
        from fuzzer_tool.core.structural_constraints import GOALS, solve_length_offset

        rng = self.f._rand_pool
        out = bytearray(buf)
        width = rng.choice((2, 4, 8))
        if len(out) < width * 2 + 1:
            return self._op_havoc(buf, byte_idx, data)

        goal = GOALS[rng.randint(0, len(GOALS) - 1)]
        solved = solve_length_offset(goal, width, len(out), rng=rng)
        if solved is None:
            return self._op_havoc(buf, byte_idx, data)
        offset_value, size_value = solved

        pos = rng.randint(0, len(out) - width * 2)
        big = rng.randint(0, 1) == 1
        order = "big" if big else "little"
        out[pos : pos + width] = offset_value.to_bytes(width, order)
        out[pos + width : pos + width * 2] = size_value.to_bytes(width, order)
        return out[: self.f.max_len]

    def _op_field_repair(self, buf, byte_idx, data):
        """Restore every derived field so they all hold simultaneously.

        The existing fixups repair one field in isolation, so a mutation
        touching coupled fields (a length inside a checksummed span, say)
        leaves at least one wrong and the target rejects the input before
        reaching the parser logic being fuzzed. This repairs them in
        dependency order.

        Applied after a havoc pass, so the operator both mutates and
        re-establishes structural validity — repairing an already-valid
        input would be a no-op execution.
        """
        from fuzzer_tool.core.field_constraints import png_fields, repair

        mutated = self._op_havoc(buf, byte_idx, data)
        if mutated is None:
            mutated = buf
        raw = bytes(mutated)
        fields = png_fields(raw)
        if not fields:
            return mutated
        fixed = repair(fields, raw)
        if fixed is None:
            return mutated
        return bytearray(fixed[: self.f.max_len])

    def _op_path_negate(self, buf, byte_idx, data):
        """Solve for an input that flips a recorded branch predicate.

        Unlike ``redqueen_xform``, which substitutes an operand that was
        already observed, this asserts the *negated* comparison over
        symbolic input bytes and lets z3 search — so it can reach the
        sibling branch of a comparison the corpus has never satisfied.

        Falls back to havoc when z3 is absent, no branch maps to an input
        offset, or the constraint is unsatisfiable; returning the buffer
        unchanged would waste the execution.
        """
        from fuzzer_tool.core.path_constraints import records_from_collector

        cmplog = getattr(self.f, "_cmplog", None)
        solver = getattr(self.f, "_path_solver", None)
        # _path_solver is None unless --path-negation was passed (or z3 is
        # missing). Respect that rather than constructing a private solver:
        # the flag is the single control, and the fuzzer's instance carries
        # the shared frontier so the two call sites do not re-solve the same
        # branch independently.
        if cmplog is None or solver is None:
            return self._op_havoc(buf, byte_idx, data)

        records = records_from_collector(cmplog)
        if not records:
            return self._op_havoc(buf, byte_idx, data)

        solved = solver.solve_first(records, bytes(buf))
        if solved is None:
            return self._op_havoc(buf, byte_idx, data)
        return bytearray(solved[: self.f.max_len])

    def _op_elf_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.elf import ElfMutator

        if not hasattr(self.f, "_elf_mutator"):
            self.f._elf_mutator = ElfMutator()
        rng = self.f._rand_pool
        mutated = self.f._elf_mutator.mutate(bytes(buf), max_len=self.f.max_len, rng=rng)
        return bytearray(mutated[: self.f.max_len])

    def _op_recompress_zlib(self, buf, byte_idx, data):
        """Inflate, mutate the plaintext, re-deflate as a valid zlib stream.

        Falls back to havoc when the input is not an inflatable stream: the
        registry's bootstrap trickle can offer this operator before any real
        compressed input has been seen, and emitting the buffer unchanged
        would waste the execution.
        """
        from fuzzer_tool.core.mutations.recompress import recompress_zlib

        rng = self.f._rand_pool
        out = recompress_zlib(bytes(buf), max_len=self.f.max_len, rng=rng)
        if out is None:
            return self._op_havoc(buf, byte_idx, data)
        return bytearray(out[: self.f.max_len])

    def _op_recompress_gzip(self, buf, byte_idx, data):
        """Inflate, mutate the plaintext, re-deflate as a valid gzip member."""
        from fuzzer_tool.core.mutations.recompress import recompress_gzip

        rng = self.f._rand_pool
        out = recompress_gzip(bytes(buf), max_len=self.f.max_len, rng=rng)
        if out is None:
            return self._op_havoc(buf, byte_idx, data)
        return bytearray(out[: self.f.max_len])

    def _op_png_crc_fix(self, buf, _byte_idx, _data):
        rng = self.f._rand_pool
        from fuzzer_tool.core.mutations.png import parse_png_chunks, serialize_png_chunks

        if buf:
            chunks = parse_png_chunks(bytes(buf))
            if chunks and len(chunks) > 1:
                candidates = [i for i, c in enumerate(chunks) if c.chunk_type != b"IEND"]
                if candidates:
                    idx = rng.choice(candidates)
                    chunk = chunks[idx]
                    if chunk.data:
                        data = bytearray(chunk.data)
                        for _ in range(rng.randint(1, min(4, len(data)))):
                            data[rng.randint(0, len(data) - 1)] ^= 1 << rng.randint(0, 7)
                        chunk.data = bytes(data)
                    else:
                        chunk.data = bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 32)))
                    return bytearray(serialize_png_chunks(chunks)[: self.f.max_len])

    def _op_redqueen(self, buf, _byte_idx, data):
        rng = self.f._rand_pool
        parent_meta = self.f.seed_meta.get(data)
        if not (buf and parent_meta):
            return
        matches = parent_meta.get("redqueen_matches", [])
        offsets = parent_meta.get("redqueen_offsets", [])
        if matches:
            for _ in range(rng.randint(1, min(4, len(matches)))):
                off, op_a, op_b = rng.choice(matches)
                end = off + len(op_a)
                if end <= len(buf) and bytes(buf[off:end]) == op_a:
                    for j, b_val in enumerate(op_b):
                        if off + j < len(buf):
                            buf[off + j] = b_val
        elif offsets and self.f._cmplog and self.f._cmplog.tokens:
            for _ in range(rng.randint(1, min(4, len(offsets)))):
                off = rng.choice(offsets)
                if off < len(buf):
                    token = rng.choice(self.f._cmplog.tokens)
                    for j, b_val in enumerate(token):
                        if off + j < len(buf):
                            buf[off + j] = b_val
        elif offsets:
            for _ in range(rng.randint(1, min(4, len(offsets)))):
                off = rng.choice(offsets)
                if off < len(buf):
                    buf[off] ^= 0xFF

    # ── Regularity operators (diehard/dieharder inverses) ──────────────
    # Each one delegates to a length-preserving construction in
    # core/mutations/structured.py; see that module for what statistic each
    # inverts and why the resulting shape is interesting to a parser.

    def _regularity(self, fn, buf):
        """Run a structured.py construction over *buf*.

        The constructions are length-preserving, so the ``max_len`` clamp is
        a formality here -- it is applied anyway to match every other
        buffer-returning handler, rather than relying on a property the
        module happens to have today.
        """
        return bytearray(fn(bytes(buf), rng=self.f._rand_pool)[: self.f.max_len])

    def _op_gcd_worst_case(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import fibonacci_pairs

        return self._regularity(fibonacci_pairs, buf)

    def _op_monotone_fill(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import monotone_fill

        return self._regularity(monotone_fill, buf)

    def _op_kmer_saturate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import de_bruijn_fill

        return self._regularity(de_bruijn_fill, buf)

    def _op_kmer_saturate_bits(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import kmer_saturate_bits

        return self._regularity(kmer_saturate_bits, buf)

    def _op_kmer_starve(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import kmer_starve

        return self._regularity(kmer_starve, buf)

    def _op_rank_deficient(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import rank_deficient

        return self._regularity(rank_deficient, buf)

    def _op_perm_lock(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import perm_lock

        return self._regularity(perm_lock, buf)

    def _op_lag_correlate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import lag_correlate

        return self._regularity(lag_correlate, buf)

    def _op_spectral_peak(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import spectral_peak

        return self._regularity(spectral_peak, buf)

    def _op_birthday_collide(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import birthday_collide

        return self._regularity(birthday_collide, buf)

    def _op_degenerate_geometry(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import degenerate_geometry

        return self._regularity(degenerate_geometry, buf)

    def _op_float_squeeze(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import float_squeeze

        return self._regularity(float_squeeze, buf)

    def _op_popcount_lock(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import popcount_lock

        return self._regularity(popcount_lock, buf)

    def corpus_invariants(self):
        """Cached ``CorpusInvariants`` for the current corpus, or None.

        Recomputed only when the corpus grows: the scan is O(samples x
        length) and the invariant set moves slowly, so recomputing per
        mutation would dominate the operator's cost. Returns None when the
        corpus is too small for the measurement to mean anything -- the
        availability predicate normally prevents that, but a direct
        dispatch call must not scribble over an entire file on the strength
        of two samples agreeing by chance.
        """
        from fuzzer_tool.core.randomness import corpus_invariants

        corpus = getattr(self.f, "corpus", None)
        if not corpus or len(corpus) < _INVARIANT_MIN_SAMPLES:
            return None
        if self._invariants is None or self._invariants_corpus_len != len(corpus):
            samples = corpus[-_INVARIANT_SAMPLE_CAP:]
            self._invariants = corpus_invariants(samples, min_samples=_INVARIANT_MIN_SAMPLES)
            self._invariants_corpus_len = len(corpus)
        return self._invariants

    def _op_invariant_break(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations.structured import invariant_break

        invariants = self.corpus_invariants()
        if invariants is None:
            return None
        return bytearray(
            invariant_break(bytes(buf), invariants, rng=self.f._rand_pool)[: self.f.max_len]
        )

    def elite_seeds(self):
        """Cached top-``_ELITE_FUSE_POOL_SIZE`` corpus seeds by coverage_edges.

        Recomputed only when the corpus grows, mirroring corpus_invariants():
        ranking is O(corpus) via seed_meta lookups, so paying that per
        mutation instead of per growth would dominate the operator's cost on
        large corpora. Falls back to an all-tied ranking (stable, corpus
        order preserved) when seed_meta has no entry for a seed or is
        missing/empty entirely, rather than refusing to run -- coverage
        metadata not being populated yet (e.g. very early in a run, or a
        corpus seeded directly rather than through save_to_corpus) shouldn't
        make this operator unavailable, only unable to discriminate.
        """
        import heapq  # noqa: PLC0415

        corpus = getattr(self.f, "corpus", None)
        if not corpus or len(corpus) < _ELITE_FUSE_MIN_CORPUS:
            return []
        seed_meta = getattr(self.f, "seed_meta", None) or {}
        if self._elite_pool is None or self._elite_pool_corpus_len != len(corpus):
            pool_size = min(_ELITE_FUSE_POOL_SIZE, len(corpus))
            # seed_meta is keyed by bytes (see Fuzzer._seed_key/save_to_corpus);
            # corpus entries are bytes in production but bytearray in some
            # test harnesses, so normalize before the dict lookup -- a plain
            # bytearray key would raise (unhashable) rather than just miss.
            self._elite_pool = heapq.nlargest(
                pool_size,
                corpus,
                key=lambda s: seed_meta.get(bytes(s), {}).get("coverage_edges", 0),
            )
            self._elite_pool_corpus_len = len(corpus)
        return self._elite_pool

    def _op_elite_fuse(self, buf, _byte_idx, data):
        """Fuse the two highest-coverage corpus seeds into a hybrid third seed.

        Unlike splice/splice_diff_located, which draw both parents uniformly
        at random from the whole corpus, this ranks seeds by
        seed_meta['coverage_edges'] and draws both parents from the top of
        that ranking. The bet: two inputs that already exercise a lot of
        distinct edges are more likely to combine into something that trips
        a *third* combination of edges than pairing an elite seed with a
        mediocre one, or splicing at random.

        Reuses splice_diff_located's diff-located cut points rather than a
        blind random cut, since fusing at a point the two parents actually
        differ produces a more structurally coherent hybrid than an
        arbitrary offset -- particularly relevant here since the two
        highest-coverage seeds are also the likeliest to share a lot of
        common structure (e.g. shared headers).
        """
        rng = self.f._rand_pool
        pool = self.elite_seeds()
        if len(pool) < 2:
            return None
        a, b = rng.sample(pool, 2)
        if a == b:
            return None
        return bytearray(splice_diff_located(bytes(a), bytes(b), rng=rng)[: self.f.max_len])

    def _op_havoc(self, buf, _byte_idx, data):
        """Havoc mutation with deterministic dedup: retry if fully redundant."""
        original = bytes(buf)
        self.havoc_mutate(buf)
        result = bytes(buf)
        if len(result) == len(original) and self._is_deterministically_redundant(original, result):
            # Fully redundant — apply one more random mutation
            self._apply_single_mutation(buf)
            result = bytes(buf)
        return result

    # ── Dispatch table: op name → handler method ───────────────────────
    def build_dispatch(self):
        return REGISTRY.dispatch(self)

    def havoc_mutate(self, buf: bytearray) -> bytearray:
        """Apply 2-8 random mutations (scaled up during stall recovery).

        During normal operation: 2-8 mutations.
        During stall recovery: 8-16 mutations (honggfuzz-style escalation).
        """
        rng = self.f._rand_pool
        if self.f._stall_recovery_active:
            n = rng.randint_list(8, 16, 1)[0]
        else:
            n = rng.randint_list(2, 8, 1)[0]
        for _ in range(n):
            self._apply_single_mutation(buf)
        # Defense in depth: every _apply_single_mutation branch that can grow
        # buf is individually bounds-checked against max_len, but a single
        # missed edge case here has bitten this codebase before (see the
        # swap-regions out-of-bounds slice above) and havoc runs 2-16 times
        # per call, so a per-iteration slip compounds. Every other operator
        # in this module clamps its result to max_len; havoc should too.
        if len(buf) > self.f.max_len:
            del buf[self.f.max_len :]
        return buf

    @staticmethod
    def _is_deterministically_redundant(original: bytes, candidate: bytes) -> bool:
        """Check if all byte diffs are covered by deterministic stages.

        Uses AFL++ duplicate-elimination helpers. If every differing byte
        could be produced by a bitflip, arithmetic, or interest-value
        mutation, the candidate is redundant with deterministic stages.
        """
        if len(original) != len(candidate):
            return False
        for a, b in zip(original, candidate, strict=True):
            if a == b:
                continue
            xor_val = a ^ b
            if could_be_bitflip(xor_val):
                continue
            if could_be_arith(a, b, 1):
                continue
            if could_be_interest(a, b, 1):
                continue
            return False
        return True

    def _apply_single_mutation(self, buf: bytearray):
        rng = self.f._rand_pool
        if not buf:
            buf.extend(rng.randint(0, 255) for _ in range(rng.randint(1, 16)))
            return
        # Pre-fetch 4 random values in one vectorized call.
        # Each branch uses 2-4 values from this batch, avoiding N
        # individual randint/randrange Python calls.
        r = self.f._rand_pool.randint_list(0, 1 << 30, 4)
        f = self.f
        if f._adaptive_havoc:
            # Same draw (r[0]) as the uniform path, so RNG consumption per
            # sub-mutation is unchanged and seeded runs stay comparable.
            op = self._havoc_table[r[0] & 255]
            self._havoc_trials[op] += 1
            f._last_havoc_subops |= _HAVOC_BITS[op]
        else:
            op = r[0] % _HAVOC_N
        if op == 0:  # bit flip
            buf[r[1] % len(buf)] ^= 1 << (r[2] % 8)
        elif op == 1:  # byte set
            buf[r[1] % len(buf)] = r[2] % 256
        elif op == 2 and len(buf) > 1:  # byte swap
            i = r[1] % len(buf)
            j = (i + 1 + r[2] % (len(buf) - 1)) % len(buf)
            buf[i], buf[j] = buf[j], buf[i]
        elif op == 3 and len(buf) < self.f.max_len:  # insert byte
            idx = r[1] % (len(buf) + 1)
            buf.insert(idx, r[2] % 256)
        elif op == 4 and len(buf) > 1:  # delete block
            idx = r[1] % len(buf)
            size = 1 + r[2] % min(len(buf) - 1, len(buf) - idx)
            del buf[idx : idx + size]
        elif op == 5 and len(buf) >= 4:  # CRC32 repair
            pos = r[1] % max(1, len(buf) - 3)
            buf[pos : pos + 4] = crc32(bytes(buf[:pos])).to_bytes(4, "big")
        elif op == 6 and len(buf) >= 2:  # swap regions
            i = r[1] % (len(buf) - 1)
            j = i + 1 + r[2] % (len(buf) - i - 1)
            # size must also respect len(buf) - j, or buf[j : j + size] comes
            # back shorter than `size` and the slice assignments below change
            # buf's length instead of swapping two equal-sized regions.
            size = 1 + r[3] % min(j - i, 8, len(buf) - j)
            a = buf[i : i + size]
            b = buf[j : j + size]
            buf[i : i + size] = b
            buf[j : j + size] = a
        elif op == 7 and buf:  # endianness swap
            width = 2 if r[1] % 2 == 0 else 4
            if len(buf) >= width:
                idx = r[2] % (len(buf) - width + 1)
                val = int.from_bytes(buf[idx : idx + width], "little")
                buf[idx : idx + width] = val.to_bytes(width, "big")
        elif op == 8 and buf:  # byte insert
            if len(buf) < self.f.max_len:
                idx = r[1] % (len(buf) + 1)
                buf.insert(idx, r[2] % 256)
        elif op == 9 and buf:  # random byte
            idx = r[1] % len(buf)
            buf[idx] = r[2] % 256
        elif op == 10 and len(buf) >= 2:  # shuffle range
            start = r[1] % (len(buf) - 1)
            end = min(start + 2 + r[2] % 7, len(buf))
            region = buf[start:end]
            rng.shuffle(region)
            buf[start:end] = region

    def _rebuild_havoc_table(self) -> None:
        """Rebuild the havoc sub-mutation inverse-CDF table from hit ratios.

        Weights are ``hits / trials`` per branch, normalized and mixed with
        the uniform distribution at ``_HAVOC_EXPLORE`` so no branch is ever
        starved to zero. Called every ``_HAVOC_TABLE_REFRESH`` draws, not
        per update -- see the module comment on the hot path.
        """
        self._havoc_rounds_since_rebuild = 0
        hits = self._havoc_hits
        trials = self._havoc_trials
        n = _HAVOC_N
        if max(trials) > _HAVOC_DECAY_AT:
            # Halve in place: preserves every ratio, keeps the window moving.
            # Trials stay >= 1 (the prior is 2), so no ratio divides by zero.
            for i in range(n):
                hits[i] = max(1, hits[i] >> 1)
                trials[i] = max(2, trials[i] >> 1)
        weights = [hits[i] / trials[i] for i in range(n)]
        total = sum(weights)  # > 0: Laplace priors keep every weight positive
        floor = _HAVOC_EXPLORE / n
        scale = (1.0 - _HAVOC_EXPLORE) / total
        table = self._havoc_table
        slots = _HAVOC_TABLE_SLOTS
        acc = 0.0
        idx = 0
        last = n - 1
        for i in range(n):
            acc += weights[i] * scale + floor
            # The floor guarantees each branch >= _HAVOC_EXPLORE/n of the
            # table (3 of 256 slots), so no branch rounds away to zero.
            end = slots if i == last else min(slots, int(acc * slots + 0.5))
            while idx < end:
                table[idx] = i
                idx += 1

    def credit_havoc_subops(self, mask: int) -> None:
        """Record a successful execution against the havoc branches it used.

        ``mask`` is the per-round bitmask accumulated in
        ``_apply_single_mutation``; bit *i* means ``HAVOC_SUB_OPS[i]`` was
        applied at least once this round. Called from
        ``Fuzzer._record_outcome`` because havoc mutates long before the
        coverage verdict for that input exists.

        Rebuilds the sampling table immediately: new hits are the only thing
        that can promote a branch, they are rare relative to draws, and this
        is off the hot path.
        """
        hits = self._havoc_hits
        i = 0
        while mask:
            if mask & 1:
                hits[i] += 1
            mask >>= 1
            i += 1
        self._rebuild_havoc_table()

    def havoc_stats(self) -> list[tuple[str, float, int]]:
        """Per-branch (name, hit ratio, trials) sorted by ratio, best first."""
        hits = self._havoc_hits
        trials = self._havoc_trials
        rows = [(HAVOC_SUB_OPS[i], hits[i] / trials[i], trials[i]) for i in range(_HAVOC_N)]
        rows.sort(key=lambda row: row[1], reverse=True)
        return rows

    # ── Operator selection logic ───────────────────────────────────────

    def build_ops(self, data: bytes) -> list[str]:
        """Build the list of available mutation operators from the registry."""
        return REGISTRY.available(self.f, data)

    @staticmethod
    def _classify_format(data: bytes) -> str:
        """Coarse format bucket for the contextual bandit's one-hot feature."""
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        if data[:2] == b"\xff\xd8":
            return "jpeg"
        if data[:2] == b"\x1f\x8b" or (len(data) > 1 and data[0] == 0x78):
            return "compressed"
        if (
            data[:2] == b"BM"
            or data[:3] == b"GIF"
            or data[:4] == b"RIFF"
            or data[:4] == b"\x1a\x45\xdf\xa3"
        ):
            return "image_other"
        if data[:2] == b"PK" or data[4:8] == b"ftyp":
            return "archive"
        if len(data) >= 64 and data[:4] == b"\x7fELF":
            return "elf"
        return "other"

    def _build_shared_context(self, data: bytes) -> list[float]:
        """Build the seed-level context feature vector for the LinUCB bandit.

        Computed once per mutate() call (the seed doesn't change across the
        n_mutations loop within one round) and cached on the fuzzer instance
        as ``f._current_context_shared``; per-op vectors are derived from it
        cheaply in ``_context_vector()`` rather than rebuilt from scratch.

        Features: log seed size, byte entropy, edge-coverage fraction,
        lineage depth, whether cmplog pairs exist, position in the corpus
        size distribution, format one-hot. All normalized to roughly [0, 1]
        so no single feature dominates the ridge regression.
        """
        f = self.f
        max_len = max(getattr(f, "max_len", 1), 1)
        log_size = math.log1p(len(data)) / math.log1p(max_len)

        if data:
            counts = [0] * 256
            for byte in data:
                counts[byte] += 1
            n = len(data)
            entropy = 0.0
            for c in counts:
                if c:
                    p = c / n
                    entropy -= p * math.log2(p)
            entropy_norm = entropy / 8.0
        else:
            entropy_norm = 0.0

        meta = f.seed_meta.get(data) if hasattr(f, "seed_meta") else None
        edge_tracker = getattr(f, "_edge_tracker", None)
        map_size = getattr(edge_tracker, "map_size", 0) if edge_tracker else 0
        edge_count = meta.get("coverage_edges", 0) if meta else 0
        edge_frac = min(edge_count / map_size, 1.0) if map_size else 0.0

        lineage_depth = meta.get("lineage_depth", 0) if meta else 0
        lineage_norm = min(lineage_depth / 20.0, 1.0)

        cmplog = getattr(f, "_cmplog", None)
        cmplog_exists = 1.0 if (cmplog and getattr(cmplog, "pairs", None)) else 0.0

        # Corpus-size percentile: logistic approximation of the CDF from
        # running mean/stddev, updated incrementally in
        # corpus_manager.save_to_corpus(). Avoids sorting the whole corpus
        # (which can be tens of thousands of seeds) on every mutation.
        stats = getattr(f, "_corpus_size_stats", None)
        if stats is not None and stats.count >= 5 and stats.stddev > 1e-9:
            z = (len(data) - stats.mean) / stats.stddev
            pctile = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))
        else:
            pctile = 0.5

        fmt = self._classify_format(data)
        fmt_onehot = [1.0 if fmt == cat else 0.0 for cat in _CONTEXT_FORMAT_CATEGORIES]

        return [
            log_size,
            entropy_norm,
            edge_frac,
            lineage_norm,
            cmplog_exists,
            pctile,
            *fmt_onehot,
        ]

    def _op_cost_feature(self, op: str) -> float:
        """Per-arm cost feature: log10(seconds) squashed to roughly [0, 1].

        1us -> ~0.14, 1ms -> ~0.43, 1s -> ~0.86, 10s -> 1.0. Neutral 0.5
        default when the op hasn't been timed yet (see _op_time_ema, fed by
        the per-call timing in mutate()). Lets the ridge regression learn
        seed-dependent cost sensitivity directly, on top of the reward
        already being cost-adjusted by _cost_adjusted_weight().
        """
        f = self.f
        cost = f._op_time_ema.get(op) if hasattr(f, "_op_time_ema") else None
        if cost is None or cost <= 0.0:
            return 0.5
        log_cost = math.log10(cost + 1e-6)
        return max(0.0, min(1.0, (log_cost + 6.0) / 7.0))

    def _context_vector(self, op: str) -> list[float]:
        """Full per-arm context: cached shared seed features + this op's cost."""
        f = self.f
        shared = getattr(f, "_current_context_shared", None) or [0.0] * (CONTEXT_DIM - 1)
        return [*shared, self._op_cost_feature(op)]

    def select_op(self, ops: list[str]) -> str:
        """Select a mutation operator using the active scheduling strategy."""
        f = self.f

        if f._stall_recovery_active:
            f._meta_strategy = "random_stall"
            return f._rand_pool.choice(ops)

        available = []
        if f._use_replicator and f._replicator:
            available.append("replicator")
        if f.mc and f.mc_bandit:
            available.append("bandit")
        if f._use_mopt and f._mopt:
            available.append("mopt")
        if f.mc and f.mc_cem and f.mc.cem_fitted:
            available.append("cem")
        if f._use_exp3 and f._exp3:
            available.append("exp3")
        if f._use_eps_greedy and f._eps_greedy:
            available.append("eps_greedy")
        if f._use_hierarchical and f._hierarchical:
            available.append("hierarchical")
        if f._use_gp_ucb and f._gp_ucb:
            available.append("gp_ucb")
        # cmaes was missing from this list while `strategy == "cmaes"` had a
        # dispatch branch below and `_use_cmaes` had a branch in the no-Elo
        # fallback chain. The effect was not a preference, it was a
        # disappearance: with --elo on and any other scheduler enabled, Elo
        # picked a strategy from this list, the chain matched that strategy's
        # branch, and the fallback chain -- the only place cmaes could ever
        # be reached -- was never evaluated. `--cma-es --elo` ran CMA-ES that
        # was arm-registered and fed record() on every outcome, and let it
        # select nothing at all. Its dispatch branch was dead code.
        if f._use_cmaes and f._cmaes:
            available.append("cmaes")
        if f._use_contextual and f._contextual:
            available.append("contextual")
        if f._use_ducb and f._ducb:
            available.append("ducb")
        if f._use_swucb and f._swucb:
            available.append("swucb")
        if f._use_cucb and f._cucb:
            available.append("cucb")

        if f._use_elo and f._elo and len(available) >= 2:
            # Resolve the meta-strategy once per exec and reuse it for all
            # mutations within it. Elo ratings move once per mutation, so
            # re-sampling the strategy on every select_op call is
            # over-frequent; mutate() resets _meta_strategy_cached each exec.
            strategy = f._meta_strategy_cached
            if strategy is None or strategy not in available:
                strategy = f._elo.select_strategy(available)
                f._meta_strategy_cached = strategy
            f._meta_strategy = strategy
        elif f._use_elo and f._elo and available:
            strategy = available[0]
            f._meta_strategy = strategy
        else:
            strategy = None

        if f._use_elo and f._elo and strategy:
            f._meta_strategy_used.add(strategy)

        if strategy == "replicator" and f._replicator:
            op = f._replicator.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "mopt" and f._mopt:
            op, pid = f._mopt.select_op(ops)
            f._last_mopt_particles.append(pid)
        elif strategy == "bandit" and f.mc and f.mc_bandit:
            op = f.mc.select_op(ops, prev_op=f._prev_bandit_op)
            f._prev_bandit_op = op
            f._last_mopt_particles.append(None)
        elif strategy == "cem" and f.mc and f.mc_cem:
            op = (
                f.mc.select_op(ops, prev_op=f._prev_bandit_op)
                if f.mc_bandit
                else f._rand_pool.choice(ops)
            )
            if f.mc_bandit:
                f._prev_bandit_op = op
            f._last_mopt_particles.append(None)
        elif strategy == "exp3" and f._exp3:
            op = f._exp3.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "eps_greedy" and f._eps_greedy:
            op = f._eps_greedy.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "hierarchical" and f._hierarchical:
            op = f._hierarchical.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "gp_ucb" and f._gp_ucb:
            op = f._gp_ucb.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "cmaes" and f._cmaes:
            op = f._cmaes.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "contextual" and f._contextual:
            op = f._contextual.select_op(ops, self._context_vector)
            f._last_mopt_particles.append(None)
        elif strategy == "ducb" and f._ducb:
            op = f._ducb.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "swucb" and f._swucb:
            op = f._swucb.select_op(ops)
            f._last_mopt_particles.append(None)
        elif strategy == "cucb" and f._cucb:
            op = f._cucb.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_replicator and f._replicator:
            op = f._replicator.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_mopt and f._mopt:
            op, pid = f._mopt.select_op(ops)
            f._last_mopt_particles.append(pid)
        elif f.mc and f.mc_bandit:
            op = f.mc.select_op(ops, prev_op=f._prev_bandit_op)
            f._prev_bandit_op = op
            f._last_mopt_particles.append(None)
        elif f._use_exp3 and f._exp3:
            op = f._exp3.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_eps_greedy and f._eps_greedy:
            op = f._eps_greedy.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_hierarchical and f._hierarchical:
            op = f._hierarchical.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_gp_ucb and f._gp_ucb:
            op = f._gp_ucb.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_cmaes and f._cmaes:
            op = f._cmaes.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_contextual and f._contextual:
            op = f._contextual.select_op(ops, self._context_vector)
            f._last_mopt_particles.append(None)
        elif f._use_ducb and f._ducb:
            op = f._ducb.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_swucb and f._swucb:
            op = f._swucb.select_op(ops)
            f._last_mopt_particles.append(None)
        elif f._use_cucb and f._cucb:
            op = f._cucb.select_op(ops)
            f._last_mopt_particles.append(None)
        else:
            op = f._rand_pool.choice(ops)
            f._last_mopt_particles.append(None)
        return op

    def region_weights(self, data: bytes):
        """Cumulative region weights for *data*, cached by content hash.

        Returns ``(cumulative_weights, bounds, total)`` or None when the seed
        is too short to profile or every region weighs zero. The profile is
        what ``randomness.profile_buffer`` recommends in its own docstring:
        computed once per seed and reused for every mutation round against it.
        """
        if len(data) < _REGION_MIN_LEN:
            return None
        key = xxhash.xxh3_64_intdigest(data)
        if key in self._region_cache:
            return self._region_cache[key]

        from fuzzer_tool.core.randomness import profile_buffer

        entry = None
        profiles = profile_buffer(data)
        if profiles:
            bounds = []
            cumulative = []
            total = 0.0
            for profile in profiles:
                total += profile.mutation_weight() * profile.length
                cumulative.append(total)
                bounds.append((profile.offset, profile.offset + profile.length))
            if total > 0.0:
                entry = (cumulative, bounds, total)
        if len(self._region_cache) >= _REGION_CACHE_MAX:
            self._region_cache.clear()
            # Evicted together: a liveness list surviving past its region
            # bounds/cumulative arrays would silently misattribute the next
            # seed's diffs to a stale region layout.
            self._region_liveness.clear()
        self._region_cache[key] = entry
        return entry

    def record_coverage_diff(
        self,
        data: bytes,
        offset: int,
        baseline_edges: set,
        mutant_edges: set,
        map_size: int = _LIVENESS_MAP_BITS,
    ) -> tuple[int, int] | None:
        entry = self.region_weights(data)
        """Fold one mutation's coverage-edge diff into the region liveness
        estimator for whichever region *offset* falls in.

        Item 4 (`core/live_bit_mask.py`) wiring, per
        `docs/handover_skittercreek_tailslayer_port.md`: `baseline_edges`
        is the parent seed's known edge set (e.g.
        `edge_tracker.seed_edges[seed_key]`), `mutant_edges` is the edge
        set from executing the mutant derived from it, and `offset` is the
        byte position the mutation touched (`f._last_mutation_offset`).

        The symmetric difference of the two edge sets is folded into a
        `map_size`-bit mask via `edge_id % map_size` and handed to
        `LiveBitMaskEstimator.observe(0, diff_bits)` -- passing 0 as the
        baseline is exact, not an approximation: `observe` only ever uses
        `baseline ^ mutant`, so `0 ^ diff_bits == diff_bits`, and this
        avoids materializing two full-width bitmasks per call when only
        their difference is needed. Cost is O(|symmetric difference|), not
        O(map_size) or O(|edges|) -- the point of this module.

        Returns the region's `(offset, width)` bounds iff this call is the
        one that pushed the region's estimator from not-converged to
        converged-with-an-empty-mask -- i.e. the edge-triggered moment a
        caller (see `fuzzer.py`'s exec loop) should forward to
        `FormatLearner.record_liveness()` as padding-corroborating
        evidence. Returns `None` on every other call, including "already
        was converged-dead" (report the transition once, not every
        subsequent no-growth sample) and "converged but mask is nonzero"
        (that's a confirmed-*live* verdict, not dead).

        A no-op (returns `None`) if *data* is too short to have a region
        profile (nothing to attribute the diff to) or the offset falls
        outside every known region (can happen if a later operator in the
        same round resized the buffer past the profiled seed's length).

        NOTE (validation status): the convergence threshold this feeds
        (`_LIVENESS_SWITCH_AFTER`) is checked only against the synthetic
        sweep in `tests/test_live_bit_mask.py`. Per the handover doc's
        Sequencing step 6, a real-corpus sweep is still open; the
        conservative `_LIVENESS_DEAD_WEIGHT` down-weight (not a hard
        exclusion) is deliberately chosen to bound the damage if that
        sweep finds the synthetic threshold doesn't transfer.
        """
        entry = self.region_weights(data)
        if entry is None:
            return None
        _cumulative, bounds, _total = entry

        region_idx = None
        for i, (lo, hi) in enumerate(bounds):
            if lo <= offset < hi:
                region_idx = i
                break
        if region_idx is None:
            return None

        key = xxhash.xxh3_64_intdigest(data)
        estimators = self._region_liveness.setdefault(key, [None] * len(bounds))

        diff_edges = baseline_edges ^ mutant_edges
        if not diff_edges:
            diff_bits = 0
        else:
            diff_bits = 0
            for edge_id in diff_edges:
                diff_bits |= 1 << (edge_id % map_size)

        # Temporary env-gated instrumentation for the item 4 real-corpus
        # sensitivity sweep. Logs every (region, diff_bits) observation the
        # production estimator consumes, using the same sparse TSV format as
        # the round-9/10 sweep data. Zero-cost when FUZZER_LIVENESS_LOG is unset.
        _liveness_log = os.getenv("FUZZER_LIVENESS_LOG")
        if _liveness_log and region_idx is not None:
            if diff_bits == 0:
                _line = f"{region_idx}\t\n"
            else:
                _bits = sorted({e % map_size for e in diff_edges})
                _line = f"{region_idx}\t{','.join(map(str, _bits))}\n"
            try:
                with open(_liveness_log, "a") as _lf:
                    _lf.write(_line)
            except OSError:
                pass

        est = estimators[region_idx]
        if est is None:
            est = LiveBitMaskEstimator(n_bits=map_size, switch_after=_LIVENESS_SWITCH_AFTER)
            estimators[region_idx] = est
        was_converged_dead = est.is_converged and est.mask == 0
        est.observe(0, diff_bits)
        now_converged_dead = est.is_converged and est.mask == 0
        if now_converged_dead and not was_converged_dead:
            lo, hi = bounds[region_idx]
            return (lo, hi - lo)
        return None

    def _region_liveness_factor(self, data: bytes, region_idx: int) -> float:
        """Down-weight factor for region *region_idx* of *data*.

        Returns `_LIVENESS_DEAD_WEIGHT` iff that region's estimator has
        converged with an empty mask (confirmed-dead, per the module's
        own fail-closed discipline: convergence is required, not just an
        empty mask so far -- see `LiveBitMaskEstimator`'s docstring on why
        a not-yet-converged empty mask is unresolved, not a negative
        claim). Returns 1.0 (no-op) in every other case, including "no
        estimator recorded for this region yet" -- absence of an
        estimator is not evidence of anything.
        """
        key = xxhash.xxh3_64_intdigest(data)
        estimators = self._region_liveness.get(key)
        if not estimators or region_idx >= len(estimators):
            return 1.0
        est = estimators[region_idx]
        if est is None:
            return 1.0
        if est.is_converged and est.mask == 0:
            return _LIVENESS_DEAD_WEIGHT
        return 1.0

    def _region_weighted_position(self, data: bytes, buf_len: int) -> int | None:
        """Draw a byte offset weighted by each region's mutation_weight(),
        further down-weighted by observed coverage liveness (item 4).

        Regions are picked proportionally to
        `mutation_weight() * length * liveness_factor`, then a uniform
        offset is taken inside the winner. The bounds come from the *seed*,
        while the position must land in the *buffer*, which earlier operators
        in the same round may already have resized -- hence the clamp.

        The static `mutation_weight() * length` term is cached in
        region_weights() since it depends only on seed content. Liveness is
        dynamic -- it strengthens as record_coverage_diff() accumulates more
        mutation outcomes for this seed -- so it's re-applied fresh on every
        draw rather than baked into that cache; this is cheap (one pass over
        a handful of regions, not the profile_buffer() battery the cache
        actually saves the cost of).
        """
        entry = self.region_weights(data)
        if entry is None:
            return None
        cumulative, bounds, total = entry
        estimators = self._region_liveness.get(xxhash.xxh3_64_intdigest(data))
        if not estimators or not any(
            e is not None and e.is_converged and e.mask == 0 for e in estimators
        ):
            # Fast path: no region has confirmed-dead liveness data yet,
            # so the cached cumulative/total (mutation_weight-only) is
            # already correct -- skip rebuilding it.
            idx = bisect.bisect_left(cumulative, self.f._rand_pool.random() * total)
        else:
            adjusted_cumulative = []
            adjusted_total = 0.0
            prev = 0.0
            for i, c in enumerate(cumulative):
                region_weight = c - prev
                prev = c
                adjusted_total += region_weight * self._region_liveness_factor(data, i)
                adjusted_cumulative.append(adjusted_total)
            if adjusted_total <= 0.0:
                return None
            idx = bisect.bisect_left(
                adjusted_cumulative, self.f._rand_pool.random() * adjusted_total
            )
        lo, hi = bounds[min(idx, len(bounds) - 1)]
        hi = min(hi, buf_len)
        if lo >= hi:
            return None
        return self.f._rand_pool.randint(lo, hi - 1)

    # ── Deterministic stage ──────────────────────────────────────────────

    def _next_deterministic_mutation(self, data: bytes, seed_key: str) -> bytes | None:
        """Pop the next mutant from *seed_key*'s deterministic queue.

        Creates the queue on first call for a given seed_key. Returns None
        once the schedule is exhausted, at which point the queue entry is
        dropped -- the caller is expected to mark the seed's metadata
        ``seed_passed_det`` so should_det_fuzz is not re-consulted for it.
        """
        q = self._det_queues.get(seed_key)
        if q is None:
            q = _deterministic_mutation_stream(bytes(data))
            self._det_queues[seed_key] = q
        try:
            return next(q)
        except StopIteration:
            del self._det_queues[seed_key]
            return None

    def maybe_deterministic_mutation(self, data: bytes) -> bytes | None:
        """Return the next deterministic-stage mutant for *data*, or None.

        None means: no deterministic stage is configured (f._skip_detector
        is None), the seed already passed one, or SkipDetector's
        undetermined-bit gate said to skip it -- in every case mutate()
        should fall through to its normal bandit-driven mutation instead.

        Routed through mutate() rather than run as a separate blocking loop:
        the mutant this returns goes through the exact same
        execution/coverage/corpus-save path fuzz_one already gives every
        mutation, so a deterministic-stage discovery gets queued into the
        corpus like any other -- a standalone loop that runs its own execs
        and updates edge-tracking bookkeeping without ever calling
        save_to_corpus would find new coverage and then throw the mutant
        that found it away.

        The should_det_fuzz gate is consulted exactly once per seed, when
        its queue is first created, not on every mutation drawn from an
        already-running queue: should_det_fuzz has side effects (it marks
        bits as deterministically explored and advances a decay timer), and
        re-running it every call would both re-decide a question already
        answered and corrupt those side effects with a seed's own
        in-progress coverage.
        """
        f = self.f
        skip_detector = getattr(f, "_skip_detector", None)
        if skip_detector is None:
            return None
        # seed_meta is keyed by the raw seed bytes (see corpus_manager.py's
        # init_seed_metadata / fuzz_one's `self.seed_meta.get(data)`) --
        # seed_key is a separate hash-string identity used only by
        # _favored and _edge_tracker.seed_edges. Mixing the two up here
        # would make meta always None and silently disable this gate.
        meta = f.seed_meta.get(data)
        if meta is None or meta.get("seed_passed_det", False):
            return None
        seed_key = f._seed_key(data)

        if seed_key in self._det_queues:
            mutant = self._next_deterministic_mutation(data, seed_key)
            if mutant is None:
                meta["seed_passed_det"] = True
            return mutant

        favored = seed_key in f._favored
        edge_ids = f._edge_tracker.seed_edges.get(seed_key)
        trace_mini = trace_mini_from_edges(edge_ids, skip_detector.map_size) if edge_ids else None
        should_run = skip_detector.should_det_fuzz(
            seed_trace_mini=trace_mini,
            seed_favored=favored,
            seed_passed_det=False,
            current_time_ms=time.monotonic() * 1000.0,
        )
        if not should_run:
            # AFL marks a skipped seed as "done" too -- otherwise the same
            # not-enough-new-bits seed gets re-evaluated (and re-skipped)
            # on every single mutate() call for as long as it stays favored.
            meta["seed_passed_det"] = True
            return None

        mutant = self._next_deterministic_mutation(data, seed_key)
        if mutant is None:
            meta["seed_passed_det"] = True
        return mutant

    def select_position(self, buf: bytearray, data: bytes) -> int:
        """Select a byte position for mutation using MI/TE/sensitivity/crash-MI/random."""
        f = self.f
        if not buf:
            return 0
        buf_len = len(buf)
        te_pos = f._get_te_weighted_position(buf_len) if f._use_transfer_entropy and f._te else None
        mi_pos = f._mi.weighted_position(buf_len) if f._use_mi and f._mi else None
        # Sensitivity is a per-seed score cache: when disabled the tracker is
        # never populated, so the call would always return None.  Gate it like
        # MI/TE instead of paying the lookup + branches on every mutation.
        sens_pos = (
            f._sensitivity.get_weighted_position(data, buf_len)
            if f._use_sensitivity and f._sensitivity
            else None
        )
        crash_mi_pos = None
        if f._crash_mi and f._crash_mi.total_execs >= f._crash_mi.min_observations:
            crash_mi_pos = f._crash_mi.weighted_position(buf_len)
        region_pos = (
            self._region_weighted_position(data, buf_len)
            if getattr(f, "_use_region_profile", False)
            else None
        )
        candidates = [
            p for p in [sens_pos, te_pos, mi_pos, crash_mi_pos, region_pos] if p is not None
        ]
        if candidates:
            byte_idx = f._rand_pool.choice(candidates)
        else:
            byte_idx = f._rand_pool.randint(0, buf_len - 1)
        if getattr(f, "debug", False):
            print(
                f"[select_position] buf_len={buf_len} sens={sens_pos} te={te_pos} "
                f"mi={mi_pos} crash_mi={crash_mi_pos} region={region_pos} "
                f"candidates={candidates} fallback={not candidates} byte_idx={byte_idx}"
            )
        return byte_idx

    # ── Main mutation orchestrator ─────────────────────────────────────

    def mutate(self, data: bytes) -> bytes:
        from fuzzer_tool.core.similarity import hamming_distance

        f = self.f

        # Deterministic stage: drains before any bandit-driven mutation for
        # a favored, not-yet-determinized seed (see maybe_deterministic_mutation
        # for the gating). Bypasses build_ops()/select_op() entirely -- this
        # is AFL's systematic per-position walk, not a bandit-scored pick,
        # so it doesn't feed the bandits' win/loss bookkeeping. Leaving
        # _last_ops_used empty means every scheduler's "if not
        # self._last_ops_used: return" guard no-ops for this round, same as
        # if nothing had run -- the deterministic pass simply isn't part of
        # that tournament.
        det_mutant = self.maybe_deterministic_mutation(data)
        if det_mutant is not None:
            f._last_ops_used = []
            f._last_ops_with_sites = []
            f._last_mopt_particles = []
            f._last_ops_effective = set()
            f._last_ops_applicable = set()
            f._last_havoc_subops = 0
            f._last_op_costs = {}
            f._last_mutation_offset = 0
            f._last_hamming_distance = (
                hamming_distance(data, det_mutant) if len(data) == len(det_mutant) else -1
            )
            f._det_execs = getattr(f, "_det_execs", 0) + 1
            return det_mutant

        buf = bytearray(data)
        if not buf:
            buf = bytearray(b"\x00" * f._rand_pool.randint_list(1, 32, 1)[0])

        ops = self.build_ops(data)
        f._last_ops_used = []
        f._last_ops_with_sites = []
        f._last_mopt_particles = []
        # Operators that actually changed the buffer this round. An operator
        # can be selected and still be a no-op: its guard fails (swap_bytes on
        # a 1-byte input), or it needs corpus state it doesn't have (splice
        # and crossover with a single-seed corpus are no-ops 100% of the time,
        # measured). Without this set every scheduler credits a no-op operator
        # exactly as much as the operator that did the work -- see
        # fuzzer.py::_record_outcome.
        f._last_ops_effective = set()
        # Ops that were handed input their own format sniffer matched, i.e.
        # that ran in their mutate-this-file regime rather than their
        # synthesise-one-from-scratch fallback. Recorded here rather than
        # against the round's parent seed because with mutations_per_input
        # > 1 the operators chain: op N sees whatever op N-1 left behind,
        # not the seed, and a png op can perfectly well be handed a PNG
        # that an earlier op in the same round produced.
        f._last_ops_applicable = set()
        # Bitmask of havoc sub-mutations applied this round (bit i ->
        # HAVOC_SUB_OPS[i]). A bitmask rather than a list because
        # _apply_single_mutation runs 2-16 times per havoc selection and an
        # OR against an int is the cheapest per-application record available.
        f._last_havoc_subops = 0
        # Table refresh lives here rather than in _apply_single_mutation:
        # once per round instead of 2-16 times, for a distribution that only
        # changes when trials accumulate or a hit lands (which rebuilds
        # directly -- see credit_havoc_subops).
        if f._adaptive_havoc:
            self._havoc_rounds_since_rebuild += 1
            if self._havoc_rounds_since_rebuild >= _HAVOC_TABLE_REFRESH:
                self._rebuild_havoc_table()
        # Per-round wall-clock cost per operator, keyed by op name, summed
        # across repeats within this round. Feeds the cost-aware reward:
        # a 10ms operator and a 2us operator shouldn't be scored on the
        # same win/loss scale (see fuzzer.py::_cost_adjusted_weight).
        f._last_op_costs = {}
        # Shared LinUCB context for this round: the seed doesn't change
        # across the n_mutations loop, so build it once here instead of
        # once per select_op() call.
        f._current_context_shared = (
            self._build_shared_context(data) if f._use_contextual and f._contextual else None
        )
        if not hasattr(f, "_prev_bandit_op"):
            f._prev_bandit_op = None
        f._meta_strategy = None
        f._meta_strategy_cached = None  # reset per-exec elo strategy cache

        n_mutations = f.mutations_per_input
        # Apply seed-level energy multiplier from SeedScorer
        if hasattr(f, "_last_perf_score") and f._last_perf_score != 100.0:
            n_mutations = max(1, int(n_mutations * f._last_perf_score / 100.0))
        if f._stall_recovery_active:
            n_mutations = max(n_mutations, 16)

        # Pre-fetch dictionary indices for dict-aware operators in one
        # vectorized call, replacing N individual random.choice(f.dictionary)
        # calls across _op_dict_* methods.
        if f.dictionary:
            f._dict_scratch = f._rand_pool.randint_list(
                0, len(f.dictionary) - 1, max(n_mutations * 8, 64)
            )
            f._dict_scratch_idx = 0

        # Pre-generate buffer lengths for select_position fallback.
        # select_position is called once per mutation, and when no
        # MI/TE/sensitivity is active it falls back to randrange(len(buf)).
        # Pre-computing positions doesn't work here because len(buf) changes
        # during mutation, so we handle this per-call below.

        track_effect = f._track_op_effect

        for _ in range(n_mutations):
            op = self.select_op(ops)
            f._last_ops_used.append(op)

            byte_idx = self.select_position(buf, data)
            f._last_mutation_offset = byte_idx
            f._last_ops_with_sites.append((op, byte_idx))
            old_len = len(buf)

            # Digest over a memoryview: no copy, ~3.4us at 64KiB. Only paid
            # when a scheduler consumes the signal (_track_op_effect).
            _h_before = xxhash.xxh3_64_intdigest(memoryview(buf)) if track_effect else 0

            # Evaluated on the buffer actually handed to the operator, and
            # before the call, because the operator may replace it. Returns
            # None for the ~85% of ops that are not sniffer-gated, which is
            # a dict miss and nothing more.
            # `is not False`, not a truth test: format_gate_matches returns
            # None for an operator that is not sniffer-gated, and None is
            # falsy. Testing truthiness dropped every ungated operator out
            # of the applicable set, so the ~85% of the table that has only
            # one regime reported Applic 0 and RateA n/a.
            if format_gate_matches(op, buf) is not False:
                f._last_ops_applicable.add(op)

            _t0 = time.perf_counter()
            result = f._op_dispatch[op](buf, byte_idx, data)
            _dt = time.perf_counter() - _t0

            if track_effect:
                # None means the handler mutated `buf` in place (the dominant
                # convention here); anything else replaces the buffer.
                _after = buf if result is None else result
                if xxhash.xxh3_64_intdigest(memoryview(_after)) != _h_before:
                    f._last_ops_effective.add(op)
            f._last_op_costs[op] = f._last_op_costs.get(op, 0.0) + _dt
            # EMA of per-call cost, seeded on first observation so a single
            # early sample doesn't get dragged toward zero.
            prev_ema = f._op_time_ema.get(op)
            f._op_time_ema[op] = _dt if prev_ema is None else (0.9 * prev_ema + 0.1 * _dt)

            if result is not None:
                if op == "havoc":
                    # Havoc's internal sub-mutations (2-16 per call, see
                    # havoc_mutate) can each touch a different position, so
                    # its frameshift bookkeeping is a full resync
                    # (apply_to_buffer) rather than the single
                    # on_insert/on_delete pair the other branch below uses
                    # for a single-site op.
                    #
                    # This used to `return result` here immediately,
                    # discarding whatever was left of n_mutations for this
                    # round. Since n_mutations is scaled by
                    # _last_perf_score (the seed-energy multiplier from
                    # SeedScorer), a highly-scored seed that earned extra
                    # mutation budget got none of the extra whenever havoc
                    # was drawn early in the loop -- which is often, since
                    # havoc is the most-drawn operator. Falling through to
                    # the shared loop tail instead (same as every other
                    # operator) makes the energy multiplier actually do
                    # something on havoc rounds; the existing loop-end
                    # hamming-distance computation after the for-loop
                    # already covers the havoc-selected case correctly, so
                    # nothing here needs to duplicate it.
                    buf = result if isinstance(result, bytearray) else bytearray(result)
                    if len(buf) > f.max_len:
                        # _op_havoc's redundant-mutation retry path calls
                        # _apply_single_mutation directly, bypassing
                        # havoc_mutate's own end-of-call clamp -- so this
                        # can't be assumed already true the way it is for
                        # havoc_mutate's normal return.
                        del buf[f.max_len :]
                    if f._frameshift.relations:
                        f._frameshift.apply_to_buffer(buf)
                    continue
                new_len = min(len(result), f.max_len)
                if f._frameshift.relations:
                    if new_len > old_len:
                        f._frameshift.on_insert(byte_idx, new_len - old_len)
                    elif new_len < old_len:
                        f._frameshift.on_delete(byte_idx, old_len - new_len)
                buf = (
                    result[: f.max_len]
                    if isinstance(result, bytearray)
                    else bytearray(result[: f.max_len])
                )
            elif len(buf) > f.max_len:
                # In-place handlers (result is None, the dominant convention
                # here) are individually responsible for staying within
                # max_len, and most do -- but a single missed bounds check
                # in one of them (see _op_swap_regions) previously slipped
                # straight through uncaught, since only the result-replaces-
                # buf branch above was clamped. Enforce the invariant here
                # too, for every in-place op, not just the ones we've
                # already found bugs in.
                del buf[f.max_len :]

        if f._frameshift.relations:
            f._frameshift.apply_to_buffer(buf)

        result = bytes(buf)
        f._last_hamming_distance = (
            hamming_distance(data, result) if len(data) == len(result) else -1
        )
        return result
