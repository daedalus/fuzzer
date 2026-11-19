"""Past fuzzer-tool bugs regenerated with random parameters.

Each mutator rebuilds the input class of one fixed bug and checks an
oracle that the bug violated, so a regression in any variant, not just the
pinned unit-test case, is caught.

| mutator      | bug (regression test)                                         |
|--------------|---------------------------------------------------------------|
| none_offset  | backtest TypeError on delocalised ops                         |
|              |   (test_regression_format_learner_none_offset.py)             |
| mg_flood     | unbounded signature counters under a prefix flood             |
|              |   (test_regression_format_learner_misra_gries.py)             |
| wide_length  | int->str crash on multi-KB `length` fields                    |
|              |   (_length_candidates; docs/TODO.md 2026-09-24)               |
| rp_batch     | batch draws truncated past one pool refill                    |
|              |   (test_regression_rand_pool_oversized_batch.py)              |
| weighted_choice | IndexError when float rounding hits the total              |
|              |   (test_regression_weighted_choice_rounding.py)               |
| delta_roundtrip | v2 delta must rebuild the child; length-gap early out      |
|              |   (test_regression_delta_v2_length_bound.py)                  |
| myers_script | truncated Myers backtrack script                              |
|              |   (test_regression_myers_backtrack.py)                        |
| dict_quotes  | AFL quotes kept, `=` split, `\\x41` mis-decoded               |
|              |   (test_regression_dict_quotes.py)                            |
| tiff_ifd     | IFD offset past the buffer raised struct.error                |
|              |   (test_regression_tiff_ifd_offset_oob.py)                    |
| bloom_roundtrip | false negatives after persist / fold                       |
|              |   (test_regression_bloom_audit.py)                            |
| edge_prune   | prune lost the seed bound / stale owner counts                |
|              |   (test_regression_edge_tracker_prune.py)                     |
| lineage_cycle | parent cycles hung every tree walker                         |
|              |   (test_regression_lineage_cycle.py)                          |
| vu64_roundtrip | vu64 encode/decode round-trip, truncation                   |
|              |   (test_regression_vu64.py)                                   |
| chi2_sparse  | sparse rows (expected < 5) entered the chi2 test              |
|              |   (test_regression_chi2_sparse_rows.py)                       |
| elf_load_segment | phdr table past the image raised                          |
|              |   (test_regression_elf_load_segment_bounds.py)                |
| grammar_escapes | quoted literals: escapes kept verbatim; `\\x+1` as hex      |
|              |   (test_regression_grammar_escapes.py, _hex_escape_strict.py) |
| gf2n_modulus | reducible modulus spun forever (assert-only guard)            |
|              |   (test_regression_primitive_root_bound.py)                   |
| bwt_mtf      | BWT inverse / MTF round-trip, length preservation             |
|              |   (test_regression_bwt_rle_vectorized.py)                     |
| huffman_decode | decode spun on short tails; codes > 16 bits                 |
|              |   (test_regression_huffman_decode.py)                         |
| entropy_codes | gamma rejected N=1; delta scaled values                      |
|              |   (test_regression_entropy_code_tables.py)                    |
| cmp_parse    | CMP fast path must match the strip/split parser               |
|              |   (test_regression_cmp_parse_fastpath.py)                     |
| parser_truncation | short buffers raised struct.error / IndexError           |
|              |   (test_regression_parser_truncation.py)                      |
| minhash_lsh  | stale band entries after re-add / remove                      |
|              |   (test_regression_minhash_stale_buckets.py)                  |
| murmur3_chain | inlined mix/finalizer must stay bit-exact (scalar + numpy)   |
|              |   (test_regression_murmur3_inlined.py)                        |
| xxhash       | XXH32 / XXH64 inlining must match the spec digests            |
|              |   (test_regression_xxh32_iter_unpack.py, _xxh64_inlined.py)   |
| lz_tokenize  | hash-chain LZ parse must equal the naive greedy parse         |
|              |   (test_regression_lz_tokenize_speed.py)                      |
| modal_value  | first-seen pair adopted instead of the plurality              |
|              |   (test_regression_modal_value_plurality.py)                  |
| fold_bits    | numpy edge fold must equal the per-edge OR                    |
|              |   (test_regression_liveness_fold_bits.py)                     |
| log_gamma    | spurious log(2**3.5) term in the Lanczos fallback             |
|              |   (test_regression_log_gamma_constant.py)                     |
| snoob        | snoob_prev ran ~2^63 iterations on a high bit                 |
|              |   (test_regression_hackers_delight.py)                        |
| set_cover    | bitset rewrite must stay an exact, irredundant cover          |
|              |   (test_regression_set_cover_bitset.py)                       |
| grammar_cap  | empty-literal repeat chains grew as R^depth                   |
|              |   (test_regression_grammar_zero_byte_budget.py)               |
| sanitizer_stack | echoed text parsed; alloc stack in frames; XOR hash        |
|              |   (test_regression_sanitizer_parse.py)                        |
| span_relocate | destination shifted by -span (negative, tail-biased)        |
|              |   (test_regression_span_relocate_dest.py)                     |
| parity_lock  | lock table must flip only wrong-parity bytes                  |
|              |   (test_regression_parity_ops.py)                             |
| diverse_copy | single shifted draw must never keep a byte                    |
|              |   (test_regression_colorization_replace_identity.py)          |
| partial_parse | vectorized delimiter scan == per-byte parser                 |
|              |   (test_regression_partial_parse_vectorized.py)               |
| crc_checksum | table / braid CRC paths == bitwise CRC                        |
|              |   (test_regression_checksum_learner.py, _cost_bound.py)       |
| lfsr_recover | Berlekamp-Massey recovery predicts lane 0                     |
|              |   (test_regression_ensure_poly_no_rerecovery.py)              |
| dominators   | CHK stalled on irreducible CFGs; idom == definition           |
|              |   (test_regression_dominators_worst_case.py)                  |
| crash_cluster | LSH pruning under-merged; == all-pairs single linkage        |
|              |   (test_regression_crash_cluster_exact.py)                    |
| cdf_walk     | numpy CDF norms bit-identical to the loop                     |
|              |   (test_regression_cdf_walk_vectorized.py)                    |
| checksum_state | pairs lost across resume; byte budget; junk dropped         |
|              |   (test_regression_checksum_learner_state.py)                 |
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
import pkgutil
import struct
from importlib import import_module
from types import SimpleNamespace
from typing import TYPE_CHECKING

import fuzzer_tool.core.mutations as mutations_pkg
from fuzzer_tool.adapters.filesystem import apply_delta_v2, compute_delta_v2
from fuzzer_tool.adapters.track_parser import cond_tuples_from_cmplog_text
from fuzzer_tool.core import (
    colorization,
    crc_engine,
    edge_tracker,
    gf2_common,
    grammar,
    hashing,
    sanitizer,
    similarity,
    tree_mutator,
)
from fuzzer_tool.core.analyzers import analyzer_checksum_learner as acl
from fuzzer_tool.core.analyzers import analyzer_format_learner as afl
from fuzzer_tool.core.analyzers.analyzer_format_learner import (
    FieldHypothesis,
    FormatCluster,
    FormatLearner,
    TimelineEntry,
)
from fuzzer_tool.core.berlekamp_massey import (
    compute_checksum,
    lfsr_step,
    linear_complexity,
    recover_lfsr,
)
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.cfg import BasicBlock, FunctionCFG
from fuzzer_tool.core.chi_squared import COCHRAN_MIN_EXPECTED, _log_gamma, drop_sparse_rows
from fuzzer_tool.core.crash_metadata import cluster_crashes
from fuzzer_tool.core.dominators import compute_idom
from fuzzer_tool.core.edge_tracker import EdgeTracker, MinHashLSH
from fuzzer_tool.core.elf import find_load_segment
from fuzzer_tool.core.format_seed_generator import generate_seeds
from fuzzer_tool.core.grammar import _SIMPLE_ESCAPES as GRAMMAR_SIMPLE
from fuzzer_tool.core.grammar import decode_quoted_literal
from fuzzer_tool.core.int_checksum_solver import _modal_value
from fuzzer_tool.core.lineage import LineageTree
from fuzzer_tool.core.mutations import generic, lz4, murmur3, parity, structured
from fuzzer_tool.core.mutations.generic import parse_dict_line
from fuzzer_tool.core.mutations.hackers_delight import snoob, snoob_prev
from fuzzer_tool.core.mutations.tiff import TiffMutator, parse_tiff
from fuzzer_tool.core.mutations.vu64 import (
    vu64_decode_bytes,
    vu64_encode_value,
    vu64_encoded_len,
)
from fuzzer_tool.core.rand_pool import _POOL_ENTRIES, RandPool
from fuzzer_tool.core.set_cover import min_cover
from fuzzer_tool.core.similarity import (
    _levenshtein_align_numpy,
    configure_diff_limits,
    crash_signature_similarity,
    frame_sequence_similarity,
    levenshtein_align,
)
from fuzzer_tool.services.operators import _fold_bits

from .corpus import err

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Mirror of `_predict_effect`'s confidence gate; the oracle re-derives it.
PREDICT_CONFIDENCE = 0.3
MAX_HYPOTHESES = 6
MAX_ENTRIES = 64
FLOOD_FACTOR = 4
SIG_BYTES = 2
WIDE_MIN = 1 << 10
WIDE_MAX = 1 << 16
SEED_LEN_MAX = 512
_OPS = ("bit_flip", "splice", "havoc", "dict_insert")

SUBSEED_RANGE = 1 << 32
U32_SPAN = 1 << 32
U64_SPAN = 1 << 64
# RandPool: batches up to 3 refills; int ranges straddling zero.
BATCH_POOLS = 3
INT_SPAN = 1 << 40
WEIGHTS_MAX = 1000
CHOICE_DRAWS = 64
TINY_WEIGHT = 1e-300
HUGE_WEIGHT = 1e300
# Diffs: sizes kept small so a case stays in the low milliseconds.
DELTA_PARENT_MAX = 4096
ALIGN_MIN = 64
ALIGN_MAX = 200
MAX_EDITS = 16
# AFL dictionary: printable ASCII stays literal, the rest is \xNN.
PRINTABLE_LO = 0x20
PRINTABLE_HI = 0x7E
TOKEN_MAX = 32
# TIFF: magic 42, 2-byte IFD entry count.
TIFF_MAGIC = 0x002A
TIFF_COUNT_BYTES = 2
TIFF_BODY_MAX = 256
BLOOM_CAP_MAX = 2000
BLOOM_KEYS = 300
BLOOM_RATES = (1e-6, 1e-3, 0.01, 0.1, 0.5)
KEY_MAX = 16
# EdgeTracker: overlapping shared pool + one unique edge per seed.
PRUNE_CAP_MAX = 100
PRUNE_OVERFILL = 5
SHARED_EDGES = 600
EDGES_PER_SEED = 50
UNIQUE_EDGE_BASE = 90_000
LINEAGE_MAX = 300
# chi2: dense rows reach Cochran's minimum on their own.
TABLE_COLS_MAX = 4
TABLE_ROWS_MAX = 6
DENSE_MIN = 50
DENSE_MAX = 5000
# ELF64: header 64 B, Elf64_Phdr 56 B.
ELF_MAGIC = b"\x7fELF"
ELF_HEADER_LEN = 64
ELF_PHDR_SIZE = 56
ELF_IMAGE_MAX = 1024
PHNUM_MAX = 12
PT_LOAD = 1
P_VADDR = 16  # Elf64_Phdr field offsets
P_FILESZ = 32  # p_filesz, then p_memsz at +40
PAGE = 0x1000
NOT_HEX = ("+", "-", " ", "_", "g", "Z", "\u0663")
GF_Q_MAX = 10
# Codecs: small alphabets are what BWT/MTF/Huffman blocks look like.
ALPHABET_MAX = 8
UNIQUE_BYTE = 0xA5
CODEC_BLOCK_MAX = 256
HUFF_BLOCK_MAX = 2048
RICE_K_MAX = 7
CMP_LINES_MAX = 40
PARSER_INPUT_MAX = 96
PARSER_PATTERNS = (b"\x00", b"\xff", b"\x82", b"\x02", b"\x7f", b"A")
# MinHash: tiny signatures over few values force band collisions.
LSH_PERM = 4
LSH_BANDS = 2
LSH_STEPS = 40
LSH_KEYS = 12
LSH_VALUES = 3
M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
# murmur3_chain switches to numpy at 64 blocks (256 B); cross it.
MURMUR_PREFIX_MAX = 40
MURMUR_LEN_MAX = 400
XXH_DATA_MAX = 300
# LZ: naive parse is O(n^2 * max_match); keep blocks short.
LZ_WINDOW = 32768
LZ_MATCH_MAX = 32
LZ_BLOCK_MAX = 120
MODAL_TOP_MAX = 12
MODAL_RANGE = 50
FOLD_ID_BITS = 63
FOLD_EDGES_MAX = 50
FOLD_MAP_SIZES = (1, 7, 64, 65536)
LGAMMA_X_MIN = 1e-3
LGAMMA_X_MAX = 1000.0
LGAMMA_N_MAX = 170
LGAMMA_TOL = 1e-9
SNOOB_BITS = 80
SNOOB_SCAN_MAX = 4096
COVER_EDGES_MAX = 80
COVER_SEED_EDGES = 10
COVER_SEEDS_MAX = 40
# Grammar: captured at import so a patched cap is caught, not followed.
EXPANSION_CAP = grammar.GENERATION_EXPANSION_CAP
GRAMMAR_REPEAT_MIN = 8
GRAMMAR_REPEAT_MAX = 32
GRAMMAR_MAX_LENS = (0, 1, 16, 64)
ASAN_DEPTH_MAX = 8
ASAN_PC_BITS = 47
ASAN_SHIFT_PAGES = 1 << 16
ASAN_ECHO = "log: 'AddressSanitizer: heap-buffer-overflow'\n"
# span_relocate: n < 3 is a no-op; spans are at most 16 bytes.
RELOCATE_MIN = 3
RELOCATE_SPAN_MAX = 16
RELOCATE_MAX = 80
PARITY_MAX = 80
PARITY_EVEN_BYTE = 0x00  # popcount 0; ^ 1 gives an odd one
DIVERSE_MAX = 80
NO_CLOSE = 0xFF  # _DELIM_CLOSE entry for a non-delimiter byte
PARSE_ALPHABET = b"abX({[\")]}'<>\x00\xff"
PARSE_SHORT_MAX = 40
PARSE_LONG_MAX = 400
CRC_WIDTHS = (8, 16, 24, 32, 40, 64)
CRC_DATA_MAX = 2000
LFSR_WIDTHS = (8, 16, 32)
LFSR_EXTRA = 16
CFG_ROOT = 0
CFG_NODES_MAX = 25
CFG_OUT_MAX = 3
CRASH_K_MAX = 14
CRASH_KINDS = ("SEGV", "ASAN:heap-buffer-overflow", "ASAN:heap-use-after-free", "FPE")
CRASH_FRAMES = (
    "malloc",
    "free",
    "memcpy",
    "av_read_frame",
    "decode_slice",
    "parse_hdr",
    "main",
    "__libc_start",
    "inflate",
    "png_read",
    "sqlite3Step",
)
CDF_OBS_MAX = 500
CDF_COUNT_MAX = 3000
# Checksum learner: captured at import so a patched budget is caught.
STATE_BYTES_MAX = acl.CHECKSUM_STATE_BYTES_MAX
PAIR_OVERHEAD = 8  # bytes charged per pair besides its data
NEVER_RECOVER = 1 << 30  # min_pairs: recovery (and its globals) never runs
CHECKSUM_JUNK = (None, 42, ("s", 1), (b"x", "y"), (b"x",))


def _flood_cap() -> int:
    """Signature-counter bound the flood oracle enforces."""
    return afl.MAX_TRACKED_SIGNATURES


def _rand_cluster(rng: RandPool) -> FormatCluster:
    """Cluster with random confident/unconfident hypotheses."""
    c = FormatCluster(signature="regress")
    for _ in range(rng.randint(1, MAX_HYPOTHESES)):
        c._add_hypothesis(
            FieldHypothesis(
                offset=rng.randint(0, 64),
                width=rng.randint(1, 16),
                field_type="data",
                confidence=rng.random(),
                sensitive_ops={rng.choice(_OPS): 1},
            )
        )
    return c


def _rand_entry(rng: RandPool) -> TimelineEntry:
    before = rng.randint(0, 100)
    return TimelineEntry(
        input_hash="0" * 16,
        mutation_op=rng.choice(_OPS),
        mutation_offset=None if rng.random() < 0.5 else rng.randint(0, 80),
        mutation_width=rng.randint(1, 8),
        coverage_before=before,
        coverage_after=before + rng.randint(-2, 2),
        new_edges=set(),
        lost_edges=set(),
    )


def _reg_none_offset(rng: RandPool, _workdir: Path) -> str | None:
    """Metamorphic: a None offset predicts from op sensitivity alone."""
    c = _rand_cluster(rng)
    c.timeline.extend(_rand_entry(rng) for _ in range(rng.randint(1, MAX_ENTRIES)))

    for entry in c.timeline:
        if entry.mutation_offset is not None:
            continue

        by_op = any(
            entry.mutation_op in h.sensitive_ops and h.confidence > PREDICT_CONFIDENCE
            for h in c.hypotheses
        )
        got = c._predict_effect(entry)
        if got is not (True if by_op else None):
            return f"None offset predicted {got}, op-only says {by_op}"

    c.backtest()
    return None


def _reg_mg_flood(rng: RandPool, _workdir: Path) -> str | None:
    """Bounded memory under a flood of distinct format signatures."""
    fl = FormatLearner()
    cap = _flood_cap()
    # Just past the cap: the overflow path is what the bug lived in. The
    # per-step check is O(1); the O(cap) set checks run once (was 3.5 s/case).
    n = afl.MAX_TRACKED_SIGNATURES + rng.randint(1, afl.MAX_TRACKED_SIGNATURES // FLOOD_FACTOR)
    for _ in range(n):
        data = rng.randbytes(SIG_BYTES) + b"payload"
        fl.record_transition(data, "op", 0, 1, 0, rng.randint(0, 1), set(), set())

        if len(fl._signature_counts) > cap:
            return f"{len(fl._signature_counts)} tracked signatures > cap {cap}"

    if not set(fl._pending) <= set(fl._signature_counts):
        return "pending buffer for an untracked signature"
    if any(len(v) > fl.promote_threshold for v in fl._pending.values()):
        return "pending buffer past promote_threshold"
    return None


def _reg_wide_length(rng: RandPool, _workdir: Path) -> str | None:
    """Seed generation survives a `length` field kilobytes wide."""
    base = rng.randbytes(rng.randint(1, SEED_LEN_MAX))
    field = {
        "offset": rng.randint(0, len(base)),
        "width": rng.randint(WIDE_MIN, WIDE_MAX),
        "type": "length",
        "confidence": 1.0,
        "observations": 50,
        "controlled_edges": 10,
    }
    for seed in generate_seeds([field], base, n_seeds=8, rng=rng):
        if not isinstance(seed.data, bytes):
            return f"seed is {type(seed.data).__name__}"
    return None


# ── RandPool ─────────────────────────────────────────────────────────────────


def _fresh_pool(rng: RandPool) -> RandPool:
    """Pool under test, seeded from the driver's: never test rng with itself."""
    return RandPool(seed=rng.randrange(SUBSEED_RANGE))


def _reg_rp_batch(rng: RandPool, _workdir: Path) -> str | None:
    """Batch draws past one pool refill keep their length and range."""
    pool = _fresh_pool(rng)
    count = rng.randint(1, BATCH_POOLS * _POOL_ENTRIES)
    a = rng.randint(-INT_SPAN, INT_SPAN)
    b = a + rng.randint(0, INT_SPAN)

    floats = pool.random_list(count)
    if len(floats) != count:
        return f"random_list({count}) gave {len(floats)}"
    if any(not 0.0 <= x < 1.0 for x in floats):
        return "random_list value outside [0, 1)"

    ints = pool.randint_list(a, b, count)
    if len(ints) != count:
        return f"randint_list({count}) gave {len(ints)}"
    if any(not a <= x <= b for x in ints):
        return f"randint_list value outside [{a}, {b}]"

    pool.randint(0, 1)  # scalar draws still work after a big batch
    return None


def _rand_weights(rng: RandPool, n: int) -> list[float]:
    """Mix of zeros, tiny, huge and repeated weights; at least one > 0."""
    palette = (0.0, TINY_WEIGHT, 1.0, HUGE_WEIGHT, 1.0 / 3.0)
    weights = [rng.choice(palette) for _ in range(n)]
    weights[rng.randrange(n)] = rng.choice(palette[1:])
    return weights


def _reg_weighted_choice(rng: RandPool, _workdir: Path) -> str | None:
    """Float-rounded totals never index past the end or pick a 0 weight."""
    pool = _fresh_pool(rng)
    n = rng.randint(1, WEIGHTS_MAX)
    seq = list(range(n))
    weights = _rand_weights(rng, n)

    for _ in range(CHOICE_DRAWS):
        pick = pool.weighted_choice(seq, weights)
        if pick not in seq:
            return f"picked {pick!r} not in seq"
        if weights[pick] == 0.0:
            return f"picked index {pick} with weight 0"
    return None


# ── diffs and dictionaries ───────────────────────────────────────────────────


def _edited(rng: RandPool, base: bytes, edits: int) -> bytes:
    """*base* with *edits* random insert/delete/replace byte edits."""
    out = bytearray(base)
    for _ in range(edits):
        pos = rng.randint(0, len(out))
        roll = rng.randrange(3)
        if roll == 0 or not out:
            out[pos:pos] = rng.randbytes(1)
        elif roll == 1:
            del out[min(pos, len(out) - 1)]
        else:
            out[min(pos, len(out) - 1)] = rng.randrange(256)
    return bytes(out)


def _reg_delta_roundtrip(rng: RandPool, _workdir: Path) -> str | None:
    """A returned v2 delta rebuilds the child; hopeless length gaps give None."""
    parent = rng.randbytes(rng.randint(0, DELTA_PARENT_MAX))
    if rng.random() < 0.3:
        child = parent[: rng.randint(0, len(parent))]
    else:
        child = _edited(rng, parent, rng.randint(0, MAX_EDITS))

    diff = compute_delta_v2(parent, child)
    if diff is not None and apply_delta_v2(parent, diff) != child:
        return f"delta of {len(parent)}->{len(child)} bytes does not rebuild child"
    if diff is not None and abs(len(parent) - len(child)) > len(child) // 2:
        return f"delta returned for length gap {len(parent)}->{len(child)}"
    return None


def _replay(base: bytes, script: list[tuple[str, int, bytes]]) -> bytes:
    """Rebuild the target from *base* by replaying an edit script."""
    out = bytearray()
    for op, pos, data in script:
        if op == "match":
            out.append(base[pos])
        elif op in ("replace", "insert"):
            out += data
        elif op != "delete":
            raise ValueError(f"unknown edit op {op!r}")
    return bytes(out)


def _reg_myers_script(rng: RandPool, _workdir: Path) -> str | None:
    """Myers backtrack rebuilds b, never shorter than the Levenshtein optimum.

    Myers minimises indels, so it may carry extra ops (not <= k); fewer than
    the exact DP would mean an invalid script.
    """
    a = rng.randbytes(rng.randint(ALIGN_MIN, ALIGN_MAX))  # < 64 B skips Myers
    k = rng.randint(0, MAX_EDITS)
    b = _edited(rng, a, k)

    saved = (similarity._DIFF_MYERS_MAX_D, similarity._DIFF_MYERS_MAX_BYTES)
    configure_diff_limits(max_d=0, max_bytes=0)  # force the Myers path
    try:
        script = levenshtein_align(a, b)
    finally:
        configure_diff_limits(*saved)

    if _replay(a, script) != b:
        return f"script does not rebuild b ({len(a)}->{len(b)} bytes, k={k})"
    edits = sum(op != "match" for op, _, _ in script)
    optimum = sum(op != "match" for op, _, _ in _levenshtein_align_numpy(a, b))
    if edits < optimum:
        return f"{edits} edit ops < Levenshtein optimum {optimum}: invalid script"
    return None


def _afl_quote(token: bytes) -> str:
    r"""AFL dictionary encoding: ``\\``, ``\"``, ``\xNN`` for non-printables."""
    out = []
    for byte in token:
        ch = chr(byte)
        if ch in '\\"':
            out.append("\\" + ch)
        elif PRINTABLE_LO <= byte <= PRINTABLE_HI:
            out.append(ch)
        else:
            out.append(f"\\x{byte:02x}")
    return '"' + "".join(out) + '"'


def _reg_dict_quotes(rng: RandPool, _workdir: Path) -> str | None:
    """parse_dict_line inverts AFL quoting for any token and name prefix."""
    alphabet = b'=\\"x0A' + bytes(range(256))
    token = bytes(rng.choice(alphabet) for _ in range(rng.randint(1, TOKEN_MAX)))
    prefix = rng.choice(("", "kw=", "kw@3=", "name_1="))
    line = prefix + _afl_quote(token)

    got = parse_dict_line(line)
    if got != token:
        return f"{line!r} parsed as {got!r}, want {token!r}"
    if parse_dict_line(prefix + '""') is not None:
        return "empty token not dropped"
    return None


# ── formats, filters, trackers ───────────────────────────────────────────────


def _reg_tiff_ifd(rng: RandPool, _workdir: Path) -> str | None:
    """Any IFD offset parses in bounds or yields None; mutate returns bytes."""
    order = rng.choice(("<", ">"))
    offset = rng.choice((rng.randrange(U32_SPAN), rng.randint(0, TIFF_BODY_MAX)))
    head = (b"II" if order == "<" else b"MM") + struct.pack(f"{order}HI", TIFF_MAGIC, offset)
    data = head + rng.randbytes(rng.randint(0, TIFF_BODY_MAX))

    hdr = parse_tiff(data)
    if hdr is not None and hdr.offset + TIFF_COUNT_BYTES > len(data):
        return f"IFD offset {hdr.offset} past {len(data)}-byte buffer"
    if hdr is not None and len(hdr.ifd_entries) > hdr.num_entries:
        return f"{len(hdr.ifd_entries)} entries > count {hdr.num_entries}"

    out = TiffMutator(seed=rng.randrange(SUBSEED_RANGE)).mutate(data)
    if not isinstance(out, bytes):
        return f"mutate returned {type(out).__name__}"
    return None


def _reg_bloom_roundtrip(rng: RandPool, _workdir: Path) -> str | None:
    """No false negatives across to_dict/from_dict and fold_to; FPR never tightens."""
    bf = BloomFilter(rng.randint(1, BLOOM_CAP_MAX), rng.choice(BLOOM_RATES))
    keys = [rng.randbytes(rng.randint(0, KEY_MAX)) for _ in range(rng.randint(1, BLOOM_KEYS))]
    for key in keys:
        bf.add_bytes(key)

    bf = BloomFilter.from_dict(bf.to_dict())
    if not all(bf.query_bytes(k) for k in keys):
        return "false negative after to_dict/from_dict"

    before = bf.error_rate
    bf.fold_to(rng.choice(BLOOM_RATES))
    if not all(bf.query_bytes(k) for k in keys):
        return "false negative after fold_to"
    if bf.error_rate < before:
        return f"fold tightened error_rate {before} -> {bf.error_rate}"
    return None


def _owner_recount(et: EdgeTracker) -> dict[int, int]:
    """Owners per edge, derived from seed_edges independently of the tracker."""
    counts: dict[int, int] = {}
    for edges in et.seed_edges.values():
        for e in edges:
            counts[e] = counts.get(e, 0) + 1
    return counts


def _reg_edge_prune(rng: RandPool, _workdir: Path) -> str | None:
    """Pruning holds the seed bound, companion maps and owner counts."""
    cap = rng.randint(2, PRUNE_CAP_MAX)
    et = EdgeTracker(max_tracked_seeds=cap)
    for i in range(rng.randint(2 * cap, PRUNE_OVERFILL * cap)):
        edges = {rng.randrange(SHARED_EDGES) for _ in range(rng.randint(1, EDGES_PER_SEED))}
        edges.add(UNIQUE_EDGE_BASE + i)
        et.record_edges(f"s{i}", edges, stack_depth=i + 1, path_hash=i + 1)

        if len(et.seed_edges) > cap:
            return f"{len(et.seed_edges)} seeds tracked > cap {cap}"

    for name in ("seed_stack_depth", "seed_path_hash"):
        if set(getattr(et, name)) != set(et.seed_edges):
            return f"{name} out of sync with seed_edges"
    live = {e: n for e, n in et._edge_owner_count.items() if n > 0}
    if live != _owner_recount(et):
        return "_edge_owner_count disagrees with seed_edges"
    if any(n <= 0 for n in et._edge_owner_count.values()):
        return "_edge_owner_count keeps non-positive entries"
    return None


def _unit_credit(_key: str) -> tuple[int, int]:
    return (1, 0)


def _reg_lineage_cycle(rng: RandPool, _workdir: Path) -> str | None:
    """Random parent graphs (self loops, cycles, orphans): every walk is bounded."""
    n = rng.randint(1, LINEAGE_MAX)
    keys = [f"k{i}" for i in range(n)]
    meta = {
        k.encode(): {
            "parent_key": rng.choice((None, "ghost", *keys)),
            "new_edge_count": rng.randint(0, 3),
        }
        for k in keys
    }

    tree = LineageTree()
    tree.rebuild_from_meta(meta, bytes.decode)
    total = len(tree)
    for k in keys:
        if len(tree.subtree_keys(k)) > total:
            return f"subtree_keys({k}) longer than the tree"
        if len(tree.chain_from(k)) > total:
            return f"chain_from({k}) longer than the tree"
        if tree.recent_credit(k, _unit_credit) > total:
            return f"recent_credit({k}) exceeds node count"
    return None


def _reg_vu64_roundtrip(rng: RandPool, _workdir: Path) -> str | None:
    """Encode/decode round-trip; truncations raise ValueError only."""
    value = rng.choice((rng.randrange(U64_SPAN), rng.randrange(1 << rng.randint(1, 64))))
    enc = vu64_encode_value(value)

    got = vu64_decode_bytes(enc)
    if got != (value, vu64_encoded_len(value)):
        return f"vu64 {value} decoded as {got}"
    if len(enc) != vu64_encoded_len(value):
        return f"vu64 {value}: {len(enc)} bytes, encoded_len says {vu64_encoded_len(value)}"

    cut = rng.randrange(len(enc))
    try:
        vu64_decode_bytes(enc[:cut])
    except ValueError:
        return None
    return f"truncated vu64 ({cut}/{len(enc)} bytes) decoded"


# ── statistics, ELF, grammar, finite fields ──────────────────────────────────


def _sparse_table(rng: RandPool) -> list[list[float]]:
    """Dense rows plus one planted sparse row (an op run once: [1, 0])."""
    cols = rng.randint(2, TABLE_COLS_MAX)
    rows = [
        [float(rng.randint(DENSE_MIN, DENSE_MAX)) for _ in range(cols)]
        for _ in range(rng.randint(1, TABLE_ROWS_MAX))
    ]
    sparse = [0.0] * cols
    sparse[0] = 1.0
    rows.insert(rng.randint(0, len(rows)), sparse)
    return rows


def _expected_ok(rows: list[list[float]], minimum: float) -> bool:
    """Every expected cell >= *minimum*, recomputed from *rows* alone."""
    if not rows:
        return True
    cols = [sum(c) for c in zip(*rows, strict=True)]
    grand = sum(cols)
    return all(sum(r) * c / grand >= minimum for r in rows for c in cols)


def _reg_chi2_sparse(rng: RandPool, _workdir: Path) -> str | None:
    """Kept rows are an in-order subset, all cells meet Cochran, idempotent."""
    table = _sparse_table(rng)
    kept = drop_sparse_rows(table)

    it = iter(table)
    if not all(any(r is row for row in it) for r in kept):
        return "kept rows are not an in-order subset of the input"
    if not _expected_ok(kept, COCHRAN_MIN_EXPECTED):
        return f"kept table has expected cells < {COCHRAN_MIN_EXPECTED}"
    if drop_sparse_rows(kept) != kept:
        return "drop_sparse_rows is not idempotent"
    return None


def _rand_elf(rng: RandPool, vaddr: int) -> bytearray:
    """Random phdr geometry (in/out of bounds); planted PT_LOADs cover *vaddr*."""
    size = rng.randint(ELF_HEADER_LEN, ELF_IMAGE_MAX)
    buf = bytearray(rng.randbytes(size))
    buf[:4] = ELF_MAGIC
    phoff = rng.choice((rng.randrange(U64_SPAN), rng.randint(0, size), ELF_HEADER_LEN))
    phnum = rng.randint(0, PHNUM_MAX)
    struct.pack_into("<Q", buf, 32, phoff)
    struct.pack_into("<HH", buf, 54, ELF_PHDR_SIZE, phnum)

    for i in range(phnum):
        off = phoff + i * ELF_PHDR_SIZE
        if off + ELF_PHDR_SIZE > size or rng.random() < 0.5:
            continue
        start = vaddr - rng.randint(0, PAGE)
        struct.pack_into("<I", buf, off, PT_LOAD)
        struct.pack_into("<Q", buf, off + P_VADDR, start)
        struct.pack_into("<QQ", buf, off + P_FILESZ, PAGE, 2 * PAGE)
    return buf


def _in_bounds_loads(buf: bytes, vaddr: int) -> set[tuple[int, int, int]]:
    """PT_LOAD answers wholly inside *buf* containing *vaddr* (reference)."""
    (phoff,) = struct.unpack_from("<Q", buf, 32)
    (phnum,) = struct.unpack_from("<H", buf, 56)
    hits = set()
    for i in range(phnum):
        off = phoff + i * ELF_PHDR_SIZE
        if off + ELF_PHDR_SIZE > len(buf):
            break
        (p_type,) = struct.unpack_from("<I", buf, off)
        (p_vaddr,) = struct.unpack_from("<Q", buf, off + P_VADDR)
        p_filesz, p_memsz = struct.unpack_from("<QQ", buf, off + P_FILESZ)
        if p_type == PT_LOAD and p_vaddr <= vaddr < p_vaddr + p_memsz:
            hits.add((p_vaddr, p_filesz, p_memsz))
    return hits


def _reg_elf_load_segment(rng: RandPool, _workdir: Path) -> str | None:
    """Any phdr geometry: never raises; a hit is a real in-bounds PT_LOAD."""
    vaddr = rng.randint(2 * PAGE, U32_SPAN)
    buf = _rand_elf(rng, vaddr)

    got = find_load_segment(bytes(buf), vaddr)
    if got is not None and got not in _in_bounds_loads(buf, vaddr):
        return f"{got} is not an in-bounds PT_LOAD containing {vaddr:#x}"
    return None


def _grammar_quote(token: bytes, rng: RandPool) -> str:
    """Escape *token* with a random mix of the decoder's escape forms."""
    simple = {v: k for k, v in GRAMMAR_SIMPLE.items()}
    out = []
    for byte in token:
        ch = chr(byte)
        if ch in "\\\"'":
            out.append("\\" + ch)
        elif byte in simple and rng.random() < 0.5:
            out.append("\\" + simple[byte])
        elif PRINTABLE_LO <= byte <= PRINTABLE_HI and rng.random() < 0.7:
            out.append(ch)
        else:
            out.append(f"\\x{byte:02x}" if rng.random() < 0.5 else f"\\x{byte:02X}")
    return "".join(out)


def _reg_grammar_escapes(rng: RandPool, _workdir: Path) -> str | None:
    """Escape round-trip; `\\x` + non-hex-digits keeps its backslash."""
    token = rng.randbytes(rng.randint(0, TOKEN_MAX)) + b"\x00"  # >= 1 escape
    text = _grammar_quote(token, rng)
    got = decode_quoted_literal(text)
    if got != token:
        return f"{text!r} decoded as {got!r}, want {token!r}"

    bogus = "\\x" + "".join(rng.choice(NOT_HEX) for _ in range(2))
    if decode_quoted_literal(bogus) != bogus.encode():
        return f"{bogus!r} decoded as an escape"
    return None


def _poly_mod(a: int, b: int) -> int:
    """a mod b over GF(2)[x]."""
    db = b.bit_length()
    while a.bit_length() >= db:
        a ^= b << (a.bit_length() - db)
    return a


def _irreducible(poly: int, q: int) -> bool:
    """Trial division by every polynomial of degree 1..q//2 (reference)."""
    return all(_poly_mod(poly, d) for d in range(2, 1 << (q // 2 + 1)))


def _prime_factors(n: int) -> set[int]:
    out, p = set(), 2
    while p * p <= n:
        while n % p == 0:
            out.add(p)
            n //= p
        p += 1
    if n > 1:
        out.add(n)
    return out


def _check_field(f: gf2_common.GF2n, rng: RandPool) -> str | None:
    """mul agrees with the poly path, inverses invert, gen is primitive."""
    a, b = rng.randint(1, f.m), rng.randint(0, f.m)
    if f.mul(a, b) != f._mul_poly(a, b):
        return f"mul({a}, {b}) disagrees with _mul_poly"
    if f.mul(a, f.inv(a)) != 1:
        return f"a * inv(a) != 1 for a={a}"
    if any(f.pow(f.gen, f.m // p) == 1 for p in _prime_factors(f.m)):
        return f"gen {f.gen} is not primitive"
    return None


def _rejects_modulus(q: int, modulus: int) -> bool:
    """GF2n refuses *modulus* with ValueError (other errors propagate)."""
    try:
        gf2_common.GF2n(q, modulus=modulus)
    except ValueError:
        return True
    return False


def _reg_gf2n_modulus(rng: RandPool, _workdir: Path) -> str | None:
    """Reducible modulus -> ValueError (no hang); irreducible -> a field."""
    q = rng.randint(2, GF_Q_MAX)
    lo = rng.randint(1, q - 1)
    left = (1 << lo) | rng.randrange(1 << lo)
    right = (1 << (q - lo)) | rng.randrange(1 << (q - lo))
    reducible = 0
    for i in range(right.bit_length()):  # carryless product: degree q, reducible
        if right >> i & 1:
            reducible ^= left << i
    if not _rejects_modulus(q, reducible):
        return f"reducible modulus {reducible:#x} (q={q}) accepted"

    irreducible = next(p for p in range(1 << q, 1 << (q + 1)) if _irreducible(p, q))
    return _check_field(
        gf2_common.GF2n(q, modulus=irreducible, seed=rng.randrange(SUBSEED_RANGE)), rng
    )


# ── entropy codecs (mutations/structured.py) ─────────────────────────────────


def _small_alphabet_block(rng: RandPool, max_len: int) -> bytes:
    """Repetitive block (what codecs see) with one unique byte planted."""
    alphabet = rng.randbytes(rng.randint(1, ALPHABET_MAX))
    block = bytearray(rng.choice(alphabet) for _ in range(rng.randint(1, max_len)))
    block[rng.randrange(len(block))] = UNIQUE_BYTE
    return bytes(block)


def _length_kept(rng: RandPool, data: bytes, mutators) -> str | None:
    """Each codec mutator preserves length."""
    for fn in mutators:
        out = fn(data, rng)
        if len(out) != len(data):
            return f"{fn.__name__} changed length {len(data)} -> {len(out)}"
    return None


def _reg_bwt_mtf(rng: RandPool, _workdir: Path) -> str | None:
    """BWT and MTF round-trip; inverse of any L keeps len(L); mutators keep length."""
    s = structured
    block = _small_alphabet_block(rng, CODEC_BLOCK_MAX)
    if s._bwt_inverse(*s._bwt(block)) != block:
        return f"BWT round-trip failed on {len(block)} bytes"
    if s._mtf_decode(s._mtf_encode(block, bytearray(range(256))), bytearray(range(256))) != block:
        return "MTF round-trip failed"

    junk = rng.randbytes(rng.randint(1, CODEC_BLOCK_MAX))
    if len(s._bwt_inverse(junk, rng.randrange(len(junk)))) != len(junk):
        return "_bwt_inverse changed length on arbitrary input"
    return _length_kept(rng, block, (s.bwt, s.mtf, s.rle))


def _huff_bits(block: bytes, codes: list[int], lengths: list[int]) -> bytearray:
    """MSB-first bit list of *block* under (codes, lengths)."""
    bits = bytearray()
    for sym in block:
        n = lengths[sym]
        bits.extend((codes[sym] >> (n - 1 - i)) & 1 for i in range(n))
    return bits


def _prefix_free(codes: list[int], lengths: list[int]) -> bool:
    words = sorted(format(codes[s], f"0{lengths[s]}b") for s in range(256) if lengths[s])
    return not any(b.startswith(a) for a, b in zip(words, words[1:], strict=False))


def _reg_huffman_decode(rng: RandPool, _workdir: Path) -> str | None:
    """Codes <= 16 bits, prefix-free; round-trip; corrupt streams terminate."""
    s = structured
    block = _small_alphabet_block(rng, HUFF_BLOCK_MAX)
    freq = [0] * 256
    for sym in block:
        freq[sym] += 1
    if rng.random() < 0.3:  # Fibonacci counts force depth past 16 bits
        a, b = 1, 1
        for sym in sorted(set(block)):
            freq[sym], a, b = a, b, a + b

    codes, lengths = s._huffman_codebook(freq)
    if max(lengths) > s._HUFF_MAX_BITS:
        return f"code length {max(lengths)} > {s._HUFF_MAX_BITS}"
    if not _prefix_free(codes, lengths):
        return "codebook is not prefix-free"

    bits = _huff_bits(block, codes, lengths)
    got = s._huff_decode(bits, codes, lengths, len(block))
    if got != (bytearray(block), len(block)):
        return f"Huffman round-trip failed on {len(block)} bytes"

    corrupt = bits[: rng.randint(0, len(bits))] + bytearray(rng.randint(0, 1) for _ in range(8))
    _, ridx = s._huff_decode(corrupt, codes, lengths, len(block))
    if not 0 <= ridx <= len(block):
        return f"corrupt stream decoded {ridx} of {len(block)}"
    return None


def _reg_entropy_codes(rng: RandPool, _workdir: Path) -> str | None:
    """Rice / Elias gamma / Elias delta round-trip any block; mutators keep length."""
    s = structured
    block = rng.randbytes(rng.randint(1, CODEC_BLOCK_MAX))
    want = (bytearray(block), len(block))
    k = rng.randint(0, RICE_K_MAX)

    if s._rice_decode(s._rice_encode(bytearray(block), k), k, len(block)) != want:
        return f"Rice k={k} round-trip failed"
    if s._gamma_decode(s._gamma_encode(block), len(block)) != want:
        return "Elias gamma round-trip failed"
    if s._delta_decode(s._delta_encode(block), len(block)) != want:
        return "Elias delta round-trip failed"
    return _length_kept(rng, block, (s.golomb, s.elias_gamma, s.elias_delta))


# ── cmplog, format parsers, MinHash ──────────────────────────────────────────


def _cmp_line(rng: RandPool) -> str:
    """CMP line: valid, padded, odd hex, junk fields, short or empty operands."""
    a = rng.randbytes(rng.randint(0, 4)).hex()
    b = rng.randbytes(rng.randint(0, 4)).hex()
    fields = ["CMP", a or "", b or ""]
    for _ in range(rng.randint(0, 3)):
        fields.append(rng.choice((str(rng.randint(0, 64)), f"{rng.randrange(U32_SPAN):x}", "zz")))
    if rng.random() < 0.1:
        fields[1] += "f"  # odd-length hex
    pad = rng.choice(("", " ", "\t", "  "))
    return pad + " ".join(fields) + pad


def _cmp_reference(lines: list[str]) -> list[tuple]:
    """Independent strip/split parse with order-preserving dedup."""
    out, seen = [], set()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3 or parts[0] != "CMP":
            continue
        try:
            a, b = bytes.fromhex(parts[1]), bytes.fromhex(parts[2])
            res = int(parts[3]) if len(parts) > 3 else None
            width = int(parts[4]) if len(parts) > 4 else None
            pc = int(parts[5], 16) if len(parts) > 5 else None
        except ValueError:
            continue
        key = (a, b, width, res, pc)
        if (a or b) and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _reg_cmp_parse(rng: RandPool, _workdir: Path) -> str | None:
    """Uncached, cold-cache and warm-cache parses match a reference parser."""
    lines = [_cmp_line(rng) for _ in range(rng.randint(1, CMP_LINES_MAX))]
    lines.append("CMP 41 42 1 8 400000")  # >= 1 valid line
    if rng.random() < 0.5:
        lines += lines[: rng.randint(0, len(lines))]  # duplicates

    want = _cmp_reference(lines)
    cache: dict = {}
    for label, got in (
        ("uncached", cond_tuples_from_cmplog_text(lines)),
        ("cold cache", cond_tuples_from_cmplog_text(lines, cache)),
        ("warm cache", cond_tuples_from_cmplog_text(lines, cache)),
    ):
        if got != want:
            return f"{label} parse disagrees with reference ({len(got)} vs {len(want)})"
    return None


@functools.cache
def _parsers() -> list[tuple[str, Callable[[bytes], object], tuple[type, ...]]]:
    """Every bytes-taking ``parse_*`` in core/mutations with its own errors."""
    found = []
    for info in pkgutil.iter_modules(mutations_pkg.__path__):
        mod = import_module(f"{mutations_pkg.__name__}.{info.name}")
        own = tuple(
            v
            for v in vars(mod).values()
            if isinstance(v, type) and issubclass(v, BaseException) and v.__module__ == mod.__name__
        )
        for name, fn in vars(mod).items():
            if _bytes_parser(name, fn, mod.__name__):
                found.append((f"{info.name}.{name}", fn, own))
    return found


def _bytes_parser(name: str, fn: object, module: str) -> bool:
    """A module's own ``parse_*(data: bytes)`` with one required argument."""
    if not name.startswith("parse_") or not inspect.isfunction(fn) or fn.__module__ != module:
        return False
    required = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    ]
    return len(required) == 1 and required[0].annotation in (bytes, "bytes")


def _reg_parser_truncation(rng: RandPool, _workdir: Path) -> str | None:
    """Short/degenerate buffers: a parser returns or raises its own error only."""
    label, fn, own = rng.choice(_parsers())
    data = rng.choice(
        (
            rng.randbytes(rng.randint(0, PARSER_INPUT_MAX)),
            rng.choice(PARSER_PATTERNS) * rng.randint(0, PARSER_INPUT_MAX),
        )
    )
    try:
        fn(data)
    except own:
        return None
    except Exception as exc:
        return f"{label}({data[:16].hex()}.. {len(data)} B): {err(exc)}"
    return None


def _lsh_reference(lsh: MinHashLSH, live: dict[str, list[int]]) -> list[dict]:
    """Bands of a fresh LSH holding only *live*, normalised to key sets."""
    fresh = MinHashLSH(num_perm=LSH_PERM, num_bands=LSH_BANDS)
    for key, sig in live.items():
        fresh.add(key, sig)
    return _norm_bands(fresh)


def _norm_bands(lsh: MinHashLSH) -> list[dict]:
    return [
        {h: frozenset([v] if isinstance(v, str) else v) for h, v in band.items()}
        for band in lsh._bands
    ]


def _reg_minhash_lsh(rng: RandPool, _workdir: Path) -> str | None:
    """Add/update/remove keeps bands equal to a fresh build; remove-all empties."""
    lsh = MinHashLSH(num_perm=LSH_PERM, num_bands=LSH_BANDS)
    live: dict[str, list[int]] = {}
    for _ in range(rng.randint(1, LSH_STEPS)):
        key = f"k{rng.randrange(LSH_KEYS)}"
        if live and rng.random() < 0.3:
            key = rng.choice(sorted(live))
            lsh.remove(key)
            del live[key]
            continue

        sig = [rng.randrange(LSH_VALUES) for _ in range(LSH_PERM)]
        lsh.add(key, sig)
        live[key] = sig

    if _norm_bands(lsh) != _lsh_reference(lsh, live):
        return "bands differ from a fresh LSH of the live keys"
    for key in list(live):
        lsh.remove(key)
    if any(lsh._bands):
        return "bands not empty after removing every key"
    return None


# ── hashes vs spec references ────────────────────────────────────────────────


def _rotl(x: int, r: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return ((x << r) | (x >> (bits - r))) & mask


def _ref_murmur3(data: bytes, seed: int) -> int:
    """MurmurHash3_x86_32, straight from the reference algorithm."""
    c1, c2, m = 0xCC9E2D51, 0x1B873593, M32
    h = seed & m
    n = len(data)
    tail_at = n - n % 4
    for i in range(0, tail_at, 4):
        k = (int.from_bytes(data[i : i + 4], "little") * c1) & m
        h ^= (_rotl(k, 15, 32) * c2) & m
        h = (_rotl(h, 13, 32) * 5 + 0xE6546B64) & m

    tail = data[tail_at:]
    if tail:
        k = int.from_bytes(tail, "little")
        h ^= (_rotl((k * c1) & m, 15, 32) * c2) & m

    h ^= n
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & m
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & m
    return h ^ (h >> 16)


def _ref_xxh32(data: bytes, seed: int = 0) -> int:
    """XXH32 per the xxHash spec."""
    p1, p2, p3, p4, p5 = 2654435761, 2246822519, 3266489917, 668265263, 374761393
    m, n, i = M32, len(data), 0
    if n >= 16:
        v = [(seed + p1 + p2) & m, (seed + p2) & m, seed & m, (seed - p1) & m]
        while i + 16 <= n:
            for j in range(4):
                lane = int.from_bytes(data[i + 4 * j : i + 4 * j + 4], "little")
                v[j] = (_rotl((v[j] + lane * p2) & m, 13, 32) * p1) & m
            i += 16
        h = sum(_rotl(x, r, 32) for x, r in zip(v, (1, 7, 12, 18), strict=True)) & m
    else:
        h = (seed + p5) & m

    h = (h + n) & m
    while i + 4 <= n:
        h = (h + int.from_bytes(data[i : i + 4], "little") * p3) & m
        h = (_rotl(h, 17, 32) * p4) & m
        i += 4
    for b in data[i:]:
        h = (_rotl((h + b * p5) & m, 11, 32) * p1) & m

    h = ((h ^ (h >> 15)) * p2) & m
    h = ((h ^ (h >> 13)) * p3) & m
    return h ^ (h >> 16)


_X64 = (
    0x9E3779B185EBCA87,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0x85EBCA77C2B2AE63,
    0x27D4EB2F165667C5,
)


def _x64_round(acc: int, lane: int) -> int:
    p1, p2 = _X64[0], _X64[1]
    return (_rotl((acc + lane * p2) & M64, 31, 64) * p1) & M64


def _ref_xxh64(data: bytes, seed: int = 0) -> int:
    """XXH64 per the xxHash spec."""
    p1, p2, p3, p4, p5 = _X64
    m, n, i = M64, len(data), 0
    if n >= 32:
        v = [(seed + p1 + p2) & m, (seed + p2) & m, seed & m, (seed - p1) & m]
        while i + 32 <= n:
            for j in range(4):
                v[j] = _x64_round(v[j], int.from_bytes(data[i + 8 * j : i + 8 * j + 8], "little"))
            i += 32
        h = sum(_rotl(x, r, 64) for x, r in zip(v, (1, 7, 12, 18), strict=True)) & m
        for x in v:
            h = ((h ^ _x64_round(0, x)) * p1 + p4) & m
    else:
        h = (seed + p5) & m

    h = (h + n) & m
    while i + 8 <= n:
        h ^= _x64_round(0, int.from_bytes(data[i : i + 8], "little"))
        h = (_rotl(h, 27, 64) * p1 + p4) & m
        i += 8
    if i + 4 <= n:
        h ^= (int.from_bytes(data[i : i + 4], "little") * p1) & m
        h = (_rotl(h, 23, 64) * p2 + p3) & m
        i += 4
    for b in data[i:]:
        h = (_rotl(h ^ ((b * p5) & m), 11, 64) * p1) & m

    h = ((h ^ (h >> 33)) * p2) & m
    h = ((h ^ (h >> 29)) * p3) & m
    return h ^ (h >> 32)


def _reg_murmur3_chain(rng: RandPool, _workdir: Path) -> str | None:
    """Chained blocks (scalar and numpy paths) match the docstring definition."""
    prefix = rng.randbytes(rng.randint(0, MURMUR_PREFIX_MAX))
    length = rng.randint(1, MURMUR_LEN_MAX)
    seed = rng.randrange(U32_SPAN)

    # b0 = H(prefix, seed); bt = H(b0..b(t-1) packed, seed ^ 4t).
    blocks = [_ref_murmur3(prefix, seed)]
    for t in range(1, -(-length // 4)):
        blocks.append(_ref_murmur3(struct.pack(f"<{t}I", *blocks), seed ^ (4 * t)))
    want = struct.pack(f"<{len(blocks)}I", *blocks)[:length]

    got = murmur3.murmur3_chain(prefix, length, seed)
    if got != want:
        return f"murmur3_chain(len={length}, seed={seed:#x}) differs from reference"
    return None


def _reg_xxhash(rng: RandPool, _workdir: Path) -> str | None:
    """XXH32 (LZ4 checksums) and XXH64 (corpus names) match the spec."""
    data = rng.randbytes(rng.randint(0, XXH_DATA_MAX))
    s32 = rng.choice((0, M32, rng.randrange(U32_SPAN)))
    s64 = rng.choice((0, M64, rng.randrange(U64_SPAN)))

    if lz4._xxh32_py(data, s32) != _ref_xxh32(data, s32):
        return f"xxh32 of {len(data)} B, seed {s32:#x} differs from spec"
    if hashing.xxh64(data, s64) != _ref_xxh64(data, s64):
        return f"xxh64 of {len(data)} B, seed {s64:#x} differs from spec"
    return None


# ── LZ, statistics, bit tricks ───────────────────────────────────────────────


def _naive_lz(block: bytes, min_match: int, max_match: int) -> list[tuple]:
    """Greedy earliest-longest parse (the pre-speedup parser, verbatim)."""
    tokens: list[tuple] = []
    i, n = 0, len(block)
    while i < n:
        best_dist, best_len = 0, 0
        for j in range(max(0, i - LZ_WINDOW), i):
            k = 0
            while k < max_match and i + k < n and block[j + k] == block[i + k]:
                k += 1
            if k > best_len:
                best_dist, best_len = i - j, k
        if best_len >= min_match:
            tokens.append(("ref", best_dist, best_len))
            i += best_len
        else:
            tokens.append(("lit", block[i]))
            i += 1
    return tokens


def _lz_rebuild(tokens: list[tuple]) -> bytes:
    out = bytearray()
    for tok in tokens:
        if tok[0] == "lit":
            out.append(tok[1])
            continue
        _, dist, length = tok
        for _ in range(length):
            out.append(out[-dist])
    return bytes(out)


def _reg_lz_tokenize(rng: RandPool, _workdir: Path) -> str | None:
    """Hash-chain parse == naive greedy parse; refs in bounds; tokens rebuild."""
    mn = rng.randint(2, 4)
    mx = rng.randint(mn, LZ_MATCH_MAX)
    alphabet = rng.randbytes(rng.randint(1, ALPHABET_MAX))
    block = bytearray(rng.choice(alphabet) for _ in range(rng.randint(mx, LZ_BLOCK_MAX)))
    block += block[: mx + 1]  # a guaranteed back-reference

    tokens = structured._lz_tokenize(bytearray(block), mn, mx)
    if _lz_rebuild(tokens) != block:
        return "tokens do not rebuild the block"
    if any(t[0] == "ref" and not (mn <= t[2] <= mx and t[1] >= 1) for t in tokens):
        return f"ref outside [{mn}, {mx}] or dist < 1"
    if tokens != _naive_lz(block, mn, mx):
        return f"parse differs from naive greedy (min={mn}, max={mx}, {len(block)} B)"
    return None


def _reg_modal_value(rng: RandPool, _workdir: Path) -> str | None:
    """Plurality, not first-seen: the result's count is the maximum count."""
    top = rng.randint(2, MODAL_TOP_MAX)
    values = [rng.randrange(MODAL_RANGE)] * top
    values += [rng.randrange(MODAL_RANGE) for _ in range(rng.randint(0, top - 1))]
    rng.shuffle(values)
    values.append(MODAL_RANGE + 1)  # unique tail: never the mode

    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    got = _modal_value(values)
    if counts.get(got, 0) != max(counts.values()):
        return f"modal {got} seen {counts.get(got, 0)}x, max is {max(counts.values())}"
    return None


def _reg_fold_bits(rng: RandPool, _workdir: Path) -> str | None:
    """numpy fold == OR of 1 << (e % map_size)."""
    bits = rng.randint(1, FOLD_ID_BITS)
    edges = {rng.randrange(1 << bits) for _ in range(rng.randint(0, FOLD_EDGES_MAX))}
    map_size = rng.choice(FOLD_MAP_SIZES)

    want = 0
    for e in edges:
        want |= 1 << (e % map_size)
    got = _fold_bits(edges, map_size)
    if got != want:
        return f"fold of {len(edges)} edges into {map_size} differs"
    return None


def _reg_log_gamma(rng: RandPool, _workdir: Path) -> str | None:
    """Lanczos fallback == lgamma; (n-1)! identity; x <= 0 -> inf."""
    x = rng.random() * LGAMMA_X_MAX + LGAMMA_X_MIN
    want = math.lgamma(x)
    if abs(_log_gamma(x) - want) > LGAMMA_TOL * max(1.0, abs(want)):
        return f"_log_gamma({x}) = {_log_gamma(x)}, lgamma {want}"

    n = rng.randint(1, LGAMMA_N_MAX)
    fact = math.log(math.factorial(n - 1))
    if abs(_log_gamma(n) - fact) > LGAMMA_TOL * max(1.0, fact):
        return f"_log_gamma({n}) != log({n - 1}!)"
    if _log_gamma(-rng.random()) != math.inf:
        return "_log_gamma(x <= 0) is not inf"
    return None


def _reg_snoob(rng: RandPool, _workdir: Path) -> str | None:
    """Next/previous same-popcount: monotone, inverse, closed form (no 2^63 loop)."""
    x = rng.choice((rng.randint(1, (1 << rng.randint(1, SNOOB_BITS)) - 1), 1 << rng.randint(0, 79)))
    pop = x.bit_count()

    y = snoob(x)
    if not (y > x and y.bit_count() == pop and snoob_prev(y) == x):
        return f"snoob({x:#x}) = {y:#x}: not the next same-popcount value"
    p = snoob_prev(x)
    if p and not (p < x and p.bit_count() == pop and snoob(p) == x):
        return f"snoob_prev({x:#x}) = {p:#x}: not the previous same-popcount value"
    if x < SNOOB_SCAN_MAX and any(v.bit_count() == pop for v in range(x + 1, y)):
        return f"snoob({x:#x}) skipped a same-popcount value"
    return None


def _reg_set_cover(rng: RandPool, _workdir: Path) -> str | None:
    """Exact, irredundant, duplicate-free, deterministic cover."""
    universe = rng.randint(1, COVER_EDGES_MAX)
    seeds = {
        f"s{i}": {rng.randrange(universe) for _ in range(rng.randint(0, COVER_SEED_EDGES))}
        for i in range(rng.randint(1, COVER_SEEDS_MAX))
    }
    seeds["s0"].add(0)  # at least one non-empty seed
    sizes = {k: rng.randint(1, 100) for k in seeds}

    got = min_cover(seeds, sizes)
    union = set().union(*seeds.values())
    if set().union(*(seeds[k] for k in got)) != union:
        return "chosen seeds do not cover every edge"
    if len(set(got)) != len(got) or any(not seeds[k] for k in got):
        return "duplicate or empty seed in the cover"
    for k in got:
        rest = set().union(*(seeds[o] for o in got if o != k))
        if rest == union:
            return f"cover is redundant: {k} can be dropped"
    if min_cover(seeds, sizes) != got:
        return "min_cover is not deterministic"
    return None


# ── grammar budget, sanitizer reports ────────────────────────────────────────


def _reg_grammar_cap(rng: RandPool, _workdir: Path) -> str | None:
    """Repeat chains over empty literals stop at the expansion cap."""
    r = rng.randint(GRAMMAR_REPEAT_MIN, GRAMMAR_REPEAT_MAX)
    levels = math.ceil(math.log(2 * EXPANSION_CAP) / math.log(r)) + 1
    leaf = rng.choice(('""', '"" | ""'))
    rules = [f"start = r0{{{r}}}"]
    rules += [f"r{i} = r{i + 1}{{{r}}}" for i in range(levels - 1)]
    rules.append(f"r{levels - 1} = {leaf}")

    g = grammar.Grammar()
    g.parse("\n".join(rules))
    calls = [0]
    expand = g._expand_rule

    def counted(name, depth):
        calls[0] += 1
        return expand(name, depth)

    g._expand_rule = counted  # instance attribute: nothing global to restore
    max_len = rng.choice(GRAMMAR_MAX_LENS)
    logger = logging.getLogger(grammar.__name__)
    level = logger.level
    logger.setLevel(logging.CRITICAL)  # the cap logs a warning per case
    try:
        out = g.generate("start", max_len=max_len)
    finally:
        logger.setLevel(level)

    if calls[0] > EXPANSION_CAP + 1:
        return f"{calls[0]} expansions > cap {EXPANSION_CAP} (R={r}, L={levels})"
    if max_len and len(out) > max_len:
        return f"{len(out)} bytes > max_len {max_len}"
    return None


def _asan(pcs: list[int], names: list[str], alloc: list[int]) -> str:
    """ASAN report: crash stack, blank line, allocation stack."""
    lines = ["==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n"]
    lines += [
        f"    #{i} {pc:#x} in {nm}\n" for i, (pc, nm) in enumerate(zip(pcs, names, strict=True))
    ]
    lines += ["\n", "freed by thread T0 here:\n"]
    lines += [f"    #{i} {pc:#x} in free_{i}\n" for i, pc in enumerate(alloc)]
    return "".join(lines)


def _reg_sanitizer_stack(rng: RandPool, _workdir: Path) -> str | None:
    """Frames = crash stack only; hash is ASLR-blind; echoed text is no report."""
    depth = rng.randint(1, ASAN_DEPTH_MAX)
    pcs = [rng.randrange(1 << ASAN_PC_BITS) for _ in range(depth)]
    names = [f"fn_{rng.randrange(U32_SPAN):x}" for _ in range(depth)]
    alloc = [rng.randrange(1 << ASAN_PC_BITS) for _ in range(rng.randint(0, ASAN_DEPTH_MAX))]

    rep = sanitizer.SanitizerReport.parse(_asan(pcs, names, alloc))
    if rep is None or rep.frames != names:
        return f"frames {rep and rep.frames} != crash stack {names}"

    shift = (2 * rng.randint(0, ASAN_SHIFT_PAGES) + 1) * PAGE  # odd pages: ASLR slide
    moved = [pc + shift for pc in pcs]
    other_alloc = [rng.randrange(1 << ASAN_PC_BITS) for _ in alloc]
    moved_rep = sanitizer.SanitizerReport.parse(_asan(moved, names, other_alloc))
    if moved_rep is None or moved_rep.stack_hash() != rep.stack_hash():
        return "stack_hash changed under an ASLR slide or a new alloc stack"
    if sanitizer.SanitizerReport.parse(ASAN_ECHO) is not None:
        return "echoed sanitizer text parsed as a report"
    return None


# ── seeded operators, replayed draw for draw ─────────────────────────────────


def _reg_span_relocate(rng: RandPool, _workdir: Path) -> str | None:
    """Span lands at the drawn index of the remainder (was shifted by -span)."""
    data = rng.randbytes(rng.randint(0, RELOCATE_MAX))
    seed = rng.randrange(SUBSEED_RANGE)
    got = generic.span_relocate(data, RandPool(seed))

    # Replay the operator's three draws on the same seed.
    n = len(data)
    want = data
    if n >= RELOCATE_MIN:
        r = RandPool(seed)
        span = r.randint(1, min(RELOCATE_SPAN_MAX, n - 1))
        src = r.randint(0, n - span)
        dest = r.randint(0, n - span)
        if dest == src:
            dest = (src + 1) % (n - span + 1)
        rest = data[:src] + data[src + span :]
        want = rest[:dest] + data[src : src + span] + rest[dest:]

    if got != want:
        return f"span_relocate({n} B, seed {seed}) misplaced the span"
    return None


def _reg_parity_lock(rng: RandPool, _workdir: Path) -> str | None:
    """Only wrong-parity bytes of the window flip, on the drawn carrier bit."""
    data = bytearray(rng.randbytes(rng.randint(parity.MIN_LEN, PARITY_MAX)))
    seed = rng.randrange(SUBSEED_RANGE)

    # Window and carrier draws do not depend on content: replay them first.
    r = RandPool(seed)
    off, length = structured._region(len(data), r, min_len=parity.MIN_LEN)
    if length < parity.MIN_LEN:
        return None
    carrier = r.choice(parity.CARRIERS)
    want_parity = r.choice(tuple(parity.Parity)).value
    data[off] = PARITY_EVEN_BYTE ^ want_parity  # one byte already right: must not flip

    got = parity.parity_lock(bytes(data), RandPool(seed))
    window = bytes(
        b ^ (1 << carrier) if b.bit_count() & 1 != want_parity else b
        for b in data[off : off + length]
    )
    if got != bytes(data[:off]) + window + bytes(data[off + length :]):
        return f"parity_lock window [{off}, {off + length}) carrier {carrier} wrong"
    return None


def _reg_diverse_copy(rng: RandPool, _workdir: Path) -> str | None:
    """Every byte changes, length kept, same seed -> same output."""
    data = rng.randbytes(rng.randint(1, DIVERSE_MAX))
    seed = rng.randrange(SUBSEED_RANGE)

    got = colorization._diverse_copy(data, RandPool(seed))
    if len(got) != len(data):
        return f"length {len(data)} -> {len(got)}"
    same = next((i for i, (a, b) in enumerate(zip(data, got, strict=True)) if a == b), None)
    if same is not None:
        return f"byte {same} kept its value {data[same]:#x}"
    if colorization._diverse_copy(data, RandPool(seed)) != got:
        return "not deterministic for a fixed seed"
    return None


# ── parsers vs their reference implementations ───────────────────────────────


def _ref_parse(data: bytes) -> tree_mutator._Node:
    """Per-byte delimiter parser: the pre-vectorization logic, branches merged."""
    node_cls = tree_mutator._Node
    close_of, close_table = tree_mutator._DELIM_CLOSE, tree_mutator._CLOSE_TABLE
    root = node_cls()
    stack = [root]
    buf: list[bytes] = []

    def flush() -> None:
        if buf:
            stack[-1].children.append(b"".join(buf))
            buf.clear()

    for byte in data:
        close = close_of[byte]
        top = stack[-1]
        if close != NO_CLOSE and not (byte == close and top.open == byte):
            flush()  # opener (or a self-closing delimiter opening)
            node = node_cls(byte)
            top.children.append(node)
            stack.append(node)
        elif close != NO_CLOSE or (top.open is not None and byte == close_table[top.open]):
            flush()  # closer of the open node
            if len(stack) > 1:
                top.closed = True
                stack.pop()
        else:
            buf.append(bytes((byte,)))
    flush()
    return root


def _shape(node) -> tuple:
    """Comparable tree: (open, closed, children) with byte leaves."""
    if isinstance(node, bytes):
        return ("leaf", node)
    return (node.open, node.closed, tuple(_shape(c) for c in node.children))


def _reg_partial_parse(rng: RandPool, _workdir: Path) -> str | None:
    """Vectorized delimiter scan builds the per-byte parser's tree."""
    lo, hi = rng.choice(((1, PARSE_SHORT_MAX), (tree_mutator._VECTOR_MIN_LEN, PARSE_LONG_MAX)))
    data = bytearray(rng.choice(PARSE_ALPHABET) for _ in range(rng.randint(lo, hi)))
    data[rng.randrange(len(data))] = rng.choice(b"([")  # >= 1 structural node

    tree = tree_mutator.partial_parse(bytes(data))
    if tree.flatten() != data:
        return f"flatten() does not give back the {len(data)}-byte input"
    if _shape(tree) != _shape(_ref_parse(bytes(data))):
        return f"tree differs from per-byte parse on {bytes(data[:24])!r}.."
    return None


# ── checksums, LFSR, dominators, clustering ──────────────────────────────────


def _reflect(x: int, bits: int) -> int:
    return int(format(x, f"0{bits}b")[::-1], 2)


def _ref_crc(
    data: bytes, poly: int, width: int, init: int, xorout: int, refin: bool, refout: bool
) -> int:
    """Bit-at-a-time CRC as compute_checksum documents it (no tables).

    refin: LSB-first, *poly* in reversed form, register shifts right.
    Otherwise MSB-first, normal-form *poly*. refout reverses the result.
    """
    top, mask = 1 << (width - 1), (1 << width) - 1
    reg = init & mask
    for byte in data:
        if refin:
            reg ^= byte
            for _ in range(8):
                reg = (reg >> 1) ^ poly if reg & 1 else reg >> 1
            continue

        reg ^= byte << (width - 8)
        for _ in range(8):
            reg = ((reg << 1) & mask) ^ poly if reg & top else (reg << 1) & mask
    if refout:
        reg = _reflect(reg, width)
    return (reg ^ xorout) & mask


def _reg_crc_checksum(rng: RandPool, _workdir: Path) -> str | None:
    """Table / braid paths (cold and warm engine) equal the bitwise CRC."""
    width = rng.choice(CRC_WIDTHS)
    poly = rng.randrange(1 << width) | 1
    init, xorout = rng.randrange(1 << width), rng.randrange(1 << width)
    refin, refout = rng.random() < 0.5, rng.random() < 0.5
    data = rng.randbytes(rng.choice((rng.randint(0, 63), rng.randint(64, CRC_DATA_MAX))))
    args = (poly, width, init, xorout, refin, refout)

    try:
        if rng.random() < 0.5:  # warm: enough bytes for the engine to build tables
            compute_checksum(bytes(crc_engine._BUILD_BYTES), *args)
        got = compute_checksum(data, *args)
    finally:
        crc_engine.clear()

    if got != _ref_crc(data, *args):
        return f"CRC width={width} poly={poly:#x} refin={refin} refout={refout} on {len(data)} B"
    return None


def _reg_lfsr_recover(rng: RandPool, _workdir: Path) -> str | None:
    """Recovered taps predict lane 0 past its linear complexity."""
    width = rng.choice(LFSR_WIDTHS)
    poly = rng.randrange(1 << width) | 1
    state = rng.randrange(1 << width) | 1  # bit 0 set: lane 0 is not all zero
    values = []
    for _ in range(2 * width + rng.randint(0, LFSR_EXTRA)):
        values.append(state)
        state = lfsr_step(state, poly, width)

    bits = [v & 1 for v in values]
    complexity = linear_complexity(bits)
    q = recover_lfsr(values, width)
    if q is None:
        return f"no polynomial recovered (linear complexity {complexity})"

    taps = [i + 1 for i in range(width) if q >> i & 1]
    for n in range(max(complexity, max(taps, default=0)), len(bits)):
        pred = 0
        for t in taps:
            pred ^= bits[n - t]
        if pred != bits[n]:
            return f"taps of {q:#x} mispredict bit {n} (L={complexity})"
    if complexity == width and q != poly:
        return f"full-complexity lane recovered {q:#x}, not {poly:#x}"
    return None


def _reach(edges: dict[int, list[int]], skip: int | None) -> set[int]:
    """Blocks reachable from the root without passing through *skip*."""
    if skip == CFG_ROOT:
        return set()
    seen, stack = {CFG_ROOT}, [CFG_ROOT]
    while stack:
        for s in edges[stack.pop()]:
            if s != skip and s not in seen:
                seen.add(s)
                stack.append(s)
    return seen


def _brute_idom(edges: dict[int, list[int]]) -> dict[int, int]:
    """idom by definition: a dominates v iff removing a cuts v off."""
    live = _reach(edges, None)
    doms = {v: {v} for v in live}
    for a in live:
        for v in live - _reach(edges, a):
            doms[v].add(a)
    idom = {CFG_ROOT: CFG_ROOT}
    for v in live - {CFG_ROOT}:
        idom[v] = max(doms[v] - {v}, key=lambda d: len(doms[d]))
    return idom


def _reg_dominators(rng: RandPool, _workdir: Path) -> str | None:
    """Semi-NCA idom == brute-force dominators on random (irreducible) CFGs."""
    n = rng.randint(3, CFG_NODES_MAX)
    edges = {v: [rng.randrange(n) for _ in range(rng.randint(0, CFG_OUT_MAX))] for v in range(n)}
    for v in edges:  # node 2 reachable only through 0 -> 1 -> 2
        edges[v] = [s for s in edges[v] if s != 2]
    edges[0].append(1)
    edges[1].append(2)

    blocks = {a: BasicBlock(start=a, end=a + 1, successors=list(edges[a])) for a in edges}
    blocks[CFG_ROOT].is_entry = True
    cfg = FunctionCFG(name="f", start=0, end=n, blocks=blocks)

    got, want = compute_idom(cfg), _brute_idom(edges)
    if got != want:
        bad = sorted(v for v in want if got.get(v) != want[v])[:4]
        return f"idom differs at {bad} ({n} nodes)"
    return None


def _dense_clusters(sigs: list[str], frames, threshold: float) -> list[list[int]]:
    """All-pairs single linkage (the pre-pruning oracle)."""
    parent = list(range(len(sigs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            if frames and i < len(frames) and j < len(frames):
                sim = frame_sequence_similarity(frames[i], frames[j])
            else:
                sim = crash_signature_similarity(sigs[i], sigs[j])
            if sim >= threshold:
                parent[find(i)] = find(j)

    out: dict[int, list[int]] = {}
    for i in range(len(sigs)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _crash(rng: RandPool) -> tuple[str, list[str]]:
    frames = [rng.choice(CRASH_FRAMES) for _ in range(rng.randint(1, 9))]
    frames = [f + (f"+0x{rng.randrange(PAGE):x}" if rng.random() < 0.4 else "") for f in frames]
    return rng.choice(CRASH_KINDS) + ":" + ":".join(frames), frames


def _reg_crash_cluster(rng: RandPool, _workdir: Path) -> str | None:
    """Pruned single linkage == all-pairs single linkage (same partition)."""
    pairs = [_crash(rng) for _ in range(rng.randint(1, CRASH_K_MAX))]
    pairs.append(rng.choice(pairs))  # a duplicate always merges
    sigs = [p[0] for p in pairs]
    frames = [p[1] for p in pairs]
    fl = rng.choice((None, frames, frames[: len(frames) // 2]))
    threshold = rng.choice((0.5, 0.7, 0.85, rng.random()))

    def part(clusters):
        return frozenset(frozenset(c) for c in clusters)

    got = part(cluster_crashes(sigs, fl, threshold))
    if got != part(_dense_clusters(sigs, fl, threshold)):
        return f"{len(sigs)} crashes at threshold {threshold:.3f}: partitions differ"
    return None


# ── CDF distances, checksum learner state ────────────────────────────────────


def _ref_cdf(prof_a: dict, prof_b: dict) -> tuple:
    """Merged-support CDF walk (the pre-numpy loop, verbatim)."""
    xs = sorted(set(prof_a) | set(prof_b))
    if not xs:
        return 0.0, 0.0, 0.0
    cdf_diff = wasserstein = ks = crps = 0.0
    prev = xs[0]
    for x in xs:
        gap = x - prev
        abs_diff = abs(cdf_diff)
        wasserstein += abs_diff * gap
        ks = max(ks, abs_diff)
        crps += cdf_diff * cdf_diff * gap
        cdf_diff += prof_a.get(x, 0.0) - prof_b.get(x, 0.0)
        prev = x
    return wasserstein, max(ks, abs(cdf_diff)), crps


def _profile(rng: RandPool) -> dict[float, float]:
    """log2(1 + count) support, equal mass per observation."""
    n = rng.randint(1, CDF_OBS_MAX)
    prof: dict[float, float] = {}
    for count in rng.randint_list(0, rng.randint(1, CDF_COUNT_MAX), n):
        x = math.log2(1.0 + count)
        prof[x] = prof.get(x, 0.0) + 1.0 / n
    return prof


def _reg_cdf_walk(rng: RandPool, _workdir: Path) -> str | None:
    """numpy CDF norms bit-identical to the loop, both argument orders."""
    a, b = _profile(rng), _profile(rng)
    if a == b:
        b[max(b) + 1.0] = 1.0  # distinct profiles: a zero distance proves nothing
    for x, y in ((a, b), (b, a)):
        if edge_tracker.EdgeTracker._cdf_walk(x, y) != _ref_cdf(x, y):
            return f"_cdf_walk differs from loop ({len(x)} vs {len(y)} points)"
    return None


def _kept_suffix(pairs: list[tuple[bytes, int]]) -> list[tuple[bytes, int]]:
    """Longest newest suffix whose len + 8 per pair fits the state budget."""
    kept, budget = [], STATE_BYTES_MAX
    for data, checksum in reversed(pairs):
        budget -= len(data) + PAIR_OVERHEAD
        if budget < 0:
            break
        kept.append((data, checksum))
    return kept[::-1]


def _reg_checksum_state(rng: RandPool, _workdir: Path) -> str | None:
    """Pairs persist under the byte budget, round-trip, junk dropped in order."""
    big = rng.random() < 0.5
    pairs = [
        (
            rng.randbytes(rng.randint(1, 6 * 1024) if big else rng.randint(0, 64)),
            rng.randrange(U32_SPAN),
        )
        for _ in range(rng.randint(1, acl.CHECKSUM_PAIRS_MAX))
    ]
    host = SimpleNamespace(corpus=[], seed_meta={})
    learner = acl.ChecksumLearner(host, min_pairs=NEVER_RECOVER)
    learner._pairs = list(pairs)

    want = _kept_suffix(pairs)
    state = learner.to_dict()
    if not want or state["pairs"] != want:
        return f"to_dict kept {len(state['pairs'])} pairs, budget allows {len(want)}"
    restored = acl.ChecksumLearner.from_dict(host, state)
    if restored._pairs != want or restored.to_dict()["pairs"] != want:
        return "pairs changed across to_dict/from_dict"

    noisy = list(state["pairs"])
    for junk in CHECKSUM_JUNK:
        noisy.insert(rng.randint(0, len(noisy)), junk)
    if acl.ChecksumLearner.from_dict(host, {**state, "pairs": noisy})._pairs != want:
        return "junk pairs not dropped, or order lost"
    return None


REGRESSION_MUTATORS: dict[str, Callable[[RandPool, Path], str | None]] = {
    "none_offset": _reg_none_offset,
    "mg_flood": _reg_mg_flood,
    "wide_length": _reg_wide_length,
    "rp_batch": _reg_rp_batch,
    "weighted_choice": _reg_weighted_choice,
    "delta_roundtrip": _reg_delta_roundtrip,
    "myers_script": _reg_myers_script,
    "dict_quotes": _reg_dict_quotes,
    "tiff_ifd": _reg_tiff_ifd,
    "bloom_roundtrip": _reg_bloom_roundtrip,
    "edge_prune": _reg_edge_prune,
    "lineage_cycle": _reg_lineage_cycle,
    "vu64_roundtrip": _reg_vu64_roundtrip,
    "chi2_sparse": _reg_chi2_sparse,
    "elf_load_segment": _reg_elf_load_segment,
    "grammar_escapes": _reg_grammar_escapes,
    "gf2n_modulus": _reg_gf2n_modulus,
    "bwt_mtf": _reg_bwt_mtf,
    "huffman_decode": _reg_huffman_decode,
    "entropy_codes": _reg_entropy_codes,
    "cmp_parse": _reg_cmp_parse,
    "parser_truncation": _reg_parser_truncation,
    "minhash_lsh": _reg_minhash_lsh,
    "murmur3_chain": _reg_murmur3_chain,
    "xxhash": _reg_xxhash,
    "lz_tokenize": _reg_lz_tokenize,
    "modal_value": _reg_modal_value,
    "fold_bits": _reg_fold_bits,
    "log_gamma": _reg_log_gamma,
    "snoob": _reg_snoob,
    "set_cover": _reg_set_cover,
    "grammar_cap": _reg_grammar_cap,
    "sanitizer_stack": _reg_sanitizer_stack,
    "span_relocate": _reg_span_relocate,
    "parity_lock": _reg_parity_lock,
    "diverse_copy": _reg_diverse_copy,
    "partial_parse": _reg_partial_parse,
    "crc_checksum": _reg_crc_checksum,
    "lfsr_recover": _reg_lfsr_recover,
    "dominators": _reg_dominators,
    "crash_cluster": _reg_crash_cluster,
    "cdf_walk": _reg_cdf_walk,
    "checksum_state": _reg_checksum_state,
}


def run_regression(name: str, rng: RandPool, workdir: Path) -> str | None:
    """Failure detail for one regenerated case, or None."""
    try:
        return REGRESSION_MUTATORS[name](rng, workdir)
    except Exception as exc:
        return err(exc)
