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
"""

from __future__ import annotations

import functools
import inspect
import pkgutil
import struct
from importlib import import_module
from typing import TYPE_CHECKING

import fuzzer_tool.core.mutations as mutations_pkg
from fuzzer_tool.adapters.filesystem import apply_delta_v2, compute_delta_v2
from fuzzer_tool.adapters.track_parser import cond_tuples_from_cmplog_text
from fuzzer_tool.core import gf2_common, similarity
from fuzzer_tool.core.analyzers import analyzer_format_learner as afl
from fuzzer_tool.core.analyzers.analyzer_format_learner import (
    FieldHypothesis,
    FormatCluster,
    FormatLearner,
    TimelineEntry,
)
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.chi_squared import COCHRAN_MIN_EXPECTED, drop_sparse_rows
from fuzzer_tool.core.edge_tracker import EdgeTracker, MinHashLSH
from fuzzer_tool.core.elf import find_load_segment
from fuzzer_tool.core.format_seed_generator import generate_seeds
from fuzzer_tool.core.grammar import _SIMPLE_ESCAPES as GRAMMAR_SIMPLE
from fuzzer_tool.core.grammar import decode_quoted_literal
from fuzzer_tool.core.lineage import LineageTree
from fuzzer_tool.core.mutations import structured
from fuzzer_tool.core.mutations.generic import parse_dict_line
from fuzzer_tool.core.mutations.tiff import TiffMutator, parse_tiff
from fuzzer_tool.core.mutations.vu64 import (
    vu64_decode_bytes,
    vu64_encode_value,
    vu64_encoded_len,
)
from fuzzer_tool.core.rand_pool import _POOL_ENTRIES, RandPool
from fuzzer_tool.core.similarity import (
    _levenshtein_align_numpy,
    configure_diff_limits,
    levenshtein_align,
)

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
}


def run_regression(name: str, rng: RandPool, workdir: Path) -> str | None:
    """Failure detail for one regenerated case, or None."""
    try:
        return REGRESSION_MUTATORS[name](rng, workdir)
    except Exception as exc:
        return err(exc)
