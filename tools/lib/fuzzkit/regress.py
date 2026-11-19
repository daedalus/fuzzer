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
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from fuzzer_tool.adapters.filesystem import apply_delta_v2, compute_delta_v2
from fuzzer_tool.core import similarity
from fuzzer_tool.core.analyzers import analyzer_format_learner as afl
from fuzzer_tool.core.analyzers.analyzer_format_learner import (
    FieldHypothesis,
    FormatCluster,
    FormatLearner,
    TimelineEntry,
)
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.format_seed_generator import generate_seeds
from fuzzer_tool.core.lineage import LineageTree
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
}


def run_regression(name: str, rng: RandPool, workdir: Path) -> str | None:
    """Failure detail for one regenerated case, or None."""
    try:
        return REGRESSION_MUTATORS[name](rng, workdir)
    except Exception as exc:
        return err(exc)
