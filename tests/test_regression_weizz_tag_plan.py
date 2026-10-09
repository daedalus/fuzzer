"""build_tag_map_from_cmplog redid every pair's work on every mutation.

Sorting, cmp-id hashing and counter numbering depend on the pair pool only,
and a pair whose operands miss the input claims nothing; the Weizz mutators
still paid ~850 pairs of shape heuristics per call (32 s / 1.5k execs under
--hail-mary). The pool-only part is now cached per pool (owner + length, as
``scanner_for_pairs`` does) and unmatched pairs are skipped.
"""

from fuzzer_tool.core import weizz_tags as wt
from fuzzer_tool.core.rand_pool import RandPool

SEED = 7
POOL = 300
INPUT_LEN = 512


def _old_length(op: bytes, input_len: int) -> bool:
    """The pre-change _looks_like_length, verbatim."""
    if not op or len(op) > 8:
        return False
    return any(0 < int.from_bytes(op, e) <= input_len * 2 for e in ("little", "big"))


def _old_flags(op_a: bytes, op_b: bytes, n: int) -> wt.TagFlags:
    """The pre-change _operand_flags: length per input, checksum, magic."""
    flags = wt.TagFlags.NONE
    if _old_length(op_a, n) or _old_length(op_b, n):
        flags |= wt.TagFlags.IS_LEN
    elif wt._looks_like_checksum(op_a) or wt._looks_like_checksum(op_b):
        flags |= wt.TagFlags.IS_CHECKSUM
    if wt._looks_like_magic(op_a, op_b):
        flags |= wt.TagFlags.IS_MAGIC
    return flags


def test_flags_match_old_heuristic():
    """Shape-then-pick equals the per-input heuristic for every n."""
    rng = RandPool(SEED)
    for _ in range(2000):
        a, b = rng.randbytes(rng.randint(0, 9)), rng.randbytes(rng.randint(0, 9))
        n = rng.randint(1, 1 << 12)
        assert wt._operand_flags(a, b, n) == _old_flags(a, b, n)


def _reference(data: bytes, pairs) -> list:
    """The pre-cache loop: every pair, in sort order, numbered as met."""
    cfg = wt.TagCollectorConfig()
    n = len(data)
    claims = wt._TagClaims(n)
    offsets = wt.scanner_for_pairs(pairs).scan(data, min_len=cfg.min_operand_len)
    counter_by_id: dict[int, int] = {}
    first_offset: dict[int, int] = {}
    for op_a, op_b in sorted(pairs, key=wt._pair_key):
        candidates = wt._sized_operands(op_a, op_b, cfg)
        if not (op_a or op_b) or not candidates:
            continue
        cid = wt._stable_cmp_id(op_a, op_b, None)
        counter_by_id.setdefault(cid, len(counter_by_id) + 1)
        flags = _old_flags(op_a, op_b, n)
        tag = (cid, counter_by_id[cid], flags)
        wt._claim_pair(claims, candidates, offsets, tag, first_offset, True)
    wt._assign_parents(claims.tags)
    return [(t.cmp_id, t.parent, t.counter, int(t.flags)) for t in claims.tags]


def _actual(data: bytes, pairs) -> list:
    smap = wt.build_tag_map_from_cmplog(data, pairs)
    return [(t.cmp_id, t.parent, t.counter, int(t.flags)) for t in smap.tags]


def _workload(rng: RandPool):
    """Random pool plus an input that embeds a third of its operands."""
    pairs = [
        (rng.randbytes(rng.randint(1, 6)), rng.randbytes(rng.randint(1, 6))) for _ in range(POOL)
    ]
    data = bytearray(rng.randbytes(INPUT_LEN))
    for op_a, _ in pairs[::3]:
        at = rng.randint(0, INPUT_LEN - len(op_a))
        data[at : at + len(op_a)] = op_a
    return bytes(data), pairs


def test_reference_matches_itself():
    """Control (Hard Rule 46): the oracle is deterministic on one input."""
    data, pairs = _workload(RandPool(SEED))
    assert _reference(data, pairs) == _reference(data, pairs)


def test_regression_weizz_tag_plan():
    """Cached plan gives the reference tags, byte for byte, across calls."""
    data, pairs = _workload(RandPool(SEED))
    expected = _reference(data, pairs)
    assert any(t[0] for t in expected)
    assert _actual(data, pairs) == expected
    assert _actual(data, pairs) == expected


def test_grown_pool_is_replanned():
    """Adversarial: appending to the same pool object must not reuse a stale plan."""
    data, pairs = _workload(RandPool(SEED))
    _actual(data, pairs)
    pairs.insert(0, (data[:3], b"\x01"))
    assert _actual(data, pairs) == _reference(data, pairs)


def test_unmatched_pool_tags_nothing():
    """Falsification: no operand in the input, no tag."""
    pairs = [(b"\xaa\xbb\xcc", b"\xdd\xee")]
    assert all(t[0] == 0 for t in _actual(b"\x00" * 64, pairs))


def test_repeat_call_skips_heuristics(monkeypatch):
    """Same pool, new input: no pair is shape-classified again."""
    data, pairs = _workload(RandPool(SEED))
    wt.build_tag_map_from_cmplog(data, pairs)
    calls = []
    real = wt._looks_like_magic
    monkeypatch.setattr(wt, "_looks_like_magic", lambda a, b: calls.append(a) or real(a, b))
    wt.build_tag_map_from_cmplog(data[::-1], pairs)
    assert calls == []


def test_length_flag_tracks_input_len():
    """Adversarial: IS_LEN depends on the input length, not on the cached pool."""
    pairs = [(b"\x40", b"\x41\x42\x43")]  # 0x40 = 64: a length only once 2n >= 64
    short = wt.build_tag_map_from_cmplog(b"\x40" + b"\x00" * 15, pairs)
    long = wt.build_tag_map_from_cmplog(b"\x40" + b"\x00" * 63, pairs)
    assert not short.tags[0].flags & wt.TagFlags.IS_LEN
    assert long.tags[0].flags & wt.TagFlags.IS_LEN
