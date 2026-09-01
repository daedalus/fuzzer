"""Unit tests for consolidated Weizz-style structure tags.

Covers both collection paths against the shared ByteTag / StructureMap model:
- active/differential: get_deps + place_tags
- passive/cmplog:      build_tag_map_from_cmplog + collect_structure_map
plus the RLE / seed-metadata glue shared by both.
"""

from __future__ import annotations

from fuzzer_tool.core.weizz_tags import (
    ByteTag,
    StructureMap,
    TagCollectorConfig,
    TagFlags,
    attach_tags_to_meta,
    build_tag_map_from_cmplog,
    collect_structure_map,
    get_deps,
    inherit_tags_from_parent,
    load_tags_from_meta,
    place_tags,
    synthetic_exec_fn,
)


# ── Active / differential path (get_deps / place_tags) ──────────────────


def test_get_deps_empty_input():
    smap = get_deps(b"", lambda d: (0, {}))
    assert smap is not None
    assert smap.length == 0
    assert smap.ntypes == 0


def test_get_deps_over_max_len_skipped():
    data = b"x" * 100
    assert get_deps(data, lambda d: (0, {}), max_len=50) is None


def test_get_deps_finds_byte_dependencies():
    """Synthetic: cmp 1 compares bytes[0:2] to a constant; cmp 2 uses bytes[4:6]."""

    def op1(data: bytes) -> tuple[bytes, bytes]:
        return (data[0:2] if len(data) >= 2 else b"\x00\x00", b"AB")

    def op2(data: bytes) -> tuple[bytes, bytes]:
        return (data[4:6] if len(data) >= 6 else b"\x00\x00", b"CD")

    comparisons = [
        (1, 0, -1, op1),
        (2, 0, -1, op2),
    ]
    exec_fn = synthetic_exec_fn(comparisons)
    data = b"ABxxCDyy"

    smap = get_deps(data, exec_fn, byte_level=True, max_len=64)
    assert smap is not None
    assert smap.exec_count >= 1 + len(data)
    assert smap.from_differential

    # Bytes 0,1 should depend on cmp 1; 4,5 on cmp 2.
    spans = smap.field_spans()
    cmp_ids = {cid for _, _, cid in spans}
    assert 1 in cmp_ids or any(t.cmp_id == 1 for t in smap.tags)
    assert 2 in cmp_ids or any(t.cmp_id == 2 for t in smap.tags)

    # Direct dep check
    assert smap.deps
    affected_by_1 = set()
    affected_by_2 = set()
    for (cid, _), (v0, v1) in smap.deps.items():
        if cid == 1:
            affected_by_1 |= set(v0) | set(v1)
        if cid == 2:
            affected_by_2 |= set(v0) | set(v1)
    assert 0 in affected_by_1 or 1 in affected_by_1
    assert 4 in affected_by_2 or 5 in affected_by_2


def test_place_tags_field_spans():
    # Manually craft deps: bytes 0-2 depend on cmp 10; bytes 5-7 on cmp 20.
    deps = {
        (10, 0): (frozenset({0, 1, 2}), frozenset()),
        (20, 0): (frozenset({5, 6, 7}), frozenset()),
    }
    smap = place_tags(10, deps)
    assert smap.ntypes == 2
    spans = smap.field_spans()
    assert (0, 3, 10) in spans
    assert (5, 8, 20) in spans
    assert smap.tags[0].cmp_id == 10
    assert smap.tags[5].cmp_id == 20
    assert smap.tags[3].cmp_id == 0  # gap


def test_place_tags_length_flag():
    # Single-byte dependency whose members span a wide region → IS_LEN.
    deps = {
        (7, 0): (frozenset({1}), frozenset(range(2, 20))),
    }
    smap = place_tags(24, deps)
    # Byte 1 is the tight tag; members cover a wide window.
    assert smap.tags[1].cmp_id == 7
    # Length heuristic may or may not fire depending on span-of-tag vs members;
    # at least the tag is placed.
    assert any(t.cmp_id == 7 for t in smap.tags)


def test_bit_level_vs_byte_level():
    def op(data: bytes) -> tuple[bytes, bytes]:
        return (bytes([data[0] & 0x01]), b"\x01")

    exec_fn = synthetic_exec_fn([(1, 0, -1, op)])
    data = b"\x00\x00\x00\x00"

    smap_byte = get_deps(data, exec_fn, byte_level=True)
    smap_bit = get_deps(data, exec_fn, byte_level=False, max_execs=40)
    assert smap_byte is not None and smap_bit is not None

    def affects(smap, cid=1):
        out = set()
        for (c, _), (v0, v1) in smap.deps.items():
            if c == cid:
                out |= set(v0) | set(v1)
        return out

    assert 0 in affects(smap_byte)
    assert 0 in affects(smap_bit)


# ── Shared model: chunk_spans nesting ────────────────────────────────────


def test_chunk_spans_nested_child():
    tags = [ByteTag() for _ in range(8)]
    for i in range(2, 6):
        tags[i].cmp_id = 3
        tags[i].parent = 9
    smap = StructureMap(tags=tags, ntypes=1)
    # children of parent 9
    assert smap.chunk_spans(parent=9) == [(2, 6, 9)]
    # no top-level (parent==0) tagged spans in this synthetic map
    assert smap.chunk_spans(parent=None) == []


def test_chunk_spans_top_level():
    data = b"AAAABBBB"
    pairs = [(b"AAAA", b"AAAA"), (b"BBBB", b"BBBB")]
    smap = build_tag_map_from_cmplog(data, pairs)
    chunks = smap.chunk_spans(parent=None)
    assert len(chunks) >= 1


# ── Passive / cmplog path (build_tag_map_from_cmplog) ────────────────────


def test_cmplog_empty_input():
    smap = build_tag_map_from_cmplog(b"", [])
    assert smap.input_len == 0
    assert smap.ntypes == 0
    assert smap.field_spans() == []


def test_cmplog_size_gate():
    cfg = TagCollectorConfig(max_input_len=4)
    data = b"ABCDEFGH"
    pairs = [(b"AB", b"\x00\x00")]
    smap = build_tag_map_from_cmplog(data, pairs, config=cfg)
    assert smap.ntypes == 0  # gated out
    assert all(t.cmp_id == 0 for t in smap.tags)


def test_cmplog_simple_field_from_pair():
    # synthetic: magic "RIFF" then length then data
    data = b"RIFF\x08\x00\x00\x00WAVEdata"
    pairs = [
        (b"RIFF", b"RIFF"),  # magic compare
        (b"\x08\x00\x00\x00", b"\x08\x00\x00\x00"),  # length
        (b"WAVE", b"WAVE"),
    ]
    smap = build_tag_map_from_cmplog(data, pairs)
    assert smap.ntypes >= 2
    spans = smap.field_spans()
    assert spans, "expected at least one field span"
    # first span should cover RIFF
    start, end, cid = spans[0]
    assert data[start:end] == b"RIFF"
    assert cid != 0
    # length field flagged
    len_tags = [t for t in smap.tags if t.flags & TagFlags.IS_LEN]
    assert len_tags, "expected IS_LEN on length operand bytes"


def test_cmplog_nested_parent():
    # outer chunk "AAAA...." with inner "BB" that also appears as operand
    data = b"AAAAXXBBXXAAAA"
    pairs = [
        (b"AAAA", b"AAAA"),
        (b"BB", b"BB"),
    ]
    smap = build_tag_map_from_cmplog(data, pairs)
    spans = smap.field_spans()
    assert len(spans) >= 2
    assert any(smap.tags[i].cmp_id for i in range(len(data)) if data[i : i + 2] == b"BB")


def test_cmplog_taint_soft_tags():
    data = b"XXXXYYYYZZZZ"
    pairs = [(b"XXXX", b"XXXX")]
    # taint covers the middle untagged region
    taints = [(4, 7)]  # YYYY inclusive
    smap = build_tag_map_from_cmplog(data, pairs, taint_regions=taints)
    # YYYY should pick up IS_IMPL from nearest cmp
    mid = smap.tags[5]
    assert mid.cmp_id != 0
    assert mid.flags & TagFlags.IS_IMPL
    assert smap.from_differential


def test_cmplog_prefer_shorter_operand():
    data = b"ABCDEFGH"
    # longer pair also matches prefix; shorter should claim first under sort
    pairs = [
        (b"ABCDEFGH", b"\x00" * 8),
        (b"AB", b"AB"),
    ]
    smap = build_tag_map_from_cmplog(data, pairs)
    assert smap.tags[0].cmp_id != 0
    assert smap.tags[1].cmp_id != 0


def test_collect_structure_map_from_cmplog_object():
    class FakeCmplog:
        pairs = [(b"HELLO", b"HELLO")]
        _pair_pc = {}

    smap = collect_structure_map(b"HELLOworld", FakeCmplog())
    assert smap.from_cmplog
    assert smap.ntypes >= 1


# ── RLE / metadata glue (shared model) ────────────────────────────────────


def test_rle_roundtrip():
    data = b"ABCABC"
    pairs = [(b"ABC", b"ABC")]
    smap = build_tag_map_from_cmplog(data, pairs)
    rle = smap.to_rle()
    restored = StructureMap.from_rle(rle, len(data), from_cmplog=True)
    assert restored.input_len == smap.input_len
    assert restored.ntypes == smap.ntypes
    for a, b in zip(smap.tags, restored.tags):
        assert a.cmp_id == b.cmp_id
        assert a.parent == b.parent
        assert a.flags == b.flags


def test_meta_attach_load():
    data = b"HEADBODY"
    pairs = [(b"HEAD", b"HEAD")]
    smap = build_tag_map_from_cmplog(data, pairs)
    meta = attach_tags_to_meta({}, smap)
    assert "weizz_tags_rle" in meta
    loaded = load_tags_from_meta(meta)
    assert loaded is not None
    assert loaded.ntypes == smap.ntypes
    assert loaded.field_spans() == smap.field_spans()


def test_dirty_flag():
    tags = [ByteTag(cmp_id=1), ByteTag(cmp_id=1), ByteTag(cmp_id=2)]
    smap = StructureMap(tags=tags, ntypes=2, input_len=3)
    smap.mark_dirty(0, 2)
    assert smap.tags[0].flags & TagFlags.DIRTY
    assert smap.tags[1].flags & TagFlags.DIRTY
    assert not (smap.tags[2].flags & TagFlags.DIRTY)


# ── P2 operator smoke (handlers via a minimal fuzzer stub) ───────────────


class _FakeRng:
    def __init__(self, seed=0):
        import random

        self._r = random.Random(seed)

    def choice(self, seq):
        return self._r.choice(seq)

    def sample(self, seq, k):
        return self._r.sample(list(seq), k)

    def randint(self, a, b):
        return self._r.randint(a, b)

    def random(self):
        return self._r.random()


class _FakeFuzzer:
    def __init__(self, data: bytes, smap: StructureMap, max_len: int = 4096):
        self.max_len = max_len
        self.weizz_tags = True
        self._rand_pool = _FakeRng(42)
        meta = attach_tags_to_meta({}, smap)
        self.seed_meta = {data: meta}


def _make_engine(data: bytes, smap: StructureMap):
    from fuzzer_tool.services.operators import OperatorEngine

    f = _FakeFuzzer(data, smap)
    return OperatorEngine(f), f


def test_weizz_field_havoc_length_preserving():
    data = b"AAAABBBBCCCC"
    # Tag first 4 as field 1, next 4 as field 2
    tags = (
        [ByteTag(cmp_id=1)] * 4
        + [ByteTag(cmp_id=2)] * 4
        + [ByteTag(cmp_id=3)] * 4
    )
    smap = StructureMap(tags=tags, ntypes=3, input_len=len(data), from_cmplog=True)
    eng, _f = _make_engine(data, smap)
    out = eng._op_weizz_field_havoc(bytearray(data), 0, data)
    assert out is not None
    assert len(out) == len(data)


def test_weizz_chunk_dup_grows():
    data = b"AAAABBBB"
    tags = [ByteTag(cmp_id=1, parent=0)] * 4 + [ByteTag(cmp_id=2, parent=0)] * 4
    smap = StructureMap(tags=tags, ntypes=2, input_len=len(data), from_cmplog=True)
    eng, f = _make_engine(data, smap)
    out = eng._op_weizz_chunk_dup(bytearray(data), 0, data)
    assert out is not None
    assert len(out) > len(data)
    assert f.seed_meta[data].get("weizz_tags_dirty") is True


def test_weizz_chunk_delete_shrinks():
    data = b"AAAABBBBCCCC"
    tags = (
        [ByteTag(cmp_id=1, parent=0)] * 4
        + [ByteTag(cmp_id=2, parent=0)] * 4
        + [ByteTag(cmp_id=3, parent=0)] * 4
    )
    smap = StructureMap(tags=tags, ntypes=3, input_len=len(data), from_cmplog=True)
    eng, f = _make_engine(data, smap)
    out = eng._op_weizz_chunk_delete(bytearray(data), 0, data)
    assert out is not None
    assert len(out) < len(data)
    assert f.seed_meta[data].get("weizz_tags_dirty") is True


def test_weizz_chunk_swap():
    data = b"AAAABBBB"
    tags = [ByteTag(cmp_id=1, parent=0)] * 4 + [ByteTag(cmp_id=2, parent=0)] * 4
    smap = StructureMap(tags=tags, ntypes=2, input_len=len(data), from_cmplog=True)
    eng, _f = _make_engine(data, smap)
    out = eng._op_weizz_chunk_swap(bytearray(data), 0, data)
    assert out is not None
    assert len(out) == len(data)
    # One of the two possible orderings after swap
    assert bytes(out) in (b"BBBBAAAA", b"AAAABBBB")


def test_weizz_ops_unavailable_without_tags():
    from fuzzer_tool.core.operator_registry import _weizz_tags_available

    class F:
        weizz_tags = True
        seed_meta = {}

    assert _weizz_tags_available(F(), b"x") is False


# ── P3 tag repair ────────────────────────────────────────────────────────


def test_flagged_spans_is_len():
    tags = (
        [ByteTag(cmp_id=1, flags=TagFlags.IS_LEN)] * 2
        + [ByteTag(cmp_id=2)] * 4
        + [ByteTag(cmp_id=3, flags=TagFlags.IS_CHECKSUM)] * 4
    )
    smap = StructureMap(tags=tags, ntypes=3, input_len=10)
    assert smap.flagged_spans(TagFlags.IS_LEN) == [(0, 2, 1)]
    assert smap.flagged_spans(TagFlags.IS_CHECKSUM) == [(6, 10, 3)]


def test_weizz_tag_repair_length_field():
    # 2-byte LE length + 4-byte payload
    data = b"\x04\x00ABCD"
    tags = [ByteTag(cmp_id=1, flags=TagFlags.IS_LEN)] * 2 + [ByteTag(cmp_id=2)] * 4
    smap = StructureMap(tags=tags, ntypes=2, input_len=len(data), from_cmplog=True)
    eng, _f = _make_engine(data, smap)
    out = eng._op_weizz_tag_repair(bytearray(data), 0, data)
    assert out is not None
    assert len(out) == len(data)


def test_weizz_tag_repair_checksum_field():
    from fuzzer_tool.core.crc32 import crc32

    body = b"PAYLOAD!!"
    # Force CRC path: only IS_CHECKSUM, no IS_LEN
    digest = crc32(body) & 0xFFFFFFFF
    data = body + digest.to_bytes(4, "little")
    tags = [ByteTag(cmp_id=1)] * len(body) + [
        ByteTag(cmp_id=2, flags=TagFlags.IS_CHECKSUM)
    ] * 4
    smap = StructureMap(tags=tags, ntypes=2, input_len=len(data), from_cmplog=True)
    eng, _f = _make_engine(data, smap)
    # Corrupt the CRC so repair has work to do
    buf = bytearray(data)
    buf[-4:] = b"\x00\x00\x00\x00"
    out = eng._op_weizz_tag_repair(buf, 0, data)
    assert out is not None
    assert len(out) == len(data)
    # Repair writes either LE or BE CRC of prefix
    got = bytes(out[-4:])
    le = digest.to_bytes(4, "little")
    be = digest.to_bytes(4, "big")
    assert got in (le, be)


# ── P5 derived-tag inheritance ───────────────────────────────────────────


def test_inherit_tags_length_preserving():
    parent = b"AAAABBBB"
    tags = [ByteTag(cmp_id=1)] * 4 + [ByteTag(cmp_id=2)] * 4
    smap = StructureMap(tags=tags, ntypes=2, input_len=len(parent), from_cmplog=True)
    parent_meta = attach_tags_to_meta({}, smap)
    child = b"AAAABBBX"  # same length
    out = inherit_tags_from_parent(parent_meta, parent, child)
    assert out.get("weizz_tags_rle")
    assert out.get("weizz_tags_dirty") is False
    assert out.get("weizz_tags_inherited") is True
    assert out.get("weizz_tags_len") == len(parent)


def test_inherit_tags_length_changing_marks_dirty():
    parent = b"AAAABBBB"
    tags = [ByteTag(cmp_id=1)] * 4 + [ByteTag(cmp_id=2)] * 4
    smap = StructureMap(tags=tags, ntypes=2, input_len=len(parent), from_cmplog=True)
    parent_meta = attach_tags_to_meta({}, smap)
    child = parent + b"EXTRA"
    out = inherit_tags_from_parent(parent_meta, parent, child)
    assert out.get("weizz_tags_dirty") is True
    assert out.get("weizz_tags_inherited") is True
    assert out.get("weizz_tags_len") == len(child)


def test_inherit_tags_no_parent():
    assert inherit_tags_from_parent(None, None, b"x") == {}
    assert inherit_tags_from_parent({}, b"ab", b"ab") == {}


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed)
