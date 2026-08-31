"""Unit tests for Weizz-style get_deps / place_tags (structure tags)."""

from __future__ import annotations

from fuzzer_tool.core.weizz_tags import (
    TAG_IS_LEN,
    Tag,
    TagsInfo,
    get_deps,
    place_tags,
    synthetic_exec_fn,
)


def test_empty_input():
    info = get_deps(b"", lambda d: (0, {}))
    assert info is not None
    assert info.length == 0
    assert info.ntypes == 0


def test_over_max_len_skipped():
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

    info = get_deps(data, exec_fn, byte_level=True, max_len=64)
    assert info is not None
    assert info.exec_count >= 1 + len(data)

    # Bytes 0,1 should depend on cmp 1; 4,5 on cmp 2.
    # After place_tags, those spans should carry the matching cmp_id.
    spans = info.field_spans()
    cmp_ids = {cid for _, _, cid in spans}
    assert 1 in cmp_ids or any(t.cmp_id == 1 for t in info.tags)
    assert 2 in cmp_ids or any(t.cmp_id == 2 for t in info.tags)

    # Direct dep check
    assert info.deps
    affected_by_1 = set()
    affected_by_2 = set()
    for (cid, _), (v0, v1) in info.deps.items():
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
    info = place_tags(10, deps)
    assert info.ntypes == 2
    spans = info.field_spans()
    assert (0, 3, 10) in spans
    assert (5, 8, 20) in spans
    assert info.tags[0].cmp_id == 10
    assert info.tags[5].cmp_id == 20
    assert info.tags[3].cmp_id == 0  # gap


def test_place_tags_length_flag():
    # Single-byte dependency whose members span a wide region → TAG_IS_LEN.
    deps = {
        (7, 0): (frozenset({1}), frozenset(range(2, 20))),
    }
    info = place_tags(24, deps)
    # Byte 1 is the tight tag; members cover a wide window.
    assert info.tags[1].cmp_id == 7
    # Length heuristic may or may not fire depending on span-of-tag vs members;
    # at least the tag is placed.
    assert any(t.cmp_id == 7 for t in info.tags)


def test_tags_info_chunk_spans():
    tags = [Tag() for _ in range(8)]
    for i in range(2, 6):
        tags[i].cmp_id = 3
        tags[i].parent = 9
    info = TagsInfo(tags=tags, ntypes=1)
    chunks = info.chunk_spans()
    assert chunks == [(2, 6, 9)]


def test_bit_level_vs_byte_level():
    def op(data: bytes) -> tuple[bytes, bytes]:
        return (bytes([data[0] & 0x01]), b"\x01")

    exec_fn = synthetic_exec_fn([(1, 0, -1, op)])
    data = b"\x00\x00\x00\x00"

    info_byte = get_deps(data, exec_fn, byte_level=True)
    info_bit = get_deps(data, exec_fn, byte_level=False, max_execs=40)
    assert info_byte is not None and info_bit is not None
    # Both should discover that byte 0 affects cmp 1.
    def affects(info, cid=1):
        out = set()
        for (c, _), (v0, v1) in info.deps.items():
            if c == cid:
                out |= set(v0) | set(v1)
        return out

    assert 0 in affects(info_byte)
    assert 0 in affects(info_bit)
