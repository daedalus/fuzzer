"""`_generate_random_isobmff` must emit self-consistent container sizes."""

from __future__ import annotations

import struct

from fuzzer_tool.core.mutations.isobmff import (
    IsobmffMutator,
    parse_boxes,
    serialize_boxes,
)

SEEDS = (0, 1, 2, 3)
HDR = 8


def _sizes_ok(data: bytes, pos: int, end: int, depth: int = 0) -> bool:
    """Every box header size matches its extent; containers nest exactly."""
    while pos < end:
        size = struct.unpack_from(">I", data, pos)[0]
        if size < HDR or pos + size > end:
            return False
        typ = data[pos + 4 : pos + 8]
        if typ in (b"moov", b"trak", b"mdia") and not _sizes_ok(data, pos + HDR, pos + size):
            return False
        pos += size
    return pos == end


def test_sizes_match_extent():
    for seed in SEEDS:
        out = IsobmffMutator(seed=seed)._generate_random_isobmff()
        assert _sizes_ok(out, 0, len(out)), seed


def test_round_trips():
    for seed in SEEDS:
        out = IsobmffMutator(seed=seed)._generate_random_isobmff()
        boxes = parse_boxes(out)
        assert boxes is not None
        assert serialize_boxes(boxes) == out, seed


def test_moov_keeps_children():
    out = IsobmffMutator(seed=0)._generate_random_isobmff()
    moov = next(b for b in parse_boxes(out) if b.box_type == b"moov")
    assert [c.box_type for c in moov.children] == [b"mvhd", b"trak"]


def test_truncated_still_parses():
    """Adversarial: max_len cut mid-box must not raise in the parser."""
    for cut in range(1, 120, 7):
        out = IsobmffMutator(seed=1)._generate_random_isobmff(max_len=cut)
        parse_boxes(out)
