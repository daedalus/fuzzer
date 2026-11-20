"""Tests for the MP4 config-box walker and splice (`mutations/mp4_path.py`)."""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.mp4_path import (
    find_boxes,
    parse_esds,
    splice,
    write_descr_len,
)

from .support.mp4_build import Layout, avcc, box, descr, esds, mp4, stsd

SPS = b"\x67\x42\xc0\x1e\xed\x08\xf2"
PPS = b"\x68\xce\x0f\xc8"
ASC = b"\x12\x10"
AVC = mp4(stsd(b"avc1", avcc(SPS, PPS)))
AAC = mp4(stsd(b"mp4a", esds(ASC)))


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from(">I", data, off)[0]


def _stco_entry(data: bytes) -> int:
    return _u32(data, data.index(b"stco") + 12)


class TestFind:
    def test_avcc_inside_sample_entry(self):
        (hit,) = find_boxes(AVC, frozenset({b"avcC"}))
        assert AVC[hit.start + 4 : hit.start + 8] == b"avcC"
        types = [AVC[a + 4 : a + 8] for a in hit.ancestors]
        assert types == [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd", b"avc1"]

    def test_esds_inside_mp4a(self):
        (hit,) = find_boxes(AAC, frozenset({b"esds"}))
        assert AAC[hit.ancestors[-1] + 4 : hit.ancestors[-1] + 8] == b"mp4a"

    def test_quicktime_sound_v1_skips_extra_fields(self):
        entry = box(b"mp4a", bytes(8) + b"\x00\x01" + bytes(18) + bytes(16) + esds(ASC))
        data = mp4(box(b"stsd", bytes(8) + entry))
        assert len(find_boxes(data, frozenset({b"esds"}))) == 1

    # Adversarial: oversized child sizes and a nesting bomb stay bounded.
    def test_bad_sizes(self):
        data = box(b"moov", struct.pack(">I", 0xFFFF) + b"trak" + bytes(8))
        assert find_boxes(data, frozenset({b"trak"})) == []

    def test_nesting_bomb(self):
        data = avcc(SPS, PPS)
        for _ in range(200):
            data = box(b"moov", data)
        assert find_boxes(data, frozenset({b"avcC"})) == []

    def test_truncated_never_raises(self):
        for cut in range(len(AVC)):
            find_boxes(AVC[:cut], frozenset({b"avcC"}))


class TestSplice:
    def test_grow_fixes_ancestors_and_chunk_offsets(self):
        (hit,) = find_boxes(AVC, frozenset({b"avcC"}))
        payload = AVC[hit.start + 8 : hit.start + hit.size] + b"xyz"
        out = splice(AVC, hit, payload)

        (new,) = find_boxes(out, frozenset({b"avcC"}))
        assert new.size == hit.size + 3
        for old_a, new_a in zip(hit.ancestors, new.ancestors, strict=True):
            assert _u32(out, new_a) == _u32(AVC, old_a) + 3
        assert _stco_entry(out) == _stco_entry(AVC) + 3
        assert out.endswith(AVC[-24:])

    # Falsification: mdat before moov, chunk offsets must not move.
    def test_mdat_first_keeps_chunk_offsets(self):
        data = mp4(stsd(b"avc1", avcc(SPS, PPS)), layout=Layout.MDAT_FIRST)
        (hit,) = find_boxes(data, frozenset({b"avcC"}))
        out = splice(data, hit, data[hit.start + 8 : hit.start + hit.size] + b"x")
        assert _stco_entry(out) == _stco_entry(data)


class TestEsds:
    def test_layout(self):
        (hit,) = find_boxes(AAC, frozenset({b"esds"}))
        payload = AAC[hit.start + 8 : hit.start + hit.size]
        layout = parse_esds(payload)
        assert [d.tag for d in (layout.es, layout.dc, layout.dsi)] == [3, 4, 5]
        assert payload[layout.dsi.payload_off : layout.dsi.end] == ASC
        assert payload[layout.dc.payload_off] == 0x40

    def test_optional_es_fields_skipped(self):
        es = descr(
            3, b"\x00\x01\xe0" + b"\x00\x02" + b"\x03abc" + b"\x00\x04" + descr(4, bytes(13))
        )
        layout = parse_esds(bytes(4) + es)
        assert layout.dc.tag == 4
        assert layout.dsi is None

    # Adversarial: a descriptor length past the end.
    def test_overrun(self):
        assert parse_esds(bytes(4) + b"\x03\x7f\x00") is None

    @pytest.mark.parametrize(
        ("width", "value", "ok"), [(1, 127, True), (1, 128, False), (4, 300, True)]
    )
    def test_write_descr_len(self, width, value, ok):
        buf = bytearray(descr(5, b"", width))
        assert write_descr_len(buf, 1, width, value) is ok
        if ok:
            assert (
                parse_esds(bytes(4) + descr(3, b"\x00\x01\x00" + descr(4, bytes(13)))) is not None
            )
