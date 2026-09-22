"""Tests for ``core/mutations/webm.py`` -- EBML/Matroska structure parsing.

This module had zero dedicated test coverage before the WFC chunk-reorder
rollout to webm (``core/wfc_chunks.py``, P2-1 of
``docs/handover/handover_generators_2026-09-20.md``) needed it: giving it
the first input over 126 bytes surfaced a real decode bug in
``_parse_element``'s size-vint handling that every prior real-file test
(all under 127 bytes) had been too small to reach.

The bug: ``_parse_element`` stripped a size vint's marker bit with
``size_raw_val & ((1 << (7 + 8 * (length - 1))) - 1))``, which only equals
the correct mask (``(1 << (7 * length)) - 1``) at ``length == 1``. Any
element declaring a size that needs 2+ size-vint bytes -- i.e. any element
at least 127 bytes long, which in practice is every Segment and Cluster in
a real WebM file -- decoded to a value many times too large, always
failing the subsequent bounds check and making ``_parse_element`` return
``None`` for that element (and therefore usually for the whole file, since
``parse_webm`` requires both the header and the Segment to parse).
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.webm import (
    CONTAINER_IDS,
    Element,
    _encode_size_vint,
    _parse_element,
    _read_vint,
    parse_webm,
    serialize_webm,
)


def _leaf(elem_id: int, id_raw: bytes, payload: bytes) -> Element:
    return Element(
        elem_id=elem_id,
        id_raw=id_raw,
        size_raw=_encode_size_vint(len(payload)),
        size_val=len(payload),
        data=payload,
    )


def _container(elem_id: int, id_raw: bytes, children: list[Element]) -> Element:
    payload = b"".join(serialize_webm([c]) for c in children)
    return Element(
        elem_id=elem_id,
        id_raw=id_raw,
        size_raw=_encode_size_vint(len(payload)),
        size_val=len(payload),
        data=b"",
        children=children,
    )


# ═══════════════════════════════════════════════════════════════════
# Size-vint mask: the bug this file exists to pin
# ═══════════════════════════════════════════════════════════════════


class TestSizeVintMask:
    def test_single_byte_size_round_trips(self):
        """length == 1 is where the old (also-correct-here) formula lived --
        must keep working."""
        for value in (0, 1, 63, 126):
            raw = _encode_size_vint(value)
            assert len(raw) == 1
            val, size_raw, pos = _read_vint(raw, 0, max_len=8)
            length = len(size_raw)
            decoded = val & ((1 << (7 * length)) - 1)
            assert decoded == value

    def test_two_byte_size_round_trips(self):
        """value=127 is the smallest that needs a 2-byte vint -- exactly the
        boundary the old formula (correct only for length==1) first missed."""
        for value in (127, 128, 200, 1000, 16382):
            raw = _encode_size_vint(value)
            assert len(raw) == 2
            val, size_raw, pos = _read_vint(raw, 0, max_len=8)
            length = len(size_raw)
            decoded = val & ((1 << (7 * length)) - 1)
            assert decoded == value

    def test_three_byte_size_round_trips(self):
        for value in (16383, 16384, 100000, 2097150):
            raw = _encode_size_vint(value)
            assert len(raw) == 3
            val, size_raw, pos = _read_vint(raw, 0, max_len=8)
            length = len(size_raw)
            decoded = val & ((1 << (7 * length)) - 1)
            assert decoded == value

    def test_parse_element_recovers_correct_size_val_past_126_bytes(self):
        """The actual regression: an element whose payload is >=127 bytes
        must parse with the size it was given, not something wildly larger
        that then fails ``_parse_element``'s own bounds check."""
        payload = b"x" * 200
        leaf = _leaf(0x86, b"\x86", payload)  # CodecID's ID, arbitrary leaf here
        raw = serialize_webm([leaf])
        parsed, pos = _parse_element(raw, 0)
        assert parsed is not None
        assert parsed.size_val == 200
        assert parsed.data == payload
        assert pos == len(raw)


# ═══════════════════════════════════════════════════════════════════
# parse_webm / serialize_webm round-trip
# ═══════════════════════════════════════════════════════════════════


class TestParseSerializeRoundTrip:
    def _sample(self, cluster_payload_len: int = 200) -> bytes:
        header = _leaf(0x1A45DFA3, b"\x1a\x45\xdf\xa3", b"\x01\x02\x03")
        children = [
            _leaf(0x114D9B74, b"\x11\x4d\x9b\x74", b"seek"),
            _leaf(0x1549A966, b"\x15\x49\xa9\x66", b"info"),
            _leaf(0x1F43B675, b"\x1f\x43\xb6\x75", b"c" * cluster_payload_len),
        ]
        segment = _container(0x18538067, b"\x18\x53\x80\x67", children)
        return serialize_webm([header, segment])

    def test_small_file_round_trips(self):
        data = self._sample(cluster_payload_len=10)
        top = parse_webm(data)
        assert top is not None
        assert len(top[1].children) == 3
        assert serialize_webm(top) == data

    def test_file_with_a_large_cluster_round_trips(self):
        """This is the case the mask bug broke: a Cluster (or anything else)
        over 126 bytes made the whole file fail to parse."""
        data = self._sample(cluster_payload_len=500)
        top = parse_webm(data)
        assert top is not None
        assert len(top[1].children) == 3
        cluster = [c for c in top[1].children if c.elem_id == 0x1F43B675][0]
        assert cluster.size_val == 500
        assert cluster.data == b"c" * 500
        assert serialize_webm(top) == data

    def test_repeated_cluster_children_all_present(self):
        """Real files repeat Cluster; each occurrence must survive with its
        own bytes, not collapse or duplicate."""
        header = _leaf(0x1A45DFA3, b"\x1a\x45\xdf\xa3", b"")
        children = [
            _leaf(0x1F43B675, b"\x1f\x43\xb6\x75", b"cluster-1" * 20),
            _leaf(0x1F43B675, b"\x1f\x43\xb6\x75", b"cluster-2" * 20),
            _leaf(0x1F43B675, b"\x1f\x43\xb6\x75", b"cluster-3" * 20),
        ]
        segment = _container(0x18538067, b"\x18\x53\x80\x67", children)
        data = serialize_webm([header, segment])
        top = parse_webm(data)
        assert top is not None
        payloads = [c.data for c in top[1].children]
        assert payloads == [b"cluster-1" * 20, b"cluster-2" * 20, b"cluster-3" * 20]

    def test_too_short_input_is_rejected(self):
        assert parse_webm(b"") is None
        assert parse_webm(b"\x00\x00") is None

    def test_wrong_magic_is_rejected(self):
        assert parse_webm(b"not an ebml file" * 4) is None

    def test_missing_segment_is_rejected(self):
        header_only = serialize_webm([_leaf(0x1A45DFA3, b"\x1a\x45\xdf\xa3", b"")])
        assert parse_webm(header_only) is None


# ═══════════════════════════════════════════════════════════════════
# Known limitation, documented rather than exercised in production:
# _encode_size_vint's own "value too large" fallback does not round-trip.
# ═══════════════════════════════════════════════════════════════════


class TestEncodeSizeVintOverflowFallback:
    def test_overflow_fallback_does_not_round_trip_as_unknown(self):
        """``_encode_size_vint`` falls back to 8 raw 0xFF bytes for a value
        too large to represent (>= 2**56-ish, never hit by any real element
        size). That fallback is not a valid length-8 EBML vint -- its first
        byte alone (0xFF) already satisfies the marker-bit search, so
        ``_read_vint`` reads it back as a *length-1* "unknown size" and
        leaves the other 7 bytes as stray data. Pinned here as a known gap,
        not fixed: legitimate "unknown size" elsewhere in this module
        already uses the single-byte form this decodes correctly as.
        """
        raw = _encode_size_vint(2**60)
        assert raw == b"\xff" * 8
        val, consumed, pos = _read_vint(raw, 0, max_len=8)
        assert val == -1
        assert len(consumed) == 1  # not 8 -- the gap
        assert pos == 1

    def test_single_byte_unknown_marker_round_trips(self):
        """The form this module's own unknown-size elements actually use."""
        val, consumed, pos = _read_vint(b"\xff", 0, max_len=8)
        assert val == -1
        assert len(consumed) == 1
