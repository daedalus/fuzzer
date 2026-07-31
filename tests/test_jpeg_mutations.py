"""Tests for structure-aware JPEG mutations, incl. length-field overflow."""

import struct

from fuzzer_tool.core.mutations.jpeg import (
    COM,
    EOI,
    SOI,
    JpegMarker,
    JpegMutator,
    parse_jpeg_markers,
    serialize_jpeg_markers,
)

JPEG_HEADER = (
    b"\xff\xd8"  # SOI
    + b"\xff\xe0"  # APP0 marker
    + b"\x00\x10"  # length
    + b"JFIF\x00"  # identifier
    + b"\x01\x01"  # version
    + b"\x00"  # units
    + b"\x00\x01\x00\x01"  # density
    + b"\x00\x00"  # thumbnail
    + b"\xff\xd9"  # EOI
)


class TestJpegMarkerSerialize:
    def test_serialize_roundtrip(self):
        m = JpegMarker(marker=COM, data=b"hello")
        out = m.serialize()
        assert out[:2] == b"\xff\xfe"
        assert struct.unpack(">H", out[2:4])[0] == 2 + len(b"hello")
        assert out[4:] == b"hello"

    def test_regression_serialize_length_field_overflow(self):
        """A segment whose data exceeds the 16-bit length field must not crash.

        Regression: serialize() packed len(data)+2 into struct.pack(">H"),
        which raises struct.error when a mutation (or a corrupt length field
        that absorbs the rest of a large input) grows a segment past 65533
        bytes — crashing the whole fuzzer mid-run.
        """
        # Parse a JPEG whose APP0 length field is corrupt (0) so the parser
        # absorbs the remaining ~70KB of input as one segment's data
        # (parse_jpeg_markers: length < 2 -> seg_data = data[pos:]).
        big_payload = b"A" * 70000
        corrupt = (
            b"\xff\xd8"  # SOI
            + b"\xff\xe0"  # APP0
            + b"\x00\x00"  # length: 0 -> parser takes all remaining bytes
            + big_payload
            + b"\xff\xd9"  # EOI
        )
        markers = parse_jpeg_markers(corrupt)
        assert markers is not None
        seg = [m for m in markers if m.marker not in (SOI, EOI)]
        assert seg and len(seg[0].data) > 65533

        # Must serialize without raising struct.error, and the emitted length
        # field must stay within the 16-bit range. Layout: FF D8 | FF E0 | len | data
        out = serialize_jpeg_markers(markers)
        assert out[4:6] == b"\xff\xff"  # clamped to 0xFFFF
        assert struct.unpack(">H", out[4:6])[0] <= 0xFFFF

    def test_regression_mutate_oversized_segment_no_crash(self):
        """JpegMutator.mutate must not crash on inputs with huge segments."""
        corrupt = b"\xff\xd8" + b"\xff\xe0" + b"\x00\x00" + b"B" * 70000 + b"\xff\xd9"
        m = JpegMutator()
        for _ in range(200):
            out = m.mutate(corrupt, max_len=65536)
            assert isinstance(out, bytes)

    def test_duplicate_marker_still_valid(self):
        """duplicate_marker truncates clones to 64 bytes; round-trip must work."""
        m = JpegMutator()
        for _ in range(50):
            out = m.mutate(JPEG_HEADER, max_len=4096)
            assert isinstance(out, bytes)
            assert len(out) > 0
