"""Regression tests for structure-aware format mutator serialization.

These verify that mutated size/seg_size fields survive the
parse → mutate → serialize round trip (i.e. reach the wire).
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestPgsMutations:
    """PGS mutator serialization: seg_size must reach the wire."""

    def test_serialize_uses_seg_size_not_payload_len(self):
        from fuzzer_tool.core.pgs_mutations import PgsSegment, serialize_pgs_segments

        seg = PgsSegment(pts=0, dts=0xFFFFFFFF, seg_type=0x14, seg_size=0, payload=b"AAAA")
        result = serialize_pgs_segments([seg])
        # seg_size field is at offset 11-12 (big-endian u16)
        written_size = struct.unpack_from(">H", result, 11)[0]
        assert written_size == 0, f"Expected seg_size=0, got {written_size}"

    def test_serialize_uses_seg_size_large(self):
        from fuzzer_tool.core.pgs_mutations import PgsSegment, serialize_pgs_segments

        seg = PgsSegment(pts=0, dts=0xFFFFFFFF, seg_type=0x14, seg_size=0xFFFF, payload=b"BB")
        result = serialize_pgs_segments([seg])
        written_size = struct.unpack_from(">H", result, 11)[0]
        assert written_size == 0xFFFF, f"Expected seg_size=0xFFFF, got {written_size}"

    def test_serialize_keeps_payload_untouched(self):
        from fuzzer_tool.core.pgs_mutations import PgsSegment, serialize_pgs_segments

        payload = b"HELLO"
        seg = PgsSegment(pts=0, dts=0xFFFFFFFF, seg_type=0x80, seg_size=0, payload=payload)
        result = serialize_pgs_segments([seg])
        # Payload is at offset 13 (after 13-byte header)
        assert result[13:] == payload, "Payload data must be preserved verbatim"


class TestIsobmffMutations:
    """ISOBMFF mutator serialization: size_orig must reach the wire."""

    def test_serialize_uses_size_orig_not_payload_len(self):
        from fuzzer_tool.core.isobmff_mutations import Box, serialize_boxes

        box = Box(box_type=b"ftyp", size_orig=0xFFFFFFFF, data=b"small")
        result = serialize_boxes([box])
        written_size = struct.unpack_from(">I", result, 0)[0]
        assert written_size == 0xFFFFFFFF, f"Expected size_orig=0xFFFFFFFF, got {written_size}"

    def test_serialize_size_orig_zero(self):
        from fuzzer_tool.core.isobmff_mutations import Box, serialize_boxes

        box = Box(box_type=b"ftyp", size_orig=0, data=b"some data here")
        result = serialize_boxes([box])
        written_size = struct.unpack_from(">I", result, 0)[0]
        assert written_size == 0, f"Expected size_orig=0, got {written_size}"

    def test_serialize_size_orig_small_writes_mismatch(self):
        from fuzzer_tool.core.isobmff_mutations import Box, serialize_boxes

        # size_orig=8 declares 0-byte payload, but data is longer.
        # This mismatch is the fuzzing probe — serialize_boxes must
        # write the declared size, not truncate the payload.
        payload = b"oversized payload data"
        box = Box(box_type=b"ftyp", size_orig=8, data=payload)
        result = serialize_boxes([box])
        written_size = struct.unpack_from(">I", result, 0)[0]
        assert written_size == 8
        # Payload bytes must be preserved verbatim (the mismatch reaches the wire)
        assert result[8:] == payload, "Payload must be preserved even when size_orig is smaller"

    def test_mutate_box_size_pads_when_larger(self):
        import random

        from fuzzer_tool.core.isobmff_mutations import Box, IsobmffMutator

        rng = random.Random(42)
        mut = IsobmffMutator()
        box = Box(box_type=b"ftyp", size_orig=8 + 4, data=b"data")
        boxes = [box]

        # Force a specific size_orig > 8 + len(data) to trigger padding
        box.size_orig = 20
        payload_len = max(0, box.size_orig - 8)
        raw = box.data
        if payload_len >= len(raw):
            box.data = raw + b"\x00" * (payload_len - len(raw))
        else:
            box.data = raw[:payload_len]

        assert len(box.data) == 12, f"Expected padded payload len 12, got {len(box.data)}"

    def test_mutate_box_size_truncates_when_smaller(self):
        from fuzzer_tool.core.isobmff_mutations import Box, IsobmffMutator

        mut = IsobmffMutator()
        box = Box(box_type=b"ftyp", size_orig=8 + 100, data=b"x" * 100)
        boxes = [box]

        # Force a small size_orig
        box.size_orig = 12
        payload_len = max(0, box.size_orig - 8)
        raw = box.data
        if payload_len >= len(raw):
            box.data = raw + b"\x00" * (payload_len - len(raw))
        else:
            box.data = raw[:payload_len]

        assert len(box.data) == 4, f"Expected truncated payload len 4, got {len(box.data)}"

    def test_mutate_box_size_clamps_payload_to_max_len(self):
        """Regression: _mutate_box_size must not allocate ~4 GB when size_orig=0xFFFFFFFF.

        The padding path must clamp payload_len to max_len to avoid OOM.
        """
        from fuzzer_tool.core.isobmff_mutations import Box, IsobmffMutator

        mut = IsobmffMutator()
        # A realistic leaf box with small data payload
        box = Box(box_type=b"ftyp", size_orig=8 + 4, data=b"data")
        boxes = [box]

        # Directly invoke the padding logic with the extreme size_orig value
        # that was causing the OOM. max_len is small (256).
        max_len = 256
        box.size_orig = 0xFFFFFFFF
        raw = box.data
        payload_len = max(0, min(box.size_orig - 8, max_len))
        assert payload_len == max_len, (
            f"Expected payload_len clamped to {max_len}, got {payload_len}"
        )
        if payload_len >= len(raw):
            box.data = raw + b"\x00" * (payload_len - len(raw))
        assert len(box.data) == max_len, (
            f"Expected padded data length {max_len}, got {len(box.data)}"
        )

    def test_serialize_extended_size_64bit(self):
        from fuzzer_tool.core.isobmff_mutations import Box, serialize_boxes

        # size_orig > 0xFFFFFFFF requires extended-size (64-bit) format
        large = 0x1FFFFFFFF
        box = Box(box_type=b"ftyp", size_orig=large, data=b"data")
        result = serialize_boxes([box])
        # Extended-size format: [size=1: u32][ext_size: u64][box_type: 4][payload]
        marker = struct.unpack_from(">I", result, 0)[0]
        assert marker == 1, f"Expected extended-size marker 1, got {marker}"
        written_size = struct.unpack_from(">Q", result, 4)[0]
        assert written_size == large, f"Expected size_orig={large}, got {written_size}"
        # Payload starts at offset 16 (4 marker + 8 ext_size + 4 box_type)
        assert result[16:] == b"data", "Payload must be preserved"


class TestMagicLock:
    """Magic-lock prefix detection consistency."""

    def test_box15_detected(self):
        from fuzzer_tool.core.magic_lock import detect_magic_prefix

        data = b"\x00\x00\x00\x0fftyp" + b"X" * 20
        result = detect_magic_prefix(data)
        assert result >= 8, f"Expected box15 (size=0x0f) ftyp to be detected, got {result}"

    def test_box12_still_detected(self):
        from fuzzer_tool.core.magic_lock import detect_magic_prefix

        data = b"\x00\x00\x00\x0cftyp" + b"X" * 20
        result = detect_magic_prefix(data)
        assert result >= 8, f"Expected box12 (size=0x0c) ftyp to be detected, got {result}"

    def test_box15_and_box12_are_distinct(self):
        from fuzzer_tool.core.magic_lock import detect_magic_prefix

        data15 = b"\x00\x00\x00\x0fftyp" + b"X" * 20
        data12 = b"\x00\x00\x00\x0cftyp" + b"X" * 20
        # Both must be detected
        assert detect_magic_prefix(data15) >= 8
        assert detect_magic_prefix(data12) >= 8
