"""Regression: TiffMutator._corrupt_offsets wrote 4 bytes at pos + 8 after
only checking pos + 4 <= len, raising struct.error on a truncated IFD entry."""

import struct

from fuzzer_tool.core.mutations.tiff import TiffMutator, parse_tiff


def _truncated_tiff() -> bytes:
    # Header + IFD claiming 1 entry, truncated mid-entry (18 bytes total).
    data = b"II" + struct.pack("<H", 0x2A) + struct.pack("<I", 8)
    data += struct.pack("<H", 1) + struct.pack("<HHI", 0x100, 3, 1)
    assert len(data) == 18  # entry at 10; value_offset (pos+8..pos+12) cut off
    return data


def test_corrupt_offsets_truncated_entry_does_not_raise():
    data = _truncated_tiff()
    header = parse_tiff(data)
    assert header is not None
    out = TiffMutator(seed=1)._corrupt_offsets(data, header, 65536)
    assert out == data  # entry out of range -> left untouched


def test_corrupt_offsets_full_entry_still_mutated():
    data = _truncated_tiff() + struct.pack("<I", 0)  # complete 12-byte entry
    header = parse_tiff(data)
    out = TiffMutator(seed=1)._corrupt_offsets(data, header, 65536)
    assert out[18:22] == b"\xff\xff\xff\xff"
