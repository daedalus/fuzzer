"""parse_tiff must reject an IFD offset past the buffer, not raise.

Found by a fuzzgoat position-arena campaign: a mutated buffer carried a TIFF
header whose IFD offset (0x6777646C) pointed far outside the 74-byte input,
``struct.unpack_from`` raised and aborted the whole fuzz loop.
"""

import struct

from fuzzer_tool.core.mutations.tiff import TiffMutator, parse_tiff

IFD_OFFSET_OOB = 0x6777646C  # the offset seen in the aborting campaign


def _tiff(ifd_offset: int, size: int = 74) -> bytes:
    head = b"II" + struct.pack("<HI", 0x002A, ifd_offset)
    return head + bytes(size - len(head))


def test_regression_tiff_ifd_offset_oob():
    assert parse_tiff(_tiff(IFD_OFFSET_OOB)) is None


def test_ifd_count_straddling_the_end_is_rejected():
    # Adversarial: offset in range, but the 2-byte entry count is not.
    assert parse_tiff(_tiff(73)) is None


def test_ifd_offset_at_last_count_slot_parses():
    # Falsification: the tightest in-range offset must still parse.
    hdr = parse_tiff(_tiff(72))
    assert hdr is not None
    assert hdr.num_entries == 0
    assert hdr.ifd_entries == []


def test_mutator_survives_oob_offset():
    out = TiffMutator(seed=1).mutate(_tiff(IFD_OFFSET_OOB))
    assert isinstance(out, bytes)
