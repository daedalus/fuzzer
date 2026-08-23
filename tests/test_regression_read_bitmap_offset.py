"""Regression: ``read_bitmap()`` disagreed with the SHM layout on both axes.

The segment is ``SHM_METADATA_SIZE`` bytes of header, then ``num_entries *
SIZEOF_ENTRY`` bytes of table, then the distance tail.  ``read_bitmap()``
read ``shm_size`` BYTES from the segment BASE:

* wrong offset -- the first 24 bytes returned were path_hash / edge_count /
  diag, not entries;
* wrong length -- ``shm_size`` is the entry COUNT, so it covered one eighth
  of the table.

``services/runner.py`` then bounded the copy with ``len(bitmap) <= shm.size``
(bytes against entries) and memmoved to ``shm._ptr``, over the header.

All of it was inert because ``coverage_env_id`` is set from
``shm_cov.env_id``: source and destination were the same segment, so the
memmove was a self-copy.  These tests assert the offset and the length
directly, because a self-copy passes any end-to-end check.
"""

import ctypes

import pytest

from fuzzer_tool.adapters import libc_shm
from fuzzer_tool.adapters.inprocess import InProcessRunner
from fuzzer_tool.adapters.shm import SHM_METADATA_SIZE, SIZEOF_ENTRY, ShmCoverage


@pytest.fixture
def cov():
    c = ShmCoverage(size=256)
    yield c
    c.cleanup()


@pytest.fixture
def runner(cov):
    r = InProcessRunner.__new__(InProcessRunner)
    r._persistent = None
    r._bitmap_out = None
    r._shm_ptr = None
    r.coverage_env_id = cov.env_id
    r.shm_size = cov.num_entries
    yield r
    if r._shm_ptr:
        libc_shm.shmdt(r._shm_ptr)


def test_length_covers_the_whole_table_not_one_eighth(cov, runner):
    bitmap = runner.read_bitmap()
    assert bitmap is not None
    assert len(bitmap) == cov.table_bytes
    assert len(bitmap) == cov.num_entries * SIZEOF_ENTRY


def test_read_starts_at_the_table_not_at_the_header(cov, runner):
    """A planted header must not appear in the returned bytes."""
    ctypes.c_uint64.from_address(cov._ptr).value = 0xA1A2A3A4A5A6A7A8  # path_hash
    ctypes.c_uint64.from_address(cov._ptr + 16).value = 0xB1B2B3B4B5B6B7B8  # edge_count

    bitmap = bytes(runner.read_bitmap())
    assert (0xA1A2A3A4A5A6A7A8).to_bytes(8, "little") not in bitmap
    assert (0xB1B2B3B4B5B6B7B8).to_bytes(8, "little") not in bitmap


def test_planted_entry_appears_at_its_table_position(cov, runner):
    """The returned bytes are the table, aligned to entry 0."""
    cov.record_edge(7)
    idx = 7 % cov.num_entries
    bitmap = bytes(runner.read_bitmap())
    entry = bitmap[idx * SIZEOF_ENTRY : (idx + 1) * SIZEOF_ENTRY]
    edge_id = int.from_bytes(entry[:4], "little")
    assert edge_id == 7


def test_bounds_check_is_bytes_against_bytes(cov, runner):
    """The caller's guard must admit a full table, not reject 7/8 of it.

    ``len(bitmap) <= shm.size`` compared a byte length against an entry
    count, so a correctly-sized table (8x the entry count) failed the check
    and the copy was silently skipped.
    """
    bitmap = runner.read_bitmap()
    assert len(bitmap) > cov.size, "the old guard's operands differ by 8x"
    assert len(bitmap) <= cov.table_bytes


def test_destination_offset_leaves_the_header_intact(cov, runner):
    """Copying the table must not clobber the front header.

    Exercises the corrected destination against a SEPARATE segment, which is
    the case the self-copy hid.
    """
    src = ShmCoverage(size=cov.num_entries)
    try:
        src.record_edge(11)
        payload = bytes(
            (ctypes.c_uint8 * src.table_bytes).from_address(src._ptr + SHM_METADATA_SIZE)
        )

        ctypes.c_uint64.from_address(cov._ptr).value = 0xFEEDFACE  # path_hash
        diag_before = cov.read_diag()

        assert len(payload) <= cov.table_bytes
        ctypes.memmove(cov._ptr + SHM_METADATA_SIZE, payload, len(payload))

        assert ctypes.c_uint64.from_address(cov._ptr).value == 0xFEEDFACE
        assert cov.read_diag() == diag_before
        assert 11 in cov.get_edge_ids()
    finally:
        src.cleanup()
