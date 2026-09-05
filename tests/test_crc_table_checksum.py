"""Regression tests for the table-driven checksum computation.

``compute_checksum`` ran eight shift-and-conditional-xor steps per byte in
Python. It now folds a byte at a time through a 256-entry table cached per
``(poly, width, reflect_in)``. The table is exactly what the inner loop
computes, so the results must be bit-identical for every configuration --
including the non-standard ones the recovery search probes.
"""

from __future__ import annotations

import os
import random
import zlib

import pytest

from fuzzer_tool.core.berlekamp_massey import _crc_table, _reverse_bits, compute_checksum

# ---------------------------------------------------------------------------
# Oracle: the bit-at-a-time loops, verbatim.
# ---------------------------------------------------------------------------


def _bitwise_checksum(
    data: bytes,
    poly: int,
    width: int = 32,
    init: int = 0,
    final_xor: int = 0,
    reflect_in: bool = False,
    reflect_out: bool = False,
) -> int:
    mask = (1 << width) - 1
    reg = init & mask
    if reflect_in:
        for byte in data:
            reg ^= byte
            for _ in range(8):
                feedback = (reg & 1) * poly
                reg = ((reg >> 1) ^ feedback) & mask
        result = reg & mask
    else:
        for byte in data:
            reg ^= byte << (width - 8)
            for _ in range(8):
                msb = (reg >> (width - 1)) & 1
                reg = ((reg << 1) ^ (msb * poly)) & mask
        result = reg & mask
    if reflect_out:
        result = _reverse_bits(result, width)
    return (result ^ final_xor) & mask


# ---------------------------------------------------------------------------


def test_matches_the_bitwise_loop_over_random_configurations():
    rnd = random.Random(0)
    for _ in range(2000):
        width = rnd.choice([8, 16, 32])
        poly = rnd.getrandbits(width)
        init = rnd.getrandbits(width)
        final_xor = rnd.getrandbits(width)
        reflect_in = rnd.random() < 0.5
        reflect_out = rnd.random() < 0.5
        data = os.urandom(rnd.randrange(0, 40))
        assert compute_checksum(
            data, poly, width, init, final_xor, reflect_in, reflect_out
        ) == _bitwise_checksum(
            data, poly, width, init, final_xor, reflect_in, reflect_out
        )


@pytest.mark.parametrize("width", [8, 16, 32])
@pytest.mark.parametrize("reflect_in", [False, True])
def test_table_entry_is_the_inner_loop(width, reflect_in):
    """Entry b must equal a zero register after folding in byte b."""
    poly = {8: 0x07, 16: 0x8005, 32: 0x04C11DB7}[width]
    table = _crc_table(poly, width, reflect_in)
    assert len(table) == 256
    mask = (1 << width) - 1
    for b in range(256):
        assert table[b] == _bitwise_checksum(bytes([b]), poly, width, 0, 0, reflect_in)
        assert 0 <= table[b] <= mask


def test_crc32_matches_zlib():
    assert compute_checksum(
        b"123456789", 0xEDB88320, 32, 0xFFFFFFFF, 0xFFFFFFFF, True, False
    ) == zlib.crc32(b"123456789")
    payload = os.urandom(5000)
    assert compute_checksum(
        payload, 0xEDB88320, 32, 0xFFFFFFFF, 0xFFFFFFFF, True, False
    ) == zlib.crc32(payload)


def test_empty_data_returns_the_configured_init():
    for width in (8, 16, 32):
        mask = (1 << width) - 1
        assert compute_checksum(b"", 0x1234 & mask, width, 0xAB & mask, 0) == (
            0xAB & mask
        )


def test_table_is_cached_per_configuration():
    _crc_table.cache_clear()
    _crc_table(0xEDB88320, 32, True)
    misses = _crc_table.cache_info().misses
    for _ in range(20):
        compute_checksum(os.urandom(64), 0xEDB88320, 32, 0, 0, True, False)
    assert _crc_table.cache_info().misses == misses
    # A different polynomial is a different table.
    compute_checksum(b"x", 0x04C11DB7, 32, 0, 0, False, False)
    assert _crc_table.cache_info().misses == misses + 1


def test_long_buffer_is_still_exact():
    """The table path is only worth having if it holds at real sizes."""
    payload = os.urandom(20000)
    for poly, reflect_in in ((0xEDB88320, True), (0x04C11DB7, False)):
        assert compute_checksum(
            payload, poly, 32, 0xFFFFFFFF, 0, reflect_in, False
        ) == _bitwise_checksum(payload, poly, 32, 0xFFFFFFFF, 0, reflect_in, False)
