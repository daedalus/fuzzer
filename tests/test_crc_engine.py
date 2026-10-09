"""Tests for the word-sliced / numpy-lane CRC engine behind compute_checksum.

``compute_checksum`` folded one byte per Python step (~8 MiB/s). Warm models
now go through ``core/crc_engine.py``: slicing-by-8 for short inputs and
crc-braid style independent lanes (numpy) for long ones. Every path must be
bit-identical to the bitwise LFSR, for every width and bit order.
"""

from __future__ import annotations

import zlib

import pytest

from fuzzer_tool.core import crc_engine
from fuzzer_tool.core.berlekamp_massey import _reverse_bits, compute_checksum
from fuzzer_tool.core.crc_engine import (
    _BUILD_BYTES,
    _LANE_WORDS,
    _LANES_MIN_BYTES,
    _MAX_ENGINES,
    _MAX_TRACKED,
    _SLICE_MIN_BYTES,
    BitOrder,
)
from fuzzer_tool.core.rand_pool import RandPool

_WIDTHS = (8, 12, 16, 24, 31, 32, 40, 64)
_CHECK = b"123456789"


@pytest.fixture(autouse=True)
def _clean_engine_cache():
    crc_engine.clear()
    yield
    crc_engine.clear()


def _rand_int(rng: RandPool, bits: int) -> int:
    return int.from_bytes(rng.randbytes(8), "little") & ((1 << bits) - 1)


def _bitwise(data, poly, width, init=0, final_xor=0, reflect_in=False, reflect_out=False):
    """Oracle: bit-at-a-time LFSR, independent of any table."""
    mask = (1 << width) - 1
    reg = init & mask
    poly &= mask
    for byte in data:
        if reflect_in:
            reg ^= byte
            for _ in range(8):
                reg = (reg >> 1) ^ ((reg & 1) * poly)
            continue
        reg ^= byte << (width - 8)
        for _ in range(8):
            reg = ((reg << 1) ^ (((reg >> (width - 1)) & 1) * poly)) & mask
    if reflect_out:
        reg = _reverse_bits(reg, width)
    return (reg ^ final_xor) & mask


def _warm(poly, width, reflect_in):
    """Push enough bytes through a model that its engine is built."""
    compute_checksum(bytes(_BUILD_BYTES), poly, width, 0, 0, reflect_in)
    order = BitOrder.LSB_FIRST if reflect_in else BitOrder.MSB_FIRST
    assert crc_engine.cached(poly, width, order)


# ---------------------------------------------------------------------------
# Control: the oracle itself must reproduce published check values (Rule 46).
# ---------------------------------------------------------------------------


def test_oracle_matches_published_check_values():
    assert _bitwise(_CHECK, 0xEDB88320, 32, 0xFFFFFFFF, 0xFFFFFFFF, True) == 0xCBF43926
    assert _bitwise(_CHECK, 0x04C11DB7, 32, 0xFFFFFFFF, 0) == 0x0376E6E7  # CRC-32/MPEG-2
    assert _bitwise(_CHECK, 0x42F0E1EBA9EA3693, 64, 0, 0) == 0x6C40DF5F0B497347  # CRC-64/ECMA
    assert _bitwise(_CHECK, 0x80F, 12, 0, 0) == 0xF5B  # CRC-12/DECT


# ---------------------------------------------------------------------------
# Equivalence on every path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", _WIDTHS)
@pytest.mark.parametrize("reflect_in", [False, True])
def test_warm_paths_match_bitwise(width, reflect_in):
    rng = RandPool(seed=width * 2 + reflect_in)
    poly = _rand_int(rng, width) | 1
    _warm(poly, width, reflect_in)

    # Lengths straddle the slice, word, lane-block and tail boundaries.
    block = 8 * _LANE_WORDS
    lengths = (0, 1, 7, 8, 9, _SLICE_MIN_BYTES - 1, _SLICE_MIN_BYTES, 1000)
    lengths += (_LANES_MIN_BYTES - 1, _LANES_MIN_BYTES, _LANES_MIN_BYTES + block + 13)
    for n in lengths:
        init, xo = _rand_int(rng, width), _rand_int(rng, width)
        data = rng.randbytes(n)
        for reflect_out in (False, True):
            want = _bitwise(data, poly, width, init, xo, reflect_in, reflect_out)
            got = compute_checksum(data, poly, width, init, xo, reflect_in, reflect_out)
            assert got == want, (width, reflect_in, reflect_out, n)


def test_lane_path_matches_zlib_on_large_buffer():
    data = RandPool(seed=3).randbytes(300_000)
    std = (0xEDB88320, 32, 0xFFFFFFFF, 0xFFFFFFFF, True, False)
    assert compute_checksum(data, *std) == zlib.crc32(data)


# ---------------------------------------------------------------------------
# Falsification: a fast path that ignored data or model would pass the
# all-zero equivalence trivially; these must change the result.
# ---------------------------------------------------------------------------


def test_falsify_poly_and_single_byte_flip_change_result():
    data = bytearray(RandPool(seed=5).randbytes(_LANES_MIN_BYTES * 2))
    a = compute_checksum(bytes(data), 0x04C11DB7, 32)
    assert a != compute_checksum(bytes(data), 0x1EDC6F41, 32)

    # Flip one byte inside a middle lane, one inside the tail.
    for pos in (len(data) // 3, len(data) - 3):
        data[pos] ^= 0x01
        assert compute_checksum(bytes(data), 0x04C11DB7, 32) != a
        assert compute_checksum(bytes(data), 0x04C11DB7, 32) == _bitwise(data, 0x04C11DB7, 32)
        data[pos] ^= 0x01


# ---------------------------------------------------------------------------
# Adversarial inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reflect_in", [False, True])
def test_adversarial_inputs(reflect_in):
    width = 64
    poly = 0xC96C5795D7870F42 if reflect_in else 0x42F0E1EBA9EA3693
    _warm(poly, width, reflect_in)
    n = _LANES_MIN_BYTES + 77
    raw = RandPool(seed=9).randbytes(n + 3)

    cases = {
        "zeros": bytes(n),
        "ones": b"\xff" * n,
        "bytearray": bytearray(raw[:n]),
        "unaligned memoryview": memoryview(raw)[3:],
    }
    for name, data in cases.items():
        want = _bitwise(bytes(data), poly, width, (1 << 64) - 1, 0, reflect_in)
        assert compute_checksum(data, poly, width, (1 << 64) - 1, 0, reflect_in) == want, name


def test_adversarial_poly_bits_above_width_are_ignored():
    data = RandPool(seed=11).randbytes(_LANES_MIN_BYTES)
    for reflect_in in (False, True):
        want = compute_checksum(data, 0x1021, 16, 0xFFFF, 0, reflect_in)
        assert compute_checksum(data, 0xABC1021, 16, 0xFFFF, 0, reflect_in) == want
        assert want == _bitwise(data, 0x1021, 16, 0xFFFF, 0, reflect_in)


@pytest.mark.parametrize("width", [3, 7, 65])
def test_unsupported_widths_keep_byte_loop_and_never_build(width):
    poly = 0x3
    data = bytes(_LANES_MIN_BYTES)
    compute_checksum(data, poly, width, 0, 0, True)
    assert not crc_engine.cached(poly, width, BitOrder.LSB_FIRST)


# ---------------------------------------------------------------------------
# Amortisation and memory bounds
# ---------------------------------------------------------------------------


def test_cold_short_calls_do_not_build_engine():
    """The recovery search probes many polys once on short pairs: no build."""
    for poly in range(1, 200, 2):
        compute_checksum(b"x" * 64, poly, 32)
        assert not crc_engine.cached(poly, 32, BitOrder.MSB_FIRST)


def test_cumulative_bytes_build_engine():
    chunk = _BUILD_BYTES // 4
    for _ in range(3):
        compute_checksum(bytes(chunk), 0x04C11DB7, 32)
    assert not crc_engine.cached(0x04C11DB7, 32, BitOrder.MSB_FIRST)
    compute_checksum(bytes(chunk), 0x04C11DB7, 32)
    assert crc_engine.cached(0x04C11DB7, 32, BitOrder.MSB_FIRST)


def test_cache_is_bounded():
    for poly in range(1, 2 * _MAX_ENGINES * 2, 2):
        compute_checksum(bytes(_BUILD_BYTES), poly, 32)
    for poly in range(1, 2 * _MAX_TRACKED * 4, 2):
        compute_checksum(b"x" * _SLICE_MIN_BYTES, poly, 16)
    engines, tracked = crc_engine.sizes()
    assert engines <= _MAX_ENGINES
    assert tracked <= _MAX_TRACKED
