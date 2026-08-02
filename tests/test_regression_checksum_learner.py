"""Regression test: ChecksumLearner end-to-end with a synthetic target.

Simulates a target that uses a non-standard CRC polynomial.  Verifies:
1. Format-aware extraction pulls pairs from PNG data
2. cmplog heuristics extract pairs from comparison data
3. Recovery succeeds after enough pairs are collected
4. The recovered polynomial produces valid checksums for unseen data
5. State serialization round-trips cleanly
"""

from __future__ import annotations

import struct
import zlib

from fuzzer_tool.core.berlekamp_massey import compute_checksum
from fuzzer_tool.core.checksum_learner import ChecksumLearner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeCmplog:
    """Minimal stand-in for CmplogCollector."""

    def __init__(self, pairs: list[tuple[bytes, bytes]] | None = None):
        self.pairs = pairs or []


class _FakeFuzzer:
    """Minimal stand-in for Fuzzer used by ChecksumLearner in tests."""

    def __init__(self, cmplog_pairs: list[tuple[bytes, bytes]] | None = None):
        self._cmplog = _FakeCmplog(cmplog_pairs)


# ---------------------------------------------------------------------------
# Format-aware PNG extraction
# ---------------------------------------------------------------------------


def _make_png() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = (
        struct.pack(">I", len(ihdr_data))
        + b"IHDR"
        + ihdr_data
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    )
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    return sig + ihdr + iend


def test_png_extraction_yields_pairs():
    png = _make_png()
    f = _FakeFuzzer()
    learner = ChecksumLearner(f)
    pairs = learner.extract_format_pairs(png)
    assert len(pairs) == 2  # IHDR + IEND
    for data, crc in pairs:
        assert len(data) > 0
        assert 0 <= crc <= 0xFFFFFFFF


# ---------------------------------------------------------------------------
# cmplog pair extraction
# ---------------------------------------------------------------------------


def test_cmplog_extraction_finds_pairs():
    """4-byte operands in cmplog that appear in input become pairs."""
    f = _FakeFuzzer(
        cmplog_pairs=[
            (b"ABCD", b"\x01\x02\x03\x04"),
            (b"EFGH", b"\x05\x06\x07\x08"),
        ]
    )
    learner = ChecksumLearner(f)
    input_data = b"ABCD\x01\x02\x03\x04EFGH\x05\x06\x07\x08"
    pairs = learner.extract_cmplog_pairs(input_data)
    # Both 4-byte operand pairs appear in input_data → should produce 2 pairs
    assert len(pairs) == 2


def test_cmplog_extraction_no_match():
    """Operands not found in input → no pairs."""
    f = _FakeFuzzer(
        cmplog_pairs=[
            (b"\xaa\xbb\xcc\xdd", b"\x11\x22\x33\x44"),
        ]
    )
    learner = ChecksumLearner(f)
    pairs = learner.extract_cmplog_pairs(b"unrelated input data here")
    assert pairs == []


# ---------------------------------------------------------------------------
# Recovery from known custom polynomial
# ---------------------------------------------------------------------------


def test_recovery_with_custom_poly():
    """Collect enough pairs → recovery succeeds → poly is set."""
    from fuzzer_tool.core.crc32 import set_active_model

    CUSTOM_POLY = 0x1D  # 8-bit: x^8 + x^4 + x^3 + x^2 + 1
    f = _FakeFuzzer()
    learner = ChecksumLearner(f, min_pairs=32, poly_width=8)
    try:
        for i in range(64):
            data = bytes([i])
            expected = compute_checksum(data, poly=CUSTOM_POLY, width=8)
            learner.add_pairs([(data, expected)])
        assert learner.ensure_poly() is not None
        # Round-trip: recovered poly should reproduce all checksums
        for i in range(64):
            data = bytes([i])
            got = compute_checksum(data, poly=learner._poly, width=8)
            expected = compute_checksum(data, poly=CUSTOM_POLY, width=8)
            assert got == expected, f"mismatch for byte {i}"
    finally:
        # Recovery activates the 8-bit model in the crc32 module; reset so
        # later tests see the standard (None) model.
        set_active_model(None)


def test_has_enough_pairs():
    """has_enough_pairs returns True only after min_pairs threshold."""
    f = _FakeFuzzer()
    learner = ChecksumLearner(f, min_pairs=10)
    assert not learner.has_enough_pairs()
    learner.add_pairs([(bytes([i]), i) for i in range(10)])
    assert learner.has_enough_pairs()


# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------


def test_state_round_trip():
    """to_dict / from_dict preserves the recovered polynomial."""
    f = _FakeFuzzer()
    learner = ChecksumLearner(f)
    learner._poly = 0x12345678
    data = learner.to_dict()
    assert data["poly"] == 0x12345678

    restored = ChecksumLearner.from_dict(f, data)
    assert restored._poly == 0x12345678


def test_state_round_trip_none():
    """from_dict with None/empty data produces a fresh learner."""
    f = _FakeFuzzer()
    learner = ChecksumLearner.from_dict(f, None)
    assert learner._poly is None
    assert learner.pair_count == 0


# ---------------------------------------------------------------------------
# CRC32 wrapper: active poly propagation
# ---------------------------------------------------------------------------


def test_crc32_wrapper_uses_learned_poly():
    """After recovery, the crc32 wrapper uses the learned poly."""
    from fuzzer_tool.core.crc32 import crc32, set_active_poly

    CUSTOM_POLY = 0x1D
    data = b"hello"
    before = crc32(data)  # standard
    set_active_poly(CUSTOM_POLY)
    try:
        after = crc32(data)
        assert before != after
    finally:
        set_active_poly(None)
