"""Heap-chunk SIZE-field mutators (glibc smallbin-range probes).

Phrack #73 "Poor Little Chunks" (Ax): glibc 2.41/2.43 stopped validating a
chunk SIZE in the 0x20-0x3F0 smallbin range, so these operators push plausible
size/length fields into that range to probe the regressed allocator path.

Determinism follows Hard Rule 39: a ``ScriptedRng`` drives the exact draw
sequence and the expected bytes are derived by hand (independent arithmetic),
never by echoing the function's own output back.
"""

import struct

import pytest

from fuzzer_tool.core.mutations.heap_chunk import (
    MAX_SMALLBIN_CHUNK,
    MIN_CHUNK,
    SMALLBIN_SIZES,
    chunk_overlap,
    smallbin_size,
)
from tests.support.scripted_rng import ScriptedRng

CHUNK_ALIGN = 0x10
SIZE_BITS = 0x7


def _decode(field: int) -> tuple[int, int]:
    """Split an encoded chunk SIZE field into (chunk_size, flag_bits)."""
    return field & ~SIZE_BITS, field & SIZE_BITS


# ── Smallbin boundary invariants (pins the constants to glibc, not the code) ──


def test_smallbin_range_matches_glibc():
    # 64-bit glibc: MINSIZE=0x20, smallbins hold 16-aligned sizes below
    # MIN_LARGE_SIZE (0x400) -> 0x20..0x3F0.
    assert MIN_CHUNK == 0x20
    assert MAX_SMALLBIN_CHUNK == 0x3F0
    assert SMALLBIN_SIZES[0] == 0x20
    assert SMALLBIN_SIZES[-1] == 0x3F0
    assert all(s % CHUNK_ALIGN == 0 for s in SMALLBIN_SIZES)
    # 0x20, 0x30, ... 0x3F0 is (0x3F0-0x20)/0x10 + 1 = 62 sizes.
    assert len(SMALLBIN_SIZES) == 62


# ── Determinism: exact bytes from a scripted draw sequence ───────────────────


def test_smallbin_size_exact():
    data = bytes(16)  # all-zero -> every aligned 4-field is a candidate
    # width=choice(WIDTHS)->WIDTHS[1]=4; offset index 2 -> candidate 8;
    # endian 0 -> "<"; size index 3 -> 0x20+3*0x10=0x50; flag index 1 -> 1.
    rng = ScriptedRng(choice_idxs=(1,), randints=(2, 0, 3, 1))

    out = smallbin_size(data, rng)

    expected = bytearray(16)
    struct.pack_into("<I", expected, 8, 0x50 | 1)
    assert out == bytes(expected)
    size, flags = _decode(struct.unpack_from("<I", out, 8)[0])
    assert size == 0x50
    assert flags == 1


def test_chunk_overlap_exact():
    data = bytearray(16)
    struct.pack_into("<I", data, 4, 0x200)  # a plausible 512-byte size field
    # width WIDTHS[1]=4; offset index 1 -> candidate 4; endian 0 -> "<";
    # k=2; sign 0 -> +. 0x200 + 2*0x10 = 0x220, flags preserved (0).
    rng = ScriptedRng(choice_idxs=(1,), randints=(1, 0, 2, 0))

    out = chunk_overlap(bytes(data), rng)

    expected = bytearray(data)
    struct.pack_into("<I", expected, 4, 0x220)
    assert out == bytes(expected)
    assert struct.unpack_from("<I", out, 4)[0] == 0x220


# ── Falsification: the result SIZE never escapes the smallbin range ──────────


@pytest.mark.parametrize("size_idx", [0, 30, len(SMALLBIN_SIZES) - 1])
@pytest.mark.parametrize("flag_idx", [0, 1, 4])
def test_smallbin_size_always_in_range(size_idx, flag_idx):
    data = bytes(32)
    # width 8, offset candidate 0, endian 0, chosen size/flag indices.
    rng = ScriptedRng(choice_idxs=(2,), randints=(0, 0, size_idx, flag_idx))

    out = smallbin_size(data, rng)

    assert len(out) == len(data)
    size, _flags = _decode(struct.unpack_from("<Q", out, 0)[0])
    assert MIN_CHUNK <= size <= MAX_SMALLBIN_CHUNK
    assert size % CHUNK_ALIGN == 0


@pytest.mark.parametrize("start", [0x8, 0x3F0, 0x4000, 0xFFFFFFFF])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("sign_draw", [0, 1])
def test_chunk_overlap_clamps_into_range(start, k, sign_draw):
    data = bytearray(16)
    struct.pack_into("<I", data, 0, start & 0xFFFFFFFF)
    # width 4, offset candidate 0, endian 0, k, sign.
    rng = ScriptedRng(choice_idxs=(1,), randints=(0, 0, k, sign_draw))

    out = chunk_overlap(bytes(data), rng)

    assert len(out) == len(data)
    size, _flags = _decode(struct.unpack_from("<I", out, 0)[0])
    assert MIN_CHUNK <= size <= MAX_SMALLBIN_CHUNK
    assert size % CHUNK_ALIGN == 0


def test_chunk_overlap_preserves_flag_bits():
    data = bytearray(16)
    struct.pack_into("<I", data, 0, 0x200 | 0x5)  # prev_inuse | non_main_arena
    rng = ScriptedRng(choice_idxs=(1,), randints=(0, 0, 1, 0))

    out = chunk_overlap(bytes(data), rng)

    _size, flags = _decode(struct.unpack_from("<I", out, 0)[0])
    assert flags == 0x5


# ── Adversarial: hostile / degenerate inputs never raise or grow ─────────────


@pytest.mark.parametrize("op", [smallbin_size, chunk_overlap])
@pytest.mark.parametrize("data", [b"", b"\x00", b"\x00\x00\x00", b"abc"])
def test_too_small_returns_unchanged(op, data):
    # Below the 4-byte minimum field: nothing to corrupt, returned verbatim.
    rng = ScriptedRng(choice_idxs=(0,), randints=(0, 0, 0, 0))
    out = op(data, rng)
    assert out == data


@pytest.mark.parametrize("op", [smallbin_size, chunk_overlap])
def test_all_ascii_uses_fallback_offset(op):
    # No non-text window -> no candidates -> the randint fallback offset path.
    data = b"A" * 16
    rng = ScriptedRng(choice_idxs=(1,), randints=(3, 0, 1, 0))
    out = op(data, rng)
    assert len(out) == len(data)
    assert isinstance(out, bytes)


@pytest.mark.parametrize("op", [smallbin_size, chunk_overlap])
def test_width_exceeds_length(op):
    # width 8 drawn but only 5 bytes: declines rather than indexing past end.
    data = b"\x00\x01\x02\x03\x04"
    rng = ScriptedRng(choice_idxs=(2,), randints=(0, 0, 0, 0))
    out = op(data, rng)
    assert out == data


@pytest.mark.parametrize("op", [smallbin_size, chunk_overlap])
def test_length_preserved(op):
    data = bytes(range(64))
    rng = ScriptedRng(choice_idxs=(1,), randints=(0, 0, 1, 0))
    out = op(data, rng)
    assert len(out) == len(data)
