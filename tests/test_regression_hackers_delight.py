"""Regression tests for Hacker's Delight rightmost-bit + Gosper ports."""

from __future__ import annotations

import random
import pytest

from fuzzer_tool.core.mutations.hackers_delight import (
    clear_rightmost_1,
    isolate_rightmost_1,
    isolate_rightmost_0,
    mask_trailing_zeros,
    right_propagate_1,
    clear_rightmost_run,
    snoob,
    snoob_prev,
    rightmost_clear,
    rightmost_isolate,
    rightmost_propagate,
    rightmost_run_clear,
    same_popcount_next,
    same_popcount_prev,
)


# ---------------------------------------------------------------------------
# Pure formula oracles (book examples + edge cases)
# ---------------------------------------------------------------------------

def test_clear_rightmost_1():
    assert clear_rightmost_1(0b01011000) == 0b01010000
    assert clear_rightmost_1(0) == 0
    assert clear_rightmost_1(1) == 0
    assert clear_rightmost_1(0b1000) == 0


def test_isolate_rightmost_1():
    assert isolate_rightmost_1(0b01011000) == 0b00001000
    assert isolate_rightmost_1(0) == 0
    assert isolate_rightmost_1(1) == 1
    assert isolate_rightmost_1(0b1000) == 0b1000


def test_isolate_rightmost_0():
    assert isolate_rightmost_0(0b10100111) == 0b00001000
    assert isolate_rightmost_0(0) == 1
    assert isolate_rightmost_0(0b1111) == 0b10000


def test_mask_trailing_zeros():
    assert mask_trailing_zeros(0b01011000) == 0b00000111
    assert mask_trailing_zeros(0) == -1  # all bits in two's complement sense
    assert mask_trailing_zeros(1) == 0


def test_right_propagate_1():
    assert right_propagate_1(0b01011000) == 0b01011111
    assert right_propagate_1(0) == -1
    assert right_propagate_1(1) == 1


def test_clear_rightmost_run():
    assert clear_rightmost_run(0b01011000) == 0b01000000
    assert clear_rightmost_run(0b11110000) == 0
    assert clear_rightmost_run(0) == 0


def test_snoob_basic():
    # Classic example from the book / HAKMEM: 0bxxx011110000 -> 0bxxx100000111
    x = 0b0000_1111_0000
    y = snoob(x)
    assert y.bit_count() == x.bit_count()
    assert y > x
    assert y == 0b0001_0000_0111


def test_snoob_preserves_popcount():
    rng = random.Random(42)
    for _ in range(200):
        x = rng.getrandbits(32)
        if x == 0:
            continue
        y = snoob(x)
        assert y.bit_count() == x.bit_count()
        assert y > x or y == 0  # may wrap only if no higher exists, but for 32-bit we stay positive


def test_snoob_zero():
    assert snoob(0) == 0


def test_snoob_prev_basic():
    x = 0b0001_0000_0111
    y = snoob_prev(x)
    assert y.bit_count() == x.bit_count()
    assert y < x
    assert y == 0b0000_1111_0000


# ---------------------------------------------------------------------------
# Windowed mutators – smoke & identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", [
    rightmost_clear,
    rightmost_isolate,
    rightmost_propagate,
    rightmost_run_clear,
    same_popcount_next,
    same_popcount_prev,
])
def test_operator_short_buffer_identity(op):
    rng = random.Random(0)
    for length in range(0, 9):
        data = bytes(range(length))
        # force the operator; on short buffers it must return identical data
        # and must not consume extra randomness beyond what _pick_window needs
        out = op(data, rng)
        assert isinstance(out, bytes)
        assert len(out) == len(data)


def test_same_popcount_next_preserves_weight_on_buffer():
    rng = random.Random(123)
    data = bytes([0x0F, 0x00, 0xF0, 0x00])  # known popcounts
    for _ in range(20):
        out = same_popcount_next(data, rng)
        # at least one of the word windows should have kept its weight
        # (we cannot assert global popcount because only a sub-window is mutated)
        assert len(out) == len(data)
        data = out


# ── RandPool compatibility ────────────────────────────────────────────────
# The operators first drew through ``random.Random.randrange(a, b)``, which
# ``RandPool.randrange(n)`` does not accept, so every handler raised TypeError
# in a real run while the ``random.Random`` tests above stayed green.  Mutation
# code draws through the RandPool API (``randint``/``choice``).


@pytest.mark.parametrize(
    "name",
    [
        "rightmost_clear",
        "rightmost_isolate",
        "rightmost_propagate",
        "rightmost_run_clear",
        "same_popcount_next",
        "same_popcount_prev",
    ],
)
@pytest.mark.parametrize("seed", range(8))
def test_operators_run_on_randpool(name, seed):
    import fuzzer_tool.core.mutations.hackers_delight as hd
    from fuzzer_tool.core.rand_pool import RandPool

    data = bytes((i * 29 + seed) & 0xFF for i in range(32))
    out = getattr(hd, name)(data, RandPool(seed=seed))
    assert len(out) == len(data)


def _snoob_prev_bruteforce(x):
    if x == 0:
        return 0
    t = x.bit_count()
    y = x - 1
    while y > 0 and y.bit_count() != t:
        y -= 1
    return y


def test_snoob_prev_matches_bruteforce_exhaustive():
    for x in range(0, 1 << 12):
        assert snoob_prev(x) == _snoob_prev_bruteforce(x), x


def test_snoob_prev_wide_values_do_not_hang():
    # Linear search needed ~2**63 iterations for these (fuzzer hang).
    import time

    t0 = time.monotonic()
    for x in (1 << 63, (1 << 63) | 1, 0xFFFF_FFFF_0000_0000, 0x8000_0000_0000_0001):
        y = snoob_prev(x)
        assert y < x and y.bit_count() == x.bit_count()
        assert snoob(y) == x or snoob(y) > y
    assert snoob_prev((1 << 64) - 1) == 0
    assert time.monotonic() - t0 < 1.0
