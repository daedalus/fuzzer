"""Iterated input-to-state: Deutsch-style self-consistency (core/i2s_fixpoint.py).

Each fake probe models a target as "parse x, report the compares it made".
Expected bytes are derived from the target model, never echoed from the code.
"""

import struct

from fuzzer_tool.core.i2s_fixpoint import Outcome, patch_step, solve

MAGIC = b"MG"
MAX_ITERS = 8


# ── Fake targets ─────────────────────────────────────────────────────


def _sum_target(data: bytes) -> list[tuple[bytes, bytes]]:
    """``MG | len:u16 | sum:u16 | payload``; the sum covers the length field.

    Stops at the first failing check, like a real parser: a wrong length
    hides the checksum compare entirely.
    """
    if len(data) < 6 or data[:2] != MAGIC:
        return []
    stored_len, stored_sum = struct.unpack_from(">HH", data, 2)
    want_len = len(data) - 6
    if stored_len != want_len:
        return [(struct.pack(">H", want_len), struct.pack(">H", stored_len))]
    want_sum = (sum(data[2:4]) + sum(data[6:])) & 0xFFFF
    return [(struct.pack(">H", want_sum), struct.pack(">H", stored_sum))]


def _valid_sum(data: bytes) -> bool:
    return all(a == b for a, b in _sum_target(data))


def _toggle_target(data: bytes) -> list[tuple[bytes, bytes]]:
    """Wants field ``F`` to equal ``F ^ 1``: no fixed point, a 2-cycle."""
    field = data[:2]
    want = bytes([field[0], field[1] ^ 1])
    return [(want, field)]


# ── solve ────────────────────────────────────────────────────────────


def test_chained_fields_converge():
    """Fixing the length exposes the checksum; the loop fixes both."""
    seed = MAGIC + b"\x00\x09\x00\x00" + b"payload"
    assert not _valid_sum(seed)

    result = solve(seed, _sum_target, MAX_ITERS)

    assert result.outcome is Outcome.FIXED
    assert len(result.candidates) == 1
    assert _valid_sum(result.candidates[0])
    # one probe per patch (length, checksum) plus the confirming probe
    assert result.execs == 3


def test_single_shot_is_not_enough():
    """Falsification: one Redqueen patch leaves the checksum wrong."""
    seed = MAGIC + b"\x00\x09\x00\x00" + b"payload"
    once = patch_step(seed, _sum_target(seed))
    assert once is not None
    assert not _valid_sum(once)


def test_cycle_yields_deutsch_mixture():
    """A 2-cycle returns every member except the seed itself."""
    seed = b"\x41\x40rest"
    flipped = b"\x41\x41rest"

    result = solve(seed, _toggle_target, MAX_ITERS)

    assert result.outcome is Outcome.CYCLE
    assert result.candidates == (flipped,)


def test_cycle_off_seed_keeps_all_members():
    """Tail then cycle: the tail is dropped, the whole cycle is kept."""
    # step 1: AAAA->BBBB (tail); then BBBB <-> CCCC forever
    table = {b"AAAA": (b"BBBB", b"AAAA"), b"BBBB": (b"CCCC", b"BBBB"), b"CCCC": (b"BBBB", b"CCCC")}

    result = solve(b"AAAA", lambda x: [table[x]], MAX_ITERS)

    assert result.outcome is Outcome.CYCLE
    assert result.candidates == (b"BBBB", b"CCCC")


def test_consistent_seed_yields_nothing():
    """Falsification: a seed already at its fixed point produces no work."""
    seed = MAGIC + b"\x00\x07\x00\x00" + b"payload"
    seed = solve(seed, _sum_target, MAX_ITERS).candidates[0]

    result = solve(seed, _sum_target, MAX_ITERS)

    assert result.outcome is Outcome.FIXED
    assert result.candidates == ()
    assert result.execs == 1


def test_operand_absent_from_input_is_fixed():
    """Falsification: no operand occurs in the input -> nothing to patch."""
    result = solve(b"hello world", lambda _x: [(b"ZZZZ", b"YYYY")], MAX_ITERS)
    assert result.outcome is Outcome.FIXED
    assert result.candidates == ()


def test_failed_run_is_blind():
    result = solve(b"data", lambda _x: None, MAX_ITERS)
    assert result.outcome is Outcome.BLIND
    assert result.candidates == ()
    assert result.execs == 1


def test_adversarial_runaway_respects_budget():
    """Adversarial: a target demanding a fresh value every run never settles.

    Bounded execs, no candidates (nothing is self-consistent), and memory
    bounded by the budget.
    """
    calls = []

    def runaway(x: bytes) -> list[tuple[bytes, bytes]]:
        calls.append(x)
        return [(struct.pack(">I", len(calls)), x[:4])]

    result = solve(b"\xff\xff\xff\xffxx", runaway, MAX_ITERS)

    assert result.outcome is Outcome.BUDGET
    assert result.candidates == ()
    assert result.execs == MAX_ITERS
    assert len(calls) == MAX_ITERS


def test_adversarial_zero_budget_and_empty_input():
    assert solve(b"abc", _toggle_target, 0).execs == 0
    assert solve(b"", lambda _x: [(b"ab", b"cd")], MAX_ITERS).candidates == ()


# ── patch_step ───────────────────────────────────────────────────────


def test_patch_skips_equal_short_and_mismatched_pairs():
    """Adversarial: satisfied, 1-byte and width-changing pairs are noise."""
    data = b"xxABCDyy"
    pairs = [(b"AB", b"AB"), (b"C", b"Q"), (b"ABC", b"QQQQ")]
    assert patch_step(data, pairs) is None


def test_patch_both_directions():
    """The input may hold either operand."""
    assert patch_step(b"__WXYZ__", [(b"WXYZ", b"abcd")]) == b"__abcd__"
    assert patch_step(b"__WXYZ__", [(b"abcd", b"WXYZ")]) == b"__abcd__"


def test_patch_little_endian_operand():
    """Integer compares log the value; the input stores it reversed."""
    stored = struct.pack("<I", 0x11223344)
    data = b"hd" + stored + b"tl"
    want = 0x55667788
    pairs = [(struct.pack(">I", want), struct.pack(">I", 0x11223344))]
    assert patch_step(data, pairs) == b"hd" + struct.pack("<I", want) + b"tl"


def test_patch_is_deterministic():
    """The map must be a function, or cycle detection is meaningless."""
    data = b"AAAA BBBB"
    pairs = [(b"AAAA", b"1111"), (b"BBBB", b"2222")]
    assert patch_step(data, pairs) == patch_step(data, pairs) == b"1111 BBBB"
