"""Tests for tools/sha1_collision.py (birthday search on SHA-1 prefixes)."""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import sha1_collision as sc  # noqa: E402

from fuzzer_tool.core.rand_pool import RandPool  # noqa: E402
from tests.support.scripted_rng import ScriptedRng  # noqa: E402


def _ref_prefix(data: bytes, bits: int) -> int:
    """Independent oracle: full-digest int shifted down."""
    return int.from_bytes(hashlib.sha1(data).digest(), "big") >> (160 - bits)


@pytest.mark.parametrize("bits", [1, 7, 8, 9, 20, 64, 159, 160])
def test_prefix_matches_oracle(bits):
    data = b"fuzzgoat"
    assert sc.sha1_prefix(data, bits) == _ref_prefix(data, bits)


def test_prefix_control_self():
    """Hard Rule 46: oracle against itself."""
    assert _ref_prefix(b"x", 20) == _ref_prefix(b"x", 20)


def test_finds_real_collision():
    """Falsification: reported pair must differ and share the prefix."""
    bits = 16
    hit = sc.find_collision(bits, 1 << 16, RandPool(seed=1))

    assert hit is not None
    assert hit.first != hit.second
    assert _ref_prefix(hit.first, bits) == _ref_prefix(hit.second, bits) == hit.prefix
    assert hit.attempts >= 2


def test_scripted_collision_exact():
    """Scripted RNG: messages 1 and 3 collide on 1 bit -> attempt 3."""
    msgs = [bytes([i]) * sc.MSG_LEN for i in range(256)]
    by_bit: dict[int, list[bytes]] = {}
    for m in msgs:
        by_bit.setdefault(_ref_prefix(m, 1), []).append(m)
    a, b = by_bit[0][:2]
    c = by_bit[1][0]

    hit = sc.find_collision(1, 10, ScriptedRng(randbytes=[a + c + b]))

    assert (hit.first, hit.second, hit.attempts) == (a, b, 3)


def test_adversarial_duplicate_message_not_collision():
    """Same message twice is not a collision."""
    m = b"\x00" * sc.MSG_LEN
    assert sc.find_collision(160, 3, ScriptedRng(randbytes=[m * 3])) is None


def test_adversarial_batch_boundary():
    """Collision whose second message opens the next draw batch."""
    by_bit: dict[int, list[bytes]] = {}
    for i in range(256):
        m = bytes([i]) * sc.MSG_LEN
        by_bit.setdefault(_ref_prefix(m, 1), []).append(m)
    a, b = by_bit[0][:2]
    c = by_bit[1][0]
    rng = ScriptedRng(randbytes=[a + c * (sc.DRAW_BATCH - 1), b])

    hit = sc.find_collision(1, sc.DRAW_BATCH + 1, rng)

    assert (hit.first, hit.second, hit.attempts) == (a, b, sc.DRAW_BATCH + 1)


def test_adversarial_budget_exhausted():
    assert sc.find_collision(160, 1, RandPool(seed=2)) is None


@pytest.mark.parametrize("bits", [0, 161, -1])
def test_adversarial_bits_rejected(bits):
    with pytest.raises(ValueError):
        sc.find_collision(bits, 10, RandPool(seed=3))
