"""Per-byte parity operators: parity_lock / parity_break.

Expected bytes are derived bit by bit here, independently of the
translate tables under test.
"""

import pytest

from fuzzer_tool.core.mutations import parity as P
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

OPS = ("parity_lock", "parity_break")


def _ones(b):
    return sum((b >> i) & 1 for i in range(8))


def _fixed(b, carrier, odd):
    """Reference: flip the carrier bit when the byte's parity is wrong."""
    return b ^ (1 << carrier) if _ones(b) % 2 != odd else b


def _script(offset, length, carrier_idx, parity_idx, extra=()):
    # _region draws length then offset; then carrier, then parity.
    return ScriptedRng(randints=(length, offset, *extra), choice_idxs=(carrier_idx, parity_idx))


DATA = bytes(range(0x30, 0x50))


@pytest.mark.parametrize("carrier_idx", range(len(P.CARRIERS)))
@pytest.mark.parametrize("parity_idx", range(len(P.Parity)))
def test_lock_fixes_every_byte_in_window(carrier_idx, parity_idx):
    carrier = P.CARRIERS[carrier_idx]
    odd = list(P.Parity)[parity_idx].value
    out = P.parity_lock(DATA, rng=_script(4, 10, carrier_idx, parity_idx))

    want = DATA[:4] + bytes(_fixed(b, carrier, odd) for b in DATA[4:14]) + DATA[14:]
    assert out == want
    assert all(_ones(b) % 2 == odd for b in out[4:14])


def test_falsification_lock_is_not_a_blind_bit_set():
    """Bytes already of the right parity stay untouched."""
    odd_bytes = bytes(b for b in range(256) if _ones(b) % 2)[:16]
    odd_idx = list(P.Parity).index(P.Parity.ODD)
    out = P.parity_lock(odd_bytes, rng=_script(0, 16, 0, odd_idx))
    assert out == odd_bytes


def test_break_leaves_exactly_one_bad_byte():
    carrier, odd = P.CARRIERS[1], P.Parity.EVEN.value
    even_idx = list(P.Parity).index(P.Parity.EVEN)
    out = P.parity_break(DATA, rng=_script(2, 8, 1, even_idx, extra=(5,)))

    bad = [i for i in range(2, 10) if _ones(out[i]) % 2 != odd]
    assert bad == [2 + 5]
    assert out[7] == _fixed(DATA[7], carrier, odd) ^ (1 << carrier)
    assert out[:2] == DATA[:2] and out[10:] == DATA[10:]


@pytest.mark.parametrize("op", [P.parity_lock, P.parity_break])
@pytest.mark.parametrize("data", [b"", b"\x01"])
def test_adversarial_short_input_unchanged_without_draws(op, data):
    # Below MIN_LEN _region returns before drawing; an empty script proves it.
    assert op(data, rng=ScriptedRng()) == data


@pytest.mark.parametrize("op", [P.parity_lock, P.parity_break])
@pytest.mark.parametrize("seed", range(8))
def test_adversarial_randpool_length_preserved(op, seed):
    data = bytes((i * 37 + seed) & 0xFF for i in range(64))
    assert len(op(data, rng=RandPool(seed=seed))) == len(data)


def test_ops_registered_in_regularity_band():
    assert set(OPS) <= set(REGISTRY.names())
    assert set(OPS) <= OPERATOR_CATEGORIES["regularity"]
