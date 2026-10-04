"""``tail_append``: add bytes at the end only.

On an instruction-stream input (one record per move/call) inserting in the
middle changes the state every later record runs against, so most inserts
fail. Appending leaves the executed prefix untouched. Measured upstream on
a dense Rush Hour puzzle: random insert 98% failure, append-only 88%.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.mutations.generic import tail_append
from fuzzer_tool.core.operator_registry import _CATEGORIES, REGISTRY
from tests.support.scripted_rng import ScriptedRng

MAX_LEN = 64


class _Fuzzer:
    def __init__(self, on: bool):
        self.op_append = on


def test_registered_in_block_category():
    assert "tail_append" in _CATEGORIES["block"]
    assert "tail_append" in REGISTRY.names()


def test_gated_on_flag():
    assert "tail_append" not in REGISTRY.available(_Fuzzer(False), b"abc")
    assert "tail_append" in REGISTRY.available(_Fuzzer(True), b"abc")


def test_prefix_preserved_and_bytes_appended():
    data = b"abcdef"
    rng = ScriptedRng(randints=[0, 3], randbytes=[b"\x01\x02\x03"])
    out = tail_append(data, MAX_LEN, rng=rng)
    assert out == data + b"\x01\x02\x03"


def test_never_touches_prefix_over_many_seeds():
    import random

    for t in range(500):
        rng = random.Random(t)
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 40)))
        out = tail_append(data, MAX_LEN, rng=rng)
        assert out.startswith(data)
        assert len(data) <= len(out) <= MAX_LEN


def test_at_max_len_returns_input_unchanged():
    data = bytes(MAX_LEN)
    assert tail_append(data, MAX_LEN, rng=ScriptedRng()) == data


def test_append_clamped_to_remaining_room():
    data = bytes(MAX_LEN - 2)
    rng = ScriptedRng(randints=[0, 9], randbytes=[b"\xaa\xbb"])
    out = tail_append(data, MAX_LEN, rng=rng)
    assert len(out) == MAX_LEN
    assert out[-2:] == b"\xaa\xbb"


def test_empty_input_gets_bytes():
    rng = ScriptedRng(randints=[0, 2], randbytes=[b"\x07\x08"])
    assert tail_append(b"", MAX_LEN, rng=rng) == b"\x07\x08"


@pytest.mark.parametrize("max_len", [0, 1])
def test_degenerate_max_len(max_len):
    out = tail_append(b"", max_len, rng=ScriptedRng(randints=[0, 1], randbytes=[b"\x09"]))
    assert len(out) <= max_len
