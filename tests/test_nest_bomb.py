"""Tests for tree_mutator.nest_bomb: deep delimiter nesting (stack exhaustion)."""

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.tree_mutator import NEST_DEPTHS, nest_bomb
from tests.support.scripted_rng import ScriptedRng

_BALANCED = 0
_UNCLOSED = 1


def _depth(d):
    return NEST_DEPTHS.index(d)


def test_balanced_wrap_of_matching_pair():
    rng = ScriptedRng(choice_idxs=[_depth(64)], randints=[_BALANCED])
    out = nest_bomb(b'x{"a":[1]}y', 0, rng, 65536)
    assert out == b"x" + b"{" * 64 + b'{"a":[1]}' + b"}" * 64 + b"y"


def test_unclosed_run_has_only_openers():
    rng = ScriptedRng(choice_idxs=[_depth(64)], randints=[_UNCLOSED])
    out = nest_bomb(b"a[1]", 0, rng, 65536)
    assert out == b"a" + b"[" * 64 + b"[1]"


def test_opener_search_starts_at_byte_idx():
    rng = ScriptedRng(choice_idxs=[_depth(64)], randints=[_BALANCED])
    out = nest_bomb(b"(a)[b]", 3, rng, 65536)
    assert out == b"(a)" + b"[" * 64 + b"[b]" + b"]" * 64


def test_unmatched_opener_degrades_to_unclosed():
    rng = ScriptedRng(choice_idxs=[_depth(64)], randints=[_BALANCED])
    out = nest_bomb(b"(ab", 0, rng, 65536)
    assert out == b"(" * 64 + b"(ab"


def test_depth_capped_by_budget():
    data = b"[]"
    rng = ScriptedRng(choice_idxs=[_depth(max(NEST_DEPTHS))], randints=[_BALANCED])
    out = nest_bomb(data, 0, rng, 12)
    # (12 - 2) // 2 = 5 extra levels on each side.
    assert out == b"[" * 5 + b"[]" + b"]" * 5


# Falsification: no bracket, no work.
def test_regression_no_bracket_declines():
    assert nest_bomb(b'plain "q"', 0, ScriptedRng(), 65536) is None


def test_no_room_declines():
    rng = ScriptedRng(choice_idxs=[0], randints=[_UNCLOSED])
    assert nest_bomb(b"[]", 0, rng, 2) is None


# Adversarial: pathological nesting and tiny budgets stay bounded.
@pytest.mark.parametrize("max_len", [1, 2, 8, 4096])
def test_adversarial_never_exceeds_max_len(max_len):
    blobs = [b"[" * 3000, b"]" * 50 + b"[", b"({[" * 100, bytes(range(256))]
    for seed in range(30):
        rng = RandPool(seed=seed)
        for blob in blobs:
            data = blob[:max_len]
            out = nest_bomb(data, seed, rng, max_len)
            assert out is None or len(out) <= max_len


def test_long_input_uses_same_match_as_short():
    """Past _NUMPY_SCAN_MIN the cumsum scan must find the same close."""
    body = b"[" + b"(a)" * 200 + b"]"
    rng = ScriptedRng(choice_idxs=[_depth(64)], randints=[_BALANCED])
    out = nest_bomb(body + b"[z]", 0, rng, 65536)
    assert out == b"[" * 64 + body + b"]" * 64 + b"[z]"
