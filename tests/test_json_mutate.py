"""Tests for core/mutations/json_struct.json_mutate (fuzzgoat's input format)."""

import json
import time

import pytest

from fuzzer_tool.core.mutations.json_struct import (
    JSON_NUMBERS,
    MODES,
    STRING_EDGES,
    VALUE_SWAPS,
    json_mutate,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_MAX = 65536


def _mode(name):
    return [m.__name__ for m in MODES].index(name)


def _run(data, *choice_idxs, max_len=_MAX):
    return json_mutate(data, ScriptedRng(choice_idxs=choice_idxs), max_len)


def test_type_swap_skips_keys():
    out = _run(b'{"a":1}', _mode("type_swap"), 0, VALUE_SWAPS.index(b"null"))
    assert out == b'{"a":null}'
    assert json.loads(out) == {"a": None}


def test_number_edge():
    out = _run(b"[7, 8]", _mode("number_edge"), 1, JSON_NUMBERS.index(b"1e309"))
    assert out == b"[7, 1e309]"


def test_string_edge_hits_keys_too():
    edge = STRING_EDGES.index(b'"\\ud800"')
    assert _run(b'{"k":"v"}', _mode("string_edge"), 0, edge) == b'{"\\ud800":"v"}'


def test_dup_member_copies_whole_value():
    out = _run(b'{"a":[1,2],"b":3}', _mode("dup_member"), 0)
    assert out == b'{"a":[1,2],"a":[1,2],"b":3}'


def test_dup_member_last_member():
    out = _run(b'{"b":{"c":3}}', _mode("dup_member"), 0)
    assert out == b'{"b":{"c":3},"b":{"c":3}}'


def test_trailing_comma():
    assert _run(b"[1]", _mode("trailing_comma"), 0) == b"[1,]"


def test_drop_punct():
    assert _run(b'{"a":1}', _mode("drop_punct"), 0) == b'{"a"1}'


def test_truncate_mid_string():
    # Token 1 is '"abcd"' at [1, 7): cut at 1 + 6 // 2 = 4.
    assert _run(b'{"abcd":1}', _mode("truncate"), 1) == b'{"ab'


def test_escaped_quote_stays_inside_string():
    # '"a\"b"' is one string token, so the only number is 1.
    out = _run(b'["a\\"b", 1]', _mode("number_edge"), 0, JSON_NUMBERS.index(b"-0"))
    assert out == b'["a\\"b", -0]'


# Falsification: a mode with no candidate must decline, not guess.
def test_regression_no_candidate_declines():
    assert _run(b'{"a":"b"}', _mode("number_edge")) is None


def test_no_tokens_declines():
    assert json_mutate(b"   ", ScriptedRng(), _MAX) is None
    assert json_mutate(b"", ScriptedRng(), _MAX) is None


def test_over_budget_declines():
    edge = JSON_NUMBERS.index(b"1" + b"0" * 400)
    assert _run(b"[1]", _mode("number_edge"), 0, edge, max_len=16) is None


# Adversarial: hostile lexer inputs stay bounded in size and time.
def test_adversarial_escape_run_is_fast():
    data = b'"' + b"\\" * 60000
    t0 = time.perf_counter()
    for seed in range(20):
        out = json_mutate(data, RandPool(seed=seed), _MAX)
        assert out is None or len(out) <= _MAX
    assert time.perf_counter() - t0 < 2.0


@pytest.mark.parametrize("max_len", [1, 2, 8, 256])
def test_adversarial_never_exceeds_max_len(max_len):
    blobs = [b'{"a":' * 100, b"[" * 500, b'"unterminated', bytes(range(256)), b'{"":[,,]}']
    for seed in range(40):
        rng = RandPool(seed=seed)
        for blob in blobs:
            out = json_mutate(blob[:max_len], rng, max_len)
            assert out is None or len(out) <= max_len
