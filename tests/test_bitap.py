"""Bitap near-miss repair: a token off by <= k bytes is overwritten with the exact token."""

import pytest

from fuzzer_tool.core.bitap import (
    MAX_ERRORS,
    MAX_REPAIR_PROBES,
    MIN_REPAIR_LEN,
    max_errors,
    near_matches,
    token_repair,
)
from fuzzer_tool.core.operator_registry import _CATEGORIES, REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer
from tests.support.scripted_rng import ScriptedRng

_RNG = RandPool(seed=0x5A11)
TOK = b"IHDRchnk"


def _bitap(text: bytes, pat: bytes, k: int) -> list[int]:
    """Scalar Wu-Manber shift-and, k substitutions: independent oracle."""
    m = len(pat)
    masks = [0] * 256
    for i, c in enumerate(pat):
        masks[c] |= 1 << i
    state = [0] * (k + 1)
    hit = 1 << (m - 1)
    out = []
    for j, c in enumerate(text):
        mc = masks[c]
        prev = state[0]
        state[0] = ((prev << 1) | 1) & mc
        for d in range(1, k + 1):
            cur = state[d]
            state[d] = (((cur << 1) | 1) & mc) | ((prev << 1) | 1)
            prev = cur
        if state[k] & hit:
            out.append(j - m + 1)
    return out


def _exact(text: bytes, pat: bytes) -> set[int]:
    m = len(pat)
    return {i for i in range(len(text) - m + 1) if text[i : i + m] == pat}


def _plant(data: bytes, at: int, tok: bytes) -> bytes:
    return data[:at] + tok + data[at + len(tok) :]


def test_matches_scalar_bitap_on_random_text():
    # 4-letter alphabet: dense near misses.
    text = bytearray(b % 4 for b in _RNG.randbytes(4096))
    pat = bytes(b % 4 for b in _RNG.randbytes(6))
    want = [p for p in _bitap(bytes(text), pat, 2) if p not in _exact(bytes(text), pat)]
    assert near_matches(bytes(text), pat, 2).tolist() == want


def test_control_bitap_against_itself():
    # Hard Rule 46: the oracle agrees with a second run of itself.
    text = _RNG.randbytes(512)
    assert _bitap(text, b"abcd", 1) == _bitap(bytes(text), b"abcd", 1)


def test_one_byte_off_found():
    bad = TOK[:3] + b"X" + TOK[4:]
    data = _plant(bytes(64), 20, bad)
    assert near_matches(data, TOK, 1).tolist() == [20]


def test_falsification_exact_and_too_far_ignored():
    far = b"XXXRchnk"  # 3 errors
    data = _plant(_plant(bytes(64), 0, TOK), 30, far)
    assert near_matches(data, TOK, MAX_ERRORS).tolist() == []


def test_max_errors_scales_with_length():
    assert max_errors(MIN_REPAIR_LEN) == 1
    assert max_errors(2 * MIN_REPAIR_LEN) == MAX_ERRORS
    assert max_errors(1000) == MAX_ERRORS


def test_repair_overwrites_with_exact_token():
    bad = b"IHxRchnk"
    data = _plant(bytes(48), 10, bad)
    out = token_repair(data, [TOK], 0, rng=ScriptedRng(randints=[0]))
    assert out == _plant(data, 10, TOK)


def test_repair_prefers_match_at_or_after_site():
    bad1, bad2 = b"IHxRchnk", b"IHDRchnX"
    data = _plant(_plant(bytes(64), 4, bad1), 40, bad2)
    out = token_repair(data, [TOK], 20, rng=ScriptedRng(randints=[0]))
    assert out == _plant(data, 40, TOK)


def test_repair_probes_next_tokens():
    # Token 0 has no near match; the probe moves on to token 1.
    other = b"zzzzQQQQ"
    data = _plant(bytes(48), 8, b"IHDRchnX")
    out = token_repair(data, [other, TOK], 0, rng=ScriptedRng(randints=[0]))
    assert out == _plant(data, 8, TOK)


def test_adversarial_no_match_and_short_tokens():
    rng = ScriptedRng(randints=[0])
    assert token_repair(bytes(32), [TOK], 0, rng=rng) is None
    assert token_repair(b"abc", [b"ab", "abc"], 0, rng=ScriptedRng(randints=[0])) is None
    assert token_repair(b"", [TOK], 0, rng=ScriptedRng(randints=[0])) is None


def test_adversarial_str_token_and_probe_bound():
    tokens = [b"q" * 8] * (MAX_REPAIR_PROBES + 5) + [TOK.decode()]
    data = _plant(bytes(48), 8, b"IHDRchnX")
    # The str token sits past the probe window: not reached.
    assert token_repair(data, tokens, 0, rng=ScriptedRng(randints=[0])) is None
    hit = token_repair(data, tokens, 0, rng=ScriptedRng(randints=[len(tokens) - 1]))
    assert hit == _plant(data, 8, TOK)


# ── registry / handler wiring ───────────────────────────────────────────────


def test_registered_in_dict_category():
    assert "token_repair" in _CATEGORIES["dict"]
    assert "token_repair" in REGISTRY.names()


def test_gated_on_literals():
    from types import SimpleNamespace

    bare = SimpleNamespace(dictionary=[], _cmplog=None)
    with_dict = SimpleNamespace(dictionary=[TOK], _cmplog=None)
    assert "token_repair" not in REGISTRY.available(bare, b"seed")
    assert "token_repair" in REGISTRY.available(with_dict, b"seed")


def test_handler_repairs_buffer():
    engine = OperatorEngine(make_minimal_fuzzer())
    engine.f.dictionary = [TOK]
    engine.f._rng = ScriptedRng(randints=[0])
    data = _plant(bytes(32), 5, b"IHDRchXk")
    out = engine._op_token_repair(bytearray(data), 0, data)
    assert bytes(out) == _plant(data, 5, TOK)


@pytest.mark.parametrize(
    ("n", "m", "alphabet"),
    [
        (64, 4, 256),  # pigeonhole filter, few candidates
        (1024, 8, 256),
        (8192, 32, 256),  # long pieces: filter past PIGEON_MAX_LEN
        (512, 8, 2),  # dense: candidate cap -> scan
        (8192, 6, 256),  # short pieces, large input -> scan
    ],
)
def test_every_path_matches_scalar_bitap(n, m, alphabet):
    text = bytearray(b % alphabet for b in _RNG.randbytes(n))
    pat = bytes(b % alphabet for b in _RNG.randbytes(m))
    k = max_errors(m)
    for at in (n // 3, 2 * n // 3):  # plant two near misses
        text[at : at + m] = pat
        text[at] ^= 1
    want = [p for p in _bitap(bytes(text), pat, k) if p not in _exact(bytes(text), pat)]
    assert near_matches(bytes(text), pat, k).tolist() == want


def test_adversarial_zero_fill_run_piece():
    # "00 00" piece matches everywhere: scan path, same answer.
    tok = bytes(7) + b"\x01"
    data = bytes(300)
    want = [p for p in _bitap(data, tok, 2) if p not in _exact(data, tok)]
    assert near_matches(data, tok, 2).tolist() == want
