"""``literal_mismatch``: keep a literal's first k bytes, break byte k or cut there.

fuzzgoat's frontier (DEEP_DIVE §Coverage frontier) left one-sided the
``*(++ptr) != 'r'`` chains of ``true``/``false``/``null``: a dictionary op
inserts the whole token and takes the matching side of every byte, a random
byte op rarely lands a near-match. This op exercises each non-matching side.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from fuzzer_tool.core.mutations.generic import Mismatch, literal_mismatch
from fuzzer_tool.core.operator_registry import _CATEGORIES, REGISTRY
from fuzzer_tool.services.operators import _DELOCALISED_OPS, OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer
from tests.support.scripted_rng import ScriptedRng

MAX_LEN = 64
TRUE = b"true"


def _flip(tok: bytes, k: int, x: int) -> bytes:
    # Independent oracle: same length, only byte k differs.
    return tok[:k] + bytes([tok[k] ^ x]) + tok[k + 1 :]


# draw order: choice -> token, randint -> k, randint -> mode, [randint -> xor]
def _rng(k: int, mode: Mismatch, x: int | None = None) -> ScriptedRng:
    return ScriptedRng(choice_idxs=[0], randints=[k, int(mode)] + ([x] if x else []))


def test_flip_found_literal_in_place():
    data = b"[1," + TRUE + b"]"
    out = literal_mismatch(data, [TRUE], 0, MAX_LEN, rng=_rng(2, Mismatch.FLIP, 0x01))
    assert out == b"[1," + _flip(TRUE, 2, 0x01) + b"]"
    assert out[3:5] == TRUE[:2] and out[5] != TRUE[2]  # falsification: prefix kept, k broken


def test_cut_found_literal_ends_input():
    data = b"[" + TRUE + b",2]"
    out = literal_mismatch(data, [TRUE], 0, MAX_LEN, rng=_rng(3, Mismatch.CUT))
    assert out == b"[" + TRUE[:3]


def test_absent_literal_inserted_at_start():
    data = b"[1,2]"
    out = literal_mismatch(data, [TRUE], 3, MAX_LEN, rng=_rng(1, Mismatch.FLIP, 0xFF))
    assert out == data[:3] + _flip(TRUE, 1, 0xFF) + data[3:]
    cut = literal_mismatch(data, [TRUE], 3, MAX_LEN, rng=_rng(1, Mismatch.CUT))
    assert cut == data[:3] + TRUE[:1]


def test_search_starts_at_offset_then_wraps():
    data = TRUE + b"," + TRUE
    out = literal_mismatch(data, [TRUE], 1, MAX_LEN, rng=_rng(1, Mismatch.CUT))
    assert out == TRUE + b"," + TRUE[:1]  # second occurrence: at/after offset 1
    wrap = literal_mismatch(TRUE + b",", [TRUE], 2, MAX_LEN, rng=_rng(1, Mismatch.CUT))
    assert wrap == TRUE[:1]  # none after offset 2: wraps to the first


def test_str_tokens_encoded():
    out = literal_mismatch(b"x", ["null"], 0, MAX_LEN, rng=_rng(1, Mismatch.CUT))
    assert out == b"n"


def test_adversarial_short_token_and_full_buffer():
    # One-byte tokens have no prefix to keep; a full buffer cannot grow.
    assert literal_mismatch(b"ab", [b"a"], 0, MAX_LEN, rng=ScriptedRng(choice_idxs=[0])) is None
    full = bytes(MAX_LEN)
    out = literal_mismatch(full, [TRUE], 0, MAX_LEN, rng=_rng(1, Mismatch.FLIP, 0x01))
    assert out is not None and len(out) <= MAX_LEN


def test_adversarial_offset_past_end_clamped():
    out = literal_mismatch(b"ab", [TRUE], 99, MAX_LEN, rng=_rng(1, Mismatch.CUT))
    assert out == b"ab" + TRUE[:1]


# ── registry / handler wiring ───────────────────────────────────────────────


def test_registered_in_dict_category():
    assert "literal_mismatch" in _CATEGORIES["dict"]
    assert "literal_mismatch" in REGISTRY.names()


def test_gated_on_dictionary_or_cmplog_tokens():
    bare = SimpleNamespace(dictionary=[], _cmplog=None)
    with_dict = SimpleNamespace(dictionary=[TRUE], _cmplog=None)
    with_cmp = SimpleNamespace(dictionary=[], _cmplog=SimpleNamespace(tokens=[TRUE], pairs=[]))
    assert "literal_mismatch" not in REGISTRY.available(bare, b"seed")
    assert "literal_mismatch" in REGISTRY.available(with_dict, b"seed")
    assert "literal_mismatch" in REGISTRY.available(with_cmp, b"seed")


def test_handler_uses_byte_idx_as_site():
    # Localised: the arena's offset is where the search starts.
    assert "literal_mismatch" not in _DELOCALISED_OPS
    params = list(inspect.signature(OperatorEngine._op_literal_mismatch).parameters)
    assert params[2] == "byte_idx"


def test_handler_mutates_buffer():
    engine = OperatorEngine(make_minimal_fuzzer())
    engine.f.dictionary = [TRUE]
    engine.f._rng = _rng(2, Mismatch.FLIP, 0x01)
    buf = bytearray(b"[" + TRUE + b"]")
    out = engine._op_literal_mismatch(buf, 0, bytes(buf))
    assert bytes(out) == b"[" + _flip(TRUE, 2, 0x01) + b"]"
