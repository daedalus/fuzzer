"""Tests for the poly1305_fix operator (core/mutations/poly1305_fix.py)."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.mutations.poly1305_fix import (
    DEFAULT_TAG_AT,
    message_of,
    parse_key,
    poly1305_fix,
    poly1305_mac,
    retag,
    tag_start,
)
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

# RFC 8439 section 2.5.2.
RFC_KEY = bytes.fromhex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b")
RFC_MSG = b"Cryptographic Forum Research Group"
RFC_TAG = bytes.fromhex("a8061dc1305136c6c22b8baf0c0127a9")
KEY_B = bytes(range(32))


def _sample(n: int) -> bytes:
    return bytes((i * 7 + 3) % 256 for i in range(n))


class TestMac:
    def test_rfc8439_vector(self):
        assert poly1305_mac(RFC_KEY, RFC_MSG) == RFC_TAG

    def test_rfc8439_appendix_a3_zero_key(self):
        assert poly1305_mac(bytes(32), bytes(64)) == bytes(16)

    def test_matches_reference_library_all_lengths(self):
        pytest.importorskip("cryptography")
        from cryptography.hazmat.primitives.poly1305 import Poly1305

        for n in range(0, 70):
            msg = _sample(n)
            assert poly1305_mac(KEY_B, msg) == Poly1305.generate_tag(KEY_B, msg), n

    def test_matches_reference_with_unclamped_r_bits(self):
        pytest.importorskip("cryptography")
        from cryptography.hazmat.primitives.poly1305 import Poly1305

        key = b"\xff" * 32
        for n in (0, 1, 15, 16, 17, 48):
            msg = b"\xff" * n
            assert poly1305_mac(key, msg) == Poly1305.generate_tag(key, msg), n

    def test_falsification_wrong_key_wrong_tag(self):
        assert poly1305_mac(KEY_B, RFC_MSG) != RFC_TAG

    def test_bad_key_length(self):
        with pytest.raises(ValueError):
            poly1305_mac(b"short", b"x")


class TestKeyParse:
    def test_roundtrip(self):
        assert parse_key(RFC_KEY.hex()) == RFC_KEY
        assert parse_key("  " + RFC_KEY.hex() + "\n") == RFC_KEY

    @pytest.mark.parametrize("bad", ["", "zz" * 32, "00" * 31, "00" * 33, "0"])
    def test_rejects(self, bad):
        with pytest.raises(ValueError):
            parse_key(bad)


class TestLayout:
    def test_tag_start_end_and_start(self):
        assert tag_start(40, -16) == 24
        assert tag_start(40, 0) == 0
        assert tag_start(40, 24) == 24

    @pytest.mark.parametrize(("total", "at"), [(15, -16), (16, 1), (20, 5), (40, -15), (40, 100)])
    def test_tag_start_does_not_fit(self, total, at):
        assert tag_start(total, at) is None

    def test_message_of_middle_tag(self):
        data = b"AAAA" + bytes(16) + b"BBBB"
        assert message_of(data, 4) == b"AAAABBBB"


class TestRetag:
    def test_end_tag_valid(self):
        data = RFC_MSG + bytes(16)
        out = retag(data, RFC_KEY)
        assert out == RFC_MSG + RFC_TAG

    def test_start_tag_valid(self):
        out = retag(bytes(16) + RFC_MSG, RFC_KEY, 0)
        assert out == RFC_TAG + RFC_MSG

    def test_middle_tag_valid(self):
        data = b"Cryptographic " + bytes(16) + b"Forum Research Group"
        out = retag(data, RFC_KEY, 14)
        assert out[14:30] == RFC_TAG

    def test_empty_message(self):
        assert retag(bytes(16), RFC_KEY) == poly1305_mac(RFC_KEY, b"")

    def test_too_short_declines(self):
        assert retag(b"short", RFC_KEY) is None


class TestFix:
    def test_scripted_bit_flip_end_tag(self):
        msg = _sample(20)
        data = msg + bytes(16)
        # op=flip, 1 flip, index 3, bit 5.
        out = poly1305_fix(data, KEY_B, ScriptedRng(randints=[0, 1, 3, 5]), 4096)
        want = bytearray(msg)
        want[3] ^= 1 << 5
        assert out == bytes(want) + poly1305_mac(KEY_B, bytes(want))

    def test_scripted_insert(self):
        msg = _sample(20)
        out = poly1305_fix(msg + bytes(16), KEY_B, ScriptedRng(randints=[1, 5, 0xAB]), 4096)
        want = msg[:5] + b"\xab" + msg[5:]
        assert out == want + poly1305_mac(KEY_B, want)

    def test_scripted_delete(self):
        msg = _sample(20)
        out = poly1305_fix(msg + bytes(16), KEY_B, ScriptedRng(randints=[2, 7]), 4096)
        want = msg[:7] + msg[8:]
        assert out == want + poly1305_mac(KEY_B, want)

    def test_insert_skipped_at_cap(self):
        msg = _sample(20)
        data = msg + bytes(16)
        # op=insert but no room: falls through to bit flips (1 flip, idx 0, bit 0).
        out = poly1305_fix(data, KEY_B, ScriptedRng(randints=[1, 1, 0, 0]), len(data))
        assert len(out) == len(data)

    def test_tag_valid_and_message_changed_over_seeds(self):
        data = _sample(33) + bytes(16)
        rng = RandPool(seed=9)
        for _ in range(200):
            out = poly1305_fix(data, KEY_B, rng, 4096)
            assert out is not None
            assert out[-16:] == poly1305_mac(KEY_B, out[:-16])

    def test_start_tag_variable_length(self):
        data = bytes(16) + _sample(30)
        rng = RandPool(seed=4)
        for _ in range(100):
            out = poly1305_fix(data, KEY_B, rng, 4096, tag_at=0)
            assert out[:16] == poly1305_mac(KEY_B, out[16:])

    def test_falsification_wrong_key_tag_invalid(self):
        out = poly1305_fix(_sample(20) + bytes(16), KEY_B, RandPool(seed=2), 4096)
        assert out[-16:] != poly1305_mac(RFC_KEY, out[:-16])

    def test_adversarial_short_input_declines(self):
        assert poly1305_fix(b"\x00" * 15, KEY_B, RandPool(seed=1), 4096) is None
        assert poly1305_fix(b"", KEY_B, RandPool(seed=1), 4096) is None

    def test_adversarial_empty_message_keeps_valid_tag(self):
        rng = RandPool(seed=6)
        for _ in range(30):
            out = poly1305_fix(bytes(16), KEY_B, rng, 4096)
            assert out[-16:] == poly1305_mac(KEY_B, out[:-16])

    def test_default_tag_position(self):
        assert DEFAULT_TAG_AT == -16


class _MockFuzzer:
    def __init__(self, key=None, tag_at=-16):
        self.dictionary = []
        self.markov_trained = False
        self.mc = None
        self.mc_cem = False
        self.grammar = None
        self._cmplog = None
        self.enable_regex_bomb = False
        self.enable_x86_mutator = False
        self.enable_arm_mutator = False
        self.seed_meta = {}
        self.corpus = []
        self.max_len = 4096
        self.poly1305_key = key
        self.poly1305_tag_at = tag_at
        self._op_declines = {}


class TestWiring:
    def test_registered_format(self):
        assert "poly1305_fix" in REGISTRY.names()
        assert REGISTRY.category_of("poly1305_fix") == "format"

    def test_available_only_with_key(self):
        data = _sample(20) + bytes(16)
        assert "poly1305_fix" not in REGISTRY.available(_MockFuzzer(), data)
        assert "poly1305_fix" in REGISTRY.available(_MockFuzzer(KEY_B), data)

    def test_dispatch_repairs_tag(self):
        fuzzer = _MockFuzzer(KEY_B)
        fuzzer._rng = RandPool(seed=5)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = _sample(40) + bytes(16)
        out = dispatch["poly1305_fix"](bytearray(data), 0, data)
        assert out is not None and bytes(out[-16:]) == poly1305_mac(KEY_B, bytes(out[:-16]))

    def test_dispatch_declines_on_short_input(self):
        fuzzer = _MockFuzzer(KEY_B)
        fuzzer._rng = RandPool(seed=5)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        buf = bytearray(b"tiny")
        dispatch["poly1305_fix"](buf, 0, bytes(buf))
        assert bytes(buf) == b"tiny"
        assert fuzzer._op_declines == {"poly1305_fix": 1}
