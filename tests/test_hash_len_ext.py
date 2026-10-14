"""Tests for the hash_len_ext operator (core/mutations/hash_len_ext.py)."""

from __future__ import annotations

import hashlib

import pytest

from fuzzer_tool.core.mutations.hash_len_ext import (
    ALGOS,
    digest_start,
    hash_len_ext,
    md_pad,
    md_resume,
    message_of,
)
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

KEY = b"0123456789abcdef"
ALGO_NAMES = sorted(ALGOS)


def _sample(n: int) -> bytes:
    return bytes((i * 7 + 3) % 256 for i in range(n))


def _seed(algo: str, msg: bytes, key: bytes = KEY) -> bytes:
    """A valid seed: msg followed by H(key || msg)."""
    return msg + hashlib.new(algo, key + msg).digest()


class TestPad:
    @pytest.mark.parametrize("algo", ALGO_NAMES)
    @pytest.mark.parametrize("n", [0, 1, 55, 56, 63, 64, 65, 119, 120, 128])
    def test_block_aligned(self, algo, n):
        assert (n + len(md_pad(algo, n))) % 64 == 0

    def test_md5_pad_is_little_endian_bit_length(self):
        pad = md_pad("md5", 3)
        assert pad[0] == 0x80
        assert pad[-8:] == (24).to_bytes(8, "little")

    @pytest.mark.parametrize("algo", ["sha1", "sha256"])
    def test_sha_pad_is_big_endian_bit_length(self, algo):
        pad = md_pad(algo, 3)
        assert pad[0] == 0x80
        assert pad[-8:] == (24).to_bytes(8, "big")


class TestResume:
    @pytest.mark.parametrize("algo", ALGO_NAMES)
    @pytest.mark.parametrize("n", [0, 1, 20, 55, 56, 63, 64, 100])
    @pytest.mark.parametrize("suffix_len", [0, 1, 9, 55, 56, 64, 130])
    def test_matches_hashlib(self, algo, n, suffix_len):
        msg, suffix = _sample(n), _sample(suffix_len)[::-1]
        digest = hashlib.new(algo, msg).digest()
        prior = n + len(md_pad(algo, n))
        want = hashlib.new(algo, msg + md_pad(algo, n) + suffix).digest()
        assert md_resume(algo, digest, prior, suffix) == want

    def test_falsification_wrong_prior_length(self):
        msg = _sample(10)
        digest = hashlib.sha256(msg).digest()
        prior = 10 + len(md_pad("sha256", 10))
        want = hashlib.sha256(msg + md_pad("sha256", 10) + b"x").digest()
        assert md_resume("sha256", digest, prior + 64, b"x") != want

    def test_adversarial_bad_digest_length(self):
        with pytest.raises(ValueError):
            md_resume("sha1", b"short", 64, b"x")

    def test_adversarial_unaligned_prior(self):
        with pytest.raises(ValueError):
            md_resume("md5", bytes(16), 65, b"x")

    def test_adversarial_unknown_algo(self):
        with pytest.raises(ValueError):
            md_resume("crc32", bytes(4), 64, b"x")


class TestLayout:
    def test_digest_start_end_and_start(self):
        assert digest_start("sha1", 40, -20) == 20
        assert digest_start("sha1", 40, 0) == 0
        assert digest_start("md5", 40, 24) == 24

    @pytest.mark.parametrize(("total", "at"), [(15, -16), (16, 1), (40, -15), (40, 100)])
    def test_does_not_fit(self, total, at):
        assert digest_start("md5", total, at) is None

    def test_message_of_middle_slot(self):
        data = b"AAAA" + bytes(16) + b"BBBB"
        assert message_of("md5", data, 4) == b"AAAABBBB"


class TestExtend:
    @pytest.mark.parametrize("algo", ALGO_NAMES)
    def test_scripted_end_slot_is_valid_extension(self, algo):
        msg = _sample(23)
        suffix = b"&admin=1"
        rng = ScriptedRng(randints=[len(suffix)], randbytes=[suffix])
        out = hash_len_ext(_seed(algo, msg), algo, len(KEY), rng, 4096)
        want_msg = msg + md_pad(algo, len(KEY) + len(msg)) + suffix
        assert out == want_msg + hashlib.new(algo, KEY + want_msg).digest()

    @pytest.mark.parametrize("algo", ALGO_NAMES)
    def test_start_slot_is_valid_extension(self, algo):
        msg = _sample(30)
        size = hashlib.new(algo).digest_size
        data = hashlib.new(algo, KEY + msg).digest() + msg
        out = hash_len_ext(data, algo, len(KEY), RandPool(seed=3), 4096, digest_at=0)
        new_msg = out[size:]
        assert out[:size] == hashlib.new(algo, KEY + new_msg).digest()
        assert new_msg.startswith(msg + md_pad(algo, len(KEY) + len(msg)))

    @pytest.mark.parametrize("algo", ALGO_NAMES)
    def test_chains_over_seeds(self, algo):
        data = _seed(algo, _sample(17))
        rng = RandPool(seed=11)
        for _ in range(20):
            data = hash_len_ext(data, algo, len(KEY), rng, 1 << 16)
            assert data is not None
            size = hashlib.new(algo).digest_size
            assert data[-size:] == hashlib.new(algo, KEY + data[:-size]).digest()

    def test_empty_message(self):
        out = hash_len_ext(_seed("md5", b""), "md5", len(KEY), RandPool(seed=1), 4096)
        assert out is not None and out[-16:] == hashlib.md5(KEY + out[:-16]).digest()

    def test_zero_key_len(self):
        data = b"abc" + hashlib.sha1(b"abc").digest()
        out = hash_len_ext(data, "sha1", 0, RandPool(seed=2), 4096)
        assert out[-20:] == hashlib.sha1(out[:-20]).digest()

    def test_suffix_clamped_to_max_len(self):
        data = _seed("sha1", _sample(10))
        pad = len(md_pad("sha1", len(KEY) + 10))
        cap = len(data) + pad + 3
        rng = ScriptedRng(randints=[16], randbytes=[b"abc"])
        out = hash_len_ext(data, "sha1", len(KEY), rng, cap)
        assert out is not None and len(out) == cap

    def test_falsification_wrong_key_len_digest_invalid(self):
        data = _seed("sha256", _sample(20))
        out = hash_len_ext(data, "sha256", len(KEY) + 1, RandPool(seed=4), 4096)
        assert out[-32:] != hashlib.sha256(KEY + out[:-32]).digest()

    def test_falsification_plain_edit_without_ext_is_invalid(self):
        data = bytearray(_seed("md5", _sample(20)))
        data[0] ^= 1
        assert bytes(data[-16:]) != hashlib.md5(KEY + bytes(data[:-16])).digest()

    def test_adversarial_no_room_for_pad_declines(self):
        data = _seed("md5", _sample(20))
        assert hash_len_ext(data, "md5", len(KEY), RandPool(seed=1), len(data)) is None

    def test_adversarial_short_input_declines(self):
        assert hash_len_ext(b"\x00" * 15, "md5", 16, RandPool(seed=1), 4096) is None
        assert hash_len_ext(b"", "sha256", 16, RandPool(seed=1), 4096) is None

    def test_adversarial_bad_algo(self):
        with pytest.raises(ValueError):
            hash_len_ext(bytes(64), "crc32", 16, RandPool(seed=1), 4096)


class _MockFuzzer:
    def __init__(self, algo=None, key_len=16, digest_at=None):
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
        self.hlx_algo = algo
        self.hlx_key_len = key_len
        self.hlx_digest_at = digest_at
        self._op_declines = {}


class TestWiring:
    def test_registered_format(self):
        assert "hash_len_ext" in REGISTRY.names()
        assert REGISTRY.category_of("hash_len_ext") == "format"

    def test_available_only_with_algo(self):
        data = _seed("sha1", _sample(20))
        assert "hash_len_ext" not in REGISTRY.available(_MockFuzzer(), data)
        assert "hash_len_ext" in REGISTRY.available(_MockFuzzer("sha1"), data)

    def test_dispatch_extends(self):
        fuzzer = _MockFuzzer("sha256")
        fuzzer._rng = RandPool(seed=5)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = _seed("sha256", _sample(40))
        out = dispatch["hash_len_ext"](bytearray(data), 0, data)
        assert out is not None
        assert bytes(out[-32:]) == hashlib.sha256(KEY + bytes(out[:-32])).digest()

    def test_dispatch_declines_unconfigured(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=5)
        eng = OperatorEngine(fuzzer)
        buf = bytearray(b"x" * 40)
        eng._op_hash_len_ext(buf, 0, bytes(buf))
        assert bytes(buf) == b"x" * 40
        assert fuzzer._op_declines == {"hash_len_ext": 1}

    def test_dispatch_declines_short_input(self):
        fuzzer = _MockFuzzer("md5")
        fuzzer._rng = RandPool(seed=5)
        eng = OperatorEngine(fuzzer)
        buf = bytearray(b"tiny")
        eng._op_hash_len_ext(buf, 0, bytes(buf))
        assert bytes(buf) == b"tiny"
        assert fuzzer._op_declines == {"hash_len_ext": 1}


class TestFuzzerPlumbing:
    def test_flags_reach_operator_engine(self, tmp_path):
        from unittest.mock import patch

        from fuzzer_tool.services.fuzzer import Fuzzer

        with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=str(tmp_path / "corpus"),
                crashes_dir=str(tmp_path / "crashes"),
                max_len=4096,
                timeout=1,
                hlx_algo="sha256",
                hlx_key_len=len(KEY),
            )
        assert (f.hlx_algo, f.hlx_key_len, f.hlx_digest_at) == ("sha256", len(KEY), None)
        data = _seed("sha256", _sample(33))
        out = OperatorEngine(f)._op_hash_len_ext(bytearray(data), 0, data)
        assert bytes(out[-32:]) == hashlib.sha256(KEY + bytes(out[:-32])).digest()
        assert "hash_len_ext" in REGISTRY.available(f, data)

    def test_default_is_off(self, tmp_path):
        from unittest.mock import patch

        from fuzzer_tool.services.fuzzer import Fuzzer

        with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=str(tmp_path / "corpus"),
                crashes_dir=str(tmp_path / "crashes"),
                max_len=4096,
                timeout=1,
            )
        assert f.hlx_algo is None
        assert "hash_len_ext" not in REGISTRY.available(f, b"x" * 40)
