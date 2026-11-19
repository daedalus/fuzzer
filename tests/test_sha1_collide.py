"""Tests for the sha1_collide operator (core/mutations/sha1_collide.py)."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from fuzzer_tool.core.mutations import sha1_collide as sc
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng


def _sample(n: int) -> bytes:
    return bytes((i * 7 + 3) % 256 for i in range(n))


def _ref_prefix(data: bytes, bits: int) -> int:
    """Independent oracle: whole digest as int, shifted down."""
    return int.from_bytes(hashlib.sha1(data).digest(), "big") >> (160 - bits)


def _splice(data: bytes, off: int, win: bytes) -> bytes:
    return data[:off] + win + data[off + sc.WINDOW :]


def _windows(data: bytes, off: int, bits: int):
    """Split candidate windows into (colliding, non-colliding) for this seed."""
    target = _ref_prefix(data, bits)
    hit, miss = [], []
    for i in range(256):
        win = bytes([i]) * sc.WINDOW
        if win == data[off : off + sc.WINDOW]:
            continue
        (hit if _ref_prefix(_splice(data, off, win), bits) == target else miss).append(win)
    return hit, miss


class TestPrefix:
    @pytest.mark.parametrize("bits", [1, 7, 8, 9, 16, 64, 159, 160])
    def test_matches_oracle(self, bits):
        assert sc.sha1_prefix(b"fuzzgoat", bits) == _ref_prefix(b"fuzzgoat", bits)

    def test_oracle_control(self):
        """Hard Rule 46: oracle against itself."""
        assert _ref_prefix(b"x", 20) == _ref_prefix(b"x", 20)


class TestCollide:
    @pytest.mark.parametrize("n", [sc.WINDOW, 64, 1000])
    def test_real_collision(self, n):
        """Falsification: output differs, same length, same digest prefix."""
        data = _sample(n)
        bits = 10
        out = sc.sha1_collide(data, bits, RandPool(seed=n))

        assert out is not None
        assert out != data
        assert len(out) == len(data)
        assert _ref_prefix(out, bits) == _ref_prefix(data, bits)

    def test_scripted_exact(self):
        """Offset and windows scripted: second window is the first hit."""
        data = _sample(40)
        off = 5
        hit, miss = _windows(data, off, 2)
        budget = sc.budget(2)
        blob = miss[0] + hit[0] + miss[0] * (budget - 2)

        out = sc.sha1_collide(data, 2, ScriptedRng(randints=[off], randbytes=[blob]))

        assert out == _splice(data, off, hit[0])

    def test_window_near_tail(self):
        """Offset is drawn so at most MAX_TAIL bytes follow the window."""
        data = _sample(sc.MAX_TAIL * 4)
        seen = []

        class _Rng(RandPool):
            def randint(self, a, b):
                seen.append((a, b))
                return super().randint(a, b)

        sc.sha1_collide(data, 4, _Rng(seed=1))

        assert seen[0] == (len(data) - sc.WINDOW - sc.MAX_TAIL, len(data) - sc.WINDOW)

    def test_adversarial_original_window_rejected(self):
        """Rewriting the window with its own bytes is no mutation."""
        data = _sample(40)
        off = 3
        orig = data[off : off + sc.WINDOW]
        budget = sc.budget(1)
        rng = ScriptedRng(randints=[off], randbytes=[orig * budget])

        assert sc.sha1_collide(data, 1, rng) is None

    def test_adversarial_budget_exhausted(self):
        data = _sample(40)
        off = 0
        _, miss = _windows(data, off, 3)
        rng = ScriptedRng(randints=[off], randbytes=[miss[0] * sc.budget(3)])

        assert sc.sha1_collide(data, 3, rng) is None

    def test_adversarial_short_input(self):
        assert sc.sha1_collide(b"x" * (sc.WINDOW - 1), 4, RandPool(seed=1)) is None

    @pytest.mark.parametrize("bits", [0, sc.MAX_BITS + 1, -1])
    def test_adversarial_bits_rejected(self, bits):
        with pytest.raises(ValueError):
            sc.sha1_collide(_sample(32), bits, RandPool(seed=1))


class _MockFuzzer:
    def __init__(self, bits=None):
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
        self.sha1_collide_bits = bits
        self._op_declines = {}


class TestWiring:
    def test_registered_format(self):
        assert "sha1_collide" in REGISTRY.names()
        assert REGISTRY.category_of("sha1_collide") == "format"

    def test_available_only_with_bits(self):
        data = _sample(32)
        assert "sha1_collide" not in REGISTRY.available(_MockFuzzer(), data)
        assert "sha1_collide" in REGISTRY.available(_MockFuzzer(8), data)

    def test_dispatch_collides(self):
        fuzzer = _MockFuzzer(8)
        fuzzer._rng = RandPool(seed=5)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = _sample(64)

        out = dispatch["sha1_collide"](bytearray(data), 0, data)

        assert bytes(out) != data
        assert _ref_prefix(bytes(out), 8) == _ref_prefix(data, 8)

    def test_dispatch_declines_unconfigured(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=5)
        buf = bytearray(_sample(32))
        OperatorEngine(fuzzer)._op_sha1_collide(buf, 0, bytes(buf))

        assert bytes(buf) == _sample(32)
        assert fuzzer._op_declines == {"sha1_collide": 1}

    def test_dispatch_declines_short_input(self):
        fuzzer = _MockFuzzer(4)
        fuzzer._rng = RandPool(seed=5)
        buf = bytearray(b"tiny")
        OperatorEngine(fuzzer)._op_sha1_collide(buf, 0, bytes(buf))

        assert bytes(buf) == b"tiny"
        assert fuzzer._op_declines == {"sha1_collide": 1}


def _fuzzer(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=4096,
            timeout=1,
            **kw,
        )


class TestPlumbing:
    def test_flag_reaches_operator(self, tmp_path):
        f = _fuzzer(tmp_path, sha1_collide_bits=6)
        data = _sample(48)

        out = OperatorEngine(f)._op_sha1_collide(bytearray(data), 0, data)

        assert f.sha1_collide_bits == 6
        assert _ref_prefix(bytes(out), 6) == _ref_prefix(data, 6)
        assert "sha1_collide" in REGISTRY.available(f, data)

    def test_default_is_off(self, tmp_path):
        f = _fuzzer(tmp_path)
        assert f.sha1_collide_bits is None
        assert "sha1_collide" not in REGISTRY.available(f, _sample(32))

    def test_cli_flag(self):
        from fuzzer_tool.cli.commands import _sha1_bits

        assert _sha1_bits("12") == 12
        with pytest.raises(Exception, match="1..16"):
            _sha1_bits(str(sc.MAX_BITS + 1))
