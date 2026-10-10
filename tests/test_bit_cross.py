"""Tests for the bit_cross operator: bit-level uniform crossover of two
seeds over their conflict set (generalizes a & b and a | b).

Randomness is scripted (Hard Rule 39): every expected child is exact.
"""

import pytest

from fuzzer_tool.core.mutations.generic import bit_cross
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.operators import OperatorEngine

AND, QUARTER, HALF, THREE_QUARTER, OR = range(5)
_LEVELS = (AND, QUARTER, HALF, THREE_QUARTER, OR)
_PREFIX = b"\x89PNG\r\n\x1a\n"


def _int(x: bytes) -> int:
    return int.from_bytes(x, "big")


def _bytes(x: int, n: int) -> bytes:
    return x.to_bytes(n, "big")


class ScriptedRng:
    """Replays a fixed level and fixed random words; nothing is random."""

    def __init__(self, level, words=(), pick=0):
        self._level = level
        self._words = list(words)
        self._pick = pick
        self.sizes = []

    def randrange(self, n):
        assert n == len(_LEVELS)
        return self._level

    def randbytes(self, n):
        self.sizes.append(n)
        return self._words.pop(0)[:n].ljust(n, b"\x00")

    def choice(self, seq):
        return seq[self._pick]


class _Fuzzer:
    def __init__(self, corpus, rng=None, max_len=65536):
        self.corpus = corpus
        self.seed_meta = {}
        self.max_len = max_len
        self._rng = rng


A = b"\xf0\x0f\xaa\x55" * 4
B = b"\x3c\xc3\x55\xaa" * 4
D = _int(A) ^ _int(B)
N = len(A)
W1, W2 = b"\x96" * N, b"\x0f\xf0" * (N // 2)


class TestBitCross:
    def test_and_level_is_bitwise_and(self):
        assert bit_cross(A, B, ScriptedRng(AND)) == _bytes(_int(A) & _int(B), N)

    def test_or_level_is_bitwise_or(self):
        assert bit_cross(A, B, ScriptedRng(OR)) == _bytes(_int(A) | _int(B), N)

    def test_half_level_takes_the_word_on_conflicts(self):
        want = (_int(A) & ~D) | (D & _int(W1))
        assert bit_cross(A, B, ScriptedRng(HALF, [W1])) == _bytes(want, N)

    def test_quarter_level_ands_two_words(self):
        want = (_int(A) & ~D) | (D & (_int(W1) & _int(W2)))
        assert bit_cross(A, B, ScriptedRng(QUARTER, [W1, W2])) == _bytes(want, N)

    def test_three_quarter_level_ors_two_words(self):
        want = (_int(A) & ~D) | (D & (_int(W1) | _int(W2)))
        assert bit_cross(A, B, ScriptedRng(THREE_QUARTER, [W1, W2])) == _bytes(want, N)

    def test_random_bits_cover_only_the_differing_span(self):
        a, b = bytes(8), b"\x00\x00\xff\x0f\x00\x00\x00\x00"
        rng = ScriptedRng(HALF, [b"\xa5\x5a"])
        assert bit_cross(a, b, rng) == b"\x00\x00\xa5\x0a\x00\x00\x00\x00"
        assert rng.sizes == [2]

    def test_interior_child_is_neither_extreme_nor_parent(self):
        child = bit_cross(A, B, ScriptedRng(HALF, [W1]))
        assert child not in (A, B, _bytes(_int(A) & _int(B), N), _bytes(_int(A) | _int(B), N))

    @pytest.mark.parametrize("level", _LEVELS)
    def test_child_keeps_shared_bits_and_stays_in_and_or_interval(self, level):
        lo, hi = _int(A) & _int(B), _int(A) | _int(B)
        c = _int(bit_cross(A, B, ScriptedRng(level, [W1, W2])))
        assert c & lo == lo
        assert c | hi == hi
        assert c & ~D == _int(A) & ~D

    @pytest.mark.parametrize("level", _LEVELS)
    def test_shared_prefix_survives(self, level):
        """a ^ b would zero the shared header; bit_cross must not."""
        a, b = _PREFIX + b"\x01\x02\x03\x04", _PREFIX + b"\x10\x20\x30\x40"
        got = bit_cross(a, b, ScriptedRng(level, [W1, W2]))
        assert got[: len(_PREFIX)] == _PREFIX

    @pytest.mark.parametrize(
        ("a", "b"),
        [(b"", b"abc"), (b"abc", b""), (b"", b""), (b"same", b"same")],
    )
    def test_degenerate_inputs_return_base_without_drawing(self, a, b):
        assert bit_cross(a, b, None) == a  # rng=None: any draw would raise

    @pytest.mark.parametrize("b", [b"\x55", b"\x55" * 31, b"\x55" * 33, b"\x55" * 4096])
    def test_length_is_base_length(self, b):
        a = b"\xaa" * 32
        assert len(bit_cross(a, b, ScriptedRng(OR))) == len(a)

    def test_short_donor_tail_comes_from_base(self):
        a, b = b"\xaa" * 8, b"\x55" * 2
        assert bit_cross(a, b, ScriptedRng(OR)) == b"\xff\xff" + a[2:]

    def test_large_input(self):
        a, b = bytes(range(256)) * 4096, bytes(reversed(range(256))) * 4096
        assert bit_cross(a, b, ScriptedRng(AND)) == _bytes(_int(a) & _int(b), len(a))


class TestOpBitCross:
    def _engine(self, corpus, rng, **kw):
        return OperatorEngine(_Fuzzer(corpus, rng, **kw))

    def test_registered_structural(self):
        assert "bit_cross" in REGISTRY.names()
        assert REGISTRY.category_of("bit_cross") == "structural"

    def test_declines_with_single_seed(self):
        assert self._engine([A], None)._op_bit_cross(bytearray(A), 0, A) is None

    def test_declines_on_empty_buffer(self):
        assert self._engine([A, B], None)._op_bit_cross(bytearray(), 0, A) is None

    @pytest.mark.parametrize("level", _LEVELS)
    def test_declines_when_child_equals_a_parent(self, level):
        """One conflicting bit: every level yields a or b -- nothing new."""
        a, b = b"\x0c" * 4, b"\x0c\x0c\x0c\x0e"
        eng = self._engine([a, b], ScriptedRng(level, [W1, W2]))
        assert eng._op_bit_cross(bytearray(a), 0, a) is None

    def test_produces_the_scripted_child(self):
        eng = self._engine([A, B], ScriptedRng(HALF, [W1]))
        want = _bytes((_int(A) & ~D) | (D & _int(W1)), N)
        assert eng._op_bit_cross(bytearray(A), 0, A) == bytearray(want)

    def test_bytearray_corpus_entries(self):
        corpus = [bytearray(A), bytearray(B)]
        eng = self._engine(corpus, ScriptedRng(OR))
        got = eng._op_bit_cross(bytearray(A), 0, corpus[0])
        assert got == bytearray(_bytes(_int(A) | _int(B), N))

    def test_respects_max_len(self):
        eng = self._engine([A, B], ScriptedRng(OR), max_len=5)
        got = eng._op_bit_cross(bytearray(A), 0, A)
        assert got == bytearray(_bytes(_int(A) | _int(B), N)[:5])
