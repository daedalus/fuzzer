"""Tests for the power_mid operator: harmonic or quadratic mean of two seeds.

Per field of width 1/2/4 bytes (BE or LE) over the differing span, with
X = x + 1, Y = y + 1 (the +1 keeps 0 from absorbing):
  harmonic   ceil(2XY / (X + Y)) - 1
  quadratic  floor(sqrt((X^2 + Y^2) / 2)) - 1
Rounding goes toward the arithmetic mean, so a gap of 2 stays reachable.
When every field lands within 1 of floor((x + y) / 2) the child repeats
arith_mid's and the operator returns the base unchanged (declined).

Randomness is scripted (Hard Rule 39): every expected child is exact.
"""

import math

import numpy as np
import pytest

from fuzzer_tool.core.mutations.generic import _harm_fields, _quad_fields, power_mid
from fuzzer_tool.core.operator_registry import _AVAILABLE, REGISTRY
from fuzzer_tool.services.corpus_manager import seed_key
from fuzzer_tool.services.operators import OperatorEngine

B1, B2, L2, B4, L4 = range(5)
SHAPES = {B1: (1, "big"), B2: (2, "big"), L2: (2, "little"), B4: (4, "big"), L4: (4, "little")}
HARM, QUAD = range(2)
_PREFIX = b"\x89PNG\r\n\x1a\x0a"
U32_MAX = 0xFFFFFFFF


class ScriptedRng:
    """Replays shape, kind, then (width > 1 only) phase draws; nothing is random."""

    def __init__(self, shape, kind=HARM, phase=0, pick=0):
        self._draws = [shape, kind, phase]
        self._pick = pick
        self.bounds = []

    def randrange(self, n):
        self.bounds.append(n)
        return self._draws.pop(0)

    def choice(self, seq):
        return seq[self._pick]

    def sample(self, population, k):
        return list(population)[:k]


class _Fuzzer:
    def __init__(self, corpus, rng=None, max_len=65536, parents=None):
        self.corpus = corpus
        # seed_meta is keyed by bytes; parent_key holds the parent's content hash.
        self.seed_meta = {c: {"parent_key": seed_key(p)} for c, p in (parents or {}).items()}
        self._use_lineage = True
        self.max_len = max_len
        self._rng = rng


def _harm(x: int, y: int) -> int:
    big_x, big_y = x + 1, y + 1
    return -(-2 * big_x * big_y // (big_x + big_y)) - 1


def _quad(x: int, y: int) -> int:
    big_x, big_y = x + 1, y + 1
    return math.isqrt((big_x * big_x + big_y * big_y) // 2) - 1


MEANS = {HARM: _harm, QUAD: _quad}


def _oracle(a: bytes, b: bytes, shape: int, kind: int, phase: int = 0) -> bytes:
    """Reference from the grid start to the end; None-free: callers pass undeclined pairs."""
    width, order = SHAPES[shape]
    first = next(i for i in range(len(a)) if a[i] != b[i])
    start = max(first - (phase if width > 1 else 0), 0)
    out = bytearray(a)
    for i in range(start, len(a) - (len(a) - start) % width, width):
        x = int.from_bytes(a[i : i + width], order)
        y = int.from_bytes(b[i : i + width], order)
        out[i : i + width] = MEANS[kind](x, y).to_bytes(width, order)
    return bytes(out)


A = b"\xf0\x0f\xaa\x55" * 4
B = b"\x3c\xc3\x55\xaa" * 4
_COMBOS = [(s, k) for s in SHAPES for k in (HARM, QUAD)]
_EDGE = (0, 1, 2, 3, 255, 256, 65535, 65536, 1 << 24, (1 << 31) - 1, 1 << 31, U32_MAX - 1, U32_MAX)


class TestFields:
    """The exact u64 field means, before the equal-field override."""

    PAIRS = [(x, y) for x in _EDGE for y in _EDGE if x != y]

    def _arrays(self):
        fa = np.array([p[0] for p in self.PAIRS], dtype=np.uint64)
        fb = np.array([p[1] for p in self.PAIRS], dtype=np.uint64)
        return fa, fb

    def test_harmonic_matches_integer_reference_at_the_edges(self):
        fa, fb = self._arrays()
        assert _harm_fields(fa, fb).tolist() == [_harm(x, y) for x, y in self.PAIRS]

    def test_quadratic_matches_integer_reference_at_the_edges(self):
        fa, fb = self._arrays()
        assert _quad_fields(fa, fb).tolist() == [_quad(x, y) for x, y in self.PAIRS]

    def test_control_reference_is_symmetric(self):
        """Control: the oracle against itself with parents swapped."""
        assert all(_harm(x, y) == _harm(y, x) and _quad(x, y) == _quad(y, x) for x, y in self.PAIRS)

    def test_quadratic_sum_of_squares_does_not_wrap(self):
        """Adversarial: X^2 + Y^2 reaches 2^65; a wrapped sum would give a tiny root."""
        fa = np.array([U32_MAX], dtype=np.uint64)
        fb = np.array([0], dtype=np.uint64)
        assert int(_quad_fields(fa, fb)[0]) == 0xB504F332


class TestPowerMid:
    def test_known_values(self):
        a, b = (1000).to_bytes(2, "big"), (5000).to_bytes(2, "big")
        assert power_mid(a, b, ScriptedRng(B2, HARM)) == (1668).to_bytes(2, "big")
        assert power_mid(a, b, ScriptedRng(B2, QUAD)) == (3605).to_bytes(2, "big")

    def test_split_points_order_harmonic_below_quadratic(self):
        a, b = b"\x00" * 4, b"\xff" * 4
        assert power_mid(a, b, ScriptedRng(B4, HARM)) == b"\x00\x00\x00\x01"
        assert power_mid(a, b, ScriptedRng(B4, QUAD)) == b"\xb5\x04\xf3\x32"

    @pytest.mark.parametrize(("shape", "kind"), _COMBOS)
    def test_matches_reference_for_every_shape_and_kind(self, shape, kind):
        assert power_mid(A, B, ScriptedRng(shape, kind)) == _oracle(A, B, shape, kind)

    @pytest.mark.parametrize("kind", [HARM, QUAD])
    def test_close_parents_are_declined(self, kind):
        """Falsification partner: both means collapse onto arith_mid's child."""
        a, b = b"\x64" * 4, b"\x66" * 4  # 100 and 102: arithmetic 101
        assert power_mid(a, b, ScriptedRng(B1, kind)) == a

    @pytest.mark.parametrize("kind", [HARM, QUAD])
    def test_far_parents_are_not_declined(self, kind):
        a, b = b"\x00" * 4, b"\xff" * 4
        assert power_mid(a, b, ScriptedRng(B1, kind)) != a

    def test_one_far_field_among_close_ones_is_enough(self):
        a = b"\x64\x64\x64\x00"
        b = b"\x66\x66\x66\xff"
        assert power_mid(a, b, ScriptedRng(B1, HARM)) != a

    @pytest.mark.parametrize(("shape", "kind"), _COMBOS)
    def test_shared_prefix_survives(self, shape, kind):
        a, b = _PREFIX + b"\x01\x02\x03\x04", _PREFIX + b"\x10\x20\x30\x40"
        got = power_mid(a, b, ScriptedRng(shape, kind))
        assert got[: len(_PREFIX)] == _PREFIX

    @pytest.mark.parametrize(("shape", "kind"), _COMBOS)
    def test_equal_maximal_fields_survive(self, shape, kind):
        """Adversarial: 2^32 * 2^32 wraps u64; equal fields must still be fixed."""
        a = b"\xff\xff\xff\xff" + b"\x00\x00\x00\x01"
        b = b"\xff\xff\xff\xff" + b"\x00\x00\x00\xf9"
        assert power_mid(a, b, ScriptedRng(shape, kind))[:4] == b"\xff\xff\xff\xff"

    @pytest.mark.parametrize(("shape", "kind"), _COMBOS)
    def test_each_field_stays_between_the_parents_at_the_extremes(self, shape, kind):
        """Adversarial: 0x00/0xff fields must neither wrap nor leave [a, b]."""
        a = b"\xff\x00\xff\xf0\x00\x00\xff\xe0"
        b = b"\x00\xff\xe0\xff\xff\x00\xe0\xff"
        child = power_mid(a, b, ScriptedRng(shape, kind))
        assert child != a
        width, order = SHAPES[shape]
        for i in range(0, len(a), width):
            x, y, c = (int.from_bytes(v[i : i + width], order) for v in (a, b, child))
            assert min(x, y) <= c <= max(x, y)

    @pytest.mark.parametrize("kind", [HARM, QUAD])
    def test_gap_of_two_is_reachable_when_the_means_are_not_declined(self, kind):
        """Falsification: wrong rounding maps (x, x+2) onto a parent; interior is x+1."""
        fa = np.array([6], dtype=np.uint64)
        fb = np.array([8], dtype=np.uint64)
        fn = _harm_fields if kind == HARM else _quad_fields
        assert int(fn(fa, fb)[0]) == 7

    def test_bytes_outside_the_differing_span_are_untouched(self):
        a = bytes(range(64))
        b = a[:20] + b"\xee\xdd\xcc\xbb" + a[24:]
        child = power_mid(a, b, ScriptedRng(B4, QUAD))
        assert child[:20] == a[:20]
        assert child[24:] == a[24:]
        assert child[20:24] != a[20:24]

    def test_diff_only_in_a_partial_tail_falls_back_to_one_byte(self):
        a, b = b"\x00" * 6, b"\x00" * 5 + b"\xff"
        assert power_mid(a, b, ScriptedRng(B4, QUAD)) == b"\x00" * 5 + bytes([_quad(0, 255)])

    @pytest.mark.parametrize("b", [b"\x55", b"\x55" * 31, b"\x55" * 33, b"\x55" * 4096])
    def test_length_is_base_length(self, b):
        a = b"\xaa" * 32
        assert len(power_mid(a, b, ScriptedRng(B4, QUAD))) == len(a)

    def test_short_donor_tail_comes_from_base(self):
        a, b = b"\xaa" * 8, b"\x01" * 2
        want = bytes([_quad(0xAA, 1)]) * 2 + a[2:]
        assert power_mid(a, b, ScriptedRng(B1, QUAD)) == want

    @pytest.mark.parametrize(
        ("a", "b"),
        [(b"", b"abc"), (b"abc", b""), (b"", b""), (b"same", b"same")],
    )
    def test_degenerate_inputs_return_base_without_drawing(self, a, b):
        assert power_mid(a, b, None) == a  # rng=None: any draw would raise

    def test_control_same_script_gives_the_same_child(self):
        first = power_mid(A, B, ScriptedRng(L4, QUAD))
        assert power_mid(A, B, ScriptedRng(L4, QUAD)) == first

    def test_large_input(self):
        a, b = bytes(range(256)) * 4096, bytes(reversed(range(256))) * 4096
        assert power_mid(a, b, ScriptedRng(B4, HARM)) == _oracle(a, b, B4, HARM)


class TestDraws:
    def test_kind_and_phase_are_drawn_phase_only_for_wide_fields(self):
        rng = ScriptedRng(B4, QUAD, phase=2)
        power_mid(A, B, rng)
        assert rng.bounds == [len(SHAPES), 2, 4]
        rng = ScriptedRng(B1, HARM)
        power_mid(A, B, rng)
        assert rng.bounds == [len(SHAPES), 2]

    @pytest.mark.parametrize("phase", range(4))
    def test_matches_reference_at_every_phase(self, phase):
        a = bytes(range(8, 40))
        b = a[:11] + bytes(reversed(a[11:27])) + a[27:]
        assert power_mid(a, b, ScriptedRng(B4, QUAD, phase)) == _oracle(a, b, B4, QUAD, phase)


class TestOpPowerMid:
    def _engine(self, corpus, rng, **kw):
        return OperatorEngine(_Fuzzer(corpus, rng, **kw))

    def test_registered_structural(self):
        assert "power_mid" in REGISTRY.names()
        assert REGISTRY.category_of("power_mid") == "structural"

    def test_declines_with_single_seed(self):
        assert self._engine([A], None)._op_power_mid(bytearray(A), 0, A) is None

    def test_declines_on_empty_buffer(self):
        assert self._engine([A, B], None)._op_power_mid(bytearray(), 0, A) is None

    def test_declines_when_the_means_collapse_onto_arith_mid(self):
        a, b = b"\x64" * 4, b"\x66" * 4
        eng = self._engine([a, b], ScriptedRng(B1, QUAD))
        assert eng._op_power_mid(bytearray(a), 0, a) is None

    def test_produces_the_scripted_child(self):
        eng = self._engine([A, B], ScriptedRng(B2, QUAD))
        want = _oracle(A, B, B2, QUAD)
        assert eng._op_power_mid(bytearray(A), 0, A) == bytearray(want)

    def test_respects_max_len(self):
        eng = self._engine([A, B], ScriptedRng(B1, HARM), max_len=5)
        got = eng._op_power_mid(bytearray(A), 0, A)
        assert got == bytearray(_oracle(A, B, B1, HARM)[:5])

    def test_related_donor_is_declined_unrelated_is_used(self):
        root, kid, other = b"\x11" * 16, b"\x22" * 16, b"\xc8" * 16
        parents = {kid: root}
        eng = self._engine([root, kid], ScriptedRng(B1, QUAD), parents=parents)
        assert eng._op_power_mid(bytearray(root), 0, root) is None
        eng = self._engine([root, kid, other], ScriptedRng(B1, QUAD), parents=parents)
        got = eng._op_power_mid(bytearray(root), 0, root)
        assert got == bytearray(_oracle(root, other, B1, QUAD))

    def test_available_only_with_lineage(self):
        class F:
            pass

        f = F()
        assert not _AVAILABLE["power_mid"](f, b"")
        f._use_lineage = True
        assert _AVAILABLE["power_mid"](f, b"")
