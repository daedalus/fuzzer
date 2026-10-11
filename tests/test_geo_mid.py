"""Tests for the geo_mid operator: geometric mean of two seeds.

Per field of width 1/2/4 bytes (BE or LE) over the differing span:
child = ceil(sqrt((a+1) * (b+1))) - 1.  The +1 keeps zero from absorbing;
rounding up keeps a gap of 2 reachable.
The field grid starts at (first differing byte - phase), phase in [0, width).

Randomness is scripted (Hard Rule 39): every expected child is exact.
"""

import math

import numpy as np
import pytest

from fuzzer_tool.core.mutations.generic import _isqrt_u64, arith_mid, geo_mid
from fuzzer_tool.core.operator_registry import _AVAILABLE, REGISTRY
from fuzzer_tool.services.corpus_manager import seed_key
from fuzzer_tool.services.operators import OperatorEngine

B1, B2, L2, B4, L4 = range(5)
SHAPES = {B1: (1, "big"), B2: (2, "big"), L2: (2, "little"), B4: (4, "big"), L4: (4, "little")}
_PREFIX = b"\x89PNG\r\n\x1a\x0a"
U32_MAX = 0xFFFFFFFF


class ScriptedRng:
    """Replays shape, then (width > 1 only) phase draws; nothing is random."""

    def __init__(self, shape, phase=0, pick=0):
        self._draws = [shape, phase]
        self._pick = pick
        self.bounds = []

    def randrange(self, n):
        self.bounds.append(n)
        return self._draws.pop(0)

    def choice(self, seq):
        return seq[self._pick]

    def sample(self, population, k):
        return list(population)[:k]


class ArithRng(ScriptedRng):
    """arith_mid draws shape, weight (k=2: midpoint), then phase."""

    def __init__(self, shape, phase=0):
        super().__init__(shape, phase)
        self._draws = [shape, 1, phase]


class _Fuzzer:
    def __init__(self, corpus, rng=None, max_len=65536, parents=None):
        self.corpus = corpus
        # seed_meta is keyed by bytes; parent_key holds the parent's content hash.
        self.seed_meta = {c: {"parent_key": seed_key(p)} for c, p in (parents or {}).items()}
        self._use_lineage = True
        self.max_len = max_len
        self._rng = rng


def _geo(x: int, y: int) -> int:
    prod = (x + 1) * (y + 1)
    root = math.isqrt(prod)
    return root + (root * root < prod) - 1


def _oracle(a: bytes, b: bytes, shape: int, phase: int = 0) -> bytes:
    """Reference from the grid start to the end: equal chunks are fixed points."""
    width, order = SHAPES[shape]
    first = next(i for i in range(len(a)) if a[i] != b[i])
    start = max(first - (phase if width > 1 else 0), 0)
    out = bytearray(a)
    for i in range(start, len(a) - (len(a) - start) % width, width):
        x = int.from_bytes(a[i : i + width], order)
        y = int.from_bytes(b[i : i + width], order)
        out[i : i + width] = _geo(x, y).to_bytes(width, order)
    return bytes(out)


A = b"\xf0\x0f\xaa\x55" * 4
B = b"\x3c\xc3\x55\xaa" * 4
N = len(A)


class TestIsqrt:
    """The exact u64 square root under the field means."""

    EDGES = (
        0,
        1,
        2,
        3,
        4,
        (1 << 53) - 1,
        1 << 53,
        (1 << 53) + 1,
        (1 << 62) - 1,
        1 << 62,
        (U32_MAX - 1) ** 2,
        (U32_MAX - 1) ** 2 - 1,
        U32_MAX**2 - 1,
        U32_MAX**2,
        U32_MAX**2 + 1,
        (1 << 32) * U32_MAX,
    )

    def test_matches_math_isqrt_at_the_edges(self):
        n = np.array(self.EDGES, dtype=np.uint64)
        want = [math.isqrt(int(v)) for v in n]
        assert _isqrt_u64(n).tolist() == want

    def test_perfect_squares_and_neighbours(self):
        roots = [0, 1, 2, 3, 94906265, 94906266, (1 << 31) - 1, 1 << 31, U32_MAX - 1, U32_MAX]
        n = np.array([r * r + d for r in roots for d in (-1, 0, 1) if 0 <= r * r + d < 1 << 64])
        n = n.astype(np.uint64)
        assert _isqrt_u64(n).tolist() == [math.isqrt(int(v)) for v in n]

    def test_control_plain_float_sqrt_is_not_trusted_alone(self):
        """Falsification: the fix-up pass is what guarantees r*r <= n < (r+1)^2."""
        n = np.array(self.EDGES, dtype=np.uint64)
        r = _isqrt_u64(n).astype(object)
        for v, root in zip(self.EDGES, r, strict=True):
            assert root * root <= v < (root + 1) ** 2


class TestGeoMid:
    def test_known_value_geometric_mean_of_shifted_fields(self):
        a, b = b"\x03" * 4, b"\x23" * 4  # (3+1)=4 and (35+1)=36 -> 12 -> 11
        assert geo_mid(a, b, ScriptedRng(B1)) == b"\x0b" * 4

    @pytest.mark.parametrize("shape", list(SHAPES))
    def test_matches_reference_for_every_shape(self, shape):
        assert geo_mid(A, B, ScriptedRng(shape)) == _oracle(A, B, shape)

    def test_zero_does_not_absorb(self):
        """Falsification: plain isqrt(a*b) returns 0 == a; the +1 shift does not."""
        a, b = b"\x00" * 4, b"\xff" * 4
        assert math.isqrt(0 * 0xFF) == 0
        assert geo_mid(a, b, ScriptedRng(B1)) == b"\x0f" * 4  # isqrt(1 * 256) - 1

    def test_halves_the_exponent_not_the_value(self):
        a, b = b"\x00" * 4, b"\xff" * 4
        assert geo_mid(a, b, ScriptedRng(B4)) == b"\x00\x00\xff\xff"  # 2^16 - 1
        assert arith_mid(a, b, ArithRng(B4)) != geo_mid(a, b, ScriptedRng(B4))

    def test_be16_carries_across_bytes(self):
        a, b = b"\x00\xff", b"\x01\x01"  # 255 and 257: isqrt(256 * 258) - 1 = 255
        assert geo_mid(a, b, ScriptedRng(B2)) == _oracle(a, b, B2)

    def test_le16_reads_little_endian(self):
        a, b = b"\x00\x00", b"\xff\xff"  # LE 0 and 65535 -> isqrt(65536) - 1 = 255
        assert geo_mid(a, b, ScriptedRng(L2)) == b"\xff\x00"

    @pytest.mark.parametrize("shape", list(SHAPES))
    def test_shared_prefix_survives(self, shape):
        a, b = _PREFIX + b"\x01\x02\x03\x04", _PREFIX + b"\x10\x20\x30\x40"
        assert geo_mid(a, b, ScriptedRng(shape))[: len(_PREFIX)] == _PREFIX

    @pytest.mark.parametrize("shape", list(SHAPES))
    def test_equal_fields_are_fixed_points_even_at_the_maximum(self, shape):
        """Adversarial: (2^32) * (2^32) wraps u64; equal fields must still survive."""
        a = b"\xff\xff\xff\xff" + b"\x00\x00\x00\x01"
        b = b"\xff\xff\xff\xff" + b"\x00\x00\x00\x09"
        # phase 0 keeps the grid aligned to the unequal field for width 4.
        child = geo_mid(a, b, ScriptedRng(shape))
        assert child[:4] == b"\xff\xff\xff\xff"

    @pytest.mark.parametrize("shape", list(SHAPES))
    def test_each_field_stays_between_the_parents_at_the_extremes(self, shape):
        """Adversarial: 0x00/0xff fields must neither wrap nor leave [a, b]."""
        a = b"\xff\x00\xff\xff\x00\x00\xff\xfe"
        b = b"\x00\xff\xfe\xff\xff\x00\xfe\xff"
        child = geo_mid(a, b, ScriptedRng(shape))
        width, order = SHAPES[shape]
        for i in range(0, len(a), width):
            x, y, c = (int.from_bytes(v[i : i + width], order) for v in (a, b, child))
            assert min(x, y) <= c <= max(x, y)

    def test_largest_unequal_pair_does_not_overflow(self):
        """(2^32) * (2^32 - 1) is the largest product; the root rounds up to 2^32."""
        a, b = b"\xff\xff\xff\xff", b"\xff\xff\xff\xfe"
        assert geo_mid(a, b, ScriptedRng(B4, phase=3)) == a  # 2^32 - 1: the other parent
        assert geo_mid(a, b, ScriptedRng(B4, phase=3)) == _oracle(a, b, B4, 3)

    def test_gap_of_two_is_bisected(self):
        """Falsification: floor isqrt maps (x, x+2) back to x and never reaches x+1."""
        a, b = b"\x00\x00\x00\x07", b"\x00\x00\x00\x09"
        assert math.isqrt(8 * 10) - 1 == 7  # floor returns the parent
        assert geo_mid(a, b, ScriptedRng(B4, phase=3)) == b"\x00\x00\x00\x08"

    def test_bytes_outside_the_differing_span_are_untouched(self):
        a = bytes(range(64))
        b = a[:20] + b"\xee\xdd\xcc\xbb" + a[24:]
        child = geo_mid(a, b, ScriptedRng(B4))
        assert child[:20] == a[:20]
        assert child[24:] == a[24:]
        assert child[20:24] != a[20:24]

    def test_partial_tail_chunk_is_left_alone_when_the_diff_precedes_it(self):
        a, b = b"\x00" * 7, b"\x0f" + b"\x00" * 6  # field 0 .. 0x0f000000
        want = _geo(0, 0x0F000000).to_bytes(4, "big") + b"\x00" * 3
        assert geo_mid(a, b, ScriptedRng(B4)) == want

    def test_diff_only_in_a_partial_tail_falls_back_to_one_byte(self):
        a, b = b"\x00" * 6, b"\x00" * 5 + b"\x0f"
        assert geo_mid(a, b, ScriptedRng(B4)) == b"\x00" * 5 + b"\x03"

    @pytest.mark.parametrize("b", [b"\x55", b"\x55" * 31, b"\x55" * 33, b"\x55" * 4096])
    def test_length_is_base_length(self, b):
        a = b"\xaa" * 32
        assert len(geo_mid(a, b, ScriptedRng(B4))) == len(a)

    def test_short_donor_tail_comes_from_base(self):
        a, b = b"\xaa" * 8, b"\x56" * 2  # ceil(sqrt(171 * 87)) - 1 = 121
        want = b"\x79" * 2 + a[2:]
        assert geo_mid(a, b, ScriptedRng(B1)) == want

    @pytest.mark.parametrize(
        ("a", "b"),
        [(b"", b"abc"), (b"abc", b""), (b"", b""), (b"same", b"same")],
    )
    def test_degenerate_inputs_return_base_without_drawing(self, a, b):
        assert geo_mid(a, b, None) == a  # rng=None: any draw would raise

    def test_symmetric_in_the_parents(self):
        assert geo_mid(A, B, ScriptedRng(B4)) == geo_mid(B, A, ScriptedRng(B4))

    def test_control_same_script_gives_the_same_child(self):
        first = geo_mid(A, B, ScriptedRng(L4))
        assert geo_mid(A, B, ScriptedRng(L4)) == first

    def test_large_input(self):
        a, b = bytes(range(256)) * 4096, bytes(reversed(range(256))) * 4096
        assert geo_mid(a, b, ScriptedRng(B4)) == _oracle(a, b, B4)


class TestFieldGrid:
    """The grid hangs off the first differing byte, not off offset 0."""

    def test_field_at_an_odd_offset_is_one_integer(self):
        a = b"K" + b"\x00\x00\x00\x00" + b"\x07" * 3
        b = b"K" + b"\x12\x34\x56\x78" + b"\x07" * 3
        want = b"K" + _geo(0, 0x12345678).to_bytes(4, "big") + b"\x07" * 3
        assert geo_mid(a, b, ScriptedRng(B4)) == want

    def test_phase_backs_the_grid_up_to_the_true_field_start(self):
        """High bytes equal: first diff is mid-field; phase 1 realigns."""
        a = b"K\x12\x00\x00\x00" + b"\x07" * 3
        b = b"K\x12\xff\xff\xff" + b"\x07" * 3
        want = b"K" + _geo(0x12000000, 0x12FFFFFF).to_bytes(4, "big") + b"\x07" * 3
        assert geo_mid(a, b, ScriptedRng(B4, phase=1)) == want
        assert geo_mid(a, b, ScriptedRng(B4, phase=0)) != want  # misaligned: falsification

    def test_phase_is_clamped_at_the_buffer_start(self):
        a, b = b"\x00\x00\x00\x00", b"\x00\x00\x00\x0f"
        assert geo_mid(a, b, ScriptedRng(B4, phase=3)) == b"\x00\x00\x00\x03"

    def test_phase_is_drawn_per_width_and_not_for_bytes(self):
        rng = ScriptedRng(B4, phase=2)
        geo_mid(A, B, rng)
        assert rng.bounds == [len(SHAPES), 4]
        rng = ScriptedRng(B1)
        geo_mid(A, B, rng)
        assert rng.bounds == [len(SHAPES)]

    @pytest.mark.parametrize("phase", range(4))
    def test_matches_reference_at_every_phase(self, phase):
        a = bytes(range(8, 40))
        b = a[:11] + bytes(reversed(a[11:27])) + a[27:]
        assert geo_mid(a, b, ScriptedRng(B4, phase)) == _oracle(a, b, B4, phase)


class TestBisection:
    """Halving the exponent reaches a small boundary in fewer steps."""

    K = 0x00001234

    def _steps(self, fn):
        lo, hi = b"\x00" * 4, b"\xff" * 4
        for steps in range(1, 80):
            lo_n, hi_n = int.from_bytes(lo, "big"), int.from_bytes(hi, "big")
            if hi_n - lo_n <= 1:
                return hi_n, steps
            first = next(i for i in range(4) if lo[i] != hi[i])
            mid = fn(lo, hi, first)
            if int.from_bytes(mid, "big") >= self.K:
                hi = mid
            else:
                lo = mid
        return None, steps

    def test_geo_finds_a_small_boundary(self):
        boundary, _steps = self._steps(lambda a, b, p: geo_mid(a, b, ScriptedRng(B4, p)))
        assert boundary == self.K

    def test_geo_needs_fewer_steps_than_arith_for_a_small_boundary(self):
        """Falsification: not a general win -- the arithmetic mean needs ~2x more."""
        _b, geo = self._steps(lambda a, b, p: geo_mid(a, b, ScriptedRng(B4, p)))
        _b, arith = self._steps(lambda a, b, p: arith_mid(a, b, ArithRng(B4, p)))
        assert geo < arith


class TestOpGeoMid:
    def _engine(self, corpus, rng, **kw):
        return OperatorEngine(_Fuzzer(corpus, rng, **kw))

    def test_registered_structural(self):
        assert "geo_mid" in REGISTRY.names()
        assert REGISTRY.category_of("geo_mid") == "structural"

    def test_declines_with_single_seed(self):
        assert self._engine([A], None)._op_geo_mid(bytearray(A), 0, A) is None

    def test_declines_on_empty_buffer(self):
        assert self._engine([A, B], None)._op_geo_mid(bytearray(), 0, A) is None

    def test_declines_when_child_equals_a_parent(self):
        """Adjacent values: the floor lands on one parent -- already executed."""
        a, b = b"\x04" * 4, b"\x05" * 4  # ceil(sqrt(5 * 6)) - 1 = 5
        eng = self._engine([a, b], ScriptedRng(B1))
        assert eng._op_geo_mid(bytearray(b), 0, b) is None

    def test_produces_the_scripted_child(self):
        eng = self._engine([A, B], ScriptedRng(B2))
        assert eng._op_geo_mid(bytearray(A), 0, A) == bytearray(_oracle(A, B, B2))

    def test_respects_max_len(self):
        eng = self._engine([A, B], ScriptedRng(B1), max_len=5)
        got = eng._op_geo_mid(bytearray(A), 0, A)
        assert got == bytearray(_oracle(A, B, B1)[:5])

    def test_related_donor_is_declined_unrelated_is_used(self):
        root, kid, other = b"\x11" * 16, b"\x22" * 16, b"\x5a" * 16
        parents = {kid: root}
        eng = self._engine([root, kid], ScriptedRng(B1), parents=parents)
        assert eng._op_geo_mid(bytearray(root), 0, root) is None
        eng = self._engine([root, kid, other], ScriptedRng(B1), parents=parents)
        got = eng._op_geo_mid(bytearray(root), 0, root)
        assert got == bytearray(_oracle(root, other, B1))

    def test_available_only_with_lineage(self):
        class F:
            pass

        f = F()
        assert not _AVAILABLE["geo_mid"](f, b"")
        f._use_lineage = True
        assert _AVAILABLE["geo_mid"](f, b"")
