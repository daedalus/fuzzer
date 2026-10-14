"""Tests for the arith_mid operator: weighted arithmetic mean of two seeds.

Per field of width 1/2/4 bytes (BE or LE) over the differing span:
child = floor(((4-k)*a + k*b) / 4), k in {1, 2, 3}.  k=2 is the midpoint.
The field grid starts at (first differing byte - phase), phase in [0, width).

Randomness is scripted (Hard Rule 39): every expected child is exact.
"""

import pytest

from fuzzer_tool.core.mutations.generic import arith_mid
from fuzzer_tool.core.operator_registry import _AVAILABLE, REGISTRY
from fuzzer_tool.services.corpus_manager import seed_key
from fuzzer_tool.services.operators import OperatorEngine

B1, B2, L2, B4, L4 = range(5)
SHAPES = {B1: (1, "big"), B2: (2, "big"), L2: (2, "little"), B4: (4, "big"), L4: (4, "little")}
QUARTER, MID, THREE_QUARTER = range(3)  # draw index; k = index + 1
_PREFIX = b"\x89PNG\r\n\x1a\x0a"


class ScriptedRng:
    """Replays shape, weight, then (width > 1 only) phase draws; nothing is random."""

    def __init__(self, shape, weight=MID, phase=0, pick=0):
        self._draws = [shape, weight, phase]
        self._pick = pick
        self.bounds = []
        self.sample_calls = []

    def randrange(self, n):
        self.bounds.append(n)
        return self._draws.pop(0)

    def choice(self, seq):
        return seq[self._pick]

    def sample(self, population, k):
        self.sample_calls.append((len(population), k))
        return list(population)[:k]


class _Fuzzer:
    def __init__(self, corpus, rng=None, max_len=65536, parents=None):
        self.corpus = corpus
        # seed_meta is keyed by bytes; parent_key holds the parent's content hash.
        self.seed_meta = {c: {"parent_key": seed_key(p)} for c, p in (parents or {}).items()}
        self._use_lineage = True
        self.max_len = max_len
        self._rng = rng


def _oracle(a: bytes, b: bytes, shape: int, weight: int, phase: int = 0) -> bytes:
    """Reference from the grid start to the end: equal chunks are fixed points."""
    width, order = SHAPES[shape]
    k = weight + 1
    first = next(i for i in range(len(a)) if a[i] != b[i])
    start = max(first - (phase if width > 1 else 0), 0)
    out = bytearray(a)
    for i in range(start, len(a) - (len(a) - start) % width, width):
        x = int.from_bytes(a[i : i + width], order)
        y = int.from_bytes(b[i : i + width], order)
        out[i : i + width] = ((x * (4 - k) + y * k) // 4).to_bytes(width, order)
    return bytes(out)


A = b"\xf0\x0f\xaa\x55" * 4
B = b"\x3c\xc3\x55\xaa" * 4
N = len(A)
_COMBOS = [(s, w) for s in SHAPES for w in (QUARTER, MID, THREE_QUARTER)]


class TestArithMid:
    def test_byte_midpoint_is_the_and_xor_identity(self):
        want = bytes((x & y) + ((x ^ y) >> 1) for x, y in zip(A, B, strict=True))
        assert arith_mid(A, B, ScriptedRng(B1, MID)) == want

    @pytest.mark.parametrize(("shape", "weight"), _COMBOS)
    def test_matches_reference_for_every_shape_and_weight(self, shape, weight):
        assert arith_mid(A, B, ScriptedRng(shape, weight)) == _oracle(A, B, shape, weight)

    def test_weights_pull_toward_the_named_parent(self):
        a, b = b"\x00" * 4, b"\x64" * 4  # 0 and 100
        assert arith_mid(a, b, ScriptedRng(B1, QUARTER)) == b"\x19" * 4  # 25
        assert arith_mid(a, b, ScriptedRng(B1, MID)) == b"\x32" * 4  # 50
        assert arith_mid(a, b, ScriptedRng(B1, THREE_QUARTER)) == b"\x4b" * 4  # 75

    def test_be16_carries_across_bytes(self):
        a, b = b"\x00\xff", b"\x01\x01"  # 255 and 257 -> 256
        assert arith_mid(a, b, ScriptedRng(B2)) == b"\x01\x00"

    def test_le16_carries_across_bytes(self):
        a, b = b"\xff\x00", b"\x01\x01"  # 255 and 257 -> 256
        assert arith_mid(a, b, ScriptedRng(L2)) == b"\x00\x01"

    def test_bytewise_is_not_the_numeric_mean(self):
        """Falsifies 'average bytes' as a stand-in for averaging the integer."""
        a, b = b"\x00\xff", b"\x01\x01"
        bytewise = arith_mid(a, b, ScriptedRng(B1))
        numeric = arith_mid(a, b, ScriptedRng(B2))
        assert bytewise == b"\x00\x80"
        assert numeric == b"\x01\x00"
        assert int.from_bytes(bytewise, "big") < int.from_bytes(a, "big")  # left [a, b]

    @pytest.mark.parametrize(("shape", "weight"), _COMBOS)
    def test_shared_prefix_survives(self, shape, weight):
        a, b = _PREFIX + b"\x01\x02\x03\x04", _PREFIX + b"\x10\x20\x30\x40"
        assert arith_mid(a, b, ScriptedRng(shape, weight))[: len(_PREFIX)] == _PREFIX

    @pytest.mark.parametrize(("shape", "weight"), _COMBOS)
    def test_each_field_stays_between_the_parents_at_the_extremes(self, shape, weight):
        """Adversarial: 0x00/0xff fields must neither wrap nor leave [a, b]."""
        a = b"\xff\x00\xff\xff\x00\x00\xff\xfe"
        b = b"\x00\xff\xfe\xff\xff\x00\xfe\xff"
        child = arith_mid(a, b, ScriptedRng(shape, weight))
        width, order = SHAPES[shape]
        for i in range(0, len(a), width):
            x, y, c = (int.from_bytes(v[i : i + width], order) for v in (a, b, child))
            assert min(x, y) <= c <= max(x, y)

    def test_bytes_outside_the_differing_span_are_untouched(self):
        a = bytes(range(64))
        b = a[:20] + b"\xee\xdd\xcc\xbb" + a[24:]
        child = arith_mid(a, b, ScriptedRng(B4))
        assert child[:20] == a[:20]
        assert child[24:] == a[24:]
        assert child[20:24] != a[20:24]

    def test_partial_tail_chunk_is_left_alone_when_the_diff_precedes_it(self):
        a, b = b"\x00" * 7, b"\x08" + b"\x00" * 6
        assert arith_mid(a, b, ScriptedRng(B4)) == b"\x04" + b"\x00" * 6

    def test_diff_only_in_a_partial_tail_falls_back_to_one_byte(self):
        a, b = b"\x00" * 6, b"\x00" * 5 + b"\x09"
        assert arith_mid(a, b, ScriptedRng(B4)) == b"\x00" * 5 + b"\x04"

    @pytest.mark.parametrize("b", [b"\x55", b"\x55" * 31, b"\x55" * 33, b"\x55" * 4096])
    def test_length_is_base_length(self, b):
        a = b"\xaa" * 32
        assert len(arith_mid(a, b, ScriptedRng(B4))) == len(a)

    def test_short_donor_tail_comes_from_base(self):
        a, b = b"\xaa" * 8, b"\x56" * 2
        want = b"\x80" * 2 + a[2:]  # (0xaa + 0x56) // 2 = 0x80
        assert arith_mid(a, b, ScriptedRng(B1)) == want

    @pytest.mark.parametrize(
        ("a", "b"),
        [(b"", b"abc"), (b"abc", b""), (b"", b""), (b"same", b"same")],
    )
    def test_degenerate_inputs_return_base_without_drawing(self, a, b):
        assert arith_mid(a, b, None) == a  # rng=None: any draw would raise

    def test_control_same_script_gives_the_same_child(self):
        first = arith_mid(A, B, ScriptedRng(L4, THREE_QUARTER))
        assert arith_mid(A, B, ScriptedRng(L4, THREE_QUARTER)) == first

    def test_large_input(self):
        a, b = bytes(range(256)) * 4096, bytes(reversed(range(256))) * 4096
        assert arith_mid(a, b, ScriptedRng(B4)) == _oracle(a, b, B4, MID)


class TestFieldGrid:
    """The grid hangs off the first differing byte, not off offset 0."""

    def test_field_at_an_odd_offset_is_averaged_as_one_integer(self):
        a = b"K" + b"\x00\x00\x00\x00" + b"\x07" * 3
        b = b"K" + b"\x12\x34\x56\x78" + b"\x07" * 3
        want = b"K" + (0x12345678 // 2).to_bytes(4, "big") + b"\x07" * 3
        assert arith_mid(a, b, ScriptedRng(B4)) == want

    def test_phase_backs_the_grid_up_to_the_true_field_start(self):
        """High bytes equal: first diff is mid-field; phase 1 realigns."""
        a = b"K\x12\x00\x00\x00" + b"\x07" * 3
        b = b"K\x12\xff\xff\xff" + b"\x07" * 3
        want = b"K" + ((0x12000000 + 0x12FFFFFF) // 2).to_bytes(4, "big") + b"\x07" * 3
        assert arith_mid(a, b, ScriptedRng(B4, phase=1)) == want
        assert arith_mid(a, b, ScriptedRng(B4, phase=0)) != want  # misaligned: falsification

    def test_phase_is_clamped_at_the_buffer_start(self):
        a, b = b"\x00\x00\x00\x00", b"\x00\x00\x00\x08"
        assert arith_mid(a, b, ScriptedRng(B4, phase=3)) == b"\x00\x00\x00\x04"

    def test_phase_is_drawn_per_width_and_not_for_bytes(self):
        rng = ScriptedRng(B4, phase=2)
        arith_mid(A, B, rng)
        assert rng.bounds == [len(SHAPES), 3, 4]
        rng = ScriptedRng(B1)
        arith_mid(A, B, rng)
        assert rng.bounds == [len(SHAPES), 3]

    @pytest.mark.parametrize("phase", range(4))
    def test_matches_reference_at_every_phase(self, phase):
        a = bytes(range(8, 40))
        b = a[:11] + bytes(reversed(a[11:27])) + a[27:]
        assert arith_mid(a, b, ScriptedRng(B4, MID, phase)) == _oracle(a, b, B4, MID, phase)


class TestBisection:
    """The midpoint halves the interval around a comparison boundary."""

    K = 0x12345678

    def _find(self, shape):
        lo, hi = b"\x00" * 4, b"\xff" * 4
        for steps in range(1, 40):
            if int.from_bytes(hi, "big") - int.from_bytes(lo, "big") <= 1:
                return int.from_bytes(hi, "big"), steps
            first = next(i for i in range(4) if lo[i] != hi[i])
            mid = arith_mid(lo, hi, ScriptedRng(shape, phase=first))
            if int.from_bytes(mid, "big") >= self.K:
                hi = mid
            else:
                lo = mid
        return None, steps

    def test_be32_finds_the_boundary_in_32_steps(self):
        boundary, steps = self._find(B4)
        assert boundary == self.K
        assert steps <= 33

    def test_bytewise_mean_cannot_bisect_a_wide_field(self):
        """Falsification: width-1 averaging never converges to K."""
        boundary, _steps = self._find(B1)
        assert boundary != self.K


class TestOpArithMid:
    def _engine(self, corpus, rng, **kw):
        return OperatorEngine(_Fuzzer(corpus, rng, **kw))

    def test_registered_structural(self):
        assert "arith_mid" in REGISTRY.names()
        assert REGISTRY.category_of("arith_mid") == "structural"

    def test_declines_with_single_seed(self):
        assert self._engine([A], None)._op_arith_mid(bytearray(A), 0, A) is None

    def test_declines_on_empty_buffer(self):
        assert self._engine([A, B], None)._op_arith_mid(bytearray(), 0, A) is None

    def test_declines_when_child_equals_a_parent(self):
        """Adjacent values: floor lands on one parent -- already executed."""
        a, b = b"\x04" * 4, b"\x05" * 4
        eng = self._engine([a, b], ScriptedRng(B1))
        assert eng._op_arith_mid(bytearray(b), 0, b) is None

    def test_produces_the_scripted_child(self):
        eng = self._engine([A, B], ScriptedRng(B2, THREE_QUARTER))
        want = _oracle(A, B, B2, THREE_QUARTER)
        assert eng._op_arith_mid(bytearray(A), 0, A) == bytearray(want)

    def test_respects_max_len(self):
        eng = self._engine([A, B], ScriptedRng(B1), max_len=5)
        got = eng._op_arith_mid(bytearray(A), 0, A)
        assert got == bytearray(_oracle(A, B, B1, MID)[:5])

    def test_related_donor_is_declined_unrelated_is_used(self):
        root, kid, other = b"\x11" * 16, b"\x22" * 16, b"\x5a" * 16
        parents = {kid: root}
        eng = self._engine([root, kid], ScriptedRng(B1), parents=parents)
        assert eng._op_arith_mid(bytearray(root), 0, root) is None
        eng = self._engine([root, kid, other], ScriptedRng(B1), parents=parents)
        got = eng._op_arith_mid(bytearray(root), 0, root)
        assert got == bytearray(_oracle(root, other, B1, MID))

    def test_available_without_lineage(self):
        # parent_key only filters ancestors out of the donor pool; with no
        # lineage the filter is a no-op, so the op is ungated (no registry gate).
        assert "arith_mid" not in _AVAILABLE
        assert _AVAILABLE.get("geo_mid") is not None  # still lineage-gated
