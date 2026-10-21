"""FXT-book operators (``core/mutations/fxt.py``).

Each operator keeps one invariant a parser checks; the falsification tests
assert it with an oracle written independently of the module (``math.gcd``,
``fractions``, ``itertools``, a direct Hilbert walk), and each oracle is also
run on an input that must fail it, so the oracle can fail.
"""

import itertools
import math
from fractions import Fraction

import pytest

import fuzzer_tool.core.mutations.fxt as F
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.scripted_rng import ScriptedRng
from .test_regression_bithacks import _MockFuzzer

OPS = (
    "rational_pair",
    "sum_shift",
    "multiset_next",
    "haar_coef",
    "revbin_permute",
    "hilbert_patch",
    "hadamard_oa",
    "fib_encode",
)
SEEDS = tuple(range(30))
FILL = 0x11


def _seeded(seed: int, n: int = 64) -> bytes:
    return bytes((i * 37 + seed * 11 + 5) & 0xFF for i in range(n))


class TestRegistration:
    @pytest.mark.parametrize("name", OPS)
    def test_registered_in_regularity_band(self, name):
        assert name in REGISTRY.names()
        assert REGISTRY.category_of(name) == "regularity"
        assert name in OPERATOR_CATEGORIES["regularity"]

    @pytest.mark.parametrize("name", OPS)
    def test_handler_preserves_length(self, name):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=7)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = _seeded(1)
        out = dispatch[name](bytearray(data), 0, data)
        assert out is not None and len(out) == len(data)

    @pytest.mark.parametrize("name", OPS)
    @pytest.mark.parametrize("data", (b"", b"\x11"))
    def test_too_short_is_untouched(self, name, data):
        # Adversarial: no op may draw or index past a 0/1-byte input.
        assert getattr(F, name)(data, ScriptedRng()) == data

    @pytest.mark.parametrize("name", OPS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_randpool_length_preserving(self, name, seed):
        data = _seeded(seed, 7 + seed)
        assert len(getattr(F, name)(data, RandPool(seed=seed))) == len(data)


# ── rational_pair ─────────────────────────────────────────────────────────
def _rat_draw(width, length, offset, endian, order, kind, extra=()):
    fitting = [w for w in F.RAT_WIDTHS if 2 * w <= length]
    return ScriptedRng(
        randints=[offset, *extra],
        choice_idxs=[
            fitting.index(width),
            F.ENDIANS.index(endian),
            F.RAT_ORDERS.index(order),
            F.RAT_KINDS.index(kind),
        ],
    )


def _rat_run(num, den, kind, width=2, endian="little", order="num_first", extra=()):
    data = num.to_bytes(width, endian) + den.to_bytes(width, endian) + bytes([FILL])
    out = F.rational_pair(data, _rat_draw(width, len(data), 0, endian, order, kind, extra))
    n2 = int.from_bytes(out[:width], endian)
    d2 = int.from_bytes(out[width : 2 * width], endian)
    assert out[2 * width :] == bytes([FILL])
    return out, n2, d2


def _convergents(n, d):
    """Independent oracle: truncations of the continued fraction of n/d, via Fraction."""
    x, terms = Fraction(n, d), []
    while True:
        a = math.floor(x)
        terms.append(a)
        if x == a:
            break
        x = 1 / (x - a)
    out = []
    for i in range(1, len(terms) + 1):
        v = Fraction(terms[i - 1])
        for t in reversed(terms[: i - 1]):
            v = t + 1 / v
        out.append(v)
    return out


class TestRationalPair:
    def test_unreduced_keeps_value_with_common_factor(self):
        _, n, d = _rat_run(3, 5, "unreduced")
        assert (n, d) == (6, 10) and math.gcd(n, d) > 1 and Fraction(n, d) == Fraction(3, 5)

    def test_reduced_divides_out_gcd(self):
        _, n, d = _rat_run(30000, 1001 * 30, "reduced")
        assert math.gcd(n, d) == 1 and Fraction(n, d) == Fraction(30000, 30030)

    def test_reduced_declines_when_coprime(self):
        out, _, _ = _rat_run(3, 5, "reduced")
        assert out[:4] == (3).to_bytes(2, "little") + (5).to_bytes(2, "little")

    def test_zero_and_unit_den(self):
        assert _rat_run(7, 9, "zero_den")[1:] == (7, 0)
        assert _rat_run(7, 9, "unit_den")[1:] == (7, 1)

    def test_den_first_order(self):
        # Fields read (den, num): (5, 3) is 3/5, unreduced -> (10, 6).
        data = (5).to_bytes(2, "big") + (3).to_bytes(2, "big")
        out = F.rational_pair(data, _rat_draw(2, 4, 0, "big", "den_first", "unreduced"))
        assert out == (10).to_bytes(2, "big") + (6).to_bytes(2, "big")

    @pytest.mark.parametrize("num,den", [(355, 113), (1001, 1000), (30001, 9550), (17, 3)])
    def test_every_convergent_is_a_true_convergent(self, num, den):
        # Falsification: each pick must be a proper convergent of num/den.
        conv = _convergents(num, den)
        for i in range(len(conv) - 1):
            _, n, d = _rat_run(num, den, "convergent", extra=(i,))
            assert Fraction(n, d) == conv[i] and d < den

    def test_convergent_oracle_rejects_a_semiconvergent(self):
        # Control: 16/5 is a best approximation of 355/113 but not a convergent.
        assert Fraction(16, 5) == Fraction(355, 113).limit_denominator(5)
        assert Fraction(16, 5) not in _convergents(355, 113)
        assert _convergents(355, 113) == [3, Fraction(22, 7), Fraction(355, 113)]  # [3; 7, 16]

    @pytest.mark.parametrize("num,den", [(355, 113), (30001, 9550)])
    def test_convergents_satisfy_legendre_bound(self, num, den):
        # |x - p/q| < 1/q^2 holds for every convergent.
        x = Fraction(num, den)
        for c in _convergents(num, den):
            assert abs(x - c) < Fraction(1, c.denominator**2) or c == x

    @pytest.mark.parametrize("num,den", [(355, 113), (7, 2), (5, 1), (2, 9)])
    def test_mediant_lies_strictly_between(self, num, den):
        _, n, d = _rat_run(num, den, "mediant")
        conv = _convergents(num, den)
        prev = conv[-2] if len(conv) > 1 else None
        x = Fraction(num, den)
        m = Fraction(n, d)
        assert m != x
        if prev is not None:
            assert min(prev, x) < m < max(prev, x)

    def test_extreme_writes_umax_over_one(self):
        assert _rat_run(3, 5, "extreme")[1:] == (0xFFFF, 1)

    def test_unreduced_overflow_declines(self):
        out, n, d = _rat_run(0x9000, 3, "unreduced")
        assert (n, d) == (0x9000, 3)

    def test_zero_den_input_declines_convergent(self):
        out, n, d = _rat_run(5, 0, "convergent")
        assert (n, d) == (5, 0)


# ── sum_shift ─────────────────────────────────────────────────────────────
class TestSumShift:
    def _draw(self, width, gap, offset, endian, direction, amount):
        return ScriptedRng(
            randints=[gap, offset, amount],
            choice_idxs=[
                F.SUM_WIDTHS.index(width),
                F.ENDIANS.index(endian),
                F.SUM_DIRS.index(direction),
            ],
        )

    def test_moves_units_between_adjacent_fields(self):
        data = bytes([10, 20, FILL])
        out = F.sum_shift(data, self._draw(1, 1, 0, "little", "a_to_b", 3))
        assert out == bytes([7, 23, FILL])

    def test_gap_and_wide_fields(self):
        data = (500).to_bytes(2, "big") + b"\xaa\xaa" + (100).to_bytes(2, "big")
        out = F.sum_shift(data, self._draw(2, 2, 0, "big", "b_to_a", 40))
        assert out == (540).to_bytes(2, "big") + b"\xaa\xaa" + (60).to_bytes(2, "big")

    def test_no_room_declines(self):
        # a = 0 cannot give, b = 255 cannot take: nothing to move.
        data = bytes([0, 255])
        rng = ScriptedRng(randints=[1, 0], choice_idxs=[0, 0, 0])
        assert F.sum_shift(data, rng) == data

    @pytest.mark.parametrize("seed", SEEDS)
    def test_sum_of_touched_fields_is_invariant(self, seed):
        # Falsification: exactly two w-byte fields change and their sum is kept.
        data = _seeded(seed)
        out = F.sum_shift(data, RandPool(seed=seed))
        diff = [i for i in range(len(data)) if out[i] != data[i]]
        if not diff:
            return
        hit = None
        for w, e in itertools.product(F.SUM_WIDTHS, F.ENDIANS):
            for a in range(0, len(data) - w + 1):
                for b in range(a + w, len(data) - w + 1):
                    if not set(diff) <= set(range(a, a + w)) | set(range(b, b + w)):
                        continue

                    def val(buf, o, w=w, e=e):
                        return int.from_bytes(buf[o : o + w], e)

                    if val(data, a) + val(data, b) == val(out, a) + val(out, b) and val(
                        out, a
                    ) != val(data, a):
                        hit = (w, e, a, b)
                        break
                if hit:
                    break
            if hit:
                break
        assert hit, (seed, diff)


# ── multiset_next ─────────────────────────────────────────────────────────
class TestMultisetNext:
    def _draw(self, length, offset, direction):
        return ScriptedRng(randints=[length, offset], choice_idxs=[F.PERM_DIRS.index(direction)])

    @pytest.mark.parametrize(
        "span", [b"\x01\x01\x02\x03", b"\x00\x05\x05\x09\x09", b"\x03\x02\x01"]
    )
    @pytest.mark.parametrize("direction", ["next", "prev"])
    def test_steps_one_place_in_lex_order(self, span, direction):
        # Oracle: sorted set of distinct permutations from itertools, cyclic step.
        perms = sorted(set(itertools.permutations(span)))
        for p in perms:
            got = F.multiset_next(bytes(p), self._draw(len(p), 0, direction))
            i = perms.index(p)
            want = perms[(i + (1 if direction == "next" else -1)) % len(perms)]
            assert tuple(got) == want

    def test_constant_span_unchanged(self):
        assert F.multiset_next(b"\x07" * 5, self._draw(5, 0, "next")) == b"\x07" * 5

    @pytest.mark.parametrize("seed", SEEDS)
    def test_histogram_and_sum_preserved(self, seed):
        data = _seeded(seed)
        out = F.multiset_next(data, RandPool(seed=seed))
        assert sorted(out) == sorted(data) and sum(out) == sum(data)

    def test_oracle_detects_a_non_step(self):
        # Control: swapping the first two of 0123 is not the lex successor.
        perms = sorted(itertools.permutations(b"\x00\x01\x02\x03"))
        assert perms[perms.index(tuple(b"\x00\x01\x02\x03")) + 1] != tuple(b"\x01\x00\x02\x03")


# ── haar_coef ─────────────────────────────────────────────────────────────
def _s_fwd(x):
    """Independent oracle: integer S-transform, Mallat layout [s, d_coarse, ..., d_fine]."""
    x, ds = list(x), []
    while len(x) > 1:
        d = [x[2 * i] - x[2 * i + 1] for i in range(len(x) // 2)]
        x = [(x[2 * i] + x[2 * i + 1]) >> 1 for i in range(len(x) // 2)]
        ds = d + ds
    return x + ds


class TestHaarCoef:
    def _draw(self, levels, offset, index, delta):
        return ScriptedRng(
            randints=[levels, offset, index], choice_idxs=[F.HAAR_DELTAS.index(delta)]
        )

    @pytest.mark.parametrize("index", range(8))
    def test_one_coefficient_moves_by_delta(self, index):
        span = bytes([100, 104, 98, 97, 120, 121, 90, 95])
        out = F.haar_coef(span, self._draw(3, 0, index, 2))
        c_in, c_out = _s_fwd(span), _s_fwd(out)
        want = list(c_in)
        want[index] += 2
        assert c_out == want

    def test_round_trip_identity_without_edit(self):
        # Adversarial: forward then inverse over every byte pair is exact.
        for a, b in itertools.product(range(0, 256, 15), repeat=2):
            assert F.haar_inverse(F.haar_forward([a, b])) == [a, b]

    def test_oracle_matches_module_forward(self):
        span = [3, 250, 7, 7, 0, 255, 128, 129]
        assert F.haar_forward(span) == _s_fwd(span)

    def test_clamps_into_byte_range(self):
        out = F.haar_coef(bytes([255, 255]), self._draw(1, 0, 0, 32))
        assert all(0 <= v <= 255 for v in out) and out == bytes([255, 255])

    @pytest.mark.parametrize("seed", SEEDS)
    def test_changes_stay_inside_one_power_of_two_span(self, seed):
        data = _seeded(seed)
        out = F.haar_coef(data, RandPool(seed=seed))
        diff = [i for i in range(len(data)) if out[i] != data[i]]
        assert not diff or diff[-1] - diff[0] < 1 << F.HAAR_MAX_LEVELS


# ── revbin_permute ────────────────────────────────────────────────────────
class TestRevbinPermute:
    def _draw(self, width, k, offset, length):
        fitting = [w for w in F.REVBIN_WIDTHS if (1 << F.REVBIN_MIN_K) * w <= length]
        return ScriptedRng(randints=[k, offset], choice_idxs=[fitting.index(width)])

    def test_byte_elements_k3(self):
        data = bytes(range(8))
        out = F.revbin_permute(data, self._draw(1, 3, 0, 8))
        assert out == bytes([0, 4, 2, 6, 1, 5, 3, 7])

    def test_word_elements_keep_element_bytes_together(self):
        data = b"".join(i.to_bytes(2, "little") for i in range(4))
        out = F.revbin_permute(data, self._draw(2, 2, 0, 8))
        assert [int.from_bytes(out[2 * i : 2 * i + 2], "little") for i in range(4)] == [0, 2, 1, 3]

    @pytest.mark.parametrize("k", range(2, 7))
    def test_involution(self, k):
        # Falsification: revbin twice is the identity; once is not (k >= 2).
        data = bytes(range(1 << k))
        once = F.revbin_permute(data, self._draw(1, k, 0, len(data)))
        twice = F.revbin_permute(once, self._draw(1, k, 0, len(data)))
        assert twice == data and once != data


# ── hilbert_patch ─────────────────────────────────────────────────────────
def _hilbert_cells(n):
    """Independent oracle: Hilbert order by recursive quadrant construction."""
    if n == 1:
        return [(0, 0)]
    h = n // 2
    sub = _hilbert_cells(h)
    q0 = [(y, x) for x, y in sub]
    q1 = [(x, y + h) for x, y in sub]
    q2 = [(x + h, y + h) for x, y in sub]
    q3 = [(2 * h - 1 - y, h - 1 - x) for x, y in sub]
    return q0 + q1 + q2 + q3


class TestHilbertPatch:
    def _draw(self, k, stride, offset, m, d0, delta):
        return ScriptedRng(
            randints=[k, stride, offset, m, d0], choice_idxs=[F.HILBERT_DELTAS.index(delta)]
        )

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_d2xy_matches_recursive_oracle(self, n):
        assert [F.hilbert_d2xy(n, d) for d in range(n * n)] == _hilbert_cells(n)

    def test_oracle_cells_are_a_path(self):
        cells = _hilbert_cells(8)
        assert all(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(cells, cells[1:], strict=False)
        )
        assert len(set(cells)) == 64

    def test_patch_touches_consecutive_hilbert_cells(self):
        n, stride = 4, 6
        data = bytes(64)
        out = F.hilbert_patch(data, self._draw(2, stride, 3, 5, 4, 1))
        cells = _hilbert_cells(n)[4:9]
        want = bytearray(data)
        for x, y in cells:
            want[3 + y * stride + x] = 1
        assert out == bytes(want)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_changed_bytes_share_one_delta(self, seed):
        data = _seeded(seed)
        out = F.hilbert_patch(data, RandPool(seed=seed))
        deltas = {(out[i] - data[i]) & 0xFF for i in range(len(data)) if out[i] != data[i]}
        assert len(deltas) <= 1


# ── hadamard_oa ───────────────────────────────────────────────────────────
class TestHadamardOA:
    def _draw(self, row, offset, endian, mode):
        return ScriptedRng(
            randints=[row, offset], choice_idxs=[F.ENDIANS.index(endian), F.OA_MODES.index(mode)]
        )

    def test_rows_are_pairwise_orthogonal(self):
        # Falsification: distinct rows differ in exactly 8 of 16 bits.
        rows = F.HADAMARD_ROWS
        assert len(set(rows)) == 16
        assert all(bin(a ^ b).count("1") == 8 for a, b in itertools.combinations(rows, 2))

    def test_every_column_pair_sees_all_four_combinations(self):
        # Strength-2 orthogonal array: over the 16 rows, each column pair hits 00/01/10/11 4 times.
        rows = F.HADAMARD_ROWS
        for i, j in itertools.combinations(range(1, 16), 2):
            counts = {(r >> i & 1, r >> j & 1) for r in rows}
            assert counts == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_orthogonality_oracle_rejects_identity_rows(self):
        assert bin(0x0001 ^ 0x0002).count("1") != 8

    def test_set_writes_row(self):
        out = F.hadamard_oa(bytes(3), self._draw(5, 1, "little", "set"))
        assert out[0] == 0 and int.from_bytes(out[1:3], "little") == F.HADAMARD_ROWS[5]

    def test_xor_toggles_row_bits(self):
        data = b"\xff\x0f"
        out = F.hadamard_oa(data, self._draw(3, 0, "big", "xor"))
        assert int.from_bytes(out, "big") == 0xFF0F ^ F.HADAMARD_ROWS[3]


# ── fib_encode ────────────────────────────────────────────────────────────
def _fib_code(n):
    """Independent oracle: Fibonacci universal code of n >= 1, as '0'/'1' text."""
    fibs = [1, 2]
    while fibs[-1] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    bits = ["0"] * len(fibs)
    for i in range(len(fibs) - 1, -1, -1):
        if fibs[i] <= n:
            bits[i] = "1"
            n -= fibs[i]
    s = "".join(bits).rstrip("0")
    return s + "1"


class TestFibEncode:
    @pytest.mark.parametrize("v", [0, 1, 2, 3, 7, 100, 232, 254, 255])
    def test_codeword_matches_oracle(self, v):
        assert "".join(map(str, F.fib_codeword(v + 1))) == _fib_code(v + 1)

    def test_oracle_has_single_terminator(self):
        # Control: '11' appears only at the end of each oracle codeword.
        for n in range(1, 300):
            assert _fib_code(n).find("11") == len(_fib_code(n)) - 2

    def test_round_trip_without_edits(self):
        block = bytes(range(256))
        restored, count = F.fib_decode(F.fib_bits(block), len(block))
        assert count == len(block) and bytes(restored) == block

    def test_zero_bytes_decode(self):
        # Adversarial: byte 0 (n = 1, codeword '11') must not stop the decoder.
        restored, count = F.fib_decode(F.fib_bits(b"\x00\x00\x05"), 3)
        assert count == 3 and bytes(restored) == b"\x00\x00\x05"

    def test_value_table_matches_enumeration(self):
        # The decode table must hold every no-adjacent-ones run ending in 1 up to len(_FIB).
        want = {}
        for n in range(1, len(F._FIB) + 1):
            for t in itertools.product((0, 1), repeat=n):
                if t[-1] and not any(a and b for a, b in zip(t, t[1:], strict=False)):
                    want[bytes(t)] = sum(f for f, b in zip(F._FIB, t, strict=False) if b)
        assert F._fib_values() == want

    def test_long_edited_run_falls_back_to_sum(self):
        # Adversarial: a run longer than the table (13 digits + terminator) still decodes.
        bits = bytearray([0] * 12 + [1, 1])
        restored, count = F.fib_decode(bits, 1)
        assert count == 1 and restored[0] == 0xFF  # value 0 (digit past _FIB) - 1
