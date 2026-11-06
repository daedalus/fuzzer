"""Pairing-function operator (``core/mutations/pairing.py``).

Oracles enumerate the Szudzik shells and Cantor diagonals directly
(Hilbert's hotel: N x N listed as one sequence), independent of the
module's closed forms; each oracle also rejects a map that is not a
pairing, so it can fail.
"""

import pytest

import fuzzer_tool.core.mutations.pairing as P
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.scripted_rng import ScriptedRng
from .test_regression_bithacks import _MockFuzzer

SEEDS = tuple(range(30))
FILL = 0x11


def _shells(side):
    """Szudzik order of [0, side)^2: shell s is (0..s-1, s) then (s, 0..s)."""
    out = []
    for s in range(side):
        out += [(x, s) for x in range(s)]
        out += [(s, y) for y in range(s + 1)]
    return out


def _diagonals(count):
    """First *count* pairs in Cantor order: diagonal d is (d, 0) .. (0, d)."""
    out, d = [], 0
    while len(out) < count:
        out += [(d - y, y) for y in range(d + 1)]
        d += 1
    return out[:count]


def _draw(width, length, offset, endian, kind):
    fitting = [w for w in P.PAIR_WIDTHS if 2 * w <= length]
    return ScriptedRng(
        randints=[offset],
        choice_idxs=[fitting.index(width), P.ENDIANS.index(endian), P.PAIR_KINDS.index(kind)],
    )


def _run_pair(x, y, kind, width=1, endian="little"):
    data = x.to_bytes(width, endian) + y.to_bytes(width, endian) + bytes([FILL])
    out = P.pairing_fn(data, _draw(width, len(data), 0, endian, kind))
    assert out[2 * width :] == bytes([FILL])
    return out[: 2 * width]


def _fields(raw, width, endian="little"):
    return int.from_bytes(raw[:width], endian), int.from_bytes(raw[width:], endian)


class TestRegistration:
    def test_registered_in_regularity_band(self):
        assert "pairing_fn" in REGISTRY.names()
        assert REGISTRY.category_of("pairing_fn") == "regularity"
        assert "pairing_fn" in OPERATOR_CATEGORIES["regularity"]

    def test_handler_preserves_length(self):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=7)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = bytes(range(64))
        assert len(dispatch["pairing_fn"](bytearray(data), 0, data)) == len(data)


class TestOracles:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_szudzik_matches_shell_enumeration(self, bits):
        side = 1 << bits
        assert [P.unpair(z) for z in range(side * side)] == _shells(side)
        assert [P.pair(x, y) for x, y in _shells(side)] == list(range(side * side))

    def test_shell_oracle_rejects_concatenation(self):
        # Control: x * side + y is a bijection, but not Szudzik's order.
        side = 4
        concat = [(z // side, z % side) for z in range(side * side)]
        assert sorted(concat) == sorted(_shells(side))
        assert concat != _shells(side)

    def test_cantor_step_matches_diagonals(self):
        cells = _diagonals(200)
        assert [P.cantor_next(*c) for c in cells[:-1]] == cells[1:]
        assert [P.cantor_prev(*c) for c in cells[1:]] == cells[:-1]

    def test_diagonal_oracle_rejects_shell_order(self):
        # Control: Szudzik's order is not Cantor's.
        assert _diagonals(16) != _shells(4)


class TestPairingFn:
    @pytest.mark.parametrize("width", P.PAIR_WIDTHS)
    @pytest.mark.parametrize("endian", P.ENDIANS)
    def test_pack_writes_szudzik_code(self, width, endian):
        x, y = 3, 7
        raw = _run_pair(x, y, "pack", width, endian)
        assert int.from_bytes(raw, endian) == _shells(8).index((x, y))

    @pytest.mark.parametrize("width", P.PAIR_WIDTHS)
    def test_unpack_inverts_pack(self, width):
        # Falsification: pack then unpack is the identity on every field pair.
        data = (5).to_bytes(width, "big") + (2).to_bytes(width, "big")
        packed = P.pairing_fn(data, _draw(width, len(data), 0, "big", "pack"))
        back = P.pairing_fn(packed, _draw(width, len(data), 0, "big", "unpack"))
        assert packed != data and back == data

    def test_shell_next_is_next_shell_cell(self):
        cells = _shells(16)
        for i in (0, 5, 17, 100):
            assert _fields(_run_pair(*cells[i], "shell_next"), 1) == cells[i + 1]

    def test_shell_prev_is_previous_shell_cell(self):
        cells = _shells(16)
        for i in (1, 5, 17, 100):
            assert _fields(_run_pair(*cells[i], "shell_prev"), 1) == cells[i - 1]

    def test_diag_next_keeps_sum_inside_diagonal(self):
        assert _fields(_run_pair(4, 9, "diag_next"), 1) == (3, 10)

    def test_diag_next_jumps_to_next_diagonal(self):
        assert _fields(_run_pair(0, 9, "diag_next"), 1) == (10, 0)

    def test_diag_prev_jumps_to_previous_diagonal(self):
        assert _fields(_run_pair(9, 0, "diag_prev"), 1) == (0, 8)


class TestAdversarial:
    @pytest.mark.parametrize("data", (b"", b"\x11"))
    def test_too_short_is_untouched(self, data):
        assert P.pairing_fn(data, ScriptedRng()) == data

    def test_shell_next_wraps_at_umax(self):
        assert _fields(_run_pair(0xFF, 0xFF, "shell_next"), 1) == (0, 0)

    def test_shell_prev_wraps_at_zero(self):
        assert _fields(_run_pair(0, 0, "shell_prev"), 1) == (0xFF, 0xFF)

    def test_diag_next_past_umax_declines(self):
        # (0, UMAX) -> (UMAX + 1, 0) does not fit one byte.
        assert _run_pair(0, 0xFF, "diag_next") == bytes([0, 0xFF])

    def test_diag_prev_at_origin_declines(self):
        assert _run_pair(0, 0, "diag_prev") == bytes(2)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_randpool_length_preserving(self, seed):
        data = bytes((i * 37 + seed) & 0xFF for i in range(2 + seed))
        assert len(P.pairing_fn(data, RandPool(seed=seed))) == len(data)
