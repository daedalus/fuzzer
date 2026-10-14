"""Derived-compare fitters (core/derived_fit.py): GF(2)-affine, additive, byte map.

Each fake target is a model of the operand cmplog logs; expected values are
recomputed from the model, never read back from the solver.
"""

import pytest

from fuzzer_tool.core.derived_fit import (
    Gf2Affine,
    fit_gf2,
    invert_bytes,
    solve_additive,
    solve_gf2,
)

# Syndrome rows of targets/gf2_gates.c (G3 uses all 16, G2 the first 8).
SYNDROME = (
    0x7DE9FC02, 0x7EE9B8C9, 0x47682BEB, 0x0A7DC563,
    0x7ADD219D, 0xB07E1AF5, 0xB4D4D8FB, 0x3798A188,
    0x78D9D9B6, 0x00EE6FC7, 0xF9728652, 0x268251C9,
    0x7EEAACFD, 0x9C27CC92, 0x0423E37D, 0x45E9DA34,
)  # fmt: skip


def _syndrome(x: int, rows: int, mask: int) -> int:
    """Independent model: bit j is the parity of x & row j, then ^ mask."""
    s = 0
    for j in range(rows):
        s |= (bin(x & SYNDROME[j]).count("1") & 1) << j
    return s ^ mask


def _rot13(c: int) -> int:
    if 0x41 <= c <= 0x5A:
        return (c - 0x41 + 13) % 26 + 0x41
    if 0x61 <= c <= 0x7A:
        return (c - 0x61 + 13) % 26 + 0x61
    return c


# ── GF(2)-affine ─────────────────────────────────────────────────────


@pytest.mark.parametrize(("rows", "mask", "target"), [(8, 0x6E, 0x5A), (16, 0x2B91, 0xC3A5)])
def test_gf2_syndrome_is_fitted_and_solved(rows, mask, target):
    observe = lambda x: _syndrome(x, rows, mask)  # noqa: E731
    x0 = 0x01020304

    model = fit_gf2(observe, x0, 32)
    assert model is not None

    x = solve_gf2(model, target)
    assert x is not None
    assert _syndrome(x, rows, mask) == target


def test_gf2_model_predicts_unseen_points():
    observe = lambda x: _syndrome(x, 16, 0x2B91)  # noqa: E731
    model = fit_gf2(observe, 7, 32)
    for x in (0, 0xFFFFFFFF, 0xDEADBEEF, 0x12345678):
        assert model(x) == observe(x)


def test_gf2_solution_moves_few_bits():
    """The solve is a combination of columns: only bits the map needs change."""
    observe = lambda x: x & 0xFF  # noqa: E731  # identity on the low byte
    model = fit_gf2(observe, 0xAB00, 16)
    x = solve_gf2(model, 0x5A)
    assert x == 0xAB5A


def test_gf2_rejects_an_arithmetic_map():
    """Falsification: x * 3 carries between bits, so it is not GF(2)-affine."""
    assert fit_gf2(lambda x: (x * 3) & 0xFFFF, 0x1234, 16) is None


def test_gf2_rejects_a_flat_map():
    assert fit_gf2(lambda _x: 42, 5, 16) is None


def test_gf2_blind_observe_fits_nothing():
    assert fit_gf2(lambda _x: None, 5, 16) is None


def test_gf2_target_outside_the_span_is_unsolvable():
    """Adversarial: rank 1 map (bit 0 copied to bits 0 and 1) cannot reach 0b01."""
    model = fit_gf2(lambda x: 0b11 if x & 1 else 0, 0, 8)
    assert model is not None
    assert solve_gf2(model, 0b01) is None
    assert solve_gf2(model, 0b11) is not None


def test_gf2_affine_call_matches_columns():
    model = Gf2Affine(cols=(1, 2, 4), x0=0, v0=8)
    assert model(0b101) == 8 ^ 1 ^ 4


# ── additive ─────────────────────────────────────────────────────────


def test_additive_sum_reaches_target():
    xs = list(range(1, 17))
    v0 = sum(xs)
    out = solve_additive(xs, [1] * 16, v0, 0x0A57, 16)
    assert out is not None
    assert sum(out) & 0xFFFF == 0x0A57
    assert all(0 <= b <= 0xFF for b in out)


def test_additive_mixed_weights():
    xs = [10, 200, 30, 40]
    weights = [3, -1, 2, 1]
    v0 = sum(w * x for w, x in zip(weights, xs, strict=True)) & 0xFFFF
    out = solve_additive(xs, weights, v0, 500, 16)
    assert out is not None
    assert sum(w * x for w, x in zip(weights, out, strict=True)) & 0xFFFF == 500


def test_additive_wraps_through_the_modulus():
    """v0 just below 2^16, target just above 0: a short step forward, not 65k back."""
    xs = [0xF0, 0x00]
    v0 = 0xFFF0
    out = solve_additive(xs, [1, 1], v0, 0x0010, 16)
    assert out is not None
    assert (v0 + sum(out) - sum(xs)) & 0xFFFF == 0x0010


def test_additive_out_of_range_is_unsolvable():
    """Adversarial: two bytes cannot add 600."""
    assert solve_additive([0, 0], [1, 1], 0, 600, 16) is None


def test_additive_zero_weights_are_unsolvable():
    """Falsification: weights that move nothing cannot reach a new value."""
    assert solve_additive([5, 6], [0, 0], 11, 12, 16) is None


# ── byte map ─────────────────────────────────────────────────────────


def _rot13_target(base: bytearray, off: int, n: int):
    def observe(assign):
        buf = bytearray(base)
        for i, v in assign.items():
            buf[i] = v
        return bytes(_rot13(c) for c in buf[off : off + n])

    return observe


def test_bytemap_inverts_rot13():
    want = b"HelloWorld!!"
    off = 33
    base = bytearray(range(64))
    base[off : off + 12] = b"#k26;&#x23;#"
    observe = _rot13_target(base, off, 12)
    links = {j: off + j for j in range(12)}
    xs = {off + j: base[off + j] for j in range(12)}

    assign = invert_bytes(observe, links, xs, observe({}), want, max_execs=300)

    assert assign is not None
    buf = bytearray(base)
    for i, v in assign.items():
        buf[i] = v
    assert bytes(_rot13(c) for c in buf[off : off + 12]) == want


def test_bytemap_costs_one_exec_per_round_not_per_byte():
    """Positions are probed in parallel: the identity map solves in one exec."""
    calls = []
    want = b"ABCDEFGH"

    def observe(assign):
        calls.append(assign)
        return bytes(assign.get(j, 0) for j in range(8))

    links = {j: j for j in range(8)}
    assign = invert_bytes(observe, links, dict.fromkeys(range(8), 0), bytes(8), want, 10)
    assert assign == {j: want[j] for j in range(8)}
    assert len(calls) == 1


def test_bytemap_unreachable_byte_fails_closed():
    """Adversarial: a position whose map never yields the target returns None."""
    want = b"AZ"

    def observe(assign):
        return bytes([assign.get(0, 0), 0x00])  # position 1 is stuck at 0

    assign = invert_bytes(observe, {0: 0, 1: 1}, {0: 0, 1: 0}, bytes(2), want, 300)
    assert assign is None


def test_bytemap_respects_budget():
    calls = []

    def observe(assign):
        calls.append(1)
        return b"\x00"

    assert invert_bytes(observe, {0: 0}, {0: 0}, b"\x00", b"\x01", max_execs=5) is None
    assert len(calls) <= 5


def test_bytemap_blind_probe_stops():
    assert invert_bytes(lambda _a: None, {0: 0}, {0: 0}, b"\x00", b"\x01", 50) is None
