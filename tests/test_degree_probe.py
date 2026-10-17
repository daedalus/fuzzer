"""Tests for core/degree_probe.py -- GF(2) algebraic-degree probes."""

import itertools

import pytest

from fuzzer_tool.core.degree_probe import (
    Degree,
    blr_defect,
    classify,
    classify_exact,
    second_derivative,
    solve_affine,
    solve_low,
)
from fuzzer_tool.core.rand_pool import RandPool

N = 8
SEED = 1234


def _parity(x: int, mask: int) -> int:
    return bin(x & mask).count("1") & 1


def _affine(x: int) -> int:
    return _parity(x, 0b10110010) ^ 1


def _quadratic(x: int) -> int:
    return ((x >> 0) & (x >> 1) & 1) ^ ((x >> 4) & (x >> 6) & 1) ^ (x >> 7) & 1


def _cubic(x: int) -> int:
    return (x >> 0) & (x >> 1) & (x >> 2) & 1


def _quartic(x: int) -> int:
    return (x >> 0) & (x >> 1) & (x >> 2) & (x >> 3) & 1


def _point(x: int) -> int:
    return int(x == 0xA5)


def _const(_x: int) -> int:
    return 1


def _rng() -> RandPool:
    return RandPool(seed=SEED)


class TestPrimitives:
    def test_blr_zero_on_affine_for_every_triple(self):
        # Exhaustive over a 4-bit domain: affine => defect is always 0.
        for x, y, z in itertools.product(range(16), repeat=3):
            assert blr_defect(lambda v: _parity(v, 0b1011) ^ 1, x, y, z) == 0

    def test_blr_nonzero_exists_for_quadratic(self):
        f = lambda v: (v & (v >> 1)) & 1  # noqa: E731
        defects = {blr_defect(f, x, y, z) for x, y, z in itertools.product(range(16), repeat=3)}
        assert 1 in defects

    def test_second_derivative_constant_for_quadratic(self):
        f = lambda v: (v & (v >> 1)) & 1  # noqa: E731
        for a, b in itertools.product(range(1, 16), repeat=2):
            vals = {second_derivative(f, x, a, b) for x in range(16)}
            assert len(vals) == 1

    def test_second_derivative_varies_for_cubic(self):
        f = lambda v: v & (v >> 1) & (v >> 2) & 1  # noqa: E731
        varies = any(
            len({second_derivative(f, x, a, b) for x in range(16)}) > 1
            for a, b in itertools.product(range(1, 16), repeat=2)
        )
        assert varies


class TestClassify:
    def test_known_degrees(self):
        assert classify(_const, N, _rng()) is Degree.CONSTANT
        assert classify(_affine, N, _rng()) is Degree.AFFINE
        assert classify(_quadratic, N, _rng()) is Degree.QUADRATIC

    # Falsification: a degree-3 function must not be called affine/quadratic.
    def test_cubic_is_high(self):
        assert classify(_cubic, N, _rng()) is Degree.HIGH

    # Falsification: a dense degree-4 function must be neither affine nor quadratic.
    def test_quartic_is_high(self):
        assert classify(_quartic, N, _rng()) is Degree.HIGH

    def test_exec_cost_is_bounded(self):
        calls = []

        def f(x):
            calls.append(x)
            return _affine(x)

        classify(f, N, _rng(), trials=10)
        assert len(calls) <= 100

    # Regression: an affine bit whose sampled values all agree is not constant.
    def test_affine_with_even_weight_is_never_constant(self):
        f = lambda x: x & 1 ^ (x >> 1) & 1  # noqa: E731
        for seed in range(50):
            assert classify(f, N, RandPool(seed=seed)) is Degree.AFFINE


class TestAdversarial:
    # Adversarial: nondeterministic edge invalidates every identity.
    def test_noisy_callable_is_noisy(self):
        state = {"n": 0}

        def f(x):
            state["n"] += 1
            return (_affine(x) ^ state["n"]) & 1

        assert classify(f, N, _rng()) is Degree.NOISY

    def test_one_bit_domain_is_affine_or_constant(self):
        assert classify(lambda x: x & 1, 1, _rng()) is Degree.AFFINE
        assert classify(_const, 1, _rng()) is Degree.CONSTANT

    def test_output_is_masked_to_one_bit(self):
        assert classify(lambda x: _affine(x) | 0b110, N, _rng()) is Degree.AFFINE

    def test_nonpositive_width_rejected(self):
        with pytest.raises(ValueError):
            classify(_affine, 0, _rng())


class TestSolveAffine:
    @pytest.mark.parametrize("want", [0, 1])
    def test_solution_satisfies_target(self, want):
        x = solve_affine(_affine, N, want)
        assert x is not None
        assert _affine(x) == want

    def test_zero_input_when_constant_already_matches(self):
        assert solve_affine(_affine, N, _affine(0)) == 0

    def test_unreachable_target_returns_none(self):
        assert solve_affine(_const, N, 0) is None

    # Falsification: a misclassified non-affine function must not return a
    # value that fails the final check.
    def test_non_affine_never_returns_wrong_value(self):
        for want in (0, 1):
            x = solve_affine(_cubic, N, want)
            assert x is None or _cubic(x) == want

    # A rare-event gate looks constant to random probing (documented limit);
    # solve_affine must still never return a wrong value for it.
    def test_point_function_never_yields_wrong_value(self):
        for want in (0, 1):
            x = solve_affine(_point, N, want)
            assert x is None or _point(x) == want

    def test_exec_cost_is_linear(self):
        calls = []

        def f(x):
            calls.append(x)
            return _affine(x)

        solve_affine(f, N, 1)
        assert len(calls) <= N + 2


class TestClassifyExact:
    @pytest.mark.parametrize(
        ("f", "deg"),
        [
            (_const, Degree.CONSTANT),
            (_affine, Degree.AFFINE),
            (_quadratic, Degree.QUADRATIC),
            (_cubic, Degree.HIGH),
        ],
    )
    def test_known_degrees(self, f, deg):
        assert classify_exact(f, N) is deg

    # Regression: the documented blind spot -- [x == K] looks constant to
    # random probing; the full ANF sees its degree-n monomial.
    def test_regression_equality_gate_is_high_not_constant(self):
        assert classify_exact(_point, N) is Degree.HIGH

    def test_wide_field_refused(self):
        with pytest.raises(ValueError):
            classify_exact(_affine, 64)


class TestSolveLow:
    @pytest.mark.parametrize("want", [0, 1])
    def test_quadratic_solution_satisfies_target(self, want):
        x = solve_low(_quadratic, N, want)
        assert x is not None
        assert _quadratic(x) == want

    def test_pure_quadratic_with_no_linear_term_is_solved(self):
        """Falsification of solve_affine's reach: no unit vector flips x0&x1."""
        f = lambda x: x & (x >> 1) & 1  # noqa: E731
        assert solve_affine(f, N, 1) is None
        x = solve_low(f, N, 1)
        assert x is not None
        assert f(x) == 1

    def test_constant_is_unsolvable(self):
        assert solve_low(_const, N, 0) is None

    def test_exec_cost_is_the_layer(self):
        calls = []

        def f(x):
            calls.append(x)
            return 0

        solve_low(f, N, 1)
        assert len(calls) == 1 + N + N * (N - 1) // 2

    # Adversarial: a degree-n gate only flips at its key; the scan of the
    # weight <= 2 layer must not return a wrong value.
    def test_point_function_never_yields_wrong_value(self):
        for want in (0, 1):
            x = solve_low(_point, N, want)
            assert x is None or _point(x) == want
