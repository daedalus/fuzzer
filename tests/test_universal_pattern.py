"""core/bessel.py and core/universal_pattern.py.

Bessel values are checked against identities (no scipy, Hard Rule 51):
Jacobi-Anger normalisation, the three-term recurrence, parity, and the
tabulated J0(1), J1(1). The identity checks are sharp (1e-12) because the
trapezoid rule is geometrically convergent.
"""

import math

import numpy as np
import pytest

from fuzzer_tool.core.bessel import jn
from fuzzer_tool.core.universal_pattern import (
    common_symmetry,
    mode_coherence,
    universal_pattern,
)

X = np.linspace(-40.0, 40.0, 401)


def test_tabulated_values():
    assert jn(0, 1.0) == pytest.approx(0.7651976865579666, abs=1e-14)
    assert jn(1, 1.0) == pytest.approx(0.4400505857449335, abs=1e-14)
    assert jn(3, 0.0) == 0.0 and jn(0, 0.0) == 1.0


def test_jacobi_anger_normalisation():
    total = jn(0, X) ** 2 + 2.0 * sum(jn(n, X) ** 2 for n in range(1, 80))
    assert np.allclose(total, 1.0, atol=1e-12)


def test_three_term_recurrence():
    x = X[np.abs(X) > 1e-3]
    for n in (1, 5, 12):
        lhs = jn(n - 1, x) + jn(n + 1, x)
        assert np.allclose(lhs, 2.0 * n / x * jn(n, x), atol=1e-11)


def test_parity_and_negative_order():
    for n in (1, 2, 5):
        assert np.allclose(jn(n, -X), (-1) ** n * jn(n, X), atol=1e-13)
        assert np.allclose(jn(-n, X), (-1) ** n * jn(n, X), atol=1e-13)


def test_nonfinite_and_bad_order():
    out = jn(2, [1.0, np.nan, np.inf])
    assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    with pytest.raises(ValueError):
        jn(2.5, 1.0)


def _grid():
    x = np.linspace(-5, 5, 33)
    xx, yy = np.meshgrid(x, x)
    return np.hypot(xx, yy), np.arctan2(yy, xx)


def test_pattern_does_not_mutate_inputs_and_accepts_scalars_and_ints():
    r, th = _grid()
    r0, th0 = r.copy(), th.copy()
    universal_pattern(r, th, recursion_depth=3, nonlinear_lambda=0.3)
    assert np.array_equal(r, r0) and np.array_equal(th, th0)
    assert universal_pattern(1.0, 0.3).shape == ()
    assert universal_pattern(np.arange(5), np.arange(5) * 0.1).shape == (5,)


def test_pairwise_closed_form_matches_double_loop():
    r, th = _grid()
    kw = dict(modes=(2, 3, 5), amplitudes=(0.5, 0.3, 0.2), nonlinear_lambda=0.7)
    got = universal_pattern(r, th, **kw)
    fields = [
        a * jn(n, 3.0 * r) * np.cos(n * th + 0.4 * n * 0.0)
        for a, n in zip(kw["amplitudes"], kw["modes"])
    ]
    ref = sum(fields) + 0.7 * sum(
        fields[i] * fields[j] for i in range(3) for j in range(i + 1, 3)
    )
    assert np.allclose(got, ref, atol=1e-12)


def test_rotational_symmetry_is_the_gcd():
    assert common_symmetry((6, 12)) == 6
    assert common_symmetry((5, 6, 12)) == 1
    r = np.full(1, 2.0)

    def f(modes, a):
        return universal_pattern(r, np.array([a]), modes=modes)[0]

    sixfold = 2 * math.pi / 6
    assert f((6, 12), 0.4) == pytest.approx(f((6, 12), 0.4 + sixfold), abs=1e-12)
    assert abs(f((5, 6, 12), 0.4) - f((5, 6, 12), 0.4 + sixfold)) > 1e-3


@pytest.mark.parametrize(
    "kw",
    [
        dict(modes=()),
        dict(modes=(5.5,)),
        dict(modes=(-5,)),
        dict(modes=(5, 6), amplitudes=(1.0,)),
        dict(recursion_depth=-1),
    ],
)
def test_validation_rejects_bad_input(kw):
    r, th = _grid()
    with pytest.raises(ValueError):
        universal_pattern(r, th, **kw)


def test_mode_coherence_extremes():
    # At theta0 = 0 with no offsets every mode phase is 0: fully locked.
    r, _ = mode_coherence(0.0, (5, 6, 12))
    assert r == pytest.approx(1.0)
    # Modes 1,2,3 at theta0 = 2*pi/3 spread to 2pi/3, 4pi/3, 2pi: not locked.
    r, _ = mode_coherence(2 * math.pi / 3, (1, 2, 3))
    assert r < 0.5
