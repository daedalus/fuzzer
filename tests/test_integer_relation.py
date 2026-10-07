"""Tests for core/integer_relation.py: HJLS, PSOS, PSLQ."""

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from fuzzer_tool.core.integer_relation import _IMPL, Algo, find_relation, hjls, pslq, psos

ALGOS = list(Algo)
DIGITS = 60


def _dec(fn):
    """Evaluate *fn* under DIGITS precision so inputs carry enough digits."""
    with localcontext() as ctx:
        ctx.prec = DIGITS
        return fn()


def _alpha_powers():
    """1, a, a^2, a^3, a^4 for a = sqrt2 + sqrt3; minimal poly a^4 - 10a^2 + 1."""
    a = _dec(lambda: Decimal(2).sqrt() + Decimal(3).sqrt())
    return _dec(lambda: [a**k for k in range(5)])


def _primitive(v):
    """Scale *v* to the primitive integer vector with a positive first nonzero entry."""
    from math import gcd

    g = 0
    for c in v:
        g = gcd(g, c)
    sign = 1 if next(c for c in v if c) > 0 else -1
    return [sign * c // g for c in v]


def _residual(m, x):
    return _dec(lambda: abs(sum(Decimal(c) * xi for c, xi in zip(m, x, strict=True))))


@pytest.mark.parametrize("algo", ALGOS)
class TestKnownRelations:
    def test_minimal_polynomial(self, algo):
        m = find_relation(_alpha_powers(), algo, digits=DIGITS)
        assert m == _primitive([1, 0, -10, 0, 1])

    def test_logarithms(self, algo):
        x = _dec(lambda: [Decimal(2).ln(), Decimal(3).ln(), Decimal(6).ln()])
        m = find_relation(x, algo, digits=DIGITS)
        assert m == _primitive([1, 1, -1])
        assert _residual(m, x) < Decimal(10) ** -40

    def test_exact_integers(self, algo):
        assert find_relation([3, 5], algo) == _primitive([5, -3])

    def test_fractions(self, algo):
        x = [Fraction(1, 3), Fraction(1, 7), Fraction(10, 21)]
        assert find_relation(x, algo) == _primitive([1, 1, -1])

    def test_falsification_independent_returns_none(self, algo):
        """1, sqrt2, sqrt3 are Q-independent: no relation within maxcoeff."""
        x = _dec(lambda: [Decimal(1), Decimal(2).sqrt(), Decimal(3).sqrt()])
        assert find_relation(x, algo, digits=DIGITS, maxcoeff=1000) is None

    def test_adversarial_zero_entry(self, algo):
        assert find_relation([Decimal("1.5"), 0, Decimal(2).sqrt()], algo) == [0, 1, 0]

    def test_adversarial_scale_and_sign(self, algo):
        """Scaling by -1e30 must not change the primitive relation."""
        base = _dec(lambda: [Decimal(2).ln(), Decimal(3).ln(), Decimal(6).ln()])
        x = _dec(lambda: [Decimal("-1e30") * v for v in base])
        assert find_relation(x, algo, digits=DIGITS) == _primitive([1, 1, -1])

    def test_adversarial_duplicate(self, algo):
        r = _dec(lambda: Decimal(5).sqrt())
        assert find_relation([r, r], algo, digits=DIGITS) == [1, -1]

    def test_adversarial_too_short(self, algo):
        with pytest.raises(ValueError):
            find_relation([1], algo)

    def test_adversarial_step_budget(self, algo):
        """Zero steps: no relation surfaces from the initial reduction of this input."""
        assert find_relation(_alpha_powers(), algo, digits=DIGITS, maxsteps=0) is None


class TestEquivalence:
    def test_control_deterministic(self):
        """Control (Hard Rule 46): each algorithm agrees with a second run of itself."""
        x = _alpha_powers()
        for algo in ALGOS:
            assert find_relation(x, algo, digits=DIGITS) == find_relation(x, algo, digits=DIGITS)

    def test_same_gamma_same_relation(self):
        """With one gamma, the three are one iteration in three numerics (FBA 1999)."""
        x = _alpha_powers()
        g = Decimal(2).sqrt()
        out = {a: find_relation(x, a, digits=DIGITS, gamma=g) for a in ALGOS}
        assert len(set(map(tuple, out.values()))) == 1

    def test_trapezoids_agree_stepwise(self):
        """PSOS's updated D and PSLQ's H_jj^2 track HJLS's recomputed Gram-Schmidt.

        Control first (Hard Rule 46): two HJLS runs must agree exactly.
        """
        x = _alpha_powers()
        with localcontext() as ctx:
            ctx.prec = DIGITS
            g2, tol2, lim = Decimal(2), Decimal(10) ** -96, Decimal(10) ** 20
            runs = {a: _IMPL[a](x, g2, tol2, lim) for a in ALGOS}
            ctrl = _IMPL[Algo.HJLS](x, g2, tol2, lim)
            for s in runs.values():
                s._reduce_all()
            ctrl._reduce_all()

            for _ in range(8):
                if runs[Algo.HJLS]._hit():
                    break
                ms = {a: s._step() for a, s in runs.items()}
                assert ctrl._step() == ms[Algo.HJLS] and ctrl._diag2() == runs[Algo.HJLS]._diag2()
                assert len(set(ms.values())) == 1
                assert runs[Algo.PSOS].a == runs[Algo.HJLS].a == runs[Algo.PSLQ].a

                ref = runs[Algo.HJLS]._diag2()
                for a in (Algo.PSOS, Algo.PSLQ):
                    for d, r in zip(runs[a]._diag2(), ref, strict=True):
                        assert abs(d - r) <= Decimal(10) ** -40 * max(r, Decimal(1))

    def test_wrappers(self):
        x = [3, 5]
        assert pslq(x) == hjls(x) == psos(x) == _primitive([5, -3])
