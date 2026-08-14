"""Shared GF(2) polynomial arithmetic and finite-field utilities.

This module is the single source of truth for:

- GF(2) polynomial operations on ``int``-encoded polynomials (bit k =
  coefficient of ``x^k``).
- Rabin irreducibility testing and irreducible-polynomial search.
- The :class:`GF2n` field class, which wraps the polynomial layer into a
  full ``GF(2^q)`` field with ``add``/``mul``/``pow``/``inv`` methods and a
  primitive-element search.
- A generic primitive-root finder used by :class:`GF2n` and any future
  GF(2^n) consumer.

Callers in this repo
--------------------
- :mod:`fuzzer_tool.core.berlekamp_massey` — CRC polynomial recovery via
  ``poly_gcd``.
- :mod:`fuzzer_tool.core.cook_mertz` — Cook-Mertz MLE interpolation field
  layer, which imports :class:`GF2n` from here.
"""

from __future__ import annotations

import random


def poly_deg(p: int) -> int:
    """Degree of a GF(2) polynomial encoded as a Python ``int``."""
    return p.bit_length() - 1 if p else -1


def poly_mul(a: int, b: int) -> int:
    """Carryless (GF(2)) polynomial multiplication."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def poly_divmod(a: int, b: int):
    """Return ``(quotient, remainder)`` of GF(2)-polynomial division ``a / b``."""
    if b == 0:
        raise ZeroDivisionError
    db = poly_deg(b)
    r = a
    q = 0
    while poly_deg(r) >= db:
        shift = poly_deg(r) - db
        r ^= b << shift
        q ^= 1 << shift
    return q, r


def poly_mod(a: int, b: int) -> int:
    """Polynomial remainder of ``a`` divided by ``b`` over GF(2)."""
    return poly_divmod(a, b)[1]


def poly_gcd(a: int, b: int) -> int:
    """GCD of two polynomials over GF(2)."""
    while b:
        a, b = b, poly_mod(a, b)
    return a


def poly_powmod(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation ``base**exp mod mod`` over GF(2)."""
    result = 1
    base = poly_mod(base, mod)
    while exp:
        if exp & 1:
            result = poly_mod(poly_mul(result, base), mod)
        base = poly_mod(poly_mul(base, base), mod)
        exp >>= 1
    return result


def _prime_factors(n: int) -> set[int]:
    """Return the distinct prime factors of ``n`` via trial division."""
    fs: set[int] = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


def is_irreducible(poly: int, q: int) -> bool:
    """Rabin's irreducibility test for a degree-``q`` GF(2) polynomial."""
    x = 2  # the polynomial "x"
    x_mod = poly_mod(x, poly)
    # x^(2^q) mod poly must equal x mod poly
    t = x_mod
    for _ in range(q):
        t = poly_mod(poly_mul(t, t), poly)
    if t != x_mod:
        return False
    # for each prime p | q, gcd(x^(2^(q/p)) - x, poly) must be 1
    for p in _prime_factors(q):
        e = q // p
        t = x_mod
        for _ in range(e):
            t = poly_mod(poly_mul(t, t), poly)
        g = poly_gcd(t ^ x_mod, poly)
        if poly_deg(g) > 0:
            return False
    return True


def find_irreducible(q: int) -> int:
    """Return the first irreducible polynomial of degree ``q`` over GF(2)."""
    high = 1 << q
    for const_term in (1,):  # constant term must be 1 (else divisible by x)
        cand = high | const_term
        while cand < (1 << (q + 1)):
            if is_irreducible(cand, q):
                return cand
            cand += 2  # keep constant term = 1 (odd)
    raise RuntimeError(f"no irreducible polynomial of degree {q} found")


def find_primitive_root(order: int, is_primitive, rng) -> int:
    """Find a primitive root of a finite group by random search.

    Args:
        order: Size of the multiplicative group (e.g. ``2**q - 1``).
        is_primitive: Callable ``is_primitive(candidate) -> bool``.
        rng: Random source with ``randrange(lo, hi)``.  Accepted types:
            ``random.Random``, :class:`fuzzer_tool.core.rand_pool.RandPool`,
            or any duck-typed object with the same method.

    Returns:
        A primitive element in ``[1, order)``.
    """
    if order <= 1:
        return 1
    while True:
        a = rng.randrange(1, order)
        if is_primitive(a):
            return a


class GF2n:
    """The field GF(2^q) = GF(2)[x] / (modulus), modulus an irreducible degree-q poly."""

    def __init__(self, q: int, modulus: int | None = None, seed: int = 0) -> None:
        self.q = q
        self.order = 1 << q  # |F|
        self.m = self.order - 1  # |F*|, exponent of the multiplicative group
        self.mod = modulus if modulus is not None else find_irreducible(q)
        assert is_irreducible(self.mod, q), "supplied modulus is not irreducible"
        self._rng = random.Random(seed)
        self.gen = self._find_primitive()

    # ---- field ops ----
    def add(self, a: int, b: int) -> int:
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        return poly_mod(poly_mul(a, b), self.mod)

    def pow(self, a: int, e: int) -> int:
        e %= self.m if a != 0 else e
        return poly_powmod(a, e, self.mod)

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("no inverse of 0")
        return self.pow(a, self.m - 1)

    # ---- primitive element ----
    def _find_primitive(self) -> int:
        if self.m == 1:
            return 1
        factors = _prime_factors(self.m)

        def _is_primitive(a: int) -> bool:
            return all(self.pow(a, self.m // p) != 1 for p in factors)

        return find_primitive_root(self.m, _is_primitive, self._rng)

    def omega_powers(self) -> dict[int, int]:
        """Return dict ``i -> omega^i`` for ``i = 0..m``."""
        powers: dict[int, int] = {0: 1}
        cur = 1
        for i in range(1, self.m + 1):
            cur = self.mul(cur, self.gen)
            powers[i] = cur
        assert powers[self.m] == 1
        return powers
