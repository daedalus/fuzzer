"""
GF(2^q) arithmetic built from scratch (carryless / polynomial arithmetic over GF(2),
elements represented as Python ints where bit k = coefficient of x^k).

Needed by the Cook-Mertz Tree Evaluation procedure: a field of characteristic two,
large enough that its multiplicative order m = |F|-1 exceeds the degree of the
multilinear extension being interpolated, plus a *primitive element* omega so that
{omega^1, ..., omega^m} enumerates F* exactly once (this is what powers the
"worst-case to arbitrary-case" interpolation identity, eq. (5) in the paper).
"""

import random


def poly_deg(p: int) -> int:
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
    """Return (quotient, remainder) of GF(2)-polynomial division a / b."""
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
    return poly_divmod(a, b)[1]


def poly_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, poly_mod(a, b)
    return a


def poly_powmod(base: int, exp: int, mod: int) -> int:
    result = 1
    base = poly_mod(base, mod)
    while exp:
        if exp & 1:
            result = poly_mod(poly_mul(result, base), mod)
        base = poly_mod(poly_mul(base, base), mod)
        exp >>= 1
    return result


def _prime_factors(n: int):
    fs = set()
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
    """Rabin's irreducibility test for a degree-q GF(2) polynomial."""
    x = 2  # the polynomial "x"
    # x^(2^q) mod poly must equal x
    t = x
    for _ in range(q):
        t = poly_mod(poly_mul(t, t), poly)
    if t != x:
        return False
    # for each prime p | q, gcd(x^(2^(q/p)) - x, poly) must be 1
    for p in _prime_factors(q):
        e = q // p
        t = x
        for _ in range(e):
            t = poly_mod(poly_mul(t, t), poly)
        g = poly_gcd(t ^ x, poly)
        if poly_deg(g) > 0:
            return False
    return True


def find_irreducible(q: int) -> int:
    high = 1 << q
    for const_term in (1,):  # constant term must be 1 (else divisible by x)
        cand = high | const_term
        while cand < (1 << (q + 1)):
            if is_irreducible(cand, q):
                return cand
            cand += 2  # keep constant term = 1 (odd)
    raise RuntimeError(f"no irreducible polynomial of degree {q} found")


class GF2n:
    """The field GF(2^q) = GF(2)[x] / (modulus), modulus an irreducible degree-q poly."""

    def __init__(self, q: int, modulus: int = None, seed: int = 0):
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
        while True:
            a = self._rng.randrange(1, self.order)
            if all(self.pow(a, self.m // p) != 1 for p in factors):
                return a

    def omega_powers(self):
        """Return dict i -> omega^i for i = 0..m (so index 1..m enumerates F* once)."""
        powers = {0: 1}
        cur = 1
        for i in range(1, self.m + 1):
            cur = self.mul(cur, self.gen)
            powers[i] = cur
        assert powers[self.m] == 1
        return powers


def test():
    # smoke test
    F = GF2n(8)
    print("modulus:", bin(F.mod), "order:", F.order, "gen:", F.gen)
    powers = F.omega_powers()
    seen = set(powers[i] for i in range(1, F.m + 1))
    assert len(seen) == F.m == len(set(range(1, F.order))), "generator not primitive!"
    print("primitive element verified: omega^1..omega^m covers F* exactly once")
    a = 5
    assert F.mul(F.inv(a), a) == 1
    print("inverse check ok")
