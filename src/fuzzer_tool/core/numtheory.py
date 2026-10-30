"""Number theory in the stdlib: the ``sympy.ntheory`` subset the fuzzer needs.

Hard Rule 51: no sympy/gmpy.  Everything is integer-only and deterministic
(no RNG: Pollard-rho restarts walk ``c = 1, 2, 3, ...``).

    egcd / modinv / crt           Bezout, inverses, Chinese remainder
    euclid_steps                  gcd division count (Lame: Fibonacci pairs worst)
    is_prime / next_prime / ...   Miller-Rabin, fixed bases (+ strong Lucas, BPSW, above 3.3e24)
    jacobi / legendre / kronecker symbols; quadratic_residues
    factorint / divisors / ...    trial division + Brent rho + perfect powers
    sqrt_mod / nthroot_mod        Tonelli-Shanks; k-th roots mod a prime
    mult_order / primitive_root   group structure of (Z/m)*
    discrete_log                  Pohlig-Hellman over BSGS

Uses: probing parameter handling of targets (primality tests, DH/RSA
parameter checks) with adversarial moduli, and solving modular relations
recovered from observed state.

Limits: ``is_prime`` is exact below ``_MR_EXACT_BELOW``; above it the fixed
Miller-Rabin bases are joined by a strong Lucas test (Baillie-PSW: no
counterexample is known, none is proven absent).  ``factorint`` is rho-based: 64-bit inputs are
instant, balanced semiprimes beyond ~100 bits are not practical.
``discrete_log`` bounds BSGS memory: any prime factor of the group order
above ``_BSGS_MAX_Q`` raises.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Witnesses; with these 12 Miller-Rabin is exact below ``_MR_EXACT_BELOW``.
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
#: Extra fixed bases for larger inputs (probabilistic, error < 4**-20).
_MR_EXTRA = (41, 43, 47, 53, 59, 61, 67, 71)
_MR_EXACT_BELOW = 3317044064679887385961981
#: Trial-division bound before switching to rho.
_TRIAL_LIMIT = 1000
#: Largest prime-order subgroup solved by BSGS (table <= 2**22 entries).
_BSGS_MAX_Q = 1 << 44
#: Brute-force cap for discrete logs with gcd(g, n) > 1.
_BRUTE_MAX_N = 1 << 22
#: Pollard-Brent GCD batch size.
_RHO_BATCH = 128


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Return ``(g, x, y)`` with ``a*x + b*y == g == gcd(a, b)`` (g >= 0)."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q, r = divmod(a, b)
        a, b = b, r
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if a < 0:
        return -a, -x0, -y0
    return a, x0, y0


def euclid_steps(a: int, b: int) -> int:
    """Division steps of Euclid's gcd on ``(|a|, |b|)``.

    Lame: below a bound, consecutive Fibonacci pairs need the most, e.g.
    ``euclid_steps(89, 55) == 9`` against ``euclid_steps(100, 55) == 4``.
    """
    a, b = abs(a), abs(b)
    steps = 0
    while b:
        a, b = b, a % b
        steps += 1
    return steps


def modinv(a: int, m: int) -> int:
    """Inverse of *a* mod *m*; ValueError when gcd(a, m) != 1."""
    if m == 1:
        return 0
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError(f"{a} is not invertible mod {m}")
    return x % m


def crt(mods: Sequence[int], rems: Sequence[int]) -> tuple[int, int] | None:
    """Solve ``x = rems[i] (mod mods[i])``; return ``(x, lcm)`` or None.

    Moduli need not be coprime: e.g. ``x=0 (4), x=2 (6)`` -> ``(8, 12)``.
    """
    if len(mods) != len(rems):
        raise ValueError("mods and rems differ in length")
    if any(m <= 0 for m in mods):
        raise ValueError("moduli must be positive")
    x, m = 0, 1
    for mi, ri in zip(mods, rems, strict=True):
        g, p, _ = egcd(m, mi)
        if (ri - x) % g:
            return None
        lcm = m // g * mi
        x = (x + m * ((ri - x) // g * p % (mi // g))) % lcm
        m = lcm
    return x, m


def _mr_witness(n: int, d: int, s: int, a: int) -> bool:
    """True when base *a* proves *n* composite."""
    x = pow(a, d, n)
    if x in (1, n - 1):
        return False
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return False
    return True


def is_prime(n: int) -> bool:
    """Miller-Rabin primality; exact below ~3.3e24, MR + strong Lucas (BPSW) above."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    if n < _MR_EXACT_BELOW:
        return not any(_mr_witness(n, d, s, a) for a in _MR_BASES)
    if any(_mr_witness(n, d, s, a) for a in _MR_BASES + _MR_EXTRA):
        return False
    return _strong_lucas(n)


def next_prime(n: int) -> int:
    """Smallest prime strictly greater than *n*."""
    if n < 2:
        return 2
    c = n + 1 + (n % 2 == 0 and n > 2)
    c |= 1
    while not is_prime(c):
        c += 2
    return c


def prev_prime(n: int) -> int | None:
    """Largest prime strictly less than *n*; None when none exists."""
    if n <= 2:
        return None
    if n == 3:
        return 2
    c = (n - 1) | 1
    if c >= n:
        c -= 2
    while c > 2 and not is_prime(c):
        c -= 2
    return c if c >= 2 else None


def _iroot(n: int, k: int) -> int:
    """Floor of the k-th root of *n* (Newton)."""
    if k == 1 or n < 2:
        return n
    x = 1 << -(-n.bit_length() // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


def _perfect_power(n: int) -> tuple[int, int] | None:
    """Return ``(root, k)`` with ``root**k == n`` and k maximal, else None."""
    for k in range(n.bit_length(), 1, -1):
        r = _iroot(n, k)
        if r > 1 and r**k == n:
            return r, k
    return None


def _rho(n: int, c: int) -> int:
    """One Pollard-Brent run with constant *c*; returns a factor or n."""
    y, r, q, g = 2, 1, 1, 1
    x = ys = 2
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(_RHO_BATCH, r - k)):
                y = (y * y + c) % n
                q = q * abs(x - y) % n
            g = math.gcd(q, n)
            k += _RHO_BATCH
        r *= 2
    if g != n:
        return g
    return _rho_backtrack(n, c, x, ys)


def _rho_backtrack(n: int, c: int, x: int, ys: int) -> int:
    """Step the last batch one at a time when the batch gcd overshot."""
    g = 1
    while g == 1:
        ys = (ys * ys + c) % n
        g = math.gcd(abs(x - ys), n)
    return g


def _split(n: int) -> int:
    """A nontrivial factor of composite *n*."""
    c = 1
    while True:
        f = _rho(n, c)
        if 1 < f < n:
            return f
        c += 1


def _trial(n: int, out: dict[int, int]) -> int:
    """Strip primes <= ``_TRIAL_LIMIT`` from *n* into *out*."""
    for p in (2, 3, *range(5, _TRIAL_LIMIT + 1, 2)):
        if p * p > n:
            break
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
    return n


def _add(n: int, mult: int, out: dict[int, int]) -> None:
    """Factor *n* (no small factors) into *out*, each prime counted *mult*."""
    if n == 1:
        return
    if is_prime(n):
        out[n] = out.get(n, 0) + mult
        return
    power = _perfect_power(n)
    if power:
        _add(power[0], mult * power[1], out)
        return
    f = _split(n)
    _add(f, mult, out)
    _add(n // f, mult, out)


def factorint(n: int) -> dict[int, int]:
    """Prime factorisation ``{p: e}``, primes ascending."""
    if n < 1:
        raise ValueError("factorint needs n >= 1")
    out: dict[int, int] = {}
    _add(_trial(n, out), 1, out)
    return dict(sorted(out.items()))


def divisors(n: int) -> list[int]:
    """All positive divisors of *n*, ascending."""
    divs = [1]
    for p, e in factorint(n).items():
        divs = [d * p**k for d in divs for k in range(e + 1)]
    return sorted(divs)


def totient(n: int) -> int:
    """Euler phi."""
    out = n
    for p in factorint(n):
        out -= out // p
    return out


def mobius(n: int) -> int:
    """Moebius mu."""
    f = factorint(n)
    if any(e > 1 for e in f.values()):
        return 0
    return -1 if len(f) % 2 else 1


def carmichael_lambda(n: int) -> int:
    """Exponent of the group (Z/n)*."""
    out = 1
    for p, e in factorint(n).items():
        lam = (1 << (e - 2)) if p == 2 and e >= 3 else p ** (e - 1) * (p - 1)
        out = out // math.gcd(out, lam) * lam
    return out


def sqrt_mod(a: int, p: int) -> int | None:
    """A square root of *a* mod odd-or-2 prime *p* (the smaller), or None."""
    if not is_prime(p):
        raise ValueError(f"{p} is not prime")
    a %= p
    if p == 2 or a == 0:
        return a
    if pow(a, (p - 1) // 2, p) != 1:
        return None
    r = _tonelli(a, p)
    return min(r, p - r)


def _tonelli(a: int, p: int) -> int:
    """Tonelli-Shanks for a quadratic residue *a* mod prime *p*."""
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c = i, b * b % p
        t, r = t * c % p, r * b % p
    return r


def mult_order(a: int, n: int) -> int:
    """Multiplicative order of *a* mod *n*."""
    if math.gcd(a, n) != 1:
        raise ValueError(f"{a} is not a unit mod {n}")
    order = carmichael_lambda(n)
    for p in factorint(order):
        while order % p == 0 and pow(a, order // p, n) == 1:
            order //= p
    return order


def _is_cyclic(n: int) -> bool:
    """(Z/n)* is cyclic iff n in {1, 2, 4, p**k, 2*p**k}."""
    if n in (1, 2, 4):
        return True
    if n % 4 == 0:
        return False
    if n % 2 == 0:
        n //= 2
    return len(factorint(n)) == 1


def primitive_root(n: int) -> int | None:
    """Smallest primitive root mod *n*; None when (Z/n)* is not cyclic."""
    if n == 1:
        return 0
    if not _is_cyclic(n):
        return None
    phi = totient(n)
    primes = list(factorint(phi))
    g = 1
    while True:
        if math.gcd(g, n) == 1 and all(pow(g, phi // q, n) != 1 for q in primes):
            return g
        g += 1


def _bsgs_prime(g: int, h: int, q: int, n: int) -> int | None:
    """Log of *h* base *g* (prime order *q* mod *n*); None if h not in <g>."""
    if q > _BSGS_MAX_Q:
        raise ValueError(f"group order has prime factor {q} > {_BSGS_MAX_Q}")
    m = math.isqrt(q - 1) + 1
    baby: dict[int, int] = {}
    cur = 1
    for j in range(m):
        baby.setdefault(cur, j)
        cur = cur * g % n
    giant = pow(g, -m, n)
    cur = h
    for i in range(m):
        j = baby.get(cur)
        if j is not None:
            return i * m + j
        cur = cur * giant % n
    return None


def _log_prime_power(g: int, h: int, q: int, e: int, n: int) -> int | None:
    """Log mod q**e where g has order q**e (digit-by-digit lift)."""
    gamma = pow(g, q ** (e - 1), n)
    g_inv = pow(g, -1, n)
    x = 0
    for k in range(e):
        hk = pow(h * pow(g_inv, x, n) % n, q ** (e - 1 - k), n)
        digit = _bsgs_prime(gamma, hk, q, n)
        if digit is None:
            return None
        x += digit * q**k
    return x


def _brute(n: int, h: int, g: int) -> int | None:
    """Linear scan for non-unit bases (bounded by ``_BRUTE_MAX_N``)."""
    if n > _BRUTE_MAX_N:
        raise ValueError("non-coprime base with large modulus unsupported")
    cur = 1 % n
    for x in range(n):
        if cur == h % n:
            return x
        cur = cur * g % n
    return None


def discrete_log(n: int, h: int, g: int) -> int | None:
    """Least ``x >= 0`` with ``g**x == h (mod n)``, or None.

    Pohlig-Hellman over the order of *g* with BSGS per prime factor, so the
    cost is ``sum(sqrt(q))`` over prime factors ``q`` of that order.
    """
    if n < 2:
        raise ValueError("modulus must be >= 2")
    g, h = g % n, h % n
    if math.gcd(g, n) != 1:
        return _brute(n, h, g)
    order = mult_order(g, n)
    if pow(h, order, n) != 1:
        return None
    parts, mods = [], []
    for q, e in factorint(order).items():
        qe = q**e
        cofactor = order // qe
        part = _log_prime_power(pow(g, cofactor, n), pow(h, cofactor, n), q, e, n)
        if part is None:
            return None
        parts.append(part)
        mods.append(qe)
    sol = crt(mods, parts)
    if sol is None or pow(g, sol[0], n) != h:
        return None
    return sol[0]


# --- Jacobi / Legendre / Kronecker, quadratic residues ----------------------


def jacobi(a: int, n: int) -> int:
    """Jacobi symbol ``(a/n)`` for odd positive *n*: -1, 0 or 1."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be odd and positive")
    a %= n
    out = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                out = -out
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            out = -out
        a %= n
    return out if n == 1 else 0


def legendre(a: int, p: int) -> int:
    """Legendre symbol ``(a/p)`` for an odd prime *p*."""
    if p == 2 or not is_prime(p):
        raise ValueError(f"{p} is not an odd prime")
    return jacobi(a, p)


def kronecker(a: int, n: int) -> int:
    """Kronecker symbol ``(a/n)`` for any integers (extends Jacobi to even/negative n)."""
    if n == 0:
        return 1 if abs(a) == 1 else 0
    out = 1
    if n < 0:
        n = -n
        if a < 0:
            out = -out
    v = 0
    while n % 2 == 0:
        n //= 2
        v += 1
    if v:
        if a % 2 == 0:
            return 0
        if v % 2 and a % 8 in (3, 5):
            out = -out
    return out * jacobi(a, n)


def quadratic_residues(n: int) -> list[int]:
    """Sorted squares mod *n* (0 included)."""
    if n < 1:
        raise ValueError("modulus must be >= 1")
    return sorted({x * x % n for x in range(n // 2 + 1)})


# --- strong Lucas (second half of Baillie-PSW) ------------------------------


def _half(x: int, n: int) -> int:
    """``x / 2 mod n`` for odd *n*."""
    x %= n
    if x & 1:
        x += n
    return x >> 1


def _strong_lucas(n: int) -> bool:
    """Strong Lucas probable-prime test, Selfridge method A (odd *n* > 5)."""
    if math.isqrt(n) ** 2 == n:
        return False
    d_par = 5
    while True:
        j = jacobi(d_par, n)
        if j == -1:
            break
        if j == 0 and abs(d_par) != n:
            return False
        d_par = -d_par - 2 if d_par > 0 else -d_par + 2
    p_par, q_par = 1, (1 - d_par) // 4
    d, s = n + 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    u, v, qk = 1, p_par, q_par % n
    for bit in bin(d)[3:]:
        u, v, qk = u * v % n, (v * v - 2 * qk) % n, qk * qk % n
        if bit == "1":
            u, v = _half(p_par * u + v, n), _half(d_par * u + p_par * v, n)
            qk = qk * q_par % n
    if u == 0 or v == 0:
        return True
    for _ in range(s - 1):
        v, qk = (v * v - 2 * qk) % n, qk * qk % n
        if v == 0:
            return True
    return False


# --- k-th roots mod a prime --------------------------------------------------


def nthroot_mod(a: int, k: int, p: int, all_roots: bool = False) -> int | list[int] | None:
    """Solutions of ``x**k == a (mod p)`` for prime *p*.

    Returns the least root (None when there is none), or the sorted list of
    every root with ``all_roots``.  ``gcd(k, p-1) == 1`` is a single modular
    exponentiation; otherwise the log of *a* base a primitive root is taken
    (``discrete_log``, so the group-order limits of that function apply).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not is_prime(p):
        raise ValueError(f"{p} is not prime")
    a %= p
    if a == 0:
        return [0] if all_roots else 0
    if p == 2:
        return [1] if all_roots else 1
    g = math.gcd(k, p - 1)
    if g == 1:
        r = pow(a, modinv(k, p - 1), p)
        return [r] if all_roots else r
    root = primitive_root(p)
    assert root is not None
    if pow(a, (p - 1) // g, p) != 1:
        return None
    lg = discrete_log(p, a, root)
    if lg is None:
        return None
    # k*y == lg (mod p-1) has g solutions spaced (p-1)/g apart.
    step = (p - 1) // g
    y0 = lg // g * modinv(k // g, step) % step
    roots = sorted(pow(root, y0 + i * step, p) for i in range(g))
    return roots if all_roots else roots[0]
