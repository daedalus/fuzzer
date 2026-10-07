"""Number-theoretic transform (NTT) over a finite field.

Finite-field counterpart of the complex DFT used in
:mod:`fuzzer_tool.core.periodicity` (numpy ``rfft`` / ``irfft``).  Same
Cooley-Tukey butterflies and convolution theorem; the ambient field is
``Z/pZ`` instead of ``C``.

Default parameters use the common NTT-friendly prime

    p = 998244353 = 119 * 2^23 + 1,  primitive root g = 3

so every power-of-two length ``n <= 2^23`` admits a principal n-th root of
unity ``omega = g^{(p-1)/n} mod p``.

Callers
-------
- :mod:`fuzzer_tool.core.mutations.ntt_poly` — injects NTT-friendly
  constants and applies structured modular-polynomial mutations so
  targets that contain modular poly-mul / NTT code are forced through
  those paths (same role Montgomery REDC injection plays for secp256k1
  field arithmetic).

No numpy dependency: pure modular arithmetic, matching the scipy-free
Hard Rule 51 stance already taken for ``gf2_common`` and the statistical
primitives.
"""

from __future__ import annotations

# NTT-friendly modulus and a primitive root modulo it.
DEFAULT_MOD = 998244353
DEFAULT_ROOT = 3


def ntt(
    a: list[int],
    *,
    invert: bool = False,
    mod: int = DEFAULT_MOD,
    root: int = DEFAULT_ROOT,
) -> list[int]:
    """In-place iterative radix-2 Cooley-Tukey NTT (or inverse).

    ``a`` must have power-of-two length.  Coefficients are reduced
    modulo *mod* on every butterfly.  When *invert* is true the
    modular inverse root is used and the result is scaled by
    ``n^{-1} mod mod``.

    Returns the same list object for chaining convenience.
    """
    n = len(a)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"NTT length must be a power of two, got {n}")

    # Bit-reversal permutation.
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Cooley-Tukey butterflies.
    length = 2
    while length <= n:
        wlen = pow(root, (mod - 1) // length, mod)
        if invert:
            wlen = pow(wlen, mod - 2, mod)
        half = length >> 1
        for i in range(0, n, length):
            w = 1
            for j in range(i, i + half):
                u = a[j]
                v = a[j + half] * w % mod
                a[j] = (u + v) % mod
                a[j + half] = (u - v) % mod
                w = w * wlen % mod
        length <<= 1

    if invert:
        n_inv = pow(n, mod - 2, mod)
        for i in range(n):
            a[i] = a[i] * n_inv % mod
    return a


def poly_mul_ntt(
    a: list[int],
    b: list[int],
    *,
    mod: int = DEFAULT_MOD,
    root: int = DEFAULT_ROOT,
) -> list[int]:
    """Multiply two polynomials modulo *mod* via three NTTs.

    Returns the coefficient list of the product, trimmed of trailing
    zeros (at least a single zero is kept for the zero polynomial).
    """
    if not a or not b:
        return [0]
    need = len(a) + len(b) - 1
    n = 1
    while n < need:
        n <<= 1
    fa = [x % mod for x in a] + [0] * (n - len(a))
    fb = [x % mod for x in b] + [0] * (n - len(b))
    ntt(fa, invert=False, mod=mod, root=root)
    ntt(fb, invert=False, mod=mod, root=root)
    for i in range(n):
        fa[i] = fa[i] * fb[i] % mod
    ntt(fa, invert=True, mod=mod, root=root)
    while len(fa) > 1 and fa[-1] == 0:
        fa.pop()
    return fa
