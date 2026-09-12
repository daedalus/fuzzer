"""Non-adjacent-form (NAF) scalar mutator for elliptic-curve targets.

Ported from OEIS A379015 (reversed NAF representation of n) — see
``docs/handover/handover_oeis_port_candidates_2026-09-12.md`` C3.

The non-adjacent form is a signed-digit binary representation (digits in
{-1, 0, 1}, no two adjacent digits both nonzero) with the minimal possible
Hamming weight among all such representations of an integer. Scalar
multiplication (ECDSA/ECDH/Schnorr signing, EC point multiplication
generally) commonly uses NAF internally to cut the number of point-addition
steps roughly in half versus plain binary double-and-add.

Why this is a fuzzing lever and not just an encoding curiosity: a
constant-time scalar-mult ladder is *supposed* to do the same sequence of
operations regardless of the scalar's NAF weight. Implementations that
branch on individual NAF digits (rather than using unconditional
conditional-swaps) leak timing information proportional to the number of
nonzero digits, and NAF-weight extremes are exactly where that leakage is
largest and where copy/dummy-operation bookkeeping is most likely to have
an off-by-one. This mutator crafts scalars at both weight extremes so
exec-time analysis (``core/exec_time_anomaly.py``, ``core/execution_time.py``)
has a real chance of catching such a leak, instead of relying on whatever
weight distribution the corpus happens to already contain.

Two extremes are constructed:

- **Minimal weight** — a value with a long run of zero NAF digits: a small
  number of set bits spaced apart (e.g. ``(1 << a) - (1 << b)`` forms,
  which NAF represents with very few nonzero digits regardless of how many
  bits they span). Exercises the "few point additions, many doublings"
  path.
- **Maximal weight** — the theoretical NAF ceiling is alternating nonzero
  digits (never two adjacent), i.e. a bit pattern that forces a nonzero NAF
  digit roughly every other position: ``0b0101...01`` and its bit-complement
  neighbourhood. Exercises the "point addition on almost every step" path,
  the opposite tail from the minimal case.

This operates on a fixed-width big-endian integer field sized to match
common EC scalar widths (32 bytes — secp256k1's order/field size, also
Ed25519/P-256), matching the input layout used by
``targets/secp256k1_read.c`` (mode byte, param byte, then payload containing
signature components, private-key-shaped nonces, and x-only public keys).
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.structured import _region, _splice

#: Field width in bytes matching secp256k1's order/field size (also
#: Ed25519/P-256/most other curves currently fuzzed against).
SCALAR_BYTES = 32


def to_naf(k: int) -> list[int]:
    """Non-adjacent form of a non-negative integer, digits LSB-first.

    Standard algorithm: while k > 0, if k is odd take
    ``d = 2 - (k mod 4)`` (giving +1 or -1 so the result stays a multiple of
    4 after subtracting d, which guarantees the next digit is forced to
    zero -- the non-adjacency property), else d = 0; then k = (k - d) // 2.
    """
    if k < 0:
        raise ValueError("to_naf requires a non-negative integer")
    digits: list[int] = []
    while k > 0:
        if k & 1:
            d = 2 - (k % 4)
            k -= d
        else:
            d = 0
        digits.append(d)
        k >>= 1
    return digits


def from_naf(digits: list[int]) -> int:
    """Inverse of :func:`to_naf` — reconstruct the integer from NAF digits."""
    value = 0
    for i, d in enumerate(digits):
        value += d << i
    return value


def naf_weight(k: int) -> int:
    """Count of nonzero digits in the NAF of *k* (the quantity NAF minimizes)."""
    if k == 0:
        return 0
    return sum(1 for d in to_naf(k) if d != 0)


def _minimal_weight_scalar(rng, nbits: int) -> int:
    """A value of the form ``(1 << a) +/- (1 << b)`` — NAF weight <= 2."""
    a = rng.randint(nbits // 2, nbits - 1)
    b = rng.randint(0, a - 1) if a > 0 else 0
    sign = 1 if rng.choice((0, 1)) else -1
    value = (1 << a) + sign * (1 << b)
    return value if value > 0 else (1 << a)


def _maximal_weight_scalar(rng, nbits: int) -> int:
    """Alternating-bit pattern: forces a nonzero NAF digit ~every other bit.

    ``0b0101...``/``0b1010...`` are the two phases; each is nudged by
    flipping a handful of individual bits so repeated draws don't collapse
    onto exactly two fixed values (the *class* of extremal-weight scalars
    matters, not one canonical member of it).
    """
    phase = rng.choice((0, 1))
    value = 0
    for i in range(nbits):
        if (i % 2) == phase:
            value |= 1 << i
    n_flips = rng.randint(0, max(1, nbits // 16))
    for _ in range(n_flips):
        value ^= 1 << rng.randint(0, nbits - 1)
    return value


def naf_scalar_mutate(data: bytes, rng) -> bytes:
    """Overwrite a 32-byte region with a NAF-weight-extremal scalar.

    Picks minimal or maximal NAF weight with equal probability, encodes it
    big-endian in :data:`SCALAR_BYTES`, and splices it into the buffer at an
    aligned offset -- aligned to the field width so it lines up with how a
    reader parsing fixed-width scalar fields (private keys, nonces, x-only
    public key coordinates, signature r/s components) actually slices the
    buffer, rather than landing as noise at an arbitrary byte offset.

    Args:
        data: Input bytes.
        rng: Draw source, required. A ``RandPool`` or anything with the
            same API (tests inject ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    nbits = SCALAR_BYTES * 8
    offset, length = _region(len(data), rng, min_len=SCALAR_BYTES, align=SCALAR_BYTES)
    if length < SCALAR_BYTES:
        return data
    if rng.choice((0, 1)):
        value = _maximal_weight_scalar(rng, nbits)
    else:
        value = _minimal_weight_scalar(rng, nbits)
    value &= (1 << nbits) - 1
    block = value.to_bytes(SCALAR_BYTES, "big")
    return _splice(data, offset, block)
