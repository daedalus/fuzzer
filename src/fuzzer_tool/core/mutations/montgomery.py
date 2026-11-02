"""Montgomery REDC / Barrett reduction constant injector for secp256k1-shaped seeds.

Port plan: ``docs/handover/handover_math_port_plan_2026-09-21.md`` P1.

secp256k1's field arithmetic (``secp256k1_fe_mul_inner`` and friends) is a
fixed-window Montgomery multiply over the field prime
``p = 2^256 - 2^32 - 977``. The curve order ``n`` is the other 32-byte
modulus that appears in scalar reduction. Today the secp256k1 target gets
flat-byte mutations; coverage of the REDC encode/decode and Barrett paths
is shallow.

This mutator:

1. Sniffs for either modulus appearing as a big-endian 32-byte constant in
   the seed (or accepts an already-classified secp-shaped buffer via the
   registry ``_AVAILABLE`` predicate).
2. Derives the classic REDC auxiliary constants from that modulus:
   - ``R = 2^{256} mod N`` (Montgomery radix residual)
   - ``R2 = R^2 mod N``
   - ``n0' = -N^{-1} mod 2^{64}`` (the limb-width Montgomery factor used by
     the 64-bit ``secp256k1_fe_mul_inner`` path)
   and the Barrett reduction factor ``µ = floor(2^{512} / N)``.
3. Splices one of those derived blocks into a field-width-aligned slot so
   the target is forced through the REDC/Barrett branches that random
   bytes almost never reach.

No GPLv3 catalogue code is vendored — the constants are computed from the
modulus with stdlib arithmetic only (Hard Rule 51).
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.structured import _region, _splice

#: secp256k1 field prime p = 2^256 - 2^32 - 977
SECP256K1_FIELD_P = bytes.fromhex(
    "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"
)
#: secp256k1 curve order n
SECP256K1_ORDER_N = bytes.fromhex(
    "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"
)

FIELD_BYTES = 32
LIMB_BITS = 64  # matches libsecp256k1's 64-bit limb path
RADIX_BITS = FIELD_BYTES * 8  # 256


def _modulus_from_seed(data: bytes) -> int | None:
    """Return the integer modulus if *data* contains p or n as a BE literal."""
    for literal in (SECP256K1_FIELD_P, SECP256K1_ORDER_N):
        if literal in data:
            return int.from_bytes(literal, "big")
    return None


def sniff_secp_modulus(data: bytes) -> bool:
    """True when the seed embeds the secp256k1 field prime or curve order."""
    return _modulus_from_seed(data) is not None


def _redc_n0_prime(n: int) -> int:
    """Montgomery factor ``n0' = -N^{-1} mod 2^{LIMB_BITS}``."""
    # Inverse of N mod 2^k via Newton for odd N (Hensel lifting).
    inv = 1
    for _ in range(6):  # 2^6 > 64
        inv = (inv * (2 - (n * inv))) & ((1 << LIMB_BITS) - 1)
    return (-inv) & ((1 << LIMB_BITS) - 1)


def _montgomery_r(n: int) -> int:
    """``R = 2^{RADIX_BITS} mod N``."""
    return (1 << RADIX_BITS) % n


def _montgomery_r2(n: int) -> int:
    """``R^2 mod N``."""
    r = _montgomery_r(n)
    return (r * r) % n


def _barrett_mu(n: int) -> int:
    """Barrett reduction factor ``µ = floor(2^{2 * RADIX_BITS} / N)``."""
    return (1 << (2 * RADIX_BITS)) // n


def _pick_constant(n: int, rng) -> bytes:
    """Choose one derived constant and encode it as FIELD_BYTES big-endian."""
    kind = rng.randint(0, 3)
    if kind == 0:
        val = _montgomery_r(n)
    elif kind == 1:
        val = _montgomery_r2(n)
    elif kind == 2:
        # n0' is only LIMB_BITS wide; left-pad to field width.
        val = _redc_n0_prime(n)
    else:
        # Barrett µ is up to RADIX_BITS+1 bits; take the low FIELD_BYTES.
        val = _barrett_mu(n) & ((1 << RADIX_BITS) - 1)
    return val.to_bytes(FIELD_BYTES, "big")


def montgomery_mutate(data: bytes, rng) -> bytes:
    """Splice a REDC/Barrett constant into a field-aligned slot of *data*.

    If the seed does not already contain a recognised modulus literal the
    field prime is injected first so subsequent coverage has a modulus to
    reduce against; otherwise one of the derived constants is written over
    an aligned region.

    Args:
        data: Input buffer.
        rng: ``RandPool`` (or ScriptedRng in tests).

    Returns:
        Mutated buffer, length preserved when possible.
    """
    n = _modulus_from_seed(data)
    if n is None:
        # Inject the field prime at an aligned offset so the sniffer fires
        # on the next selection and the target sees a real modulus.
        offset, length = _region(len(data), rng, min_len=FIELD_BYTES, align=FIELD_BYTES)
        if length < FIELD_BYTES:
            # Buffer too short: prepend mode-compatible padding.
            return data[:2] + SECP256K1_FIELD_P + data[2:]
        return _splice(data, offset, SECP256K1_FIELD_P)

    block = _pick_constant(n, rng)
    offset, length = _region(len(data), rng, min_len=FIELD_BYTES, align=FIELD_BYTES)
    if length < FIELD_BYTES:
        return data
    return _splice(data, offset, block)


# ---------------------------------------------------------------------------
# Word-size moduli (32 / 64 bit)
# ---------------------------------------------------------------------------
#
# ``montgomery_mutate`` above only fires on the secp256k1 field prime and
# curve order.  Word-size Montgomery code (NTT butterflies, Dilithium/Kyber-
# style reductions, 64-bit modular multiplication) uses one-limb moduli
# instead.  The same REDC branch -- the conditional subtraction ``t >= n``
# -- sits behind every one of them, and flat-byte mutation reaches its edge
# (operands at ``n - 1``, ``n``, ``2n - 1``) only by luck.
#
# Source: Slotin, *Algorithms for Modern Hardware*, "Montgomery
# Multiplication" (n' by Newton iteration ``x *= 2 - n*x``, R mod n, R^2 mod
# n).  Plan: ``docs/handover/handover_algorithmica_port_2026-11-02.md`` 4.2.

#: Odd word-size moduli seen in practice, as ``(value, width in bytes)``.
#: Recognition is by exact literal, so a modulus not listed here is not found:
#: an arbitrary odd word would match almost every seed.
WORD_MODULI: tuple[tuple[int, int], ...] = (
    (998244353, 4),  # NTT prime 119 * 2^23 + 1
    (469762049, 4),  # NTT prime 7 * 2^26 + 1
    (167772161, 4),  # NTT prime 5 * 2^25 + 1
    (754974721, 4),  # NTT prime 45 * 2^24 + 1
    (1000000007, 4),  # 10^9 + 7
    (2013265921, 4),  # BabyBear, 15 * 2^27 + 1
    (2147483647, 4),  # Mersenne 2^31 - 1
    (8380417, 4),  # Dilithium q, 2^23 - 2^13 + 1
    (4294967291, 4),  # largest 32-bit prime, 2^32 - 5
    ((1 << 61) - 1, 8),  # Mersenne 2^61 - 1
    ((1 << 64) - 59, 8),  # largest 64-bit prime
    (0xFFFFFFFF00000001, 8),  # Goldilocks, 2^64 - 2^32 + 1
)

#: ``literal bytes -> (modulus, width, byteorder)`` for both endiannesses.
_WORD_LITERALS: dict[bytes, tuple[int, int, str]] = {
    n.to_bytes(w, order): (n, w, order) for n, w in WORD_MODULI for order in ("little", "big")
}


def find_word_modulus(data: bytes) -> tuple[int, int, str, int] | None:
    """Earliest known word-size modulus literal in *data*.

    Returns ``(modulus, width_bytes, byteorder, offset)`` or ``None``.
    """
    best: tuple[int, int, str, int] | None = None
    for literal, (n, w, order) in _WORD_LITERALS.items():
        pos = data.find(literal)
        if pos >= 0 and (best is None or pos < best[3]):
            best = (n, w, order, pos)
    return best


def sniff_word_modulus(data: bytes) -> bool:
    """True when the seed embeds a modulus from :data:`WORD_MODULI`."""
    return find_word_modulus(data) is not None


def _neg_inverse_pow2(n: int, bits: int) -> int:
    """``n^-1 mod 2^bits`` for odd *n*, by Newton/Hensel lifting."""
    mask = (1 << bits) - 1
    inv = n & mask  # n * n == 1 (mod 8): 3 correct bits, doubled per step
    for _ in range(5):  # 3 -> 6 -> 12 -> 24 -> 48 -> 96 >= 64
        inv = (inv * (2 - n * inv)) & mask
    return inv


def word_constants(n: int, bits: int) -> tuple[int, ...]:
    """Constants a word-size REDC/Barrett implementation reads or compares.

    In order: ``R mod n``, ``R^2 mod n``, ``n' = -n^-1 mod R``, ``n^-1 mod R``,
    the Barrett factor ``floor(R^2 / n)`` truncated to *bits*, and the REDC
    boundary operands ``n - 1``, ``n + 1``, ``2n - 1`` (mod ``R = 2^bits``),
    which sit either side of the conditional subtraction.
    """
    mask = (1 << bits) - 1
    r = (1 << bits) % n
    inv = _neg_inverse_pow2(n, bits)
    return (
        r,
        (r * r) % n,
        (-inv) & mask,
        inv,
        ((1 << (2 * bits)) // n) & mask,
        (n - 1) & mask,
        (n + 1) & mask,
        (2 * n - 1) & mask,
    )


def montgomery_word_mutate(data: bytes, rng) -> bytes:
    """Overwrite one word with a REDC/Barrett constant of the modulus in *data*.

    The slot keeps the alignment of the modulus literal and the constant is
    written in the literal's byte order, so the target reads it through the
    same field layout it reads the modulus through.  Length is preserved.
    Returns *data* unchanged when no known modulus is present.
    """
    hit = find_word_modulus(data)
    if hit is None:
        return data
    n, width, order, pos = hit
    consts = word_constants(n, width * 8)
    val = consts[rng.randint(0, len(consts) - 1)]
    first = pos % width
    slots = (len(data) - first) // width  # >= 1: the literal itself fits
    offset = first + width * rng.randint(0, slots - 1)
    return _splice(data, offset, val.to_bytes(width, order))
