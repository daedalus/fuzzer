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
