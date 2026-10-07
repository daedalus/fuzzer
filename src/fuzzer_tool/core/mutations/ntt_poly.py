"""NTT-friendly modular polynomial mutator.

Finite-field counterpart of the complex FFT already used for structural
periodicity detection (``core/periodicity.py``).  Targets that implement
modular polynomial multiplication or an NTT (post-quantum schemes such as
Dilithium / Falcon / Kyber, custom finite-field DSP, etc.) almost never
see algebraically structured inputs under flat-byte mutation.  This
operator injects the classic NTT modulus / primitive root and, when the
buffer is long enough, rewrites a power-of-two coefficient region via the
NTT so the modular butterfly paths are exercised.

Parallel to ``montgomery_mutate`` (REDC / Barrett constant injection for
secp256k1 field arithmetic) and ``naf_scalar_mutate`` (NAF-weight extremes
for EC scalar ladders).
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.structured import _region, _splice
from fuzzer_tool.core.ntt import DEFAULT_MOD, DEFAULT_ROOT, ntt

#: 4-byte big-endian encodings of the default NTT parameters.
_MOD_BE = DEFAULT_MOD.to_bytes(4, "big")
_MOD_LE = DEFAULT_MOD.to_bytes(4, "little")
_ROOT_BE = DEFAULT_ROOT.to_bytes(4, "big")
_ROOT_LE = DEFAULT_ROOT.to_bytes(4, "little")

#: Coefficient word width written into / read from the buffer.
_COEFF_BYTES = 4
#: Smallest useful NTT length (must be a power of two).
_MIN_NTT_LEN = 4


def sniff_ntt_modulus(data: bytes) -> bool:
    """True when the seed already embeds the default NTT modulus."""
    return _MOD_BE in data or _MOD_LE in data


def _inject_constants(data: bytes, rng) -> bytes:
    """Splice the NTT modulus (and optionally the root) into an aligned slot."""
    offset, length = _region(
        len(data), rng, min_len=4, align=4, max_len=8
    )
    if length < 4:
        # Buffer too short for an in-place splice: append the modulus.
        return data + _MOD_BE
    block = _MOD_BE if rng.randint(0, 1) == 0 else _MOD_LE
    if length >= 8 and rng.randint(0, 1) == 0:
        root = _ROOT_BE if rng.randint(0, 1) == 0 else _ROOT_LE
        block = block + root
    return _splice(data, offset, block)


def _coeffs_from_region(buf: bytes, offset: int, n: int) -> list[int]:
    """Decode *n* little-endian 32-bit words starting at *offset*."""
    out: list[int] = []
    for i in range(n):
        start = offset + i * _COEFF_BYTES
        chunk = buf[start : start + _COEFF_BYTES]
        if len(chunk) < _COEFF_BYTES:
            chunk = chunk + b"\x00" * (_COEFF_BYTES - len(chunk))
        out.append(int.from_bytes(chunk, "little") % DEFAULT_MOD)
    return out


def _region_from_coeffs(coeffs: list[int]) -> bytes:
    """Encode coefficients as little-endian 32-bit words."""
    return b"".join((c % DEFAULT_MOD).to_bytes(_COEFF_BYTES, "little") for c in coeffs)


def _transform_region(data: bytes, rng) -> bytes | None:
    """Apply a forward NTT to a power-of-two coefficient window, if space allows."""
    max_words = len(data) // _COEFF_BYTES
    if max_words < _MIN_NTT_LEN:
        return None
    # Largest power-of-two length that fits.
    n = _MIN_NTT_LEN
    while (n << 1) <= max_words and (n << 1) * _COEFF_BYTES <= len(data):
        n <<= 1
    # Optionally use a smaller power of two so we do not always rewrite the
    # whole buffer.
    while n > _MIN_NTT_LEN and rng.randint(0, 1) == 0:
        n >>= 1
    byte_len = n * _COEFF_BYTES
    offset, _ = _region(len(data), rng, min_len=byte_len, align=_COEFF_BYTES)
    if offset + byte_len > len(data):
        offset = 0
        if byte_len > len(data):
            return None
    coeffs = _coeffs_from_region(data, offset, n)
    # Structured transform: forward NTT, then optionally scale one bin so
    # the inverse is non-trivial if a target ever round-trips the region.
    ntt(coeffs, invert=False)
    if rng.randint(0, 1) == 0:
        idx = rng.randint(0, n - 1)
        scale = 1 + rng.randint(1, 15)
        coeffs[idx] = coeffs[idx] * scale % DEFAULT_MOD
    block = _region_from_coeffs(coeffs)
    return _splice(data, offset, block)


def ntt_poly_mutate(data: bytes, rng) -> bytes:
    """Inject NTT constants or rewrite a coefficient region via the NTT.

    Args:
        data: Input buffer.
        rng: ``RandPool`` (or ScriptedRng in tests).

    Returns:
        Mutated buffer.  Length is preserved when the transform path is
        taken; the inject path may grow a short buffer by a few bytes.
    """
    if not data:
        return _MOD_BE

    # Prefer the transform when the buffer is large enough and the modulus
    # is already present (so subsequent coverage has a recognised field);
    # otherwise inject the modulus so later selections can transform.
    if sniff_ntt_modulus(data) and len(data) >= _MIN_NTT_LEN * _COEFF_BYTES:
        transformed = _transform_region(data, rng)
        if transformed is not None:
            return transformed
    return _inject_constants(data, rng)
