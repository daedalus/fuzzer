"""secp256k1 field-level edits for ``secp256k1_read.c``.

Layout (see the target): byte 0 = surface mode bits, byte 1 = recid,
bytes 2.. = payload (pubkey 33/65, compact sig r||s, DER sig, x-only key).
``montgomery_mutate`` and ``naf_scalar_mutate`` inject reduction constants
and scalar shapes; neither aims at the parse-time range checks. This does:

    mode            payload edit                      check it hits
    --------------  --------------------------------  -----------------------
    scalar_edge     32-byte slot := 0, n, p, (n+1)/2  r,s in [1,n) / x < p
    prefix_edge     byte 0 := 00/05/06/07/ff          pubkey prefix dispatch
    s_negate        s := n - s                        low-S normalization
    payload_resize  len := 32/33/64/65/72/...         per-surface size gates

Returns None to decline when the payload is too short for the mode or the
edit is a no-op.
"""

from fuzzer_tool.core.mutations.montgomery import (
    FIELD_BYTES,
    SECP256K1_FIELD_P,
    SECP256K1_ORDER_N,
)

PAYLOAD_OFFSET = 2
_N = int.from_bytes(SECP256K1_ORDER_N, "big")
_P = int.from_bytes(SECP256K1_FIELD_P, "big")
_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
_U256_MAX = (1 << 256) - 1

SCALAR_EDGES = (
    0,
    1,
    _N - 1,
    _N,
    _N + 1,
    (_N - 1) // 2,  # largest low-S
    (_N + 1) // 2,  # smallest high-S
    _P - 1,
    _P,
    _P + 1,
    1 << 255,
    _U256_MAX,
    _GX,
)

# 02/03 compressed, 04 uncompressed, 06/07 hybrid; the rest are invalid.
PREFIXES = (0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0xFF)

# Sizes the surfaces gate on: x-only 32, compressed 33, compact 64,
# uncompressed 65, max DER 72, plus off-by-one neighbours.
PAYLOAD_LENS = (0, 31, 32, 33, 63, 64, 65, 72, 73, 96, 97)

_S_OFFSET = PAYLOAD_OFFSET + FIELD_BYTES
_SIG_END = _S_OFFSET + FIELD_BYTES


def _put(data: bytes, off: int, value: int) -> bytes:
    return data[:off] + value.to_bytes(FIELD_BYTES, "big") + data[off + FIELD_BYTES :]


def scalar_edge(data: bytes, rng) -> bytes | None:
    slots = (len(data) - PAYLOAD_OFFSET) // FIELD_BYTES
    if slots < 1:
        return None

    k = rng.choice(range(slots))
    return _put(data, PAYLOAD_OFFSET + k * FIELD_BYTES, rng.choice(SCALAR_EDGES))


def prefix_edge(data: bytes, rng) -> bytes | None:
    if len(data) <= PAYLOAD_OFFSET:
        return None
    return data[:PAYLOAD_OFFSET] + bytes((rng.choice(PREFIXES),)) + data[PAYLOAD_OFFSET + 1 :]


def s_negate(data: bytes, _rng) -> bytes | None:
    """Swap s for n - s: a low-S signature becomes high-S and vice versa."""
    if len(data) < _SIG_END:
        return None

    s = int.from_bytes(data[_S_OFFSET:_SIG_END], "big")
    return _put(data, _S_OFFSET, (_N - s) % _N)


def payload_resize(data: bytes, rng) -> bytes | None:
    """Truncate or zero-pad the payload to a size a surface gates on."""
    length = rng.choice(PAYLOAD_LENS)
    payload = data[PAYLOAD_OFFSET : PAYLOAD_OFFSET + length]
    return data[:PAYLOAD_OFFSET] + payload + bytes(length - len(payload))


MODES = (scalar_edge, prefix_edge, s_negate, payload_resize)


def ecdsa_field_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """Apply one secp256k1 range-check edit; None when it cannot apply."""
    if len(data) < PAYLOAD_OFFSET:
        return None

    out = rng.choice(MODES)(data, rng)
    if out is None or out == data or len(out) > max_len:
        return None
    return out
