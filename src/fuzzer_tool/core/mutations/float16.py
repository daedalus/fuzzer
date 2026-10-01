"""Half-precision float edges: IEEE binary16 and bfloat16.

``bitcast_float`` and ``interesting_*`` cover 32/64-bit floats; 16-bit
formats (GPU textures, OpenEXR, ML tensors, glTF) have their own subnormal,
infinity and NaN bit patterns, which byte-level operators hit by luck.

``float16_edge`` overwrites two bytes at *byte_idx* with one such pattern,
in either byte order. Length-preserving, so field offsets stay aligned.
"""

# IEEE binary16: sign | 5-bit exponent | 10-bit mantissa.
F16_EDGES = (
    0x0000,  # +0
    0x8000,  # -0
    0x0001,  # min subnormal, 2**-24
    0x03FF,  # max subnormal
    0x0400,  # min normal, 2**-14
    0x3BFF,  # largest < 1
    0x3C00,  # 1.0
    0x7BFF,  # max finite, 65504
    0x7C00,  # +inf
    0xFC00,  # -inf
    0x7E00,  # quiet NaN
    0x7C01,  # signalling NaN
)

# bfloat16: the top half of a float32 (8-bit exponent, 7-bit mantissa).
BF16_EDGES = (
    0x8000,  # -0
    0x0001,  # min subnormal
    0x0080,  # min normal
    0x7F7F,  # max finite
    0x7F80,  # +inf
    0xFF80,  # -inf
    0x7FC0,  # quiet NaN
    0x7F81,  # signalling NaN
)

TABLES = (F16_EDGES, BF16_EDGES)
ORDERS = ("little", "big")
_WIDTH = 2


def float16_edge(data: bytes, byte_idx: int, rng, max_len: int) -> bytes | None:  # noqa: ARG001
    """Overwrite the 16-bit slot at *byte_idx* with a half-float edge."""
    n = len(data)
    if n < _WIDTH:
        return None

    # Clamp so the slot fits: b"abc" at 2 writes bytes 1..2.
    pos = min(byte_idx % n, n - _WIDTH)
    bits = rng.choice(rng.choice(TABLES))
    repl = bits.to_bytes(_WIDTH, rng.choice(ORDERS))

    if repl == data[pos : pos + _WIDTH]:
        return None
    return data[:pos] + repl + data[pos + _WIDTH :]
