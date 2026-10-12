"""Reed-Solomon parity repair and symbol-error injection (``--rs-nsym``).

A target that decodes RS-protected data (QR, DataMatrix, CCSDS-style
containers) either rejects a mutated block at the integrity check or silently
"corrects" it, so plain byte flips never reach the parser behind the decoder.
This operator treats the buffer as consecutive systematic RS codewords
(data then ``nsym`` parity bytes, GF(256)) and, on one block, does one of:

* repair: edit data bytes, then re-encode the parity so the block is valid;
* correctable: re-encode, then corrupt 1..t symbols (t = nsym // 2) anywhere
  in the block, which exercises the decoder's correction path;
* uncorrectable: re-encode, then corrupt t+1..2t+1 symbols, which exercises
  its failure path (some such words decode to a different codeword).

The code is MDS (minimum distance nsym + 1), so up to t errors are always
uniquely correctable.  Lengths never change.  Generator roots are
``alpha^(fcr+i)``, ``alpha = x``.  Stdlib only (Hard Rule 51).
"""

from __future__ import annotations

import functools

from fuzzer_tool.core.gf2_common import GF2n

_Q = 8
_ORDER = 1 << _Q
#: QR / DataMatrix field polynomial x^8+x^4+x^3+x^2+1.
DEFAULT_POLY = 0x11D
#: Largest block: 2^8 - 1 symbols.
MAX_BLOCK = _ORDER - 1
DEFAULT_FCR = 0
_BYTE = 8
#: Modes: edit data and re-encode, <= t errors, > t errors.
_MODE_REPAIR, _MODE_CORRECTABLE, _MODE_UNCORRECTABLE = 0, 1, 2
_MAX_DATA_EDITS = 4


@functools.cache
def _field(poly: int) -> GF2n:
    """GF(256) over *poly*; ValueError unless irreducible and x is primitive."""
    field = GF2n(_Q, poly)
    x, order = 1, 0
    while True:
        x = field.mul(x, 2)
        order += 1
        if x == 1:
            break
    if order != MAX_BLOCK:
        raise ValueError(f"poly {poly:#x}: x has order {order}, need {MAX_BLOCK} (primitive)")
    return field


@functools.cache
def _feedback_rows(nsym: int, fcr: int, poly: int) -> tuple[int, ...]:
    """``rows[f]``: the generator's low coefficients scaled by *f*, as one int.

    With g(x) = prod (x - alpha^(fcr+i)) monic of degree nsym, the LFSR step
    for feedback byte f XORs ``f * (g_{nsym-1} .. g_0)`` into the shifted
    register.  Packing the register into one big int makes a step O(1).
    """
    field = _field(poly)
    gen = [1]  # highest degree first
    for i in range(nsym):
        root = field.pow(2, fcr + i)
        nxt = gen + [0]
        for j, c in enumerate(gen):
            nxt[j + 1] ^= field.mul(c, root)
        gen = nxt
    low = gen[1:]
    return tuple(int.from_bytes(bytes(field.mul(f, c) for c in low), "big") for f in range(_ORDER))


def rs_parity(data: bytes, nsym: int, fcr: int = DEFAULT_FCR, poly: int = DEFAULT_POLY) -> bytes:
    """Return the *nsym* parity bytes of *data* (systematic, data first)."""
    rows = _feedback_rows(nsym, fcr, poly)
    shift = _BYTE * (nsym - 1)
    mask = (1 << (_BYTE * nsym)) - 1
    reg = 0
    for d in data:
        fb = d ^ (reg >> shift)
        reg = ((reg << _BYTE) & mask) ^ rows[fb]
    return reg.to_bytes(nsym, "big")


def check_params(nsym: int, block: int) -> None:
    """ValueError unless 2 <= nsym < block <= 255 (nsym < 2 corrects nothing)."""
    if not 2 <= nsym < block <= MAX_BLOCK:
        raise ValueError(f"need 2 <= nsym < block <= {MAX_BLOCK}, got nsym={nsym} block={block}")


def parse_poly(text: str) -> int:
    """Parse a hex/decimal field polynomial and validate it; ValueError on failure."""
    try:
        poly = int(text, 0)
    except ValueError as exc:
        raise ValueError(f"field polynomial is not a number: {exc}") from exc
    if not _ORDER < poly < 2 * _ORDER:
        raise ValueError(f"field polynomial must have degree {_Q}, got {poly:#x}")
    _field(poly)
    return poly


def block_spans(total: int, nsym: int, block: int) -> list[tuple[int, int]]:
    """Spans of consecutive blocks long enough to hold data (> nsym bytes)."""
    spans = []
    for start in range(0, total, block):
        end = min(start + block, total)
        if end - start > nsym:
            spans.append((start, end))
    return spans


def _pick_positions(rng, n: int, count: int) -> list[int]:
    """*count* distinct indices in range(n) (partial Fisher-Yates)."""
    idx = list(range(n))
    for i in range(count):
        j = rng.randint(i, n - 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx[:count]


def _edit_data(blk: bytearray, k: int, rng) -> None:
    """Flip a bit or redraw a byte at 1..4 data positions."""
    for pos in _pick_positions(rng, k, rng.randint(1, min(_MAX_DATA_EDITS, k))):
        if rng.randint(0, 1):
            blk[pos] ^= 1 << rng.randint(0, _BYTE - 1)
        else:
            blk[pos] ^= rng.randint(1, _ORDER - 1)


def rs_fix(
    data: bytes,
    rng,
    nsym: int,
    block: int = MAX_BLOCK,
    fcr: int = DEFAULT_FCR,
    poly: int = DEFAULT_POLY,
) -> bytes | None:
    """Mutate one RS block of *data*; None if no block can hold data.

    Args:
        data: Buffer of consecutive ``block``-byte codewords (last may be short).
        rng: ``RandPool`` (or ScriptedRng in tests).
        nsym: Parity bytes per block; t = nsym // 2 symbol errors correctable.
        block: Codeword length including parity, at most 255.
        fcr: First consecutive root exponent (QR/DataMatrix use 0).
        poly: GF(256) field polynomial.
    """
    spans = block_spans(len(data), nsym, block)
    if not spans:
        return None
    start, end = spans[rng.randint(0, len(spans) - 1)]
    blk = bytearray(data[start:end])
    k = len(blk) - nsym
    mode = rng.randint(_MODE_REPAIR, _MODE_UNCORRECTABLE)
    if mode == _MODE_REPAIR:
        _edit_data(blk, k, rng)
    blk[k:] = rs_parity(bytes(blk[:k]), nsym, fcr, poly)
    if mode != _MODE_REPAIR:
        t = nsym // 2
        lo, hi = (1, t) if mode == _MODE_CORRECTABLE else (t + 1, 2 * t + 1)
        count = min(rng.randint(lo, hi), len(blk))
        for pos in _pick_positions(rng, len(blk), count):
            blk[pos] ^= rng.randint(1, _ORDER - 1)
    return data[:start] + bytes(blk) + data[end:]
