"""Rice/Elias codecs: table encode + C-level scans, byte-identical to bit loops.

``golomb`` / ``elias_gamma`` / ``elias_delta`` encoded and decoded one bit
per Python iteration (0.3-1.2 ms per 512-byte block each way). Encoding now
joins per-byte codewords from a 256-entry table; decoding finds unary runs
with ``bytearray.find`` and reads fields with ``int(..., 2)``. The oracles
below are the old loops, verbatim.
"""

import random

import pytest

from fuzzer_tool.core.mutations import structured as s
from fuzzer_tool.core.rand_pool import RandPool

KS = (2, 3, 4, 5, 6)


# ---------------------------------------------------------------------------
# Oracles: the bit-loop codecs, verbatim.
# ---------------------------------------------------------------------------


def _old_rice_encode(block, k):
    mask = (1 << k) - 1
    bits = bytearray()
    for b in block:
        q = b >> k
        r = b & mask
        bits.extend([1] * q + [0])
        for i in range(k):
            bits.append((r >> i) & 1)
    return bits


def _old_rice_decode(bits, k, length):
    restored = bytearray(length)
    ridx = 0
    bidx = 0
    while ridx < length and bidx < len(bits):
        q = 0
        while bidx < len(bits) and bits[bidx] == 1:
            q += 1
            bidx += 1
        if bidx >= len(bits):
            break
        bidx += 1
        r = 0
        for i in range(k):
            if bidx + i < len(bits):
                r |= bits[bidx + i] << i
        restored[ridx] = ((q << k) | r) & 0xFF
        ridx += 1
        bidx += k
    return restored, ridx


def _old_gamma_encode(block):
    bits = bytearray()
    for b in block:
        n = b + 1
        log2 = n.bit_length() - 1
        bits.extend([0] * log2)
        bits.extend(int(x) for x in format(n, f"0{log2 + 1}b"))
    return bits


def _old_gamma_decode(bits, length):
    restored = bytearray(length)
    ridx = 0
    bidx = 0
    while ridx < length and bidx < len(bits):
        log2 = 0
        while bidx < len(bits) and bits[bidx] == 0:
            log2 += 1
            bidx += 1
        if bidx >= len(bits) or log2 == 0:
            break
        total_bits = log2 + 1
        if bidx + total_bits > len(bits):
            break
        val = 0
        for i in range(total_bits):
            val = (val << 1) | bits[bidx + i]
        restored[ridx] = (val - 1) & 0xFF
        ridx += 1
        bidx += total_bits
    return restored, ridx


def _old_delta_encode(block):
    bits = bytearray()
    for b in block:
        n = b + 1
        log2 = n.bit_length() - 1
        gamma = log2 + 1
        g_log2 = gamma.bit_length() - 1
        bits.extend([0] * g_log2)
        bits.extend(int(x) for x in format(gamma, f"0{g_log2 + 1}b"))
        if log2 > 0:
            bits.extend(int(x) for x in format(n & ((1 << log2) - 1), f"0{log2}b"))
    return bits


def _old_read_gamma(bits, bidx):
    g_log2 = 0
    while bidx < len(bits) and bits[bidx] == 0:
        g_log2 += 1
        bidx += 1
    if bidx >= len(bits) or g_log2 == 0:
        return -1, bidx
    gamma_bits = g_log2 + 1
    if bidx + gamma_bits > len(bits):
        return -1, bidx
    gamma = 0
    for i in range(gamma_bits):
        gamma = (gamma << 1) | bits[bidx + i]
    return gamma, bidx + gamma_bits


def _old_delta_decode(bits, length):
    restored = bytearray(length)
    ridx = 0
    bidx = 0
    while ridx < length and bidx < len(bits):
        gamma, bidx = _old_read_gamma(bits, bidx)
        if gamma < 0:
            break
        log2 = gamma - 1
        if log2 < 0:
            break
        total_bits = log2
        if bidx + total_bits > len(bits):
            break
        val = 1 << log2
        for i in range(total_bits):
            val = (val << 1) | bits[bidx + i]
        restored[ridx] = (val - 1) & 0xFF
        ridx += 1
        bidx += total_bits
    return restored, ridx


_OLD = {
    "_rice_encode": _old_rice_encode,
    "_rice_decode": _old_rice_decode,
    "_gamma_encode": _old_gamma_encode,
    "_gamma_decode": _old_gamma_decode,
    "_delta_encode": _old_delta_encode,
    "_delta_decode": _old_delta_decode,
}


def _edited(bits, rnd):
    """Random 0/1 inserts, deletes and flips, like ``_edit_codebits``."""
    bits = bytearray(bits)
    for _ in range(rnd.randrange(0, 12)):
        if not bits:
            break
        pos = rnd.randrange(len(bits))
        op = rnd.randrange(3)
        if op == 0:
            bits.insert(pos, rnd.randrange(2))
        elif op == 1:
            del bits[pos]
        else:
            bits[pos] ^= 1
    return bits


def _blocks():
    rnd = random.Random(0)
    yield b""
    yield bytes(range(256))
    yield b"\x00" * 64
    yield b"\xff" * 64
    for _ in range(80):
        yield bytes(rnd.randrange(256) for _ in range(rnd.randrange(1, 300)))


# ---------------------------------------------------------------------------


def test_regression_entropy_code_tables():
    """Encoders emit the same bitstream as the bit loops."""
    for block in _blocks():
        for k in KS:
            assert s._rice_encode(bytearray(block), k) == _old_rice_encode(block, k)
        assert s._gamma_encode(block) == _old_gamma_encode(block)
        assert s._delta_encode(block) == _old_delta_encode(block)


def test_decoders_match_on_edited_streams():
    rnd = random.Random(1)
    for block in _blocks():
        length = max(len(block), 2)
        for k in KS:
            bits = _edited(_old_rice_encode(block, k), rnd)
            assert s._rice_decode(bits, k, length) == _old_rice_decode(bits, k, length)
        bits = _edited(_old_gamma_encode(block), rnd)
        assert s._gamma_decode(bits, length) == _old_gamma_decode(bits, length)
        bits = _edited(_old_delta_encode(block), rnd)
        assert s._delta_decode(bits, length) == _old_delta_decode(bits, length)


@pytest.mark.parametrize(
    "bits",
    [
        bytearray(),
        bytearray([1] * 40),  # unary run with no terminator
        bytearray([0] * 40),  # leading zeros with no 1
        bytearray([1]),
        bytearray([0]),
        bytearray([0, 0, 0, 1, 1]),  # truncated codeword
        bytearray([1] * 300 + [0, 1, 0]),  # quotient far past 0xFF
        bytearray([0] * 9 + [1] * 10),  # delta length prefix past the stream
    ],
)
def test_adversarial_streams(bits):
    """Unterminated, truncated and oversized codewords stop where the loops did."""
    for length in (1, 2, 50):
        for k in KS:
            assert s._rice_decode(bits, k, length) == _old_rice_decode(bits, k, length)
        assert s._gamma_decode(bits, length) == _old_gamma_decode(bits, length)
        assert s._delta_decode(bits, length) == _old_delta_decode(bits, length)


@pytest.mark.parametrize("fn", [s.golomb, s.elias_gamma, s.elias_delta])
def test_operator_output_and_draws_unchanged(fn, monkeypatch):
    """Falsification: same seed, same mutant and RNG position as the old codecs."""
    rnd = random.Random(2)
    inputs = [bytes(rnd.randrange(256) for _ in range(rnd.randrange(2, 400))) for _ in range(40)]

    new_pool = RandPool(seed=7)
    new = [fn(d, rng=new_pool) for d in inputs]
    new_tail = new_pool.random()

    with monkeypatch.context() as m:
        for name, old in _OLD.items():
            m.setattr(s, name, old)
        old_pool = RandPool(seed=7)
        old = [fn(d, rng=old_pool) for d in inputs]
        old_tail = old_pool.random()

    # Control (Hard Rule 46): the oracle against a second run of itself.
    with monkeypatch.context() as m:
        for name, old_fn in _OLD.items():
            m.setattr(s, name, old_fn)
        ctl_pool = RandPool(seed=7)
        assert [fn(d, rng=ctl_pool) for d in inputs] == old

    assert new == old
    assert new_tail == old_tail
