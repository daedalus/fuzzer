"""huffman_tree_mutate: incremental-prefix decode, table encode, no spin.

``_huff_decode`` rebuilt each candidate prefix bit by bit (up to 16x16 steps
per symbol), and when fewer bits remained than the shortest unmatched code,
its inner ``break`` skipped the ``for/else`` exit and the ``while`` spun on
the same state forever. The prefix now grows one bit per length, and a tail
no code fits ends the decode. The oracle below is the old operator,
verbatim, which terminates on every stream it is fed here.
"""

import random
import signal

import pytest

from fuzzer_tool.core.mutations import structured as s
from fuzzer_tool.core.rand_pool import RandPool

HANG_LIMIT_S = 2.0


def _old_huff_decode(bits, codes, lengths, length):
    decode_table = {}
    for sym in range(256):
        if lengths[sym] == 0:
            continue
        decode_table[(codes[sym], lengths[sym])] = sym
    restored = bytearray(length)
    ridx = 0
    bidx = 0
    while ridx < length and bidx < len(bits):
        for bit_len in range(1, 17):
            if bidx + bit_len > len(bits):
                break
            prefix = 0
            for i in range(bit_len):
                prefix = (prefix << 1) | bits[bidx + i]
            key = (prefix, bit_len)
            if key in decode_table:
                restored[ridx] = decode_table[key]
                ridx += 1
                bidx += bit_len
                break
        else:
            break
    return restored, ridx


def _old_huffman_tree_mutate(data, rng):
    if len(data) < 4:
        return data
    offset, length = s._region(len(data), rng, min_len=4)
    if length < 4:
        return data
    block = data[offset : offset + length]
    freq = [0] * 256
    for b in block:
        freq[b] += 1
    symbols = sorted(range(256), key=lambda x: (freq[x], x))
    codes, lengths = s._rank_codes(freq, symbols)
    non_zero = [x for x in symbols if freq[x] > 0]
    if len(non_zero) < 2:
        return data
    a, b = rng.sample(non_zero, 2)
    codes[a], codes[b] = codes[b], codes[a]
    lengths[a], lengths[b] = lengths[b], lengths[a]
    bits = bytearray()
    for b in block:
        sym_code = codes[b]
        sym_len = lengths[b]
        for i in range(sym_len - 1, -1, -1):
            bits.append((sym_code >> i) & 1)
    restored, ridx = _old_huff_decode(bits, codes, lengths, length)
    s._pad_random(restored, ridx, rng)
    return s._splice(data, offset, bytes(restored[:length]))


def _inputs():
    rnd = random.Random(3)
    out = [b"abcd", b"\x00\x00\x00\x01", bytes(range(256)), bytes(range(256)) * 3]
    for _ in range(60):
        alphabet = rnd.choice((2, 4, 16, 200, 256))
        out.append(bytes(rnd.randrange(alphabet) for _ in range(rnd.randrange(4, 700))))
    return out


class _Hang(Exception):
    pass


@pytest.fixture
def alarm():
    """Raise _Hang after HANG_LIMIT_S; always disarmed and restored."""

    def _fire(*_):
        raise _Hang

    prev = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, HANG_LIMIT_S)
    yield
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, prev)


# ---------------------------------------------------------------------------


def test_regression_huffman_decode_spin(alarm):
    """Adversarial: a 1-bit tail with only 2-bit codes ends the decode."""
    codes = [0] * 256
    lengths = [0] * 256
    codes[ord("A")], lengths[ord("A")] = 0b10, 2
    assert s._huff_decode(bytearray([1, 0, 1]), codes, lengths, 4) == (bytearray(b"A\0\0\0"), 1)


def test_operator_output_and_draws_unchanged():
    """Falsification: same seed, same mutant and RNG position as the old operator."""
    inputs = _inputs()
    new_pool = RandPool(seed=11)
    new = [s.huffman_tree_mutate(d, rng=new_pool) for d in inputs]
    old_pool = RandPool(seed=11)
    old = [_old_huffman_tree_mutate(d, rng=old_pool) for d in inputs]

    # Control (Hard Rule 46): the oracle against a second run of itself.
    ctl_pool = RandPool(seed=11)
    assert [_old_huffman_tree_mutate(d, rng=ctl_pool) for d in inputs] == old

    assert new == old
    assert new_pool.random() == old_pool.random()


def test_decode_matches_on_random_streams():
    """Arbitrary 0/1 streams against real rank codebooks."""
    rnd = random.Random(4)
    for block in _inputs():
        freq = [0] * 256
        for b in block:
            freq[b] += 1
        symbols = sorted(range(256), key=lambda x: (freq[x], x))
        codes, lengths = s._rank_codes(freq, symbols)
        if 1 not in lengths:
            continue  # without 1-bit codes a random tail can spin the oracle
        bits = bytearray(rnd.randrange(2) for _ in range(rnd.randrange(0, 400)))
        assert s._huff_decode(bits, codes, lengths, 64) == _old_huff_decode(
            bits, codes, lengths, 64
        )
