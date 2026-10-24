"""Word-first galloping prefix/suffix scan behind ``splice_common_prefix``.

Expected lengths come from how each input is built, not from the code under
test; the byte-loop oracle is checked against the same construction first
(Hard Rule 46).
"""

import pytest

from fuzzer_tool.core.mutations import splice_common_prefix
from fuzzer_tool.core.mutations.generic import _prefix_len, _suffix_len
from tests.support.scripted_rng import ScriptedRng

# Around the 8-byte first word and the 16/32/64 gallop steps.
SHARED_LENS = (0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 2000)
FILL = bytes(range(1, 251))


def _naive(a, b):
    """Byte-by-byte prefix/suffix (the pre-gallop algorithm)."""
    i0 = 0
    while i0 < len(a) and i0 < len(b) and a[i0] == b[i0]:
        i0 += 1
    i1 = 0
    while i1 < len(a) and i1 < len(b) and a[len(a) - i1 - 1] == b[len(b) - i1 - 1]:
        i1 += 1
    return i0, i1


def _naive_splice(a, b, rng):
    """Pre-gallop ``splice_common_prefix`` body, kept as the oracle."""
    if len(a) < 4 or len(b) < 4:
        return a
    idx0, idx1 = _naive(a, b)
    diff = min(len(a) - idx0 - idx1, len(b) - idx0 - idx1)
    if diff < 4:
        return a
    cut = idx0 + rng.randint(0, diff - 2) + 1
    n = rng.randint(0, min(diff - (cut - idx0), len(b) - idx0))
    return a[:cut] + b[idx0 : idx0 + n] + a[cut + n :]


def _pair(pre, suf):
    """Shared prefix/suffix of known length around one differing byte."""
    head = (FILL * 10)[:pre]
    tail = (FILL[::-1] * 10)[:suf]
    return head + b"\x00" + tail, bytearray(head + b"\xff" + tail)


@pytest.mark.parametrize("pre", SHARED_LENS)
@pytest.mark.parametrize("suf", SHARED_LENS)
def test_lengths_match_construction(pre, suf):
    a, b = _pair(pre, suf)

    # Control: the oracle recovers the construction.
    assert _naive(a, b) == (pre, suf)

    assert (_prefix_len(a, b), _suffix_len(a, b)) == (pre, suf)


def test_falsify_last_byte_mismatch():
    # One differing byte at the very end: the gallop must not skip it.
    a = FILL * 16
    b = bytearray(a)
    b[-1] ^= 1

    assert _prefix_len(a, b) == len(a) - 1
    assert _suffix_len(a, b) == 0


def test_adversarial_periodic_and_overlap():
    # Period-8 data matches every word-aligned window but the last byte.
    word = b"ABCDEFGH"
    a = word * 40 + b"Y"
    b = word * 40 + b"Z"
    assert (_prefix_len(a, b), _suffix_len(a, b)) == _naive(a, b)

    # One input inside the other: prefix and suffix overlap, capped at min len.
    short, long_ = b"AAAA", b"AAAAAAAAAAAAAAAAAAAA"
    assert (_prefix_len(short, long_), _suffix_len(short, long_)) == (4, 4)
    assert (_prefix_len(b"", long_), _suffix_len(long_, b"")) == (0, 0)


@pytest.mark.parametrize("pre,suf", [(0, 0), (9, 33), (2000, 65)])
def test_splice_output_unchanged(pre, suf):
    head = (FILL * 10)[:pre]
    tail = (FILL[::-1] * 10)[:suf]
    a = head + b"a" * 40 + tail
    b = bytearray(head + b"b" * 24 + tail)
    draws = [5, 3]

    expected = _naive_splice(a, b, ScriptedRng(randints=list(draws)))
    assert expected != a, "oracle must take the splice path"

    assert splice_common_prefix(a, b, rng=ScriptedRng(randints=list(draws))) == expected
