"""bwt / rle: LF-mapping inverse and vectorized runs, byte-identical.

``_bwt_inverse`` recounted ``c`` over ``bwt[:idx+1]`` at every step, O(n^2)
Python for a 256-byte block. That rank picks the rank-th ``c`` of the stable
sort of ``bwt``, i.e. LF[idx] = position of idx in the stable argsort, which
is now built once. ``rle`` found runs and rebuilt the output byte by byte;
run starts now come from one comparison and the output from one repeat.
The oracles below are the old code, verbatim.
"""

import random

import pytest

from fuzzer_tool.core.mutations import structured as s
from fuzzer_tool.core.rand_pool import RandPool


def _old_bwt_inverse(bwt_data, primary):
    n = len(bwt_data)
    if n <= 1:
        return bwt_data
    table = sorted((bwt_data[i], i) for i in range(n))
    F = [table[i][0] for i in range(n)]
    rows_f = {}
    for i, c in enumerate(F):
        rows_f.setdefault(c, []).append(i)
    idx = primary
    out = bytearray(n)
    for i in range(n):
        c = bwt_data[idx]
        out[n - 1 - i] = c
        rank = sum(1 for j in range(idx + 1) if bwt_data[j] == c)
        idx = rows_f[c][rank - 1]
    return bytes(out)


def _old_rle(data, rng):
    if len(data) < 2:
        return data
    runs = []
    i = 0
    while i < len(data):
        b = data[i]
        j = i + 1
        while j < len(data) and data[j] == b:
            j += 1
        runs.append((b, j - i))
        i = j
    if len(runs) < 2:
        return data
    n_mut = rng.randint(1, min(3, len(runs)))
    for _ in range(n_mut):
        pos = rng.randint(0, len(runs) - 1)
        if rng.randint(0, 4) < 3:
            runs[pos] = (rng.randint(0, 255), runs[pos][1])
        else:
            new_len = rng.randint(1, max(2, runs[pos][1] * 3))
            delta = new_len - runs[pos][1]
            if delta > 0 and len(runs) >= 2:
                other = rng.randint(0, len(runs) - 1)
                while other == pos:
                    other = rng.randint(0, len(runs) - 1)
                new_other = max(1, runs[other][1] - delta)
                delta -= runs[other][1] - new_other
                runs[other] = (runs[other][0], new_other)
                if delta > 0:
                    runs[pos] = (runs[pos][0], runs[pos][1] + delta)
            else:
                runs[pos] = (runs[pos][0], max(1, runs[pos][1] + delta))
    out = bytearray()
    for b, length in runs:
        out.extend([b] * length)
    if len(out) != len(data):
        return data
    return s._splice(data, 0, bytes(out))


def _inputs():
    rnd = random.Random(9)
    out = [b"ab", b"aaaa", b"abab" * 8, b"\x00" * 300, bytes(range(256))]
    for _ in range(80):
        alphabet = rnd.choice((1, 2, 3, 16, 256))
        run = rnd.choice((1, 1, 4, 20))
        body = bytearray()
        while len(body) < rnd.randrange(2, 600):
            body += bytes([rnd.randrange(alphabet)]) * rnd.randrange(1, run + 1)
        out.append(bytes(body))
    return out


# ---------------------------------------------------------------------------


def test_regression_bwt_inverse_lf_mapping():
    """Every real (BWT, primary) pair inverts exactly as before."""
    for data in _inputs():
        block = data[:256]
        bwt_data, primary = s._bwt(block)
        assert s._bwt_inverse(bwt_data, primary) == _old_bwt_inverse(bwt_data, primary)
        assert s._bwt_inverse(bwt_data, primary) == block


@pytest.mark.parametrize("n", [2, 3, 17, 256])
def test_bwt_inverse_of_edited_stream(n):
    """Adversarial: MTF-edited streams are not valid BWTs; same output anyway."""
    rnd = random.Random(n)
    for _ in range(40):
        garbage = bytes(rnd.randrange(rnd.choice((2, 5, 256))) for _ in range(n))
        primary = rnd.randrange(n)
        assert s._bwt_inverse(garbage, primary) == _old_bwt_inverse(garbage, primary)


def test_bwt_operator_unchanged(monkeypatch):
    """Falsification: same seed, same mutants and draws through the operator."""
    inputs = _inputs()
    new_pool = RandPool(seed=3)
    new = [s.bwt(d, rng=new_pool) for d in inputs]
    with monkeypatch.context() as m:
        m.setattr(s, "_bwt_inverse", _old_bwt_inverse)
        old_pool = RandPool(seed=3)
        old = [s.bwt(d, rng=old_pool) for d in inputs]
        # Control (Hard Rule 46): the oracle against a second run of itself.
        ctl_pool = RandPool(seed=3)
        assert [s.bwt(d, rng=ctl_pool) for d in inputs] == old
    assert new == old
    assert new_pool.random() == old_pool.random()


def test_regression_rle_vectorized():
    """Same mutants and RNG position as the byte loops."""
    inputs = _inputs() + [b"a", b"", b"\x01\x01"]
    new_pool = RandPool(seed=4)
    new = [s.rle(d, rng=new_pool) for d in inputs]
    old_pool = RandPool(seed=4)
    old = [_old_rle(d, rng=old_pool) for d in inputs]
    ctl_pool = RandPool(seed=4)
    assert [_old_rle(d, rng=ctl_pool) for d in inputs] == old
    assert new == old
    assert new_pool.random() == old_pool.random()


def test_rle_returns_bytes_type_unchanged():
    """Adversarial: single-run input comes back as the very same object."""
    data = b"\x07" * 50
    assert s.rle(data, rng=RandPool(seed=0)) is data
