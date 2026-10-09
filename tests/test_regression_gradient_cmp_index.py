"""gradient_cmp walked the whole cmplog pool to its first partial match.

~360 pairs per call (2.3 ms) on a JSON target: pairs sharing no byte with
the input can never match. A per-pool byte -> pair-index map, merged in
pool order, visits only pairs that can; first hit, offsets and RNG draws
are unchanged.
"""

from fuzzer_tool.core import gradient_cmp as gc
from fuzzer_tool.core.rand_pool import RandPool


def _old(data, cmp_values, rng):
    """The pre-change body, verbatim."""
    if not data or not cmp_values:
        return data
    buf = bytearray(data)
    buf_len = len(buf)
    pos_map = {}
    for idx, b in enumerate(buf):
        pos_map.setdefault(b, []).append(idx)
    for cmp_a, cmp_b in cmp_values:
        if len(cmp_a) == 0 or len(cmp_a) > 32:
            continue
        for cmp_val in (cmp_a, cmp_b):
            n = len(cmp_val)
            if n < 2 or n > buf_len:
                continue
            hit = gc._partial_match(buf, pos_map, cmp_val)
            if hit is None:
                continue
            gc._apply_gradient(buf, hit[0], hit[1], cmp_val, rng)
            return bytes(buf)
    if cmp_values:
        cmp_val = cmp_values[rng.randint(0, len(cmp_values) - 1)][0]
        if 0 < len(cmp_val) <= 32:
            pos = rng.randint(0, len(buf))
            return bytes(buf[:pos]) + cmp_val + bytes(buf[pos:])
    return bytes(buf)


def _pool(rng, n):
    out = []
    for _ in range(n):
        a = bytes(rng.randint(128, 255) for _ in range(rng.choice([0, 1, 2, 4, 8, 40])))
        b = bytes(rng.randint(0, 255) for _ in range(rng.choice([1, 2, 4])))
        out.append((a, b))
    return out


def test_regression_gradient_cmp_index():
    for seed in range(60):
        rng = RandPool(seed)
        pool = _pool(rng, 300)
        data = bytes(rng.randint(32, 127) for _ in range(rng.randint(0, 200)))
        if seed % 3 == 0 and data:
            data += pool[200][0][:2]  # a late pair becomes matchable
        assert gc.gradient_cmp(data, pool, rng=RandPool(seed)) == _old(data, pool, RandPool(seed))


def test_grown_pool_reindexed():
    """Adversarial: a pair appended to the same list is seen on the next call."""
    pool = [(b"\xf0\xf1", b"\x01")]
    data = b"abcdef"
    gc.gradient_cmp(data, pool, rng=RandPool(1))
    pool.append((b"cz", b"\x02"))
    assert gc.gradient_cmp(data, pool, rng=RandPool(1)) == _old(data, pool, RandPool(1))
