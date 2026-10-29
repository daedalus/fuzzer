"""``RandPool.randint_list(0, b, n)`` skips the int64 cast and offset add.

With ``a == 0`` the modulo result is already the answer; the cast and add
cost ~1 us of ~2 us per call (``_prefetch_dict`` draws >= 64 per round).
Oracle: the pre-change body, frozen here.
"""

import numpy as np

from fuzzer_tool.core.rand_pool import _POOL_ENTRIES, RandPool

_SEED = 0x5EED
# Widths: tiny, odd, the 256 fast path's neighbours, large, 2**32 - 1.
_WIDTHS = (1, 2, 37, 255, 257, 5000, 1 << 20, (1 << 32) - 1)
_COUNTS = (1, 64, 200, _POOL_ENTRIES - 1, _POOL_ENTRIES + 3)
_OFFSETS = (0, 1, -5, 1000)


def _oracle(pool: RandPool, a: int, b: int, count: int) -> list[int]:
    """Pre-change non-256 path of randint_list, verbatim."""
    width = b - a + 1
    return ((pool._take(count) % width).astype(np.int64) + a).tolist()


def _pairs():
    for width in _WIDTHS:
        if width == 256:
            continue
        for count in _COUNTS:
            for a in _OFFSETS:
                yield a, a + width - 1, count


def test_control_oracle_matches_itself():
    """Rule 46: two pools on one seed give the same oracle stream."""
    p, q = RandPool(seed=_SEED), RandPool(seed=_SEED)
    for a, b, count in _pairs():
        assert _oracle(p, a, b, count) == _oracle(q, a, b, count)


def test_regression_randint_list_matches_oracle():
    """Falsification: same values, same stream position, every width/offset."""
    p, q = RandPool(seed=_SEED), RandPool(seed=_SEED)
    for a, b, count in _pairs():
        assert p.randint_list(a, b, count) == _oracle(q, a, b, count)
        assert p._idx == q._idx


def test_regression_zero_base_returns_python_ints():
    """Adversarial: the uncast path still yields plain ints in range."""
    p = RandPool(seed=_SEED)
    out = p.randint_list(0, 36, 500)
    assert all(type(v) is int for v in out)
    assert min(out) >= 0 and max(out) <= 36
