"""``randrange_list(2**32, k)`` raised OverflowError.

The vectorized path guards ``n > 1 << 32``, so ``n == 2**32`` reached
``uint32_array % n`` and numpy rejected the out-of-range scalar. A full-span
draw must return the pool words unchanged, matching scalar ``randrange``.
"""

from fuzzer_tool.core.rand_pool import RandPool

U32 = 1 << 32


def test_regression_rand_pool_full_u32_list():
    got = RandPool(seed=9).randrange_list(U32, 100)
    ref = RandPool(seed=9)
    assert got == [ref.randrange(U32) for _ in range(100)]
