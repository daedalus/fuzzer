"""``murmur3_32`` and the scalar chain inline the block mix and finalizer.

One helper call per 4-byte block cost 11-20% of the hash. Oracle: the
reference implementation in ``tests/test_murmur3_chain.py`` (the operator's
original code, verbatim), which also carries the published vectors.
"""

from fuzzer_tool.core.mutations.murmur3 import _VECTOR_MIN_BLOCKS, murmur3_32, murmur3_chain
from fuzzer_tool.core.rand_pool import RandPool
from tests.test_murmur3_chain import _ref_chain, _ref_murmur3_32

_SEED = 0x3A3
_HASH_SEEDS = (0, 1, 0x7FFFFFFF, 0xFFFFFFFF, 0x9747B28C)
_MAX_LEN = 300


def test_control_oracle_matches_itself():
    """Rule 46: the reference agrees with itself."""
    rng = RandPool(seed=_SEED)
    for n in range(0, 40):
        data = rng.randbytes(n)
        assert _ref_murmur3_32(data, 7) == _ref_murmur3_32(bytes(data), 7)


def test_regression_murmur3_32_matches_reference():
    """Falsification: every length 0..300 (all tail sizes), extreme seeds,
    all-0xFF blocks that widen every product."""
    rng = RandPool(seed=_SEED)
    for n in range(_MAX_LEN):
        for data in (rng.randbytes(n), b"\xff" * n):
            for seed in _HASH_SEEDS:
                assert murmur3_32(data, seed) == _ref_murmur3_32(data, seed)


def test_regression_scalar_chain_every_block_count():
    """Adversarial: each scalar-path block count up to the vector switch,
    plus the first vector length, against the original fill loop."""
    rng = RandPool(seed=_SEED + 1)
    for blocks in range(1, _VECTOR_MIN_BLOCKS + 2):
        for length in (blocks * 4 - 3, blocks * 4):
            prefix = rng.randbytes(rng.randrange(64))
            seed = rng.randrange(1 << 32)
            assert murmur3_chain(prefix, length, seed) == _ref_chain(prefix, length, seed)
