"""Bloom filter for probabilistic membership testing."""

import hashlib
import math


class BloomFilter:
    """Bloom filter backed by a :class:`bytearray`.

    Uses a single SHA-256 hash with bit-variable slicing to produce *k*
    independent index positions from the 256-bit digest.  The filter size *m*
    is rounded up to a power of two so that fast bitwise masking (``& (m-1)``)
    can replace modulo.

    Each position *p* is mapped to a byte in the backing array and a bit
    within that byte::

        byte_idx = (p & mask) >> 3
        bit_idx  = (p & mask) & 7
    """

    def __init__(self, capacity: int, error_rate: float = 0.01) -> None:
        n = max(capacity, 1)
        m_ideal = -n * math.log(error_rate) / (math.log(2) ** 2)
        self.m = 1 << max(1, int(m_ideal).bit_length())
        self._mask = self.m - 1
        self._bits_per_slice = self.m.bit_length() - 1
        self._k = max(1, round(self.m / n * math.log(2)))

        self._byte_len = (self.m + 7) // 8
        self._bits = bytearray(self._byte_len)

    @staticmethod
    def _digest(key: str) -> int:
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")

    def _check(self, value: int) -> bool:
        v = value
        for _ in range(self._k):
            pos = v & self._mask
            byte_idx = pos >> 3
            bit_idx = pos & 7
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
            v >>= self._bits_per_slice
        return True

    def _set(self, value: int) -> None:
        v = value
        for _ in range(self._k):
            pos = v & self._mask
            byte_idx = pos >> 3
            bit_idx = pos & 7
            self._bits[byte_idx] |= 1 << bit_idx
            v >>= self._bits_per_slice

    def add(self, key: str) -> None:
        self._set(self._digest(key))

    def query(self, key: str) -> bool:
        return self._check(self._digest(key))

    def update(self, key: str) -> bool:
        """Check membership then add.  Returns ``True`` if the key was already present."""
        value = self._digest(key)
        if self._check(value):
            return True
        self._set(value)
        return False

    @property
    def load_factor(self) -> float:
        """Fraction of bits set to 1."""
        bits_set = sum(b.bit_count() for b in self._bits)
        return bits_set / self.m

    def clear(self) -> None:
        self._bits = bytearray(self._byte_len)
