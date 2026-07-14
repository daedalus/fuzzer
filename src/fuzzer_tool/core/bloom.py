"""Bloom filter for probabilistic membership testing.

Includes fuzzy near-duplicate detection via Hamming distance on stored keys.
"""

import hashlib
import math
from collections import deque

from fuzzer_tool.core.similarity import hamming_distance


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

    def add_bytes(self, key: bytes, max_hamming: int = 0) -> bool:
        """Add bytes as a key, optionally checking for near-duplicates via Hamming distance.

        When max_hamming > 0, checks the most recent N keys (where N = max_hamming's
        reciprocal heuristic, capped at 200) before adding. Returns True if a
        near-duplicate was found (key NOT added), False if unique (key added).

        Args:
            key: Raw bytes to add.
            max_hamming: Maximum Hamming distance to consider a near-duplicate.
                0 disables fuzzy checking (exact-only, same as add()).

        Returns:
            True if a near-duplicate was found and key was skipped.
        """
        if not hasattr(self, "_recent_keys"):
            self.add(key.hex())
            return False

        key_hex = key.hex()
        if self._check(self._digest(key_hex)):
            return True  # exact match already in filter

        if max_hamming <= 0:
            self._set(self._digest(key_hex))
            self._recent_keys.append(key)
            if len(self._recent_keys) > 200:
                self._recent_keys.popleft()
            return False

        for recent in self._recent_keys:
            try:
                if hamming_distance(key, recent) <= max_hamming:
                    return True
            except ValueError:
                continue

        self._set(self._digest(key_hex))
        self._recent_keys.append(key)
        if len(self._recent_keys) > 200:
            self._recent_keys.popleft()
        return False

    def init_fuzzy(self, max_recent: int = 200) -> None:
        """Initialize recent-keys buffer for fuzzy Hamming dedup.

        Must be called before add_bytes() with max_hamming > 0.

        Args:
            max_recent: Maximum recent keys to track for Hamming comparison.
        """
        self._recent_keys: deque[bytes] = deque(maxlen=max_recent)
