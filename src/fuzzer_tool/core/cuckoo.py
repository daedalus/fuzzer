"""Cuckoo filter for probabilistic membership testing with deletions.

Ported and cleaned from AIscripts/cuckoofilter.py for fuzzer-tool.
Uses hashlib (SHA-256) instead of mmh3 so no extra dependency is required
on the hot path; matches the style of core/bloom.py.

Supports:
  - insertions with cuckoo kicking
  - membership queries (false positives possible, no false negatives for inserted items)
  - deletions
  - check-then-add style update

Typical uses in the fuzzer:
  - alternative / complementary structure to BloomFilter for exec-dedup
  - corpus or path tracking that needs eviction
  - generational membership sets
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, List, Optional, Union


Key = Union[str, bytes]


class CuckooFilter:
    """Cuckoo filter backed by fixed-size buckets of fingerprints."""

    def __init__(
        self,
        capacity: int,
        bucket_size: int = 4,
        fingerprint_size: int = 8,
        max_kicks: int = 500,
    ) -> None:
        """
        Parameters
        ----------
        capacity:
            Expected number of items.
        bucket_size:
            Fingerprints per bucket (classic default 4).
        fingerprint_size:
            Bits per fingerprint (8 is common; keep small).
        max_kicks:
            Maximum relocations before insertion fails.
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if fingerprint_size < 1 or fingerprint_size > 64:
            raise ValueError("fingerprint_size must be in 1..64")

        self.capacity = capacity
        self.bucket_size = bucket_size
        self.fingerprint_size = fingerprint_size
        self.max_kicks = max_kicks

        # Number of buckets (power of two for fast masking)
        raw = max(1, capacity // bucket_size)
        self.size = self._next_power_of_two(raw)
        self._mask = self.size - 1
        self.buckets: List[List[int]] = [[] for _ in range(self.size)]
        self.count = 0
        self._fp_mask = (1 << fingerprint_size) - 1

    @staticmethod
    def _next_power_of_two(n: int) -> int:
        n = max(1, n)
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _to_bytes(item: Key) -> bytes:
        if isinstance(item, bytes):
            return item
        if isinstance(item, str):
            return item.encode("utf-8")
        raise TypeError(f"unsupported key type: {type(item)}")

    def _digest(self, data: bytes) -> int:
        return int.from_bytes(hashlib.sha256(data).digest(), "big")

    def _get_fingerprint(self, item: Key) -> int:
        data = self._to_bytes(item)
        # Use high bits of the digest so fingerprint is well mixed
        fp = (self._digest(data) >> 128) & self._fp_mask
        return fp or 1  # never zero

    def _get_index(self, item: Key) -> int:
        data = self._to_bytes(item)
        return self._digest(data) & self._mask

    def _get_alt_index(self, fingerprint: int, index: int) -> int:
        # Classic cuckoo: i2 = i1 XOR hash(fingerprint)
        h = self._digest(fingerprint.to_bytes(8, "big"))
        return (index ^ h) & self._mask

    def _insert_fingerprint(self, index: int, fingerprint: int) -> bool:
        bucket = self.buckets[index]
        if len(bucket) < self.bucket_size:
            bucket.append(fingerprint)
            return True
        return False

    def add(self, item: Key) -> bool:
        """Insert *item*. Returns True on success, False if the filter is full
        or kicking failed.
        """
        if self.count >= self.capacity:
            return False

        fingerprint = self._get_fingerprint(item)
        i1 = self._get_index(item)
        i2 = self._get_alt_index(fingerprint, i1)

        if self._insert_fingerprint(i1, fingerprint):
            self.count += 1
            return True
        if self._insert_fingerprint(i2, fingerprint):
            self.count += 1
            return True

        # Cuckoo kicking
        current_index = random.choice((i1, i2))
        for _ in range(self.max_kicks):
            bucket = self.buckets[current_index]
            if not bucket:
                # Should not happen, but be safe
                bucket.append(fingerprint)
                self.count += 1
                return True

            victim_pos = random.randrange(len(bucket))
            victim_fp = bucket[victim_pos]
            bucket[victim_pos] = fingerprint

            fingerprint = victim_fp
            current_index = self._get_alt_index(fingerprint, current_index)

            if self._insert_fingerprint(current_index, fingerprint):
                self.count += 1
                return True

        return False

    def contains(self, item: Key) -> bool:
        """Return True if *item* is probably present (possible false positive)."""
        fingerprint = self._get_fingerprint(item)
        i1 = self._get_index(item)
        i2 = self._get_alt_index(fingerprint, i1)
        return fingerprint in self.buckets[i1] or fingerprint in self.buckets[i2]

    def query(self, item: Key) -> bool:
        """Alias for :meth:`contains` (BloomFilter-compatible name)."""
        return self.contains(item)

    def remove(self, item: Key) -> bool:
        """Remove one occurrence of *item*. Returns True if a fingerprint was removed."""
        fingerprint = self._get_fingerprint(item)
        i1 = self._get_index(item)
        i2 = self._get_alt_index(fingerprint, i1)

        for idx in (i1, i2):
            bucket = self.buckets[idx]
            try:
                bucket.remove(fingerprint)
                self.count -= 1
                return True
            except ValueError:
                continue
        return False

    def update(self, item: Key) -> bool:
        """Check-then-add. Returns True if the item was already present
        (and is left untouched), False if it was newly inserted.
        """
        if self.contains(item):
            return True
        self.add(item)
        return False

    def clear(self) -> None:
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0

    def __contains__(self, item: Key) -> bool:
        return self.contains(item)

    def __len__(self) -> int:
        return self.count

    @property
    def load_factor(self) -> float:
        total_slots = self.size * self.bucket_size
        return self.count / total_slots if total_slots else 0.0
