"""Cuckoo filter for probabilistic membership testing with deletions.

Ported and cleaned from AIscripts/cuckoofilter.py for fuzzer-tool.
Uses hashlib (SHA-256) instead of mmh3 so no extra dependency is required
on the hot path; matches the style of core/bloom.py.

Supports:
  - insertions with cuckoo kicking
  - membership queries (false positives possible, no false negatives for inserted items)
  - deletions
  - check-then-add style update

Design notes (load scaling):
  - The bucket array is sized so that ``count == capacity`` never exceeds
    ``MAX_LOAD`` (0.90) of the slots.  Classic b=4 cuckoo tables start
    failing kick chains around 0.95-0.96 load, so sizing straight to
    ``capacity // bucket_size`` (the old behaviour) put capacities such as
    500_000 or any power of two right on the cliff.
  - A failed ``add()`` is transactional: the kick chain is journalled and
    rolled back, so a rejected insert never evicts an already-stored
    fingerprint (no false negatives, ever, for stored items).
  - Kicking uses a private ``random.Random`` so it neither consumes nor
    perturbs the global stream that ``--seed`` makes reproducible.
  - Default fingerprints are 16 bits: false-positive rate ~= 2*b*load/2**f,
    i.e. ~1e-4 at MAX_LOAD (8 bits would be ~3%, worse than the bloom
    backend's 1e-3).

Typical uses in the fuzzer:
  - alternative / complementary structure to BloomFilter for exec-dedup
  - corpus or path tracking that needs eviction
  - generational membership sets
"""

from __future__ import annotations

import hashlib
import math
import random

Key = str | bytes

_FP_SRC_MASK = (1 << 128) - 1


class CuckooFilter:
    """Cuckoo filter backed by fixed-size buckets of fingerprints."""

    #: Maximum fraction of slots occupied when ``count == capacity``.
    MAX_LOAD = 0.90

    def __init__(
        self,
        capacity: int,
        bucket_size: int = 4,
        fingerprint_size: int = 16,
        max_kicks: int = 500,
        rng_seed: int = 0x5EED_C0C0,
    ) -> None:
        """
        Parameters
        ----------
        capacity:
            Expected number of items.
        bucket_size:
            Fingerprints per bucket (classic default 4).
        fingerprint_size:
            Bits per fingerprint.  False-positive rate is roughly
            ``2 * bucket_size * load / 2**fingerprint_size``.
        max_kicks:
            Maximum relocations before insertion fails (and rolls back).
        rng_seed:
            Seed of the filter's private kick RNG (deterministic by
            default so campaigns stay reproducible).
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if fingerprint_size < 1 or fingerprint_size > 64:
            raise ValueError("fingerprint_size must be in 1..64")
        if bucket_size < 1:
            raise ValueError("bucket_size must be >= 1")
        if max_kicks < 0:
            raise ValueError("max_kicks must be >= 0")

        self.capacity = capacity
        self.bucket_size = bucket_size
        self.fingerprint_size = fingerprint_size
        self.max_kicks = max_kicks

        # Number of buckets (power of two for fast masking).  Sized against
        # MAX_LOAD so ``capacity`` items occupy <= 90% of the slots.
        raw = max(1, math.ceil(capacity / (bucket_size * self.MAX_LOAD)))
        self.size = self._next_power_of_two(raw)
        self._mask = self.size - 1
        self.buckets: list[list[int]] = [[] for _ in range(self.size)]
        self.count = 0
        # Distinct-insertion counter, mirroring BloomFilter.n_added.  It is
        # the quantity the generational reset gate reads (see update_bytes),
        # and it differs from `count` only in that it is not decremented by
        # remove() -- a removed item still counts towards the "this filter
        # has absorbed capacity keys" threshold, same as the bloom.
        self.n_added = 0
        # Number of add() calls rejected because the kick chain failed
        # (each was rolled back; the item was NOT stored).
        self.n_failed = 0
        self._fp_mask = (1 << fingerprint_size) - 1
        self._rng = random.Random(rng_seed)
        # alt-index hash depends only on the fingerprint; memoise when the
        # fingerprint space is small enough to enumerate.
        self._alt_cache: dict[int, int] | None = {} if fingerprint_size <= 16 else None

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

    def _fp_from_digest(self, digest: int) -> int:
        # Map onto 1..fp_mask (0 is reserved).  Reducing 128 independent
        # digest bits modulo fp_mask leaves a bias of ~2**-(128-f), unlike
        # the old "0 -> 1" fold (or reducing only f bits), which doubled
        # P(fp == 1).  Bits 64..191 don't overlap the low index bits.
        return ((digest >> 64) & _FP_SRC_MASK) % self._fp_mask + 1

    def _get_fingerprint(self, item: Key) -> int:
        return self._fp_from_digest(self._digest(self._to_bytes(item)))

    def _get_index(self, item: Key) -> int:
        return self._digest(self._to_bytes(item)) & self._mask

    def _get_alt_index(self, fingerprint: int, index: int) -> int:
        # Classic cuckoo: i2 = i1 XOR hash(fingerprint)
        cache = self._alt_cache
        if cache is not None:
            h = cache.get(fingerprint)
            if h is None:
                h = self._digest(fingerprint.to_bytes(8, "big")) & self._mask
                cache[fingerprint] = h
        else:
            h = self._digest(fingerprint.to_bytes(8, "big")) & self._mask
        return index ^ h

    def _locate(self, item: Key) -> tuple[int, int, int]:
        """Return ``(fingerprint, i1, i2)`` from a single item digest."""
        d = self._digest(self._to_bytes(item))
        fp = self._fp_from_digest(d)
        i1 = d & self._mask
        return fp, i1, self._get_alt_index(fp, i1)

    def _insert_fingerprint(self, index: int, fingerprint: int) -> bool:
        bucket = self.buckets[index]
        if len(bucket) < self.bucket_size:
            bucket.append(fingerprint)
            return True
        return False

    def _stored(self) -> None:
        self.count += 1
        self.n_added += 1

    def add(self, item: Key) -> bool:
        """Insert *item*.  Returns True on success.

        Returns False if the filter is at ``capacity`` or the kick chain
        failed.  A failed insert is rolled back: the filter is left exactly
        as it was, so no previously stored item is ever lost.
        """
        if self.count >= self.capacity:
            return False

        fingerprint, i1, i2 = self._locate(item)

        if self._insert_fingerprint(i1, fingerprint) or self._insert_fingerprint(i2, fingerprint):
            self._stored()
            return True

        # Cuckoo kicking, journalled so a failure can be undone.
        rng = self._rng
        journal: list[tuple[int, int, int]] = []  # (bucket, pos, fp displaced)
        current_index = rng.choice((i1, i2))
        for _ in range(self.max_kicks):
            bucket = self.buckets[current_index]
            victim_pos = rng.randrange(len(bucket))
            victim_fp = bucket[victim_pos]
            bucket[victim_pos] = fingerprint
            journal.append((current_index, victim_pos, victim_fp))

            fingerprint = victim_fp
            current_index = self._get_alt_index(fingerprint, current_index)

            if self._insert_fingerprint(current_index, fingerprint):
                self._stored()
                return True

        # Failure: undo every swap, newest first.  This restores the
        # displaced fingerprints and drops the new item's fingerprint.
        for idx, pos, old in reversed(journal):
            self.buckets[idx][pos] = old
        self.n_failed += 1
        return False

    def contains(self, item: Key) -> bool:
        """Return True if *item* is probably present (possible false positive)."""
        fingerprint, i1, i2 = self._locate(item)
        return fingerprint in self.buckets[i1] or fingerprint in self.buckets[i2]

    def query(self, item: Key) -> bool:
        """Alias for :meth:`contains` (BloomFilter-compatible name)."""
        return self.contains(item)

    def remove(self, item: Key) -> bool:
        """Remove one occurrence of *item*. Returns True if a fingerprint was removed."""
        fingerprint, i1, i2 = self._locate(item)

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
        self.add(item)  # a rolled-back failure leaves the item unstored
        return False

    def update_bytes(self, key: bytes, reset_on_full: bool = False) -> bool:
        """Check-then-add for a raw ``bytes`` key.  Returns ``True`` if seen.

        Hot-path variant of :meth:`update` that takes the digest over the
        raw buffer rather than round-tripping through ``str``/``utf-8``,
        matching :meth:`fuzzer_tool.core.bloom.BloomFilter.update_bytes`.

        Args:
            key: Raw bytes to test and insert.
            reset_on_full: When ``n_added`` has reached ``capacity``, wipe
                the filter before inserting.  Keeps the realised
                false-positive rate bounded for unbounded streams at the
                cost of forgetting older keys -- a generational filter in
                one array, exactly like the bloom's ``reset_on_full``.
        """
        if reset_on_full and self.n_added >= self.capacity:
            self.clear()
        if self.contains(key):
            return True
        if not self.add(key) and reset_on_full:
            # Table/kick-chain full before n_added hit capacity (removals
            # or fingerprint collisions): start a new generation instead of
            # silently not tracking the key.
            self.clear()
            self.add(key)
        return False

    def clear(self) -> None:
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        self.n_added = 0
        self.n_failed = 0

    def __contains__(self, item: Key) -> bool:
        return self.contains(item)

    def __len__(self) -> int:
        return self.count

    @property
    def load_factor(self) -> float:
        total_slots = self.size * self.bucket_size
        return self.count / total_slots if total_slots else 0.0

    @property
    def expected_fpr(self) -> float:
        """Approximate false-positive rate at the current load."""
        return min(1.0, 2 * self.bucket_size * self.load_factor / (1 << self.fingerprint_size))
