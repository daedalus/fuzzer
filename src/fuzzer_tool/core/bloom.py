"""Bloom filter for probabilistic membership testing.

Includes fuzzy near-duplicate detection via Hamming distance on stored keys.
"""

import hashlib
import math
from collections import deque

from fuzzer_tool.core.similarity import hamming_distance

_M128 = (1 << 128) - 1


class BloomFilter:
    """Bloom filter backed by a :class:`bytearray`.

    One SHA-256 digest per key supplies all *k* bit positions.  The filter
    size *m* is rounded up to a power of two so that bitwise masking
    (``& (m-1)``) can replace modulo.  The rounding is a deliberate speed
    trade-off: it can hand out up to 2x the memory the requested
    ``error_rate`` needs, in which case the realised rate is *better* than
    requested (never worse).  See :attr:`expected_fpr` / :attr:`memory_bytes`.

    Position derivation:

    * **Sliced** (``k * log2(m) <= 256``): position *i* is the *i*-th
      ``log2(m)``-bit slice of the digest.  Fully independent probes.
    * **Double hashing** (otherwise): Kirsch-Mitzenmacher,
      ``pos_i = (h1 + i * h2) & mask`` with ``h1``/``h2`` the two 128-bit
      digest halves and ``h2`` forced odd (full period on a power-of-two
      table).  Same asymptotic false-positive rate as *k* independent
      hashes, so tight ``error_rate`` values (~1e-5 and below) are honoured
      instead of silently clamping *k* to what 256 bits can slice.

    Each position *p* is mapped to a byte in the backing array and a bit
    within that byte::

        byte_idx = p >> 3
        bit_idx  = p & 7

    Overfilling (``n_added > capacity``) only degrades the filter towards
    "always maybe"; it never produces a false negative.  Callers that pair
    it with an exact set (``adapters/filesystem.py``) stay correct, so the
    corpus filter deliberately has no reset: forgetting keys there would
    turn into false negatives.  Use ``update_bytes(reset_on_full=True)`` for
    a bounded generational filter.
    """

    #: Width of the single backing digest.  Sliced mode requires
    #: ``k * bits_per_slice <= DIGEST_BITS``; beyond that, double hashing.
    DIGEST_BITS = 256

    def __init__(self, capacity: int, error_rate: float = 0.01) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not (0.0 < error_rate < 1.0):
            raise ValueError("error_rate must be in the open interval (0, 1)")
        n = capacity
        self.capacity = n
        self.error_rate = error_rate
        m_ideal = -n * math.log(error_rate) / (math.log(2) ** 2)
        # Smallest power of two >= m_ideal.  ``int(x).bit_length()`` overshoots
        # by a full doubling whenever m_ideal is already a power of two.
        self.m = 1 << max(1, (max(int(m_ideal), 1) - 1).bit_length())
        self._mask = self.m - 1
        self._bits_per_slice = self.m.bit_length() - 1
        self._k_ideal = max(1, round(self.m / n * math.log(2)))
        # Independent slicing is only possible while all k slices fit in one
        # digest; otherwise fall back to double hashing rather than shrinking
        # k (which used to inflate the realised rate above the requested one).
        self._double = self._k_ideal * self._bits_per_slice > self.DIGEST_BITS
        self._k = self._k_ideal

        self._byte_len = (self.m + 7) // 8
        self._bits = bytearray(self._byte_len)
        self.n_added = 0
        self._recent_keys: deque[bytes] | None = None

    @property
    def digest_limited(self) -> bool:
        """True when *k* is too large to slice from one digest, so double
        hashing is in use.  (*k* itself is never reduced.)"""
        return self._double

    @property
    def memory_bytes(self) -> int:
        return self._byte_len

    @property
    def over_capacity(self) -> bool:
        """True once more than ``capacity`` keys have been absorbed."""
        return self.n_added > self.capacity

    @property
    def expected_fpr(self) -> float:
        """Estimated false-positive rate at the current ``n_added``."""
        return (1.0 - math.exp(-self._k * self.n_added / self.m)) ** self._k

    @staticmethod
    def _digest(key: str) -> int:
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")

    def _check(self, value: int) -> bool:
        bits = self._bits
        mask = self._mask
        if self._double:
            h = value & _M128
            step = (value >> 128) | 1
            for _ in range(self._k):
                pos = h & mask
                if not (bits[pos >> 3] & (1 << (pos & 7))):
                    return False
                h += step
            return True
        v = value
        shift = self._bits_per_slice
        for _ in range(self._k):
            pos = v & mask
            if not (bits[pos >> 3] & (1 << (pos & 7))):
                return False
            v >>= shift
        return True

    def _set(self, value: int) -> None:
        bits = self._bits
        mask = self._mask
        if self._double:
            h = value & _M128
            step = (value >> 128) | 1
            for _ in range(self._k):
                pos = h & mask
                bits[pos >> 3] |= 1 << (pos & 7)
                h += step
        else:
            v = value
            shift = self._bits_per_slice
            for _ in range(self._k):
                pos = v & mask
                bits[pos >> 3] |= 1 << (pos & 7)
                v >>= shift
        self.n_added += 1

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

    def update_bytes(self, key: bytes, reset_on_full: bool = False) -> bool:
        """Check-then-add for a raw ``bytes`` key.  Returns ``True`` if seen.

        Hot-path variant of :meth:`update`: the digest is taken over the raw
        buffer, skipping the ``.hex()`` round-trip (which doubles the payload
        and allocates) that :meth:`add_bytes` performs.

        Args:
            key: Raw bytes to test and insert.
            reset_on_full: When the filter has absorbed ``capacity`` keys, wipe
                it before inserting.  Keeps the realised false-positive rate at
                the configured bound for unbounded streams at the cost of
                forgetting older keys — a generational filter in one array.
        """
        if reset_on_full and self.n_added >= self.capacity:
            self.clear()
        value = int.from_bytes(hashlib.sha256(key).digest(), "big")
        if self._check(value):
            return True
        self._set(value)
        return False

    @property
    def load_factor(self) -> float:
        """Fraction of bits set to 1."""
        return int.from_bytes(self._bits, "little").bit_count() / self.m

    def clear(self) -> None:
        self._bits = bytearray(self._byte_len)
        self.n_added = 0
        if self._recent_keys is not None:
            self._recent_keys.clear()

    def add_bytes(self, key: bytes, max_hamming: int = 0) -> bool:
        """Add raw bytes, optionally rejecting near-duplicates by Hamming distance.

        Uses the same keyspace as :meth:`update_bytes` (SHA-256 of the raw
        bytes), so the two APIs agree on membership.  The str API
        (:meth:`add` / :meth:`query` / :meth:`update`) is a separate keyspace.

        With ``max_hamming > 0`` the key is also compared against the most
        recently added keys of the *same length* (up to the ``max_recent``
        given to :meth:`init_fuzzy`, default 200); the buffer is created on
        first use if :meth:`init_fuzzy` was not called.

        Returns:
            True if the key was an exact or near duplicate (NOT added),
            False if it was unique (added).
        """
        value = int.from_bytes(hashlib.sha256(key).digest(), "big")
        if self._check(value):
            return True  # exact match already in filter

        if max_hamming > 0:
            if self._recent_keys is None:
                self.init_fuzzy()
            klen = len(key)
            for recent in self._recent_keys:  # type: ignore[union-attr]
                if len(recent) == klen and hamming_distance(key, recent) <= max_hamming:
                    return True

        self._set(value)
        if self._recent_keys is not None:
            self._recent_keys.append(key)
        return False

    def init_fuzzy(self, max_recent: int = 200) -> None:
        """Initialize (or reset) the recent-keys buffer for fuzzy Hamming dedup.

        Args:
            max_recent: Maximum recent keys to track for Hamming comparison.
        """
        self._recent_keys = deque(maxlen=max_recent)
