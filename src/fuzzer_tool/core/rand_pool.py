"""Numpy-accelerated batched random number pool for the mutation hotpath.

Pre-generates a large batch of random uint32 values in a single vectorized
numpy call (C-level), then dispenses them via fast array indexing.

Key optimisations over ``random`` module:
- Pool generation: ``rng.integers`` fills 4096 entries in one C call.
- Pre-computed ``% 256``: ``integers(0, 255)`` reads a uint8 array — no modulo.
- ``shuffle`` / ``sample`` delegate to the Generator's C-level functions.
- Inlined ``choice`` avoids method call indirection.

Opt-in Floyd sampling (``--rand-floyd-sample`` / ``FUZZER_RAND_FLOYD=1``):
  When enabled, the ``k >= 3`` branch of ``sample()`` uses Floyd's algorithm
  (exactly *k* draws from ``_draw()``, O(k) space, no numpy round-trip).
  The k==1 and k==2 fast paths are never touched so seeded runs remain
  byte-identical when the flag is off.

Modulo bias is acceptable for the hot-path ``randrange`` / ``randint`` —
we are generating test inputs, not cryptographic keys.  When exact
uniformity matters (weighted-choice tables, distribution tests) use
:meth:`RandPool.random_between` instead; it rejection-samples and is
bias-free.  The pool is not thread-safe.

Determinism
-----------
This module never reads system or hardware entropy (``os.urandom``, CPU
``RDRAND``, ``/dev/urandom``, etc.), directly or indirectly. A ``RandPool``
built with ``seed=None`` does NOT fall through to OS entropy the way
``np.random.default_rng(None)`` normally would -- it is seeded from
``_NO_SEED_FALLBACK`` instead, so *every* pool, seeded or not, produces a
byte-for-byte reproducible stream. This applies transitively to
:func:`get_default_rand_pool`.

Full determinism across the fuzzer also requires there being exactly one
pool in play rather than many independently-constructed ones (even if each
of those were individually deterministic, two schedulers each drawing from
their own fresh, uncoordinated stream is a different, non-reproducible
interleaving from run to run relative to everything else touching the
pool). :func:`get_default_rand_pool` is the process-wide singleton every
call site that does not receive an explicit ``rng``/``seed`` from its
caller should fall back to -- see its docstring.
"""

from __future__ import annotations

import bisect
import itertools
import math
import os

import numpy as np

_POOL_ENTRIES = 4096  # refill every 4K draws

# ``draw % n`` over-weights the lowest ``2**32 % n`` values by one count in
# ``2**32 // n``: <= 1e-7 relative bias for n <= 2**20, 2:1 near
# n = 2/3 * 2**32.  Above this bound the range APIs rejection-sample
# (_randbelow) instead.
_MODULO_SAFE_MAX = 1 << 30

# Fixed replacement for "seed from OS/hardware entropy". Arbitrary but
# constant -- what matters is that it never changes between runs, not what
# it is. Used any time a caller passes seed=None instead of an int.
_NO_SEED_FALLBACK = 0x5EED_5EED

# SeedSequence tags: keep derived streams disjoint from plain-int seeds.
# Negative seed s → entropy [-s, _NEG_SEED_TAG] (numpy rejects s < 0).
# Default pool → entropy [seed, _DEFAULT_POOL_TAG], so it never mirrors a
# RandPool(seed) built from the same campaign seed.
_NEG_SEED_TAG = 0x4E45_4721
_DEFAULT_POOL_TAG = 0xDEFA_0172

# Entropy-gate retries per refill before raising starvation.
_GATE_RETRIES = 3

# Opt-in Floyd path (Phase 1 / B1 of the seventeen-source survey).
_RAND_FLOYD: bool = os.environ.get("FUZZER_RAND_FLOYD", "").strip() in ("1", "true", "yes")


def configure_rand_floyd(enabled: bool = False) -> None:
    """Enable/disable Floyd sampling for ``RandPool.sample`` (k >= 3)."""
    global _RAND_FLOYD
    _RAND_FLOYD = bool(enabled)


def rand_floyd_enabled() -> bool:
    """Return whether Floyd sampling is currently active."""
    return _RAND_FLOYD


def _seed_source(seed):
    """Map a user seed to something ``np.random.default_rng`` accepts.

    ``None`` → fixed fallback; negative int → tagged ``SeedSequence``;
    anything else passes through, so existing seeded streams are unchanged.
    """
    if seed is None:
        return _NO_SEED_FALLBACK
    if isinstance(seed, int) and seed < 0:
        return np.random.SeedSequence([-seed, _NEG_SEED_TAG])
    return seed


def _default_source(seed) -> np.random.SeedSequence:
    """Tagged seed for the default pool: never mirrors ``RandPool(seed)``."""
    base = _NO_SEED_FALLBACK if seed is None else seed
    entropy = [-base, _NEG_SEED_TAG] if base < 0 else [base]
    return np.random.SeedSequence([*entropy, _DEFAULT_POOL_TAG])


def _min_entropy(data: bytes) -> float:
    """Empirical min-entropy H_inf = -log2(max_x freq(x)) in bits/byte."""
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    return float(-np.log2(counts.max() / len(data)))


def _shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy in bits/byte from a byte string."""
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    c = counts[counts > 0] / len(data)
    return float(-np.sum(c * np.log2(c)))


class RandPool:
    """Pre-fetched pool of random integers (numpy-backed).

    Usage::

        pool = RandPool()
        idx = pool.randrange(len(buf))    # like random.randrange(n)
        val = pool.randint(0, 255)         # like random.randint(a, b)
        pick = pool.choice(seq)            # like random.choice(seq)
        # bias-free inclusive range (rejection sampling):
        x = pool.random_between(0, 9)
    """

    __slots__ = (
        "_pool",
        "_idx",
        "_m256",
        "_pool_l",
        "_m256_l",
        "_rng",
        "_min_entropy",
        "_min_hinf",
    )

    def __init__(
        self,
        seed=None,
        min_entropy: float | None = None,
        min_hinf: float | None = None,
    ) -> None:
        self._rng = np.random.default_rng(_seed_source(seed))
        self._pool: np.ndarray = np.empty(_POOL_ENTRIES, dtype=np.uint32)
        self._m256: np.ndarray = np.empty(_POOL_ENTRIES, dtype=np.uint8)  # pre-computed % 256
        # Python-list mirrors of the pools. numpy is the right representation
        # for bulk slicing (randbytes/randint_list), but scalar element access
        # (`int(arr[i])`) is ~2x slower than a plain list because each read
        # materializes a numpy scalar object. The scalar-draw methods
        # (randint/randrange/choice/_draw) are by far the hottest callers —
        # ~1.6M calls in a 15k-exec run — so they read from these lists
        # instead. The tolist() cost is paid once per refill and amortizes
        # over _POOL_ENTRIES draws.
        self._pool_l: list[int] = []
        self._m256_l: list[int] = []
        self._idx = _POOL_ENTRIES
        self._min_entropy = min_entropy
        # Min-entropy (not Shannon) gate. A uniform 4096-byte batch reads
        # H_inf ~7.2 bits/byte, but 2.6% of batches read < 7.0 and 25%
        # < 7.1 (seed 3, 2000 batches): keep thresholds <= 6.5.
        self._min_hinf = min_hinf

    def _generate_pool(self) -> None:
        """Fill ``_pool`` and ``_m256`` from the current generator."""
        self._pool[:] = self._rng.integers(0, 2**32, size=_POOL_ENTRIES, dtype=np.uint32)
        # Unsafe uint32 → uint8 cast truncates, i.e. % 256 (~16x cheaper than np.mod).
        np.copyto(self._m256, self._pool, casting="unsafe")

    def _refill(self) -> None:
        self._generate_pool()

        # Entropy only when gated: ~9% of a refill, read by nobody otherwise.
        # Every gate is rechecked after any retry: a batch drawn to satisfy
        # one gate can fail the other. The retry seed comes from the live
        # stream (not the fixed fallback, which would retry the same seed),
        # so the sequence stays a pure function of the original seed.
        retries = 0
        while (starved := self._gate_failure()) is not None:
            if retries == _GATE_RETRIES:
                raise RuntimeError(starved)
            retries += 1
            self.reseed(int(self._rng.integers(0, 2**63)))
            self._generate_pool()

        self._pool_l = self._pool.tolist()
        self._m256_l = self._m256.tolist()
        self._idx = 0

    def _gate_failure(self) -> str | None:
        """Starvation message for the first failing entropy gate, else None."""
        if self._min_hinf is not None:
            hinf = self._hinf()
            if hinf < self._min_hinf:
                return f"RandPool min-entropy starvation: {hinf:.2f} < {self._min_hinf:.2f}"
        if self._min_entropy is not None:
            h = self._entropy()
            if h < self._min_entropy:
                return f"RandPool entropy starvation: {h:.2f} < {self._min_entropy:.2f}"
        return None

    def pool_min_entropy_now(self) -> float:
        """Empirical min-entropy (bits/byte) of the current ``_m256`` batch.

        Raises ``RuntimeError`` before the first refill (or after a reseed):
        ``_m256`` then holds uninitialized memory or the old stream.
        """
        if not self._pool_l:
            raise RuntimeError("RandPool has not been refilled")
        return self._hinf()

    def _hinf(self) -> float:
        """Min-entropy (bits/byte) of the current ``_m256`` batch."""
        return _min_entropy(self._m256.tobytes())

    def _entropy(self) -> float:
        """Shannon entropy (bits/byte) of the current ``_m256`` batch."""
        return _shannon_entropy(self._m256.tobytes())

    def _draw(self) -> int:
        if self._idx >= _POOL_ENTRIES:
            self._refill()
        val = self._pool_l[self._idx]
        self._idx += 1
        return val

    # ── Public API ────────────────────────────────────────────────────

    def reseed(self, seed: int | np.random.SeedSequence | None = None) -> None:
        """Replace the backing generator and discard the pre-fetched pool.

        Args:
            seed: New seed for the pool's private ``Generator``. ``None``
                seeds from ``_NO_SEED_FALLBACK`` (a fixed constant), not
                OS/hardware entropy -- see the module docstring.

        Replacing the generator is not enough on its own: the pool still holds
        up to ``_POOL_ENTRIES`` values already drawn from the *old* stream,
        so the next ~4096 draws would keep coming from it and the reseed
        would look like a no-op.  Invalidating ``_idx`` forces a refill from
        the new stream on the next draw. Clearing ``_pool_l`` marks the old
        batch stale for the entropy readers.
        """
        self._rng = np.random.default_rng(_seed_source(seed))
        self._idx = _POOL_ENTRIES
        self._pool_l = []

    def inject_entropy(self, raw: bytes) -> None:
        """Mix externally supplied bytes into this pool's stream.

        Wired at exactly one call site:
        ``CorpusManager.save_to_corpus`` calls this with a seed's bytes
        once the seen_hashes/bloom novelty gate has confirmed they are
        genuinely new. It was added deliberately unwired and the
        docstring was not updated when the call site landed, which is
        why it says so explicitly now — a caller passing a fuzzer-like
        object without ``_rng`` raises from inside a corpus save.

        This is a *mix-in*, not a reseed: ``raw`` is folded together with
        a snapshot of the pool's current (already-deterministic) state via
        a SHA-256 digest, so the result is a deterministic function of
        (prior state, raw) rather than of ``raw`` alone — the same bytes
        injected into two pools with different histories still diverge.
        Discards the current pre-fetched batch so the next draw reflects
        the mix immediately.

        The snapshot is taken from the underlying ``Generator``'s own
        ``bit_generator.state`` rather than from ``self._pool`` — the
        pre-fetched batch buffer is allocated with ``np.empty`` and holds
        uninitialized memory until the first refill actually happens, so
        reading it before that point would fold in non-reproducible
        garbage and defeat the very determinism this method is designed
        to preserve. ``bit_generator.state`` is well-defined immediately
        after construction, for any seed.

        Args:
            raw: Entropy bytes to mix in. A falsy/empty value is a no-op.
        """
        if not raw:
            return
        import hashlib

        state_bytes = repr(self._rng.bit_generator.state).encode()
        digest = hashlib.sha256(state_bytes + bytes(raw)).digest()
        seed_words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, len(digest), 4)]
        self._rng = np.random.default_rng(np.random.SeedSequence(seed_words))
        self._idx = _POOL_ENTRIES
        self._pool_l = []

    def pool_entropy(self) -> float | None:
        """Return the Shannon entropy (bits/byte) of the most recent refill.

        Returns ``None`` until the first refill after construction or a
        reseed. Computed on demand from the batch still held in ``_m256``.
        """
        if not self._pool_l:
            return None
        return self._entropy()

    def measure_entropy(self, data: bytes | None = None) -> float:
        """Measure Shannon entropy (bits/byte) of bytes or the current pool."""
        if data is None:
            if not self._pool_l:
                raise RuntimeError("RandPool has not been refilled")
            data = self._m256.tobytes()
        return _shannon_entropy(data)

    def randrange_list(self, n: int, count: int) -> list[int]:
        """Return *count* random integers in [0, *n*).  Vectorized.

        Equivalent to calling ``randrange(n)`` *count* times but all
        values are sliced from the pool in one C-level operation when
        ``n <= _MODULO_SAFE_MAX``.  Wider ranges fall back to the scalar
        rejection path so the full span is reachable without bias.
        """
        if n <= 0 or count <= 0:
            return []
        if n > _MODULO_SAFE_MAX:
            return [self.randrange(n) for _ in range(count)]
        # _take spans refills; a bare slice truncated for count > _POOL_ENTRIES
        # (and still advanced _idx by the full count).
        return (self._take(count) % n).tolist()

    def random(self) -> float:
        """Return a random float in [0.0, 1.0).  ``random.random()`` equivalent."""
        return self._draw() / 4294967296.0  # 2^32

    def _take(self, count: int) -> np.ndarray:
        """Return exactly *count* pool words, spanning refills when needed.

        A slice past the end of ``self._pool`` truncates rather than raising,
        so a bare ``self._pool[idx : idx + count]`` returned short whenever
        *count* exceeded the pool while ``_idx`` advanced by the full amount.

        Counts that fit in one pool keep the original refill-then-slice
        sequence exactly, so existing seeded runs are unchanged; only the
        previously-broken oversized path is new.
        """
        if count <= _POOL_ENTRIES:
            if self._idx + count > _POOL_ENTRIES:
                self._refill()
            out = self._pool[self._idx : self._idx + count]
            self._idx += count
            return out

        parts = []
        while count > 0:
            if self._idx >= _POOL_ENTRIES:
                self._refill()
            take = min(count, _POOL_ENTRIES - self._idx)
            parts.append(self._pool[self._idx : self._idx + take].copy())
            self._idx += take
            count -= take
        return np.concatenate(parts)

    def _random53(self) -> float:
        """Float in [0.0, 1.0) with 53-bit resolution (CPython's formula).

        27 bits of one word, 26 of the next: two pool draws. ``random()``
        keeps one word (2**-32 resolution) for the hot path.
        """
        hi = self._draw() >> 5
        lo = self._draw() >> 6
        return (hi * 67108864 + lo) / 9007199254740992.0  # 2**26, 2**53

    def random_sequential(self, count: int) -> list[float]:
        """*count* floats, exactly as *count* ``random()`` calls would draw them.

        ``random_list`` refills early when a batch would cross the pool end;
        this consumes the tail first, so callers replacing a ``random()`` loop
        keep a seeded run's stream unchanged.
        """
        out: list[int] = []
        while count > 0:
            if self._idx >= _POOL_ENTRIES:
                self._refill()
            take = min(count, _POOL_ENTRIES - self._idx)
            out.extend(self._pool_l[self._idx : self._idx + take])
            self._idx += take
            count -= take
        return [v / 4294967296.0 for v in out]

    def random_list(self, count: int) -> list[float]:
        """Return *count* random floats in [0.0, 1.0).  Vectorized.

        Values span pool refills when *count* exceeds one pool.
        """
        if count <= 0:
            return []
        return (self._take(count).astype(np.float64) / 4294967296.0).tolist()

    def randint_list(self, a: int, b: int, count: int) -> list[int]:
        """Generate *count* random integers in [a, b] using vectorized numpy.

        This is faster than calling ``randint(a, b)`` *count* times because
        all values are sliced from the pool in a single C-level operation
        when the width is at most ``_MODULO_SAFE_MAX``.  Wider ranges fall
        back to the scalar rejection path so the full span is reachable.
        """
        width = b - a + 1
        if width <= 0 or count <= 0:
            return []
        if width > _MODULO_SAFE_MAX:
            return [self.randint(a, b) for _ in range(count)]
        # Fast path: pre-computed % 256 array, only while the request fits
        # in the current pool -- the spanning path has no matching _m256 run.
        if width == 256 and count <= _POOL_ENTRIES:
            if self._idx + count > _POOL_ENTRIES:
                self._refill()
            result = self._m256[self._idx : self._idx + count]
            self._idx += count
            if a == 0:
                return result.tolist()
            return (result.astype(np.int64) + a).tolist()
        result = self._take(count) % width
        if a == 0:
            # Already the answer: skip the cast and add (~2.1 -> 1.1 us at 64).
            return result.tolist()
        # int64 before the offset: the pool is uint32, and numpy rejects a
        # negative Python int against an unsigned array outright.
        return (result.astype(np.int64) + a).tolist()

    def randrange(self, n: int) -> int:
        """Return a random integer in ``[0, n)``.

        For ``n <= _MODULO_SAFE_MAX`` uses a single pool word (modulo bias
        is accepted for the hot path).  For larger ``n`` rejection-samples
        via :meth:`_randbelow` so the full range is reachable — required by
        callers such as the Ogg granule-position mutator that request a
        64-bit span.

        Raises ``ValueError`` for ``n <= 0``, as ``random.randrange`` does
        (it returned 0, hiding callers' empty-range bugs).
        """
        if n <= 0:
            raise ValueError(f"empty range for randrange({n})")
        if n <= _MODULO_SAFE_MAX:
            return self._draw() % n
        return self._randbelow(n)

    def randint(self, a: int, b: int) -> int:
        """Return a random integer in ``[a, b]`` (inclusive).

        Widths up to ``_MODULO_SAFE_MAX`` stay on the single-word fast
        path.  Wider ranges (e.g. ``randint(0, 0xFFFFFFFFFFFFFFFF)``)
        rejection-sample via :meth:`_randbelow` so every value of the
        requested range is reachable without bias.

        Raises ``ValueError`` for ``b < a``, as ``random.randint`` does
        (it returned *a*).
        """
        width = b - a + 1
        if width <= 0:
            raise ValueError(f"empty range for randint({a}, {b})")
        if width <= _MODULO_SAFE_MAX:
            if self._idx >= _POOL_ENTRIES:
                self._refill()
            pos = self._idx
            self._idx += 1
            # Fast path: pre-computed % 256 — avoids modulo at draw time
            if width == 256:
                return a + self._m256_l[pos]
            return a + (self._pool_l[pos] % width)
        return a + self._randbelow(width)

    def _randbelow(self, n: int) -> int:
        """Return a uniform integer in ``[0, n)`` with no modulo bias.

        Uses rejection sampling on the 32-bit pool draws (same algorithm as
        CPython's ``Random._randbelow_with_getrandbits``).  For power-of-two
        ``n`` that fit in 32 bits the mask path is bias-free with a single
        draw; wider ranges (including power-of-two spans such as ``2**64``)
        compose successive 32-bit draws so every bit is reachable.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        k = (n - 1).bit_length()
        # Fast path: power-of-two that fits in one pool word.
        if k <= 32 and (n & (n - 1)) == 0:
            return self._draw() & (n - 1)
        mask = (1 << k) - 1
        if k <= 32:
            while True:
                r = self._draw() & mask
                if r < n:
                    return r
        # >32-bit range: compose from successive 32-bit draws.
        while True:
            r = 0
            bits = 0
            while bits < k:
                r = (r << 32) | self._draw()
                bits += 32
            r &= mask
            if r < n:
                return r

    def random_between(self, low: int, high: int) -> int:
        """Return a uniform random integer in ``[low, high]`` (inclusive).

        Unlike :meth:`randint` / :meth:`randrange`, this uses rejection
        sampling so the distribution is exactly uniform even when the
        range width does not divide the generator's period.  Slightly
        slower; prefer it when constructing weighted-choice tables or
        any place where the classic ``r % n`` bias would matter (see
        the ERSC / LWN notes on modulo bias).

        Raises ``ValueError`` if ``high < low``.
        """
        if high < low:
            raise ValueError(f"empty range: high ({high}) < low ({low})")
        width = high - low + 1
        if width == 1:
            return low
        return low + self._randbelow(width)

    def randbytes(self, n: int) -> bytes:
        """Generate *n* random bytes. Matches ``random.randbytes`` API.

        Vectorized: slices directly from the pre-computed uint8 pool array,
        refilling as needed. ~10-50x faster than the per-byte Python loop.
        """
        if n <= 0:
            return b""
        remaining = _POOL_ENTRIES - self._idx
        if n <= remaining:
            # Fast path: all bytes available in current pool
            result = self._m256[self._idx : self._idx + n].tobytes()
            self._idx += n
            return result
        # Slow path: need to refill — collect chunks
        parts = []
        # Drain current pool
        if remaining > 0:
            parts.append(self._m256[self._idx :].tobytes())
            n -= remaining
        # Fill full pools directly
        while n >= _POOL_ENTRIES:
            self._refill()
            parts.append(self._m256.tobytes())
            n -= _POOL_ENTRIES
        # Fill remaining
        if n > 0:
            self._refill()
            parts.append(self._m256[:n].tobytes())
            self._idx = n
        else:
            # Last full pool is spent; rewinding to 0 replayed it.
            self._idx = _POOL_ENTRIES
        return b"".join(parts)

    def choice(self, seq: list | tuple | bytes) -> object:
        n = len(seq)
        if n == 0:
            raise IndexError("cannot choose from empty sequence")
        if n <= 256:
            if self._idx >= _POOL_ENTRIES:
                self._refill()
            val = self._pool_l[self._idx]
            self._idx += 1
            return seq[val % n]
        # Modulo bias ≤ n / 2**32: negligible for any in-memory sequence.
        return seq[self.randrange(n)]

    def choice_list(self, seq: list | tuple | bytes, count: int) -> list:
        """Return *count* elements randomly chosen from *seq* (with replacement).

        Uses vectorized :meth:`randint_list` for index generation (one numpy
        slice), then a single list comprehension for lookups.  Faster than
        calling :meth:`choice` *count* times when *count* > 1 because the
        indices are generated in one C-level operation.
        """
        n = len(seq)
        if n == 0:
            raise IndexError("cannot choose from empty sequence")
        if count <= 0:
            return []
        indices = self.randint_list(0, n - 1, count)
        return [seq[i] for i in indices]

    def weighted_choice(self, seq: list | tuple, weights: list[float]) -> object:
        """Return one element from *seq* chosen proportional to *weights*.

        ``random.choices(seq, weights=weights)[0]`` equivalent.
        """
        n = len(seq)
        if n == 0:
            raise IndexError("cannot choose from empty sequence")
        if len(weights) != n:
            raise ValueError(f"weighted_choice: {len(weights)} weights for {n} items")
        cum = list(itertools.accumulate(weights))
        total = cum[-1]
        if total <= 0:
            # All weights zero (or net non-positive) is a caller bug, not a
            # rounding artifact -- keep raising rather than silently picking
            # an arbitrary element.
            raise IndexError("weighted_choice: weights sum to <= 0")
        # A negative weight makes ``cum`` non-monotone: bisect then returns
        # garbage ([5, -4, 1] picked 'a' and 'c' 50/50).
        if min(weights) < 0:
            raise ValueError("weighted_choice: negative weight")
        r = self.random() * total
        # ``r`` is mathematically < total (self.random() is in [0.0, 1.0)),
        # but float64 rounding in the multiplication can push it up to
        # exactly total (more likely as total grows, e.g. accumulating many
        # small positive weights), which would make bisect_right return n --
        # one past the end of seq. Clamp the search's upper bound to n - 1,
        # mirroring how CPython's random.choices() avoids the same edge case.
        return seq[bisect.bisect_right(cum, r, 0, n - 1)]

    def weighted_choice_list(self, seq: list | tuple, weights: list[float], k: int) -> list:
        """Return *k* elements from *seq* chosen proportional to *weights*.

        ``random.choices(seq, weights=weights, k=k)`` equivalent.
        Vectorized via numpy for the index generation.
        """
        n = len(seq)
        if n == 0:
            raise IndexError("cannot choose from empty sequence")
        if k <= 0:
            return []
        total = sum(weights)
        if total <= 0:
            raise IndexError("weighted_choice_list: weights sum to <= 0")
        probs = [w / total for w in weights]
        indices = self._rng.choice(n, size=k, replace=True, p=probs)
        return [seq[i] for i in indices]

    def shuffle(self, seq: list) -> None:
        n = len(seq)
        if n < 8:
            for i in range(n - 1, 0, -1):
                if self._idx >= _POOL_ENTRIES:
                    self._refill()
                j = self._pool_l[self._idx] % (i + 1)
                self._idx += 1
                seq[i], seq[j] = seq[j], seq[i]
        else:
            self._rng.shuffle(seq)

    def _floyd_indices(self, n: int, k: int) -> list[int]:
        """Floyd's sampling: exactly *k* unique indices in ``[0, n)``.

        Draws only from the pool's own stream (``randrange``: one word while
        ``n <= _MODULO_SAFE_MAX``, rejection sampling above that).  O(k)
        space, exactly k draws for ``n <= _MODULO_SAFE_MAX``.

        Permutation variant (Bentley & Floyd 1987): a fresh ``t`` goes to the
        front, a collision inserts ``j`` right after ``t``. The output order
        is then a uniform random permutation; ``list(set)`` came out
        near-sorted for small ints. Linked list via ``nxt`` keeps it O(k).
        """
        nxt: dict[int, int] = {}
        head = -1
        for j in range(n - k, n):
            t = self.randrange(j + 1)
            if t in nxt:
                nxt[j] = nxt[t]
                nxt[t] = j
                continue
            nxt[t] = head
            head = t

        out: list[int] = []
        while head != -1:
            out.append(head)
            head = nxt[head]
        return out

    def sample(self, population, k: int):
        """Return *k* unique elements from *population*.

        Accepts either an int (range size) or a sequence (list/tuple/bytes/
        bytearray/range), matching the ``random.sample`` API.

        ``range`` has to be in the sequence tuple, not just tolerated: it is
        the idiomatic population for ``random.sample`` and a dozen mutators
        call ``rng.sample(range(len(x)), 2)``. Without it a range fell
        through to the int branch below and raised
        ``TypeError: '>' not supported between instances of 'int' and
        'range'`` on the first comparison. That only fires when the pooled
        RNG is in use rather than stdlib ``random``, which made it look like
        a seed-dependent test flake instead of what it is: an exception
        raised mid-campaign from an ordinary call.

        When ``--rand-floyd-sample`` is enabled the ``k >= 3`` branch uses
        Floyd's algorithm (exactly k ``_draw()`` calls). The k==1 / k==2
        fast paths are never altered so seeded runs stay identical when
        the flag is off.
        """
        if isinstance(population, list | tuple | bytes | bytearray | range):
            n = len(population)
            if k > n:
                k = n
            if k <= 0:
                return []
            # Above _MODULO_SAFE_MAX, randrange rejection-samples so every
            # index is reachable without bias; below it, inline the draw.
            wide = n > _MODULO_SAFE_MAX
            if k == 1:
                return [population[self.randrange(n) if wide else self._draw() % n]]
            if k == 2:
                if wide:
                    a = self.randrange(n)
                    b = self.randrange(n - 1)
                else:
                    a = self._draw() % n
                    b = self._draw() % (n - 1)
                if b >= a:
                    b += 1
                return [population[a], population[b]]
            if _RAND_FLOYD:
                indices = self._floyd_indices(n, k)
            else:
                indices = list(self._rng.choice(n, size=k, replace=False))
            return [population[i] for i in indices]
        # Original: population is an int (range size)
        if k > population:
            k = population
        if k <= 0:
            return []
        if k == 1:
            return [self.randrange(population)]
        if k == 2:
            a = self.randrange(population)
            b = self.randrange(population - 1)
            return [a, b if b < a else b + 1]
        if _RAND_FLOYD:
            return self._floyd_indices(population, k)
        return self._rng.choice(population, size=k, replace=False).tolist()

    # ── Continuous distributions ──────────────────────────────────────
    # These delegate to numpy's C-level generators (no pool draws).

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Return a random float from Gaussian(*mu*, *sigma*)."""
        return float(self._rng.normal(mu, sigma))

    def gauss_list(self, mu: float, sigma: float, count: int) -> list[float]:
        """Return *count* random floats from Gaussian(*mu*, *sigma*).

        Vectorized: all values generated in one C-level numpy call.
        """
        if count <= 0:
            return []
        return self._rng.normal(mu, sigma, size=count).tolist()

    def expovariate(self, lambd: float = 1.0) -> float:
        """Return a random float from Exponential(rate=*lambd*).

        Like ``random.expovariate(lambd)`` for *lambd* > 0. Unlike CPython
        (which raises on 0 and goes negative below it), *lambd* <= 0
        returns ``float('inf')``.
        """
        if lambd <= 0:
            return float("inf")
        return float(self._rng.exponential(1.0 / lambd))

    def expovariate_list(self, lambd: float, count: int) -> list[float]:
        """Return *count* random floats from Exponential(rate=*lambd*).

        Vectorized: all values generated in one C-level numpy call.
        """
        if count <= 0:
            return []
        if lambd <= 0:
            return [float("inf")] * count
        return self._rng.exponential(1.0 / lambd, size=count).tolist()

    def betavariate(self, alpha: float, beta: float) -> float:
        """Return a random float from Beta(*alpha*, *beta*).

        ``random.betavariate(alpha, beta)`` equivalent.
        """
        return float(self._rng.beta(alpha, beta))

    def betavariate_list(self, alpha: float, beta: float, count: int) -> list[float]:
        """Return *count* random floats from Beta(*alpha*, *beta*).

        Vectorized: all values generated in one C-level numpy call.
        """
        if count <= 0:
            return []
        return self._rng.beta(alpha, beta, size=count).tolist()

    def betavariate_array(self, alphas, betas) -> np.ndarray:
        """Return one Beta(alphas[i], betas[i]) draw per element, as an array.

        Unlike :meth:`betavariate_list` (one distribution, *count* draws),
        the parameters vary per element: a Thompson-sampling step over K
        arms in one C-level numpy call rather than K Python-level calls.
        """
        return self._rng.beta(alphas, betas)

    def binomial_array(self, counts, p: float) -> np.ndarray:
        """Return one Binomial(counts[i], *p*) draw per element, as an array.

        One C-level numpy call: the per-arm pseudo-reward draw of
        perturbed-history exploration over K arms.
        """
        return self._rng.binomial(counts, p)

    def poisson_array(self, lams) -> np.ndarray:
        """Return one Poisson(lams[i]) draw per element, as an array.

        One C-level numpy call (no pool words consumed); keeps the shape of
        *lams*. Negative rates raise ``ValueError``.
        """
        return self._rng.poisson(lams)

    def geometric(self, p: float) -> int:
        """Return the number of failures before the first success.

        Inversion: ``floor(log(V) / log(1-p))`` with ``V = 1 - U`` in
        ``(0, 1]``, so ``U = 0`` never reaches ``log(0)``. ``U`` has 53 bits
        (two pool words): one 32-bit word capped the tail at
        ``log(2**-32) / log1p(-p)`` (~22x the mean at p=1e-6). ``log1p``
        keeps tiny *p* accurate. ``p == 1`` is certain and returns 0
        without drawing.

        Raises:
            ValueError: *p* outside ``(0, 1]`` (NaN included).
        """
        if not 0.0 < p <= 1.0:
            raise ValueError(f"geometric needs 0 < p <= 1, got {p}")
        if p == 1.0:
            return 0
        return math.floor(math.log(1.0 - self._random53()) / math.log1p(-p))

    def random_array(self, shape) -> np.ndarray:
        """Return uniform floats in [0.0, 1.0) as an array of *shape*.

        One C-level numpy call; float64 ndarray for vectorised masks
        (e.g. QEA collapse), skipping :meth:`random_list`'s list round-trip.
        """
        return self._rng.random(shape)

    def normal_array(self, shape) -> np.ndarray:
        """Return standard-normal floats as an array of *shape*.

        One C-level numpy call; :meth:`gauss_list`'s list round-trip costs
        ~3.5x at a few hundred values (spectral / correlated-Thompson noise).
        """
        return self._rng.standard_normal(shape)

    def dirichlet(self, alphas) -> np.ndarray:
        """Return one point on the simplex drawn from Dirichlet(*alphas*).

        One C-level numpy call; numpy's small-α path avoids the Gamma
        underflow (NaN) of naive normalisation at α ≲ 0.01.
        """
        a = np.asarray(alphas, dtype=np.float64)
        if a.size == 0 or not np.isfinite(a).all() or (a <= 0.0).any():
            raise ValueError("dirichlet needs a non-empty vector of finite alphas > 0")
        return self._rng.dirichlet(a)

    def categorical(self, probs: np.ndarray, count: int) -> list[int]:
        """Return *count* indices drawn from the (unnormalised) weights *probs*.

        Inverse-CDF over one cumsum: ~2x cheaper than
        :meth:`weighted_choice_list` on a numpy vector (no list round-trip).
        """
        if count <= 0:
            return []
        if len(probs) == 0:
            raise IndexError("categorical: empty weights")
        cum = np.cumsum(probs)
        if cum[-1] <= 0:
            raise IndexError("categorical: weights sum to <= 0")
        idx = np.searchsorted(cum, self._rng.random(count) * cum[-1], side="right")
        return np.minimum(idx, len(cum) - 1).tolist()

    def gammavariate(self, alpha: float, beta: float = 1.0) -> float:
        """Return a random float from Gamma(*alpha*, *beta*).

        NOT ``random.gammavariate``: *beta* here is the rate (mean
        ``alpha / beta``); CPython's is the scale (mean ``alpha * beta``).
        Matches :meth:`gammavariate_array`. *beta* <= 0 returns 0.0.
        """
        return float(self._rng.gamma(alpha, scale=1.0 / beta) if beta > 0 else 0.0)

    def gammavariate_list(self, alpha: float, beta: float, count: int) -> list[float]:
        """Return *count* random floats from Gamma(*alpha*, *beta*).

        Vectorized: all values generated in one C-level numpy call.
        """
        if count <= 0:
            return []
        draws = (
            self._rng.gamma(alpha, scale=1.0 / beta, size=count) if beta > 0 else np.zeros(count)
        )
        return draws.tolist()

    def gammavariate_array(self, alphas, betas) -> np.ndarray:
        """Return one Gamma(alphas[i], rate=betas[i]) draw per element, as an array.

        The Gamma counterpart of :meth:`betavariate_array`: a Thompson step
        over K Gamma-Poisson arms in one C-level numpy call. Rate <= 0
        gives 0.0, as :meth:`gammavariate` does (was ``inf`` / negative).
        """
        draws = self._rng.gamma(alphas)
        rates = np.asarray(betas, dtype=np.float64)
        # argmin+index beats .min() (350 vs 530 ns); empty arrays have no argmin.
        if rates.size == 0 or rates.flat[rates.argmin()] > 0:
            return draws / rates

        draws, rates = np.broadcast_arrays(draws, rates)
        return np.divide(draws, rates, out=np.zeros(draws.shape), where=rates > 0)

    def lognormvariate(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Return a random float from LogNormal(*mu*, *sigma*).

        ``random.lognormvariate(mu, sigma)`` equivalent.
        """
        return float(self._rng.lognormal(mu, sigma))

    def lognormvariate_list(self, mu: float, sigma: float, count: int) -> list[float]:
        """Return *count* random floats from LogNormal(*mu*, *sigma*).

        Vectorized: all values generated in one C-level numpy call.
        """
        if count <= 0:
            return []
        return self._rng.lognormal(mu, sigma, size=count).tolist()


# ── Process-wide singleton ───────────────────────────────────────────────
#
# Every call site in src/ that would otherwise construct its own fallback
# ``RandPool()`` (because no ``rng``/``seed`` was threaded down to it) must
# instead fetch this one shared instance. That is what makes "no explicit
# seed anywhere in the call chain" behave like a single coordinated PRNG
# rather than N independently-deterministic-but-uncoordinated ones: the
# latter is still non-reproducible in aggregate, because how many draws
# each fallback instance has consumed relative to the others depends on
# scheduling order, not on the campaign seed.
#
# ``Fuzzer.__init__`` still builds and owns its *own* ``RandPool(seed=seed)``
# for the campaign seed the person actually passed via ``--seed`` — that
# instance is threaded explicitly to every scheduler/mutator it constructs,
# same as before. This singleton only exists to catch the fallback path:
# anything that wasn't handed a pool explicitly.
_default_instance: RandPool | None = None


def get_default_rand_pool(seed=None) -> RandPool:
    """Return the process-wide shared ``RandPool``, creating it if needed.

    ``seed`` only has an effect on the call that actually creates the
    singleton; once created, later calls (with or without a ``seed``)
    just return the existing instance unchanged. Silently reseeding an
    already-shared instance out from under everything already holding a
    reference to it would move their stream without their knowledge,
    which is its own source of non-determinism -- so seeding after the
    fact is a hard error via :func:`configure_default_rand_pool_seed`
    instead of a silent option here.
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = RandPool(seed=_default_source(seed))
    return _default_instance


def configure_default_rand_pool_seed(seed) -> None:
    """Seed the shared default pool before its first use.

    Raises ``RuntimeError`` if the singleton already exists — call this
    (if at all) before any code path can have triggered
    :func:`get_default_rand_pool`, e.g. at the very top of campaign
    startup, right where ``Fuzzer.__init__`` seeds its own pool.
    """
    global _default_instance
    if _default_instance is not None:
        raise RuntimeError(
            "default RandPool singleton already created; cannot reconfigure its seed"
        )
    _default_instance = RandPool(seed=_default_source(seed))


def reset_default_rand_pool(seed=None) -> None:
    """Force-replace the shared singleton.

    Test-only escape hatch: production code should never need to reset a
    process-wide singleton mid-run. Existing holders of the old instance
    keep drawing from it; only future ``get_default_rand_pool()`` callers
    see the replacement.
    """
    global _default_instance
    _default_instance = RandPool(seed=_default_source(seed))


def reseed_default_rand_pool(seed=None) -> None:
    """Reseed the shared default pool from a campaign seed.

    The stream is tagged (``_DEFAULT_POOL_TAG``) so it differs from
    ``RandPool(seed)``: reseeding both with the same seed made fallback
    draws replay the hotpath stream at an offset. Every default-pool
    constructor uses the same tag (:func:`_default_source`).
    """
    get_default_rand_pool().reseed(_default_source(seed))
