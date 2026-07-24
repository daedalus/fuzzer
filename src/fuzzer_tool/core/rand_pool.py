"""Numpy-accelerated batched random number pool for the mutation hotpath.

Pre-generates a large batch of random uint32 values in a single vectorized
numpy call (C-level), then dispenses them via fast array indexing.

Key optimisations over ``random`` module:
- Pool generation: ``np.random.randint`` fills 4096 entries in one C call.
- Pre-computed ``% 256``: ``randint(0, 255)`` reads a uint8 array — no modulo.
- ``shuffle`` / ``sample`` delegate to ``np.random`` C-level functions.
- Inlined ``choice`` avoids method call indirection.

Modulo bias is acceptable for fuzzing — we are generating test inputs, not
cryptographic keys.  The pool is not thread-safe.
"""

import numpy as np

_POOL_ENTRIES = 4096  # refill every 4K draws


class RandPool:
    """Pre-fetched pool of random integers (numpy-backed).

    Usage::

        pool = RandPool()
        idx = pool.randrange(len(buf))    # like random.randrange(n)
        val = pool.randint(0, 255)         # like random.randint(a, b)
        pick = pool.choice(seq)            # like random.choice(seq)
    """

    __slots__ = ("_pool", "_idx", "_m256")

    def __init__(self) -> None:
        self._pool: np.ndarray = np.empty(_POOL_ENTRIES, dtype=np.uint32)
        self._m256: np.ndarray = np.empty(_POOL_ENTRIES, dtype=np.uint8)  # pre-computed % 256
        self._idx = _POOL_ENTRIES

    def _refill(self) -> None:
        self._pool[:] = np.random.randint(0, 2**32, size=_POOL_ENTRIES, dtype=np.uint32)
        np.mod(self._pool, 256, out=self._m256)
        self._idx = 0

    def _draw(self) -> int:
        if self._idx >= _POOL_ENTRIES:
            self._refill()
        val = int(self._pool[self._idx])
        self._idx += 1
        return val

    # ── Public API ────────────────────────────────────────────────────

    def randrange_list(self, n: int, count: int) -> list[int]:
        """Return *count* random integers in [0, *n*).  Vectorized.

        Equivalent to calling ``randrange(n)`` *count* times but all
        values are sliced from the pool in one C-level operation.
        """
        if n <= 0 or count <= 0:
            return []
        if self._idx + count > _POOL_ENTRIES:
            self._refill()
        raw = self._pool[self._idx:self._idx + count]
        self._idx += count
        return [int(x % n) for x in raw]

    def random(self) -> float:
        """Return a random float in [0.0, 1.0).  ``random.random()`` equivalent."""
        return self._draw() / 4294967296.0  # 2^32

    def random_list(self, count: int) -> list[float]:
        """Return *count* random floats in [0.0, 1.0).  Vectorized.

        All values are generated from one pool slice.
        """
        if count <= 0:
            return []
        if self._idx + count > _POOL_ENTRIES:
            self._refill()
        raw = self._pool[self._idx:self._idx + count]
        self._idx += count
        return [int(x) / 4294967296.0 for x in raw]

    def randint_list(self, a: int, b: int, count: int) -> list[int]:
        """Generate *count* random integers in [a, b] using vectorized numpy.

        This is faster than calling ``randint(a, b)`` *count* times because
        all values are sliced from the pool in a single C-level operation.
        The list comprehension conversion to Python int is also C-level
        (CPython 3.12+).
        """
        width = b - a + 1
        if width <= 0 or count <= 0:
            return []
        # Ensure we have enough values in the pool
        if self._idx + count > _POOL_ENTRIES:
            self._refill()
        # Slice: vectorized C-level read of count values
        raw = self._pool[self._idx:self._idx + count]
        self._idx += count
        # Convert to Python ints (list comprehension, C-level iteration)
        return [int(a + (x % width)) for x in raw]

    def randrange(self, n: int) -> int:
        return self._draw() % n if n > 0 else 0

    def randint(self, a: int, b: int) -> int:
        width = b - a + 1
        if width <= 0:
            return a
        if self._idx >= _POOL_ENTRIES:
            self._refill()
        pos = self._idx
        self._idx += 1
        # Fast path: pre-computed % 256 — avoids modulo at draw time
        if width == 256:
            return int(self._m256[pos])
        return a + (int(self._pool[pos]) % width)

    def choice(self, seq: list | tuple | bytes) -> object:
        n = len(seq)
        if n == 0:
            raise IndexError("cannot choose from empty sequence")
        if n <= 256:
            if self._idx >= _POOL_ENTRIES:
                self._refill()
            val = int(self._m256[self._idx])
            self._idx += 1
            return seq[val % n]
        return seq[int(np.random.randint(n))]

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

    def shuffle(self, seq: list) -> None:
        n = len(seq)
        if n < 8:
            for i in range(n - 1, 0, -1):
                if self._idx >= _POOL_ENTRIES:
                    self._refill()
                j = int(self._pool[self._idx]) % (i + 1)
                self._idx += 1
                seq[i], seq[j] = seq[j], seq[i]
        else:
            np.random.shuffle(seq)

    def sample(self, population: int, k: int) -> list[int]:
        if k > population:
            k = population
        if k <= 0:
            return []
        if k == 1:
            return [self._draw() % population]
        if k == 2:
            a = self._draw() % population
            b = self._draw() % (population - 1)
            return [a, b if b < a else b + 1]
        return list(np.random.choice(population, size=k, replace=False))

    # ── Continuous distributions ──────────────────────────────────────
    # These delegate to numpy's C-level generators (no pool draws).

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Return a random float from Gaussian(*mu*, *sigma*)."""
        return float(np.random.normal(mu, sigma))

    def gauss_list(self, mu: float, sigma: float, count: int) -> list[float]:
        """Return *count* random floats from Gaussian(*mu*, *sigma*).

        Vectorized: all values generated in one C-level numpy call.
        """
        if count <= 0:
            return []
        return list(np.random.normal(mu, sigma, size=count))

    def expovariate(self, lambd: float = 1.0) -> float:
        """Return a random float from Exponential(rate=*lambd*).

        ``random.expovariate(lambd)`` equivalent.  For *lambd* == 0
        returns ``float('inf')`` (matches CPython behaviour).
        """
        if lambd <= 0:
            return float("inf")
        return float(np.random.exponential(1.0 / lambd))

    def expovariate_list(self, lambd: float, count: int) -> list[float]:
        """Return *count* random floats from Exponential(rate=*lambd*).

        Vectorized: all values generated in one C-level numpy call.
        """
        if count <= 0:
            return []
        if lambd <= 0:
            return [float("inf")] * count
        return list(np.random.exponential(1.0 / lambd, size=count))

    def betavariate(self, alpha: float, beta: float) -> float:
        """Return a random float from Beta(*alpha*, *beta*).

        ``random.betavariate(alpha, beta)`` equivalent.
        """
        return float(np.random.beta(alpha, beta))

    def betavariate_list(self, alpha: float, beta: float, count: int) -> list[float]:
        """Return *count* random floats from Beta(*alpha*, *beta*).

        Vectorized: all values generated in one C-level numpy call.
        """
        if count <= 0:
            return []
        return list(np.random.beta(alpha, beta, size=count))

    def gammavariate(self, alpha: float, beta: float = 1.0) -> float:
        """Return a random float from Gamma(*alpha*, *beta*).

        ``random.gammavariate(alpha, beta)`` equivalent — *beta* is the
        rate parameter (not scale).
        """
        return float(np.random.gamma(alpha, scale=1.0 / beta) if beta > 0 else 0.0)

    def gammavariate_list(self, alpha: float, beta: float, count: int) -> list[float]:
        """Return *count* random floats from Gamma(*alpha*, *beta*).

        Vectorized: all values generated in one C-level numpy call.
        """
        if count <= 0:
            return []
        return list(
            np.random.gamma(alpha, scale=1.0 / beta, size=count)
            if beta > 0 else np.zeros(count)
        )

    def lognormvariate(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Return a random float from LogNormal(*mu*, *sigma*).

        ``random.lognormvariate(mu, sigma)`` equivalent.
        """
        return float(np.random.lognormal(mu, sigma))

    def lognormvariate_list(self, mu: float, sigma: float, count: int) -> list[float]:
        """Return *count* random floats from LogNormal(*mu*, *sigma*).

        Vectorized: all values generated in one C-level numpy call.
        """
        if count <= 0:
            return []
        return list(np.random.lognormal(mu, sigma, size=count))
