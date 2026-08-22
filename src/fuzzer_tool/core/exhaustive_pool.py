"""A ``RandPool``-shaped generator that enumerates every path instead of sampling one.

Port item P1-5 from ``docs/tigerbeetle_four_fuzzers_port.md``, after
matklad's ``Gen`` in *A Tale Of Four Fuzzers*. The observation the whole
thing rests on is that ``core/rand_pool.py`` is already the abstraction
required: every discrete draw carries an explicit bound, and operators
receive the pool as a parameter rather than reaching for a module-level
``random``. Substituting a different object with the same method names
therefore turns "run this operator once with random draws" into "run it
once per reachable combination of draws", with no enumeration logic
written per operator.

Usage::

    pool = ExhaustivePool()
    outputs = set()
    for _ in pool.runs():
        buf = bytearray(b"\\x00\\x01")
        engine._op_bit_offset_flip(buf, 0, b"")
        outputs.add(bytes(buf))
    assert pool.exhausted          # the space was covered, not merely sampled

## How it works

The state is an odometer: a list of ``[value, bound]`` positions, one per
bounded draw, in draw order. During a run each draw reads the value at the
current position, appending ``[0, bound]`` if this is the deepest the
enumeration has gone. Between runs, ``_advance`` increments the last
position that can still be incremented and discards everything after it,
so the next run replays an identical prefix and then diverges.

Two consequences are worth stating because they are what make this
exhaustive rather than approximate:

* Positions are defined by *draw order*, not by call site. An operator
  whose control flow depends on earlier draws simply produces different
  bounds at later positions, and the odometer handles it, because a
  replayed prefix is byte-identical and therefore takes the identical
  path.
* A bound that changes at a replayed position is impossible for a
  deterministic operator, so it is reported rather than tolerated -- see
  ``NondeterministicDrawError``. It means the code under enumeration is
  reading entropy the pool does not own.

## What is deliberately not enumerable

``RandPool`` is much larger than the Zig interface this ports from, and
two families of its methods cannot be walked. Both raise rather than
silently returning something plausible, because an enumeration that
quietly skips a draw is worse than one that refuses: it still reports
``exhausted``.

* **Continuous draws** -- ``random``, ``gauss``, ``expovariate``,
  ``betavariate``, ``gammavariate``, ``lognormvariate`` and their
  ``_list`` forms. There is no finite set of values to walk.
  ``ContinuousDrawError``.
* **Bulk draws** -- ``randbytes``, ``randrange_list``, ``randint_list``,
  ``choice_list``, ``weighted_choice_list``. Finite but combinatorial:
  ``randbytes(4)`` alone is 2**32 paths. Available under
  ``allow_bulk=True`` for callers who have checked the arithmetic.
  ``BulkDrawError``.

Note that the continuous exclusion is sharper than it looks. Several
byte-level operators are unenumerable *only* because a fair coin is
written ``rng.random() < 0.5`` rather than ``rng.randint(0, 1)``; see the
census in ``tests/test_exhaustive_pool.py``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

#: Default cap on the number of bounded draws in a single run. A run that
#: needs more is reported rather than truncated, since a truncated path is
#: an unexplored path.
DEFAULT_MAX_DEPTH = 24

#: Default cap on total runs. Reaching it clears ``exhausted``, so a test
#: asserting ``pool.exhausted`` cannot pass on a partial walk.
DEFAULT_MAX_RUNS = 1_000_000


class ExhaustivePoolError(Exception):
    """Base for every refusal this pool makes."""


class ContinuousDrawError(ExhaustivePoolError):
    """A continuous distribution was drawn; it has no finite enumeration."""


class BulkDrawError(ExhaustivePoolError):
    """A bulk ``*_list``/``randbytes`` draw was made without ``allow_bulk``."""


class DepthExceededError(ExhaustivePoolError):
    """A single run made more bounded draws than ``max_depth`` allows."""


class NondeterministicDrawError(ExhaustivePoolError):
    """A replayed prefix produced a different bound than it did last run.

    The prefix is replayed value-for-value, so a deterministic function of
    the draws must take the identical path and request the identical
    bounds. A mismatch means the code under enumeration is reading entropy
    from somewhere other than this pool -- a module-level ``random``, a
    clock, a set iteration order, a hash seed.
    """


class ExhaustivePool:
    """Drop-in replacement for ``RandPool`` that walks the space of draws.

    Args:
        max_depth: Cap on bounded draws per run. Exceeding it raises
            ``DepthExceededError`` rather than silently returning 0.
        max_runs: Cap on total runs. Reaching it stops iteration with
            ``exhausted`` false and ``budget_exhausted`` true.
        allow_bulk: Permit the ``*_list`` and ``randbytes`` draws. Off by
            default because their trees are combinatorial.
    """

    def __init__(
        self,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_runs: int = DEFAULT_MAX_RUNS,
        allow_bulk: bool = False,
    ) -> None:
        self._v: list[list[int]] = []
        self._p = 0
        self._max_depth = max_depth
        self._max_runs = max_runs
        self._allow_bulk = allow_bulk
        self._runs = 0
        self._exhausted = False
        self._budget_exhausted = False
        self._max_depth_seen = 0

    # ── Driving the enumeration ──────────────────────────────────────

    def runs(self) -> Iterator[int]:
        """Yield once per combination of draws; stops when the space is walked.

        Yields the zero-based run index. The loop body must perform the
        same computation every time, differing only through this pool.
        """
        while True:
            if self._runs >= self._max_runs:
                self._budget_exhausted = True
                return
            self._p = 0
            yield self._runs
            self._runs += 1
            self._max_depth_seen = max(self._max_depth_seen, len(self._v))
            if not self._advance():
                self._exhausted = True
                return

    def _advance(self) -> bool:
        # Positions past where this run stopped belong to a longer earlier
        # path that the last increment diverged away from.
        del self._v[self._p :]
        while self._v:
            entry = self._v[-1]
            if entry[0] + 1 < entry[1]:
                entry[0] += 1
                return True
            self._v.pop()
        return False

    @property
    def exhausted(self) -> bool:
        """True only if every combination was visited.

        False while iterating, and false if ``max_runs`` cut the walk
        short -- assert on this rather than on the run count.
        """
        return self._exhausted

    @property
    def budget_exhausted(self) -> bool:
        """True if iteration stopped because ``max_runs`` was reached."""
        return self._budget_exhausted

    @property
    def runs_completed(self) -> int:
        return self._runs

    @property
    def max_depth_seen(self) -> int:
        """Deepest run observed, for sizing ``max_depth`` on a real operator."""
        return self._max_depth_seen

    # ── The bounded-draw core ────────────────────────────────────────

    def _bounded(self, bound: int, what: str) -> int:
        """Return the current value at this position for a draw of *bound* choices.

        A bound of 1 or less is not a choice and is deliberately *not*
        recorded: recording it would double the tree depth for operators
        that call ``randint(0, 0)`` on a one-byte buffer without adding a
        single distinct outcome.
        """
        if bound <= 1:
            return 0
        if self._p == len(self._v):
            if len(self._v) >= self._max_depth:
                raise DepthExceededError(
                    f"run exceeded max_depth={self._max_depth} bounded draws "
                    f"(at {what}). Raise max_depth, shrink the input, or accept "
                    f"that this operator is not enumerable."
                )
            self._v.append([0, bound])
        entry = self._v[self._p]
        if entry[1] != bound:
            raise NondeterministicDrawError(
                f"draw {self._p} requested bound {bound} ({what}) but the "
                f"identical prefix requested bound {entry[1]} last run; the "
                f"code under enumeration is not a deterministic function of "
                f"this pool"
            )
        self._p += 1
        return entry[0]

    # ── Discrete core: the RandPool methods that can be walked ───────

    def randrange(self, n: int) -> int:
        # RandPool returns 0 for n <= 0 rather than raising; mirrored so an
        # operator cannot behave differently under enumeration.
        if n <= 0:
            return 0
        return self._bounded(n, f"randrange({n})")

    def randint(self, a: int, b: int) -> int:
        width = b - a + 1
        if width <= 0:
            return a  # RandPool's behaviour for an inverted range
        return a + self._bounded(width, f"randint({a}, {b})")

    def choice(self, seq: Sequence) -> object:
        n = len(seq)
        if n == 0:
            raise IndexError("cannot choose from empty sequence")
        return seq[self._bounded(n, f"choice(len={n})")]

    def weighted_choice(self, seq: Sequence, weights: Sequence[float]) -> object:
        """Enumerate the reachable elements, ignoring the relative weights.

        Weights shape *how often* an outcome occurs, not *whether* it can,
        and enumeration is a question about reachability. Zero-weight
        entries are the exception and are excluded, since those genuinely
        cannot be drawn.

        This is where enumeration and sampling most visibly differ: a
        weight of 1-in-1000 is visited exactly as often as a weight of
        999-in-1000, which is the point.
        """
        n = len(seq)
        if n == 0:
            raise IndexError("cannot choose from empty sequence")
        reachable = [i for i in range(n) if i >= len(weights) or weights[i] > 0]
        if not reachable:
            raise IndexError("every weight is zero; no element is reachable")
        idx = reachable[self._bounded(len(reachable), f"weighted_choice(len={n})")]
        return seq[idx]

    def shuffle(self, seq: list) -> None:
        """In-place Fisher-Yates, enumerating all ``n!`` permutations.

        ``RandPool.shuffle`` delegates to numpy above n=8 and uses this
        loop below it. The loop is used at every size here: numpy's
        shuffle draws entropy this pool does not intermediate, so
        delegating would silently produce one permutation per run instead
        of enumerating them.
        """
        n = len(seq)
        for i in range(n - 1, 0, -1):
            j = self._bounded(i + 1, f"shuffle(i={i})")
            seq[i], seq[j] = seq[j], seq[i]

    def sample(self, population, k: int):
        """Enumerate ordered selections of *k* distinct elements.

        Sequential draws without replacement -- bounds n, n-1, ..., so the
        tree is n!/(n-k)! wide rather than n**k.
        """
        is_seq = isinstance(population, list | tuple | bytes)
        n = len(population) if is_seq else population
        if k > n:
            k = n
        if k <= 0:
            return []
        remaining = list(range(n))
        picked = []
        for step in range(k):
            j = self._bounded(len(remaining), f"sample(step={step})")
            picked.append(remaining.pop(j))
        return [population[i] for i in picked] if is_seq else picked

    def reseed(self, seed: int | None = None) -> None:
        """No-op: the enumeration order is the state, and it is not a seed.

        Present so an operator that reseeds mid-run does not crash under
        enumeration. Silently ignoring it is safe here precisely because
        this pool has no entropy to reset.
        """

    # ── Bulk draws: finite, but combinatorial ────────────────────────

    def _bulk_guard(self, what: str, paths: int) -> None:
        if not self._allow_bulk:
            raise BulkDrawError(
                f"{what} would branch {paths} ways per call; pass "
                f"allow_bulk=True if that is intended, or restrict the "
                f"enumeration to operators that draw scalars"
            )

    def randbytes(self, n: int) -> bytes:
        if n <= 0:
            return b""
        self._bulk_guard(f"randbytes({n})", 256**n)
        return bytes(self._bounded(256, f"randbytes[{i}]") for i in range(n))

    def randrange_list(self, n: int, count: int) -> list[int]:
        if count <= 0:
            return []
        self._bulk_guard(f"randrange_list({n}, {count})", max(n, 1) ** count)
        return [self.randrange(n) for _ in range(count)]

    def randint_list(self, a: int, b: int, count: int) -> list[int]:
        if count <= 0:
            return []
        self._bulk_guard(f"randint_list({a}, {b}, {count})", max(b - a + 1, 1) ** count)
        return [self.randint(a, b) for _ in range(count)]

    def choice_list(self, seq: Sequence, count: int) -> list:
        if len(seq) == 0:
            raise IndexError("cannot choose from empty sequence")
        if count <= 0:
            return []
        self._bulk_guard(f"choice_list(len={len(seq)}, {count})", len(seq) ** count)
        return [self.choice(seq) for _ in range(count)]

    def weighted_choice_list(self, seq: Sequence, weights: Sequence[float], k: int) -> list:
        if len(seq) == 0:
            raise IndexError("cannot choose from empty sequence")
        if k <= 0:
            return []
        self._bulk_guard(f"weighted_choice_list(len={len(seq)}, {k})", len(seq) ** k)
        return [self.weighted_choice(seq, weights) for _ in range(k)]

    # ── Continuous draws: refused ────────────────────────────────────

    def _continuous(self, what: str):
        raise ContinuousDrawError(
            f"{what} is continuous and cannot be enumerated. If the call is a "
            f"coin flip written as `random() < p`, express it as a bounded "
            f"draw (randint/choice) and the operator becomes enumerable."
        )

    def random(self) -> float:
        self._continuous("random()")

    def random_list(self, count: int) -> list[float]:
        self._continuous("random_list()")

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        self._continuous("gauss()")

    def gauss_list(self, mu: float, sigma: float, count: int) -> list[float]:
        self._continuous("gauss_list()")

    def expovariate(self, lambd: float = 1.0) -> float:
        self._continuous("expovariate()")

    def expovariate_list(self, lambd: float, count: int) -> list[float]:
        self._continuous("expovariate_list()")

    def betavariate(self, alpha: float, beta: float) -> float:
        self._continuous("betavariate()")

    def betavariate_list(self, alpha: float, beta: float, count: int) -> list[float]:
        self._continuous("betavariate_list()")

    def gammavariate(self, alpha: float, beta: float = 1.0) -> float:
        self._continuous("gammavariate()")

    def gammavariate_list(self, alpha: float, beta: float, count: int) -> list[float]:
        self._continuous("gammavariate_list()")

    def lognormvariate(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        self._continuous("lognormvariate()")

    def lognormvariate_list(self, mu: float, sigma: float, count: int) -> list[float]:
        self._continuous("lognormvariate_list()")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "exhausted" if self._exhausted else "running"
        if self._budget_exhausted:
            state = "budget-exhausted"
        return f"<ExhaustivePool {state} runs={self._runs} depth={self._max_depth_seen}>"
