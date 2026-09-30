"""Non-adaptive group testing: which of *n* items matter, from pooled tests.

Oracle model (OR / monotone): a pool tests positive iff it contains at least
one *defective* item -- e.g. "revert these bytes to the baseline and the crash
disappears" means the pool holds a crash-critical byte.

All pools of round 1 are fixed before any answer, so they can run in parallel
(unlike ddmin / binary splitting, which need ``O(log n)`` sequential rounds).
The price is more total tests: ``~ c (d+1) ln n`` versus ``~ d log2(n/d)``.
Use it when executions parallelise and latency, not count, is the cost.

Decoding (Bernoulli design, p = 1/(d+1)):

    COMP  candidates = items in no negative pool. Never drops a defective.
    DD    a candidate that is the only candidate in some positive pool is
          certainly defective.
    fix   every other candidate is tested one by one, so the result is exact
          for *any* d; a wrong ``d`` only costs extra tests (COMP survivors
          that are not defective are few when the design is sized for d).

Wired into ``colorize(mode=POOLED)`` via ``comp`` (``--colorize-mode pooled``).
Not into ``tmin``: its oracle is conjunctive, not OR. See
``tools/bench_group_testing.py`` and
``docs/handover/handover_combinatorics_permutations_2026-09-02.md`` §6.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Pool = frozenset[int]
Oracle = Callable[[Pool], bool]

# Pools per (d+1) ln n. COMP's per-pool elimination chance for a non-defective
# is ~0.37/(d+1), so ~2.7 is the union-bound break-even; 3.5 leaves margin
# without paying for full disjunctness (correctness never depends on it).
_POOL_FACTOR = 3.5


@dataclass
class Result:
    defective: set[int] = field(default_factory=set)
    tests: int = 0
    rounds: int = 0
    truncated: bool = False


def tests_needed(n: int, d: int) -> int:
    """Round-1 pool count for *n* items and up to *d* defectives."""
    return max(1, math.ceil(_POOL_FACTOR * (d + 1) * math.log(max(n, 2))))


def _check(n: int, d: int) -> None:
    if n < 1:
        raise ValueError(f"need n >= 1, got {n}")
    if d < 1:
        raise ValueError(f"need d >= 1, got {d}")


def design(n: int, tests: int, *, d: int, rng: Any) -> list[Pool]:
    """*tests* random pools over ``range(n)``, each item in with prob 1/(d+1)."""
    _check(n, d)
    if tests < 1:
        raise ValueError(f"need tests >= 1, got {tests}")

    # Bounded draw, not random() < p: enumerable under ExhaustivePool.
    return [frozenset(i for i in range(n) if rng.randint(0, d) == 0) for _ in range(tests)]


def comp(n: int, pools: Sequence[Pool], outcomes: Sequence[bool]) -> set[int]:
    """Items in no negative pool (a superset of the defectives)."""
    out = set(range(n))
    for pool, positive in zip(pools, outcomes, strict=True):
        if not positive:
            out -= pool
    return out


def _definite(candidates: set[int], pools: Sequence[Pool], outcomes: Sequence[bool]) -> set[int]:
    """DD step: candidates that are the only candidate in some positive pool.

    Sound for any design: that pool holds a defective, and every defective is
    a candidate, so the lone candidate is it.
    """
    sure: set[int] = set()
    for pool, positive in zip(pools, outcomes, strict=True):
        if not positive:
            continue
        inside = pool & candidates
        if len(inside) == 1:
            sure |= inside
    return sure


def identify(
    n: int,
    oracle: Oracle,
    *,
    d: int,
    rng: Any,
    max_tests: int | None = None,
) -> Result:
    """Exact defective set of *n* items; round 1 is non-adaptive.

    ``truncated`` is set when *max_tests* stopped the resolution step; the
    returned set is then only the certain defectives found so far.
    """
    _check(n, d)
    res = Result()
    pools = design(n, tests_needed(n, d), d=d, rng=rng)

    # Round 1: all pools fixed up front.
    outcomes = [oracle(p) for p in pools]
    res.tests = len(pools)
    res.rounds = 1

    candidates = comp(n, pools, outcomes)
    sure = _definite(candidates, pools, outcomes)
    res.defective = set(sure)

    # Round 2: every other candidate. A candidate sharing positive pools with a
    # sure defective may be masked by it, and one in no pool has no evidence
    # at all -- neither can be assumed clean without breaking exactness.
    ambiguous = sorted(candidates - sure)
    if ambiguous:
        res.rounds = 2
    for item in ambiguous:
        if max_tests is not None and res.tests >= max_tests:
            res.truncated = True
            break
        res.tests += 1
        if oracle(frozenset({item})):
            res.defective.add(item)
    return res


def split_search(n: int, oracle: Oracle) -> Result:
    """Baseline: level-order binary splitting; one parallel round per level."""
    if n < 1:
        raise ValueError(f"need n >= 1, got {n}")

    res = Result()
    level: list[tuple[int, int]] = [(0, n)]
    while level:
        res.rounds += 1
        nxt: list[tuple[int, int]] = []
        for lo, hi in level:
            res.tests += 1
            if not oracle(frozenset(range(lo, hi))):
                continue
            if hi - lo == 1:
                res.defective.add(lo)
                continue
            mid = (lo + hi) // 2
            nxt += [(lo, mid), (mid, hi)]
        level = nxt
    return res
