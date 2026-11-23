"""Exact posterior over which items matter, from pooled-test constraints.

Minesweeper view: item = cell, defective = mine, pool test = clue.

    pool {a,b,c} positive      ->  1 <= a+b+c       (OR oracle)
    pool {a,b,c} negative      ->  a+b+c = 0
    pool {a,b,c} reads 2       ->  a+b+c = 2        (count oracle)

``group_testing`` decodes with COMP/DD: a verdict per item, a fixed design.
This module keeps ``P(item defective | all tests so far)`` instead, then
picks the next pool whose answer is closest to 50/50 (max information).
Fewer executions than a fixed design, but sequential.

Model: items independent with prior ``p``; every assignment that satisfies
all constraints is weighted ``(p/(1-p))^k`` for ``k`` defectives. Constraints
split the items into independent components (shared pools); each is
enumerated exactly with at most ``max_def`` defectives (per component). ``identify`` starts
at ``max_def = d`` and widens it when the evidence contradicts it. Items in no constraint keep marginal ``p``.

    pools     {0,1,2}+  {2,3}-  {4,5}+        components
              0 1 2 3 4 5                      {0,1}  (2,3 cleared)
              . . x x . .                      {4,5}
                                               every other item: free

A component over ``max_sol`` solutions is not enumerated: it falls back to
COMP bounds (cleared items 0, the rest the prior) and ``exact`` is False.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import chain

import numpy as np

Pool = frozenset[int]
Oracle = Callable[[Pool], bool]

DEFAULT_MAX_SOL = 20_000
# Search nodes allowed per solution budget (dead ends cost too).
_NODE_FACTOR = 8
# Marginal this close to 0/1 counts as settled.
SETTLED_EPS = 1e-3
_HALF = 0.5


class NoSolution(ValueError):
    """Observations need more defectives than ``max_def`` allows."""


@dataclass(frozen=True)
class Obs:
    """``lo <= (defectives in pool) <= hi``."""

    pool: Pool
    lo: int
    hi: int


def or_obs(pool: Pool, positive: bool) -> Obs:
    """Observation of an OR oracle."""
    return Obs(pool, 1, len(pool)) if positive else Obs(pool, 0, 0)


@dataclass
class _Comp:
    items: list[int]
    flat: np.ndarray  # defective item ids, solutions concatenated
    sol_id: np.ndarray  # solution index of each flat entry
    weight: np.ndarray  # per solution, unnormalised
    n_sol: int

    def marginals(self, n: int) -> np.ndarray:
        mass = np.bincount(self.flat, weights=self.weight[self.sol_id], minlength=n)
        return mass / self.weight.sum()

    def p_clear(self, inpool: np.ndarray) -> float:
        hits = np.bincount(self.sol_id, weights=inpool[self.flat], minlength=self.n_sol)
        return float(self.weight[hits == 0].sum() / self.weight.sum())


@dataclass
class Posterior:
    n: int
    prior: float
    cleared: set[int] = field(default_factory=set)
    free: list[int] = field(default_factory=list)
    comps: list[_Comp] = field(default_factory=list)
    loose: list[list[int]] = field(default_factory=list)  # un-enumerated comps
    exact: bool = True
    _open_cache: set[int] | None = None

    def marginals(self) -> np.ndarray:
        out = np.full(self.n, self.prior)
        out[list(self.cleared)] = 0.0
        for comp in self.comps:
            out[comp.items] = comp.marginals(self.n)[comp.items]
        return out

    def _open(self) -> set[int]:
        """Items with no exact evidence: free, or in an un-enumerated comp."""
        if self._open_cache is None:
            self._open_cache = set(self.free).union(*self.loose)
        return self._open_cache

    def p_clear(self, pool: Pool) -> float:
        """P(no defective in *pool*)."""
        inpool = np.zeros(self.n)
        inpool[list(pool)] = 1.0
        out = 1.0
        for comp in self.comps:
            out *= comp.p_clear(inpool)
        return out * (1.0 - self.prior) ** len(pool & self._open())

    def next_pool(self) -> Pool | None:
        """Pool whose OR answer is closest to 50/50, or None if all settled."""
        marg = self.marginals()
        live = [i for i in np.argsort(-marg) if SETTLED_EPS < marg[i] < 1 - SETTLED_EPS]
        if not live:
            return None

        # Grow the pool in marginal order; keep the size nearest 50/50.
        pool: set[int] = set()
        best: tuple[float, Pool] | None = None
        for i in live:
            pool.add(int(i))
            gap = abs(1.0 - self.p_clear(frozenset(pool)) - _HALF)
            if best is None or gap < best[0]:
                best = (gap, frozenset(pool))
            # P(positive) only grows with the pool: past 1/2 it cannot improve.
            if 1.0 - self.p_clear(frozenset(pool)) >= _HALF:
                break
        return best[1] if best else None

    def defective(self) -> set[int]:
        marg = self.marginals()
        return {int(i) for i in np.flatnonzero(marg > _HALF)}


def _components(obs: Sequence[Obs]) -> list[tuple[list[int], list[Obs]]]:
    """Group observations that share items (union-find over items)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for o in obs:
        items = list(o.pool)
        parent.update({i: parent.get(i, i) for i in items})
        for i in items[1:]:
            parent[find(i)] = find(items[0])

    groups: dict[int, tuple[set[int], list[Obs]]] = {}
    for o in obs:
        root = find(next(iter(o.pool)))
        items, members = groups.setdefault(root, (set(), []))
        items |= o.pool
        members.append(o)
    return [(sorted(items), members) for items, members in groups.values()]


def _enumerate(
    items: list[int], obs: list[Obs], max_def: int, max_sol: int
) -> list[tuple[int, ...]] | None:
    """All satisfying defective sets of size <= max_def, or None over budget."""
    where = {i: [] for i in items}
    for j, o in enumerate(obs):
        for i in o.pool:
            where[i].append(j)

    # Most-constrained first: prunes earliest.
    order = sorted(items, key=lambda i: -len(where[i]))
    have = [0] * len(obs)
    left = [len(o.pool) for o in obs]
    chosen: list[int] = []
    sols: list[tuple[int, ...]] = []
    nodes = [0]

    def feasible(i: int) -> bool:
        return all(have[j] <= obs[j].hi and have[j] + left[j] >= obs[j].lo for j in where[i])

    def walk(pos: int) -> bool:
        nodes[0] += 1
        if len(sols) > max_sol or nodes[0] > max_sol * _NODE_FACTOR:
            return False
        if pos == len(order):
            sols.append(tuple(chosen))
            return True

        i = order[pos]
        for j in where[i]:
            left[j] -= 1

        ok = True
        if feasible(i):
            ok = walk(pos + 1)

        # Variant: i defective.
        if ok and len(chosen) < max_def:
            for j in where[i]:
                have[j] += 1
            chosen.append(i)
            if feasible(i):
                ok = walk(pos + 1)
            chosen.pop()
            for j in where[i]:
                have[j] -= 1

        for j in where[i]:
            left[j] += 1
        return ok

    return sols if walk(0) else None


def posterior(
    n: int,
    obs: Sequence[Obs],
    *,
    prior: float,
    max_def: int,
    max_sol: int = DEFAULT_MAX_SOL,
) -> Posterior:
    """Exact posterior given *obs*; raises ValueError if they contradict."""
    if n < 1 or not 0.0 < prior < 1.0 or max_def < 1:
        raise ValueError(f"bad arguments n={n} prior={prior} max_def={max_def}")

    post = Posterior(n=n, prior=prior)
    for o in obs:
        if not o.pool <= set(range(n)) or o.lo > o.hi or o.lo > len(o.pool) or o.hi < 0:
            raise ValueError(f"bad observation {o}")

    # Zero-count pools clear their items outright.
    for o in obs:
        if o.hi == 0:
            post.cleared |= o.pool
    live: list[Obs] = []
    for o in obs:
        pool = o.pool - post.cleared
        if o.lo > len(pool):
            raise ValueError(f"contradiction: {o}")
        if pool and o.hi > 0:
            live.append(Obs(pool, o.lo, min(o.hi, len(pool))))

    odds = prior / (1.0 - prior)
    in_comp: set[int] = set()
    for items, members in _components(live):
        in_comp |= set(items)
        sols = _enumerate(items, members, max_def, max_sol)
        if sols is None:
            post.exact = False
            post.loose.append(items)
            continue
        if not sols:
            raise NoSolution("no assignment within max_def")

        sizes = np.fromiter((len(s) for s in sols), dtype=np.int64, count=len(sols))
        flat = np.fromiter(chain.from_iterable(sols), dtype=np.int64, count=int(sizes.sum()))
        post.comps.append(
            _Comp(
                items=items,
                flat=flat,
                sol_id=np.repeat(np.arange(len(sols)), sizes),
                weight=odds ** sizes.astype(float),
                n_sol=len(sols),
            )
        )

    post.free = [i for i in range(n) if i not in in_comp and i not in post.cleared]
    return post


def _verify_pool(post: Posterior) -> Pool | None:
    """Everything not proven either way: one negative answer makes the result
    exact for any d; a positive one means ``max_def`` was too low."""
    marg = post.marginals()
    sure = {int(i) for i in np.flatnonzero(marg > 1 - SETTLED_EPS)}
    rest = frozenset(range(post.n)) - post.cleared - sure
    return rest or None


@dataclass
class Result:
    defective: set[int] = field(default_factory=set)
    tests: int = 0
    rounds: int = 0
    exact: bool = True
    truncated: bool = False


def identify(
    n: int,
    oracle: Oracle,
    *,
    d: int,
    max_tests: int | None = None,
    max_sol: int = DEFAULT_MAX_SOL,
) -> Result:
    """Adaptive search for the defective set; one test per round."""
    if n < 1 or d < 1:
        raise ValueError(f"need n >= 1 and d >= 1, got n={n} d={d}")

    res = Result()
    seen: list[Obs] = []
    prior = min(d / n, 1.0 - 1e-9) if d < n else 0.5
    cap = max_tests if max_tests is not None else 4 * n
    max_def = d

    while True:
        # Trust d; if the evidence needs more defectives, widen the bound.
        try:
            post = posterior(n, seen, prior=prior, max_def=max_def, max_sol=max_sol)
        except NoSolution:
            max_def += 1
            continue
        res.exact = post.exact
        if not post.exact:
            # Over budget: marginals are only priors there. Caller falls back.
            res.truncated = True
            break
        pool = post.next_pool() or _verify_pool(post)
        if pool is None:
            break
        if res.tests >= cap:
            res.truncated = True
            break
        seen.append(or_obs(pool, oracle(pool)))
        res.tests += 1
        res.rounds += 1

    res.defective = post.defective()
    return res
