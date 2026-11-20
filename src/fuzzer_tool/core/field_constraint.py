"""Pair-relation solver for header fields: pick *which* relations a row breaks.

``covering_array`` guarantees that every pair of field values appears together
once. It cannot say "length and count agree" or "the stored-method sizes
differ, everything else is consistent": a pairwise row that breaks one
relation usually breaks several, and the parser rejects it at the first
check, so the bug behind the second check is never reached.

This module builds rows by *relation polarity*. A ``Relation`` is a predicate
on two fields (``comp_size == file_size`` when ``method == 0``). A polarity
says, for each relation, hold or violate. ``solve`` returns rows with exactly
that polarity; ``relation_rows`` sweeps "all hold" plus every way of violating
one (or k) relations, which is the input a validator needs to see to exercise
each check in isolation.

The solver is the bitmask N-Queens search: each field's domain is one int
(bit ``v`` = value index ``v`` still allowed), a placement ANDs precomputed
support masks into the other fields' domains (the way a queen clears its
column and both diagonals), the next field is the one with the fewest bits
left, candidates are peeled with ``avail & -avail``, and the state is passed
by value so backtracking costs nothing.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

Row = tuple[int, ...]


@dataclass(frozen=True)
class Relation:
    """A named predicate over the values of fields *a* and *b* (field indices)."""

    name: str
    a: int
    b: int
    holds: Callable[[int, int], bool]


def _bit_ids(mask: int) -> list[int]:
    ids = []
    while mask:
        low = mask & -mask
        ids.append(low.bit_length() - 1)
        mask ^= low
    return ids


def _shuffle(items: list[int], rng: Any) -> None:
    """Fisher-Yates on ``rng.random()`` (RandPool and ``random.Random`` both have it)."""
    for i in range(len(items) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        items[i], items[j] = items[j], items[i]


def _validate(value_sets: Sequence[Sequence[int]], relations: Sequence[Relation]) -> None:
    for i, vs in enumerate(value_sets):
        if not vs:
            raise ValueError(f"value_sets[{i}] is empty")
    names = [r.name for r in relations]
    if len(set(names)) != len(names):
        raise ValueError("duplicate relation names")
    for r in relations:
        if not (0 <= r.a < len(value_sets) and 0 <= r.b < len(value_sets)) or r.a == r.b:
            raise ValueError(f"relation {r.name!r}: bad field pair ({r.a}, {r.b})")


def _supports(
    value_sets: Sequence[Sequence[int]],
    relations: Sequence[Relation],
    violate: frozenset[str],
) -> dict[tuple[int, int], list[int]]:
    """``sup[(i, j)][x]`` = mask of value indices of field *j* compatible with value *x* of *i*."""
    sup: dict[tuple[int, int], list[int]] = {}
    for r in relations:
        want = r.name not in violate
        for i, j, swap in ((r.a, r.b, False), (r.b, r.a, True)):
            table = sup.setdefault((i, j), [(1 << len(value_sets[j])) - 1] * len(value_sets[i]))
            for x, vx in enumerate(value_sets[i]):
                m = 0
                for y, vy in enumerate(value_sets[j]):
                    ok = r.holds(vy, vx) if swap else r.holds(vx, vy)
                    if ok == want:
                        m |= 1 << y
                table[x] &= m
    return sup


def solve(
    value_sets: Sequence[Sequence[int]],
    relations: Sequence[Relation],
    violate: Sequence[str] = (),
    limit: int = 1,
    rng: Any = None,
) -> list[Row]:
    """Up to *limit* distinct rows where exactly the relations in *violate* fail.

    Every other relation holds. Rows come from a depth-first search over the
    fields; *rng* (``.random()``) randomises the value order, otherwise values
    are tried in declaration order. Returns ``[]`` when no row has that
    polarity (a violation the value sets cannot express, or contradictory
    relations).
    """
    _validate(value_sets, relations)
    unknown = set(violate) - {r.name for r in relations}
    if unknown:
        raise ValueError(f"unknown relation(s) to violate: {sorted(unknown)}")
    k = len(value_sets)
    sup = _supports(value_sets, relations, frozenset(violate))
    nbrs: list[list[int]] = [sorted({j for (i, j) in sup if i == f}) for f in range(k)]
    out: list[Row] = []
    chosen = [0] * k

    def rec(doms: list[int], todo: int) -> bool:
        if not todo:
            out.append(tuple(value_sets[f][chosen[f]] for f in range(k)))
            return len(out) >= limit
        # Fewest-candidates-first: the cell with the smallest `avail`.
        field, fewest, t = -1, 1 << 62, todo
        while t:
            low = t & -t
            f = low.bit_length() - 1
            c = doms[f].bit_count()
            if c < fewest:
                field, fewest = f, c
                if c <= 1:
                    break
            t ^= low
        if not fewest:
            return False
        ids = _bit_ids(doms[field])
        if rng is not None:
            _shuffle(ids, rng)
        rest = todo & ~(1 << field)
        for x in ids:
            nxt = list(doms)
            dead = False
            for j in nbrs[field]:
                if (rest >> j) & 1:
                    nxt[j] &= sup[(field, j)][x]
                    if not nxt[j]:
                        dead = True
                        break
            if dead:
                continue
            chosen[field] = x
            if rec(nxt, rest):
                return True
        return False

    rec([(1 << len(vs)) - 1 for vs in value_sets], (1 << k) - 1)
    return out


def relation_rows(
    value_sets: Sequence[Sequence[int]],
    relations: Sequence[Relation],
    *,
    violate: int = 1,
    per_pattern: int = 2,
    rng: Any = None,
) -> list[Row]:
    """Rows for "all relations hold" plus every way to violate 1..*violate* of them.

    *per_pattern* rows per polarity (fewer when the value sets admit fewer).
    Order: the all-hold rows first, then violated subsets by size. Duplicates
    across patterns cannot occur (a row has exactly one polarity).
    """
    _validate(value_sets, relations)
    rows: list[Row] = []
    names = [r.name for r in relations]
    for size in range(0, min(violate, len(names)) + 1):
        for pattern in itertools.combinations(names, size):
            rows.extend(solve(value_sets, relations, pattern, per_pattern, rng))
    return rows


def polarity(row: Row, relations: Sequence[Relation]) -> tuple[str, ...]:
    """Names of the relations *row* violates (inverse of ``solve``)."""
    return tuple(r.name for r in relations if not r.holds(row[r.a], row[r.b]))
