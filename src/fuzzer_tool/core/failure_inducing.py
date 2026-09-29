"""Failure-inducing combination isolation over a parameter row (FIC-style).

``core/covering_array.py`` guarantees every t-way value combination is
*exercised*; it says nothing about *which* combination made a row fail. When
a covering-array row crashes the target, all 7 IHDR fields differ from the
seed, yet usually only two of them (say ``color_type=3`` and
``bit_depth=16``) matter. ``core/root_cause.py`` answers the same question at
byte level; this module answers it at parameter level, in ``O(k)`` oracle
calls for ``k`` parameters, and names the result in terms a human can act on.

Method (after Zhang & Zhang, "Characterizing failure-causing parameter
interactions by adaptive testing", ISSTA 2011, simplified to delta
debugging over the difference set):

1. Confirm the failing row fails.
2. Find a *passing companion* that differs from it on every movable
   parameter (random draws from each domain, minus the failing value).
3. ``hybrid(S)`` = the failing row's values on ``S``, the companion's
   elsewhere. Chunked greedy removal shrinks ``S`` from "all differing
   parameters" while ``hybrid(S)`` still fails, until no single parameter
   can be dropped.

Assumption (stated, not enforced): failure is monotone in ``S`` -- once
``hybrid(S)`` fails, adding parameters keeps it failing -- and the
companion's values do not themselves trigger a second, different failure
(no masking). Under it the result is an inclusion-minimal failure-inducing
schema. Without it the result is still a schema whose ``hybrid`` row was
observed to fail, and every single-parameter removal was observed to pass,
but it may not be the only one; ``verify_samples`` probes random rows that
contain the schema to catch the case where it is not sufficient.

Pure combinatorics: no I/O, no target knowledge. The oracle is a callback,
so the same code serves a live target, a mock, or a table lookup.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Row = tuple[Any, ...]

# Statuses of a FailureSchema.
ISOLATED = "isolated"  # minimal schema found (see ``verified`` for sufficiency)
UNCONDITIONAL = "unconditional"  # no passing companion found: parameters don't explain it
TRUNCATED = "truncated"  # probe budget hit; schema still fails but may not be minimal
NOT_FAILING = "not_failing"  # the given row did not fail on replay (flaky / not reproducible)

_DEFAULT_MAX_PROBES = 500
_DEFAULT_COMPANION_TRIES = 32


@dataclass(frozen=True)
class FailureSchema:
    """Outcome of :func:`isolate`."""

    status: str
    params: dict[int, Any] = field(default_factory=dict)
    """Parameter index -> value of the failing row, for the isolated schema."""
    passing_row: Row | None = None
    """The passing companion the reduction ran against."""
    probes: int = 0
    """Distinct oracle calls made (repeats are memoized and not counted)."""
    verified: bool | None = None
    """True/False if ``verify_samples`` ran to completion, else None."""
    counterexample: Row | None = None
    """A row containing the schema that did *not* fail (set when verified is False)."""
    unresolved: tuple[int, ...] = ()
    """Parameters whose domain offers no alternative value; relevance untestable."""


class _BudgetExceeded(Exception):
    pass


def _pick(rng: Any, values: Sequence[Any]) -> Any:
    values = list(values)
    if hasattr(rng, "choice"):
        return rng.choice(values)
    return values[rng.randint(0, len(values) - 1)]  # pragma: no cover - fallback


class FailureIsolator:
    """Stateful isolation of one failing row; see the module docstring.

    State lives on the instance (probe cache, companion, current schema
    ``_s``) so a probe-budget cutoff can return the best schema reached so
    far instead of losing it with a call frame.
    """

    def __init__(
        self,
        value_sets: Sequence[Sequence[Any]],
        fails: Callable[[Row], bool],
        *,
        rng: Any = None,
        max_probes: int = _DEFAULT_MAX_PROBES,
        companion_tries: int = _DEFAULT_COMPANION_TRIES,
        verify_samples: int = 0,
    ) -> None:
        for i, vs in enumerate(value_sets):
            if not vs:
                raise ValueError(f"value_sets[{i}] is empty")
        self._vs = [list(vs) for vs in value_sets]
        self._k = len(self._vs)
        self._fails = fails
        self._rng = rng if rng is not None else random.Random(0)
        self._max_probes = max_probes
        self._companion_tries = max(1, companion_tries)
        self._verify_samples = verify_samples
        self._cache: dict[Row, bool] = {}
        self._f: Row = ()
        self._companion: Row | None = None
        self._movable: list[int] = []
        self._unresolved: tuple[int, ...] = ()
        self._s: list[int] = []

    # -- oracle -------------------------------------------------------
    def _probe(self, row: Row) -> bool:
        hit = self._cache.get(row)
        if hit is not None:
            return hit
        if len(self._cache) >= self._max_probes:
            raise _BudgetExceeded
        self._cache[row] = bool(self._fails(row))
        return self._cache[row]

    def _out(self, status: str, **kw: Any) -> FailureSchema:
        return FailureSchema(
            status=status, probes=len(self._cache), unresolved=self._unresolved, **kw
        )

    # -- steps --------------------------------------------------------
    def _find_companion(self) -> Row | None:
        alts = {i: [v for v in self._vs[i] if v != self._f[i]] for i in self._movable}
        for _ in range(self._companion_tries):
            cand = list(self._f)
            for i in self._movable:
                cand[i] = _pick(self._rng, alts[i])
            row = tuple(cand)
            if not self._probe(row):
                return row
        return None

    def _hybrid(self, keep: Sequence[int]) -> Row:
        """Failing row's values on *keep*, the companion's on other movable params."""
        assert self._companion is not None
        kept = set(keep)
        movable = set(self._movable)
        return tuple(
            self._f[i] if (i in kept or i not in movable) else self._companion[i]
            for i in range(self._k)
        )

    def _reduce(self) -> None:
        """Chunked greedy removal to a 1-minimal ``_s`` (every single removal passes).

        Chunk size halves from ``len(_s) // 2`` down to 1, then single-element
        passes repeat until one removes nothing, so the fixed point holds even
        if the predicate is not monotone.
        """
        chunk = max(1, len(self._s) // 2)
        while True:
            changed = False
            i = 0
            while i < len(self._s):
                cand = self._s[:i] + self._s[i + chunk :]
                if len(cand) < len(self._s) and cand and self._probe(self._hybrid(cand)):
                    self._s = cand
                    changed = True
                else:
                    i += chunk
            if chunk == 1:
                if not changed:
                    return
            else:
                chunk = max(1, chunk // 2)

    def _verify(self, schema: dict[int, Any]) -> tuple[bool | None, Row | None]:
        if self._verify_samples <= 0:
            return None, None
        try:
            for _ in range(self._verify_samples):
                row = tuple(
                    self._f[i] if i in schema else _pick(self._rng, self._vs[i])
                    for i in range(self._k)
                )
                if not self._probe(row):
                    return False, row
        except _BudgetExceeded:
            return None, None
        return True, None

    def run(self, fail_row: Sequence[Any]) -> FailureSchema:
        if len(fail_row) != self._k:
            raise ValueError(f"fail_row has {len(fail_row)} values for {self._k} parameters")
        self._f = tuple(fail_row)
        try:
            if not self._probe(self._f):
                return self._out(NOT_FAILING)
            self._movable = [i for i in range(self._k) if any(v != self._f[i] for v in self._vs[i])]
            self._unresolved = tuple(i for i in range(self._k) if i not in set(self._movable))
            if not self._movable:
                return self._out(UNCONDITIONAL)
            self._companion = self._find_companion()
            if self._companion is None:
                return self._out(UNCONDITIONAL)
            self._s = list(self._movable)
            self._reduce()
        except _BudgetExceeded:
            schema = {i: self._f[i] for i in self._s}
            return self._out(TRUNCATED, params=schema, passing_row=self._companion)
        schema = {i: self._f[i] for i in self._s}
        verified, counter = self._verify(schema)
        return self._out(
            ISOLATED,
            params=schema,
            passing_row=self._companion,
            verified=verified,
            counterexample=counter,
        )


def isolate(
    fail_row: Sequence[Any],
    value_sets: Sequence[Sequence[Any]],
    fails: Callable[[Row], bool],
    *,
    rng: Any = None,
    max_probes: int = _DEFAULT_MAX_PROBES,
    companion_tries: int = _DEFAULT_COMPANION_TRIES,
    verify_samples: int = 0,
) -> FailureSchema:
    """Isolate a minimal failure-inducing schema of *fail_row*.

    Args:
        fail_row: a row (value per parameter) for which ``fails`` is true.
            Its values need not lie in ``value_sets`` (a real input's
            ``width=640`` is not a boundary value); alternatives are drawn
            from ``value_sets[i]`` minus ``fail_row[i]``.
        value_sets: candidate values per parameter, as for
            ``covering_array.generate``. Values must be hashable.
        fails: oracle, ``fails(row) -> True`` iff the target fails on
            *row*. Assumed deterministic; results are memoized by row.
        rng: ``.choice(list)`` provider (``RandPool`` or ``random.Random``).
            Defaults to ``random.Random(0)`` so reports are reproducible.
        max_probes: cap on distinct oracle calls, verification included.
        companion_tries: passing-companion candidates to try before giving
            up with ``UNCONDITIONAL``.
        verify_samples: random rows containing the schema to probe for
            sufficiency. 0 skips verification (``verified`` stays None).
    """
    return FailureIsolator(
        value_sets,
        fails,
        rng=rng,
        max_probes=max_probes,
        companion_tries=companion_tries,
        verify_samples=verify_samples,
    ).run(fail_row)


def format_schema(schema: FailureSchema, names: Sequence[str] | None = None) -> str:
    """One-line human form, e.g. ``bit_depth=16 & color_type=3 (isolated, 9 probes)``."""
    if not schema.params:
        body = "<no parameter isolated>"
    else:
        body = " & ".join(
            f"{names[i] if names else f'p{i}'}={v}" for i, v in sorted(schema.params.items())
        )
    tail = schema.status
    if schema.verified is True:
        tail += ", verified"
    elif schema.verified is False:
        tail += ", NOT sufficient"
    return f"{body} ({tail}, {schema.probes} probes)"
