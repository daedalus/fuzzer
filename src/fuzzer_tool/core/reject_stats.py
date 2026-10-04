"""Reject rate per operator and per mutation position.

The validity channel (``core/validity.py``) records that a run was rejected
but not what caused it. On a linear instruction input a mid-stream edit
reinterprets every record after it, so rejection depends on *where* the
mutant diverges from its parent:

    parent  | m1 | m2 | m3 | m4 |
    insert  | m1 | m2 | X | m3 | m4 |   m3, m4 now run on a new state -> reject
    append  | m1 | m2 | m3 | m4 | X |   prefix unchanged               -> keep

Position is the first byte where parent and mutant differ, not the offset an
operator was handed: operators draw their own positions, so the proposed
offset is not where the edit landed.

Diagnostic only. Not persisted; memory is bounded by ``MAX_OPS``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from fuzzer_tool.core.validity import Validity

#: Relative-position bins over the parent length.
NUM_BINS = 8

#: Operator names tracked; further names are dropped, not evicted.
MAX_OPS = 512


def first_diff(a: bytes, b: bytes) -> int:
    """Index of the first differing byte; ``min(len)`` when one is a prefix.

    Example: ``first_diff(b"abcd", b"abXcd") == 2``.
    """
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n

    # Invariant: a[:lo] == b[:lo], and a[lo:hi] != b[lo:hi].
    lo, hi = 0, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
            continue
        hi = mid
    return lo


@dataclass(frozen=True)
class RejectRow:
    """Verdict counts for one operator or one position bin."""

    name: str
    valid: int
    invalid: int
    bin: int = -1

    @property
    def reject_rate(self) -> float:
        total = self.valid + self.invalid
        return self.invalid / total if total else 0.0


class RejectStats:
    """Accepted/rejected counts keyed by operator and by position bin."""

    def __init__(self) -> None:
        self._ops: dict[str, list[int]] = {}
        self._bins = [[0, 0] for _ in range(NUM_BINS)]

    def record(
        self,
        ops: Iterable[str],
        parent: bytes,
        mutant: bytes,
        validity: Validity,
    ) -> None:
        """Fold one classified run in. UNKNOWN is not a verdict and is skipped."""
        if validity is Validity.UNKNOWN:
            return
        slot = 0 if validity is Validity.VALID else 1

        # An op stacked twice in one round is one observation, as in op_counts.
        for op in set(ops):
            counts = self._ops.get(op)
            if counts is None and len(self._ops) < MAX_OPS:
                counts = self._ops[op] = [0, 0]
            if counts is not None:
                counts[slot] += 1

        pos = first_diff(parent, mutant)
        idx = min(NUM_BINS - 1, pos * NUM_BINS // max(len(parent), 1))
        self._bins[idx][slot] += 1

    def op_rows(self) -> list[RejectRow]:
        """Per-operator rows, most-run first."""
        rows = [RejectRow(op, c[0], c[1]) for op, c in self._ops.items()]
        return sorted(rows, key=lambda r: (-(r.valid + r.invalid), r.name))

    def bin_rows(self) -> list[RejectRow]:
        """Per-position rows in position order; empty bins omitted."""
        return [
            RejectRow(f"{i * 100 // NUM_BINS}%", c[0], c[1], bin=i)
            for i, c in enumerate(self._bins)
            if c[0] + c[1]
        ]
