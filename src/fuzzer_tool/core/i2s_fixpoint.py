"""Iterated input-to-state: drive a Redqueen patch to a self-consistent input.

Redqueen patches once: run x, see ``cmp(a, b)``, write the operand the
target wanted where the input held the other. When the wanted value depends
on the patched bytes (a checksum over a length field, a second field the
first one hid), one patch is not enough. Iterating it is a fixed-point
search -- Deutsch's consistency condition for a closed timelike curve,
``x = step(x)``::

    x0 ─probe─▶ pairs ─patch─▶ x1 ─probe─▶ pairs ─patch─▶ x2 ...
         FIXED  step(x) == x          → x is self-consistent
         CYCLE  x_j == x_i, i < j     → no fixed point; Deutsch's answer
                                        is the mixture over x_i..x_{j-1},
                                        so every member is a candidate
         BUDGET max_iters probes spent → nothing consistent found

The map must be a function of ``x`` alone (no RNG), or a repeated state
would not imply a cycle. Trajectories are short, so a seen-dict replaces
Floyd (see ``core/cycle_detect.py`` for when Floyd pays).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

# Shorter operands match the input by chance (see fuzz_round._scan_operand).
MIN_OPERAND = 2

Pair = tuple[bytes, bytes]

# x -> compare pairs from one run of x; None when the run gave no reading.
Probe = Callable[[bytes], Sequence[Pair] | None]


class Outcome(Enum):
    FIXED = "fixed"
    CYCLE = "cycle"
    BUDGET = "budget"
    BLIND = "blind"


@dataclass(frozen=True)
class Fixpoint:
    """Search result. ``candidates`` never contains the seed."""

    outcome: Outcome
    candidates: tuple[bytes, ...]
    execs: int


def _swap(data: bytes, have: bytes, want: bytes) -> bytes | None:
    """Replace the first ``have`` in ``data`` with ``want``."""
    idx = data.find(have)
    if idx < 0:
        return None
    return data[:idx] + want + data[idx + len(have) :]


def patch_step(data: bytes, pairs: Sequence[Pair]) -> bytes | None:
    """Apply the first unsatisfied pair whose operand occurs in ``data``.

    Each pair is tried as stored and byte-reversed (integer compares log
    the value; a little-endian input stores it reversed), in both
    directions. None when no pair applies.
    """
    for a, b in pairs:
        if a == b or len(a) != len(b) or len(a) < MIN_OPERAND:
            continue

        for have, want in ((b, a), (a, b), (b[::-1], a[::-1]), (a[::-1], b[::-1])):
            out = _swap(data, have, want)
            if out is not None:
                return out

    return None


def solve(data: bytes, probe: Probe, max_iters: int) -> Fixpoint:
    """Iterate probe+patch from ``data`` until fixed, cyclic or out of budget."""
    trail = [data]
    seen = {data: 0}
    x = data

    for execs in range(1, max_iters + 1):
        pairs = probe(x)
        if pairs is None:
            return Fixpoint(Outcome.BLIND, (), execs)

        nxt = patch_step(x, pairs)
        if nxt is None:
            found = (x,) if x != data else ()
            return Fixpoint(Outcome.FIXED, found, execs)

        # Revisit: trail[start:] is the cycle; the seed is already queued.
        start = seen.get(nxt)
        if start is not None:
            cycle = tuple(s for s in trail[start:] if s != data)
            return Fixpoint(Outcome.CYCLE, cycle, execs)

        seen[nxt] = len(trail)
        trail.append(nxt)
        x = nxt

    return Fixpoint(Outcome.BUDGET, (), max_iters)
