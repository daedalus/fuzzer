"""Detect coverage keyed on an input hash (AntiFuzz, USENIX Sec '19, §4.1).

AntiFuzz hashes the whole input and calls a hash-chosen chain of fake
functions, so a one-bit change anywhere yields "new" edges and nearly every
mutant enters the corpus. Real parsers rarely react that way to trailing
garbage:

    seed + b"\\x01" ─┐
    seed + b"\\x02" ─┼─▶ edge sets ─▶ all distinct?  ─▶ SUSPECTED
    ...             │                 1–2 classes    ─▶ CLEAN
    seed + b"\\x08" ─┘

:class:`AdmissionMonitor` is the runtime sibling: a campaign that asks to
admit most of its executions is being fed noise.

Pure: no I/O, no shared memory. The caller runs the target.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

# One-byte tail variants per probe. Eight values that all land on distinct
# edge sets is a byte-value dispatch on trailing garbage or a hash.
NOISE_PROBE_VARIANTS = 8


class NoiseVerdict(Enum):
    UNMEASURED = "unmeasured"
    CLEAN = "clean"
    SUSPECTED = "suspected"


def tail_variants(seed: bytes, count: int) -> list[bytes]:
    """*seed* with one appended byte, values 1..count (deterministic)."""
    return [seed + bytes((v,)) for v in range(1, count + 1)]


def classify_noise(base_runs: Sequence[set[int]], variant_sets: Sequence[set[int]]) -> NoiseVerdict:
    """Classify tail-variant edge sets against repeated runs of the seed.

    *base_runs* must agree with each other: a target that diverges on an
    identical input is nondeterministic, and that is not the input's fault.
    """
    if len(base_runs) < 2 or len(variant_sets) < 2:
        return NoiseVerdict.UNMEASURED

    first = base_runs[0]
    if any(run != first for run in base_runs[1:]):
        return NoiseVerdict.UNMEASURED

    distinct = {frozenset(s) for s in variant_sets}
    if len(distinct) == len(variant_sets):
        return NoiseVerdict.SUSPECTED

    return NoiseVerdict.CLEAN


class AdmissionMonitor:
    """Fire once when corpus admissions approach one per execution.

    Healthy campaigns admit well under 1% of executions once past the
    first seconds; hash-keyed coverage admits nearly all of them.
    """

    MIN_EXECS = 5000  # skip the early regime, where most inputs are new
    FLOOD_RATE = 0.5  # admissions per execution

    def __init__(self) -> None:
        self._base: tuple[int, int] | None = None
        self._fired = False

    def observe(self, execs: int, admissions: int) -> bool:
        """Record cumulative counters; True exactly once, on a flood."""
        if self._base is None:
            self._base = (execs, admissions)
            return False

        if self._fired:
            return False

        span = execs - self._base[0]
        if span < self.MIN_EXECS:
            return False

        rate = (admissions - self._base[1]) / span
        self._fired = rate >= self.FLOOD_RATE
        return self._fired

    def rate(self, execs: int, admissions: int) -> float:
        """Admissions per execution since the baseline (0 before one)."""
        if self._base is None or execs <= self._base[0]:
            return 0.0

        return (admissions - self._base[1]) / (execs - self._base[0])
