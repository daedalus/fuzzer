"""Orbit canonicalization for exec dedup (algebra handover G1).

The exec filter is exact-match. Inputs the target treats as equivalent still
cost one exec each. The user declares the symmetries; this module maps every
input to the minimum of its orbit, and the filter hashes that instead.

Spec grammar (items separated by ``,``)::

    swap@OFFSET:SIZExCOUNT    COUNT chunks of SIZE bytes commute
    endian@OFFSET:SIZE        field is byte-order agnostic

Example: ``swap@8:4x3,endian@0:4``.

* swap: canonical form = chunks sorted ascending. Orbit size ``COUNT!``.
* endian: canonical form = min(field, reversed field). Orbit size 2.
* Total orbit size (product over items) is capped at ``MAX_ORBIT``. A wrong
  declaration silently drops real novelty, so big orbits are refused.
* Items must not overlap. An input too short for an item is left unchanged.

A wrong declaration drops novelty silently, so 1 in ``DEFAULT_PROBE_PERIOD``
folded repeats is executed anyway (``OrbitProbe``): twice for a flakiness
control, once as its orbit minimum. Differing coverage flags the spec.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from fuzzer_tool.core.rand_pool import get_default_rand_pool

MAX_ORBIT = 16
MIN_ENDIAN_SIZE = 2
MAX_ENDIAN_SIZE = 8
DEFAULT_PROBE_PERIOD = 64

_ITEM_RE = re.compile(
    r"^(?P<kind>swap|endian)@(?P<off>0x[0-9a-fA-F]+|\d+):(?P<size>\d+)(?:x(?P<count>\d+))?$"
)


class OrbitKind(Enum):
    SWAP = "swap"
    ENDIAN = "endian"


@dataclass(frozen=True)
class OrbitSpec:
    kind: OrbitKind
    offset: int
    size: int
    count: int  # chunks; 1 for ENDIAN

    @property
    def end(self) -> int:
        return self.offset + self.size * self.count

    @property
    def orbit_size(self) -> int:
        return math.factorial(self.count) if self.kind is OrbitKind.SWAP else 2


def parse_orbit_spec(spec: str) -> list[OrbitSpec]:
    """Parse the ``--orbit-dedup`` grammar; raises ``ValueError`` on bad input."""
    items: list[OrbitSpec] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        m = _ITEM_RE.match(raw)
        if m is None:
            raise ValueError(f"bad orbit spec {raw!r}; expected swap@OFF:SIZExCOUNT|endian@OFF:SIZE")
        items.append(_build_item(raw, m))
    if not items:
        raise ValueError("empty orbit spec")

    ordered = sorted(items, key=lambda s: s.offset)
    for a, b in zip(ordered, ordered[1:], strict=False):
        if b.offset < a.end:
            raise ValueError(f"orbit items at {a.offset} and {b.offset} overlap")

    total = math.prod(s.orbit_size for s in items)
    if total > MAX_ORBIT:
        raise ValueError(f"orbit size {total} exceeds {MAX_ORBIT}")
    return items


def _build_item(raw: str, m: re.Match) -> OrbitSpec:
    kind = OrbitKind(m["kind"])
    size = int(m["size"])
    if kind is OrbitKind.ENDIAN:
        if m["count"] is not None:
            raise ValueError(f"{raw!r}: endian takes no count")
        if not MIN_ENDIAN_SIZE <= size <= MAX_ENDIAN_SIZE:
            raise ValueError(f"{raw!r}: endian size must be {MIN_ENDIAN_SIZE}..{MAX_ENDIAN_SIZE}")
        return OrbitSpec(kind, int(m["off"], 0), size, 1)

    if m["count"] is None:
        raise ValueError(f"{raw!r}: swap needs xCOUNT")
    count = int(m["count"])
    if size < 1 or count < 2:
        raise ValueError(f"{raw!r}: swap needs SIZE >= 1 and COUNT >= 2")
    return OrbitSpec(kind, int(m["off"], 0), size, count)


def canonical(data: bytes, specs: list[OrbitSpec]) -> bytes:
    """Orbit minimum of ``data``. Returns ``data`` itself when nothing moves."""
    buf = None
    for s in specs:
        if s.end > len(data):
            continue
        cur = data if buf is None else buf
        folded = _fold(cur, s)
        if folded != cur:
            buf = folded
    return data if buf is None else bytes(buf)


def _fold(data: bytes, s: OrbitSpec) -> bytes:
    """Replace one item's span by its minimum representative."""
    if s.kind is OrbitKind.ENDIAN:
        field = data[s.offset : s.end]
        low = min(field, field[::-1])
        return data[: s.offset] + low + data[s.end :]

    step = s.size
    chunks = [data[s.offset + i * step : s.offset + (i + 1) * step] for i in range(s.count)]
    return data[: s.offset] + b"".join(sorted(chunks)) + data[s.end :]


class Verdict(Enum):
    SAME = "same"
    DIFF = "diff"
    FLAKY = "flaky"


def judge(first: set, again: set, minimum: set) -> Verdict:
    """Compare edge sets of x, x again (control) and x's orbit minimum."""
    if first != again:
        return Verdict.FLAKY
    return Verdict.SAME if first == minimum else Verdict.DIFF


def run_probe(data: bytes, specs: list[OrbitSpec], run: Callable[[bytes], set]) -> Verdict:
    """Execute ``data`` twice and its orbit minimum once via ``run``."""
    first = run(data)
    again = run(data)
    return judge(first, again, run(canonical(data, specs)))


class OrbitProbe:
    """Samples folded repeats and tallies how often orbit members disagree."""

    MIN_PROBES = 10  # stable probes before a verdict on the spec
    SUSPECT_RATIO = 0.2  # share of stable probes that differ

    def __init__(self, period: int = DEFAULT_PROBE_PERIOD):
        self.period = period
        self.probes = 0
        self.diffs = 0
        self.flaky = 0
        self.warned = False

    def due(self, rng=None) -> bool:
        """True for about 1 in ``period`` calls; never when period is 0."""
        if self.period <= 0:
            return False
        rng = rng or get_default_rand_pool()
        return rng.random() * self.period < 1

    def record(self, verdict: Verdict) -> None:
        self.probes += 1
        self.diffs += verdict is Verdict.DIFF
        self.flaky += verdict is Verdict.FLAKY

    def suspect(self) -> bool:
        """True once enough stable probes differ: the declaration is likely wrong."""
        stable = self.probes - self.flaky
        if stable < self.MIN_PROBES:
            return False
        return self.diffs / stable > self.SUSPECT_RATIO
