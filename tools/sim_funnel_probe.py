#!/usr/bin/env python3
"""Funnel hashing (arXiv 2501.02305 §3) vs the shim's bounded linear probe.

Gate for docs/TODO.md "funnel hashing for the edge map's __AFL_PROBE_MAX
drops": only a drop win with no extra cache lines per lookup justifies a
shim prototype. Models one generation, no deletions, uniform edge ids.

Both tables are greedy (insert takes the first free slot of the key's probe
sequence), so a lookup stops at the first empty slot it sees.

    linear:  home -> home+1 -> ... (window slots)           drop if all full
    funnel:  A_1 bucket -> A_2 bucket -> ... -> A_alpha bucket
             -> B (loglog n random slots) -> C (2-choice buckets, interleaved)

Usage:
    python tools/sim_funnel_probe.py --loads 0.7,0.77,0.85,0.9 --trials 5
"""

from __future__ import annotations

import argparse
import math
import statistics
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, fields
from enum import Enum

from fuzzer_tool.adapters.shm import SHM_METADATA_SIZE, home_slot
from fuzzer_tool.core.rand_pool import RandPool

EMPTY = 0
ENTRY_BYTES = 8  # struct __afl_entry { u32 edge_id; u32 count; }
LINE_BYTES = 64
LINE_ENTRIES = LINE_BYTES // ENTRY_BYTES
U32_SPAN = 1 << 32
U64 = (1 << 64) - 1

# Paper §3: delta <= 1/8, alpha = ceil(4 log 1/delta + 10), beta = ceil(2 log 1/delta).
MAX_DELTA = 1 / 8
LEVEL_RATIO = 3 / 4  # a_{i+1} = 3 a_i / 4
SPECIAL_FRAC = 5 / 8  # |A_{alpha+1}| in [delta n / 2, 3 delta n / 4]: midpoint
SALT_BITS = 8

# splitmix64 finalizer constants.
SM_GAMMA = 0x9E3779B97F4A7C15
SM_MUL1 = 0xBF58476D1CE4E5B9
SM_MUL2 = 0x94D049BB133111EB


class Insert(Enum):
    PLACED = "placed"
    DROPPED = "dropped"


class Kind(Enum):
    LINEAR = "linear"
    FUNNEL = "funnel"


@dataclass(frozen=True)
class Lookup:
    found: bool
    probes: int
    lines: int


@dataclass(frozen=True)
class TrialStats:
    drop_rate: float
    hit_probes: float
    hit_probes_max: int
    hit_lines: float
    hit_lines_max: int
    miss_probes: float
    miss_probes_max: int
    miss_lines: float
    miss_lines_max: int


def line_of(slot: int) -> int:
    """Cache line of *slot*: the table starts SHM_METADATA_SIZE bytes into a page."""
    return (SHM_METADATA_SIZE + slot * ENTRY_BYTES) // LINE_BYTES


def _mix(eid: int, salt: int) -> int:
    """splitmix64 of (eid, salt): an independent hash per level/attempt."""
    z = (((eid << SALT_BITS) | salt) + SM_GAMMA) & U64
    z = ((z ^ (z >> 30)) * SM_MUL1) & U64
    z = ((z ^ (z >> 27)) * SM_MUL2) & U64
    return z ^ (z >> 31)


def unique_ids(pool: RandPool, count: int, exclude: set[int] | None = None) -> list[int]:
    """*count* distinct non-zero u32 edge ids not in *exclude*, in draw order."""
    seen = set(exclude or ())
    seen.add(EMPTY)
    out: list[int] = []
    while len(out) < count:
        for eid in pool.randrange_list(U32_SPAN, count - len(out)):
            if eid in seen:
                continue
            seen.add(eid)
            out.append(eid)
    return out


class _GreedyTable:
    """Open addressing without reordering; subclasses supply the probe sequence."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._slots = array("I", bytes(4 * size))

    def _sequence(self, eid: int) -> Iterator[int]:
        raise NotImplementedError

    def insert_probes(self, eid: int) -> int | None:
        """Place *eid* in the first free slot; return probes used, None on drop."""
        for n, slot in enumerate(self._sequence(eid), 1):
            if self._slots[slot] != EMPTY:
                continue
            self._slots[slot] = eid
            return n
        return None

    def insert(self, eid: int) -> Insert:
        return Insert.DROPPED if self.insert_probes(eid) is None else Insert.PLACED

    def lookup(self, eid: int) -> Lookup:
        """Scan until *eid* or an empty slot; count probes and distinct lines."""
        lines: set[int] = set()
        n = 0
        for slot in self._sequence(eid):
            n += 1
            lines.add(line_of(slot))
            held = self._slots[slot]
            if held == eid:
                return Lookup(True, n, len(lines))
            if held == EMPTY:
                break
        return Lookup(False, n, len(lines))

    def slot_of(self, eid: int) -> int | None:
        for slot in self._sequence(eid):
            held = self._slots[slot]
            if held == eid:
                return slot
            if held == EMPTY:
                return None
        return None


class LinearTable(_GreedyTable):
    """The shim's probe: Fibonacci home slot, then window-1 successors, wrapping."""

    def __init__(self, size: int, window: int) -> None:
        super().__init__(size)
        self._window = min(window, size)

    def _sequence(self, eid: int) -> Iterator[int]:
        home = home_slot(eid, self.size)
        size = self.size
        for k in range(self._window):
            slot = home + k
            yield slot - size if slot >= size else slot


def _geometric_counts(total: int, levels: int) -> list[int]:
    """Split *total* buckets into *levels* shrinking by LEVEL_RATIO, each >= 1."""
    if total < levels:
        raise ValueError(f"{total} buckets cannot fill {levels} levels")
    weights = [LEVEL_RATIO**i for i in range(levels)]
    scale = total / sum(weights)
    counts = [max(1, int(w * scale)) for w in weights]
    counts[0] += total - sum(counts)
    return counts


class FunnelTable(_GreedyTable):
    """Funnel hashing, paper §3, sized for final load 1 - delta.

    Layout (slots):  [A_1 | A_2 | ... | A_alpha | B | C]
    A_i holds counts[i] buckets of beta slots; B is uniform probing capped at
    loglog n attempts; C is two-choice with buckets of 2 loglog n.
    """

    def __init__(self, size: int, delta: float) -> None:
        super().__init__(size)
        if not 0 < delta <= MAX_DELTA or not math.log2(1 / delta).is_integer():
            raise ValueError(f"delta must be an inverse power of 2 <= 1/8, got {delta}")

        log_inv = round(math.log2(1 / delta))
        self._alpha = 4 * log_inv + 10
        self._beta = 2 * log_inv

        special = round(SPECIAL_FRAC * delta * size)
        main_buckets = (size - special) // self._beta
        special = size - main_buckets * self._beta
        self._levels = self._level_offsets(_geometric_counts(main_buckets, self._alpha))

        # Special array: B (uniform probing) then C (two-choice buckets).
        loglog = math.ceil(math.log2(math.log2(size)))
        self._b_tries = loglog
        self._c_bucket = 2 * loglog
        self._c_buckets = max(1, (special // 2) // self._c_bucket)
        self._b_off = main_buckets * self._beta
        self._b_size = special - self._c_buckets * self._c_bucket
        self._c_off = self._b_off + self._b_size

    def _level_offsets(self, counts: list[int]) -> list[tuple[int, int]]:
        out = []
        off = 0
        for nb in counts:
            out.append((off, nb))
            off += nb * self._beta
        return out

    def layout_size(self) -> int:
        return self._c_off + self._c_buckets * self._c_bucket

    def _sequence(self, eid: int) -> Iterator[int]:
        beta = self._beta
        for i, (off, nb) in enumerate(self._levels):
            base = off + (_mix(eid, i) % nb) * beta
            yield from range(base, base + beta)

        salt = self._alpha
        for t in range(self._b_tries):
            yield self._b_off + _mix(eid, salt + t) % self._b_size
        salt += self._b_tries

        # Two distinct buckets, probed a0, b0, a1, b1, ...
        nc, cb = self._c_buckets, self._c_bucket
        a = _mix(eid, salt) % nc
        b = (a + 1 + _mix(eid, salt + 1) % (nc - 1)) % nc if nc > 1 else a
        for k in range(cb):
            yield self._c_off + a * cb + k
            if b != a:
                yield self._c_off + b * cb + k


def make_table(kind: Kind, size: int, window: int, delta: float) -> _GreedyTable:
    if kind is Kind.LINEAR:
        return LinearTable(size, window)
    return FunnelTable(size, delta)


def _summary(found: list[Lookup]) -> tuple[float, int, float, int]:
    if not found:
        return 0.0, 0, 0.0, 0
    probes = [r.probes for r in found]
    lines = [r.lines for r in found]
    return statistics.fmean(probes), max(probes), statistics.fmean(lines), max(lines)


def run_trial(
    kind: Kind, size: int, window: int, delta: float, load: float, misses: int, pool: RandPool
) -> TrialStats:
    """Insert load*size fresh ids, then look up every placed id and *misses* absent ones."""
    table = make_table(kind, size, window, delta)
    ids = unique_ids(pool, int(load * size))
    placed = [e for e in ids if table.insert(e) is Insert.PLACED]

    hits = [table.lookup(e) for e in placed]
    absent = unique_ids(pool, misses, exclude=set(ids))
    miss = [table.lookup(e) for e in absent]

    return TrialStats(
        1 - len(placed) / len(ids),
        *_summary(hits),
        *_summary(miss),
    )


def _aggregate(trials: list[TrialStats]) -> TrialStats:
    """Means average across trials; *_max fields take the max."""
    vals = []
    for f in fields(TrialStats):
        col = [getattr(t, f.name) for t in trials]
        vals.append(max(col) if f.name.endswith("_max") else statistics.fmean(col))
    return TrialStats(*vals)


def _row(load: float, kind: Kind, s: TrialStats) -> str:
    return (
        f"{load:5.2f} {kind.value:7} {s.drop_rate:9.2e} "
        f"{s.hit_probes:6.2f} {s.hit_probes_max:4d} {s.hit_lines:6.2f} {s.hit_lines_max:4d} "
        f"{s.miss_probes:7.2f} {s.miss_probes_max:4d} {s.miss_lines:6.2f} {s.miss_lines_max:4d}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=8192, help="entries (AFL_MAP_SIZE)")
    ap.add_argument("--window", type=int, default=64, help="__AFL_PROBE_MAX")
    ap.add_argument("--delta", type=float, default=MAX_DELTA, help="funnel design slack")
    ap.add_argument("--loads", default="0.6,0.7,0.77,0.85,0.9,0.95")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--misses", type=int, default=2000, help="absent-id lookups per trial")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print("load  kind    drop_rate  hit_p  max hit_ln  max  miss_p  max miss_ln max")
    for load in (float(x) for x in args.loads.split(",")):
        for kind in Kind:
            trials = [
                run_trial(
                    kind,
                    args.size,
                    args.window,
                    args.delta,
                    load,
                    args.misses,
                    RandPool(seed=args.seed + t),
                )
                for t in range(args.trials)
            ]
            print(_row(load, kind, _aggregate(trials)))


if __name__ == "__main__":
    main()
