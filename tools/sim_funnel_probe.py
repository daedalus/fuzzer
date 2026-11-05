#!/usr/bin/env python3
"""Funnel (§3), elastic (§2) hashing (arXiv 2501.02305) and a hybrid vs the shim's linear probe.

Gate for docs/TODO.md "funnel hashing for the edge map's __AFL_PROBE_MAX
drops": only a drop win with no extra cache lines per lookup justifies a
shim prototype. Models one generation, no deletions, uniform edge ids.

Linear and funnel are greedy (insert takes the first free slot of the key's
probe sequence), so a lookup stops at the first empty slot it sees. Elastic
is not: a miss scans every (array, probe) pair in use, up to each array's
deepest probe so far (the cheapest honest bound).

    linear:  home -> home+1 -> ... (window slots)           drop if all full
    funnel:  A_1 bucket -> A_2 bucket -> ... -> A_alpha bucket
             -> B (loglog n random slots) -> C (2-choice buckets, interleaved)
    elastic: A_1 | A_2 | ... (|A_{i+1}| = |A_i|/2); probe h_{i,j} scanned in
             phi(i, j) order; batch i fills A_i to 1 - delta/2, A_{i+1} to 3/4
    hybrid:  elastic over line-aligned 8-entry buckets; an empty slot in a
             probed bucket rules its array out, so misses stop early

Growth mode replays a campaign: edges arrive one by one into a map that
starts at MAP_SIZE_DEFAULT and resizes by EdgeTracker.recommended_map_size's
policy (load >= threshold or any drop; new table empty, edges re-inserted).

Usage:
    python tools/sim_funnel_probe.py --loads 0.7,0.77,0.85,0.9 --trials 5
    python tools/sim_funnel_probe.py --mode growth --thresholds 0.7,0.9
"""

from __future__ import annotations

import argparse
import math
import statistics
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, fields
from enum import Enum

import numpy as np

from fuzzer_tool.adapters.shm import SHM_METADATA_SIZE, home_slot
from fuzzer_tool.core.elf import MAP_SIZE_DEFAULT, MAP_SIZE_MAX
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

# Paper §2: f(eps) = c * min(log^2 1/eps, log 1/delta).
ELASTIC_C = 4  # the paper's "large positive constant"
FILL_NEXT = 3 / 4  # batch 0 fills A_1, batch i fills A_{i+1}, to this fraction
NEXT_FULL = 1 / 4  # A_{i+1} free fraction at which case 3 (greedy in A_i) applies
GREEDY_CAP = 32  # draws per slot before a greedy scan gives up: misses a free slot ~e^-32
PAIR_I_SHIFT = 24  # elastic pair hash key: eid << 32 | i << 24 | j
# EdgeTracker.recommended_map_size: load trigger, minimum edges, 2x headroom.
SHIPPED_THRESHOLD = 0.7
MIN_EDGES = 100
HEADROOM = 2
PROBE_MAX = 64  # __AFL_PROBE_MAX

# Entries before the first line boundary: the table starts SHM_METADATA_SIZE into a page.
LINE_LEAD = (LINE_BYTES - SHM_METADATA_SIZE % LINE_BYTES) % LINE_BYTES // ENTRY_BYTES

# splitmix64 finalizer constants.
SM_GAMMA = 0x9E3779B97F4A7C15
SM_MUL1 = 0xBF58476D1CE4E5B9
SM_MUL2 = 0x94D049BB133111EB


class Mode(Enum):
    LOAD = "load"
    GROWTH = "growth"


class Insert(Enum):
    PLACED = "placed"
    DROPPED = "dropped"


class Kind(Enum):
    LINEAR = "linear"
    FUNNEL = "funnel"
    ELASTIC = "elastic"
    HYBRID = "hybrid"


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


def _splitmix64(x: int) -> int:
    z = (x + SM_GAMMA) & U64
    z = ((z ^ (z >> 30)) * SM_MUL1) & U64
    z = ((z ^ (z >> 27)) * SM_MUL2) & U64
    return z ^ (z >> 31)


def _splitmix64_np(x: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_splitmix64`; uint64 arithmetic wraps like ``& U64``."""
    z = x + np.uint64(SM_GAMMA)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(SM_MUL1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(SM_MUL2)
    return z ^ (z >> np.uint64(31))


def _mix(eid: int, salt: int) -> int:
    """splitmix64 of (eid, salt): an independent hash per level/attempt."""
    return _splitmix64((eid << SALT_BITS) | salt)


def phi(i: int, j: int) -> int:
    """Lemma 1 injection Z+ x Z+ -> Z+, phi(i, j) = O(i j^2).

    Bits, most significant first: 1 b_1 1 b_2 ... 1 b_q 0 a_1 ... a_p,
    with b = bits of j and a = bits of i. phi(1, 1) = 0b1110 = 14.
    """
    v = 0
    for bit in bin(j)[2:]:
        v = (v << 2) | 0b10 | int(bit)
    return (v << (1 + i.bit_length())) | i


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


class ElasticTable:
    """Elastic hashing, paper §2: non-greedy, no reordering, final load 1 - delta.

    Insert in batch i >= 1, with A_i 1-e1 full and A_{i+1} 1-e2 full:
      1. e1 > delta/2, e2 > 1/4: first free of h_{i,1..f(e1)}, else greedy in A_{i+1}
      2. e1 <= delta/2:          greedy in A_{i+1}
      3. e2 <= 1/4:              greedy in A_i (the rare expensive case)
    Search cost is the rank of the key's (i, j) in phi order among pairs in use,
    not phi itself: unused 1-D indices are never probed.
    """

    _WIDTH = 1  # entries per probe unit

    def __init__(self, size: int, delta: float, c: float = ELASTIC_C) -> None:
        if not 0 < delta < 1 or not math.log2(1 / delta).is_integer():
            raise ValueError(f"delta must be an inverse power of 2 < 1, got {delta}")

        self.size = size
        self._delta = delta
        self._c = c
        self._log_inv = math.log2(1 / delta)

        units = self._units(size)
        levels = (units - 1).bit_length()
        sizes = [max(1, units >> (k + 1)) for k in range(levels)]
        sizes[0] += units - sum(sizes)
        self._sizes = sizes
        self._offs = [sum(sizes[:k]) for k in range(levels)]
        self._sizes_np = np.array(sizes, dtype=np.uint64)
        self._offs_np = np.array(self._offs, dtype=np.int64)

        self._slots = np.zeros(size, dtype=np.uint32)
        self._fill = [0] * levels
        self._max_j = [0] * levels
        self._where: dict[int, tuple[int, int]] = {}
        self.batch = 0

        self._pairs_key: tuple[int, ...] | None = None
        self._pairs = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

    def array_sizes(self) -> list[int]:
        return list(self._sizes)

    def fill(self) -> list[int]:
        return list(self._fill)

    def max_j(self) -> list[int]:
        return list(self._max_j)

    def placement(self, eid: int) -> tuple[int, int]:
        """1-based (array, probe index) where *eid* was placed."""
        return self._where[eid]

    def _units(self, size: int) -> int:
        return size

    def _claim(self, eid: int, unit: int) -> bool:
        if self._slots[unit] != EMPTY:
            return False
        self._slots[unit] = eid
        return True

    def _slot(self, eid: int, i: int, j: int) -> int:
        key = (eid << 32) | ((i + 1) << PAIR_I_SHIFT) | j
        return self._offs[i] + _splitmix64(key) % self._sizes[i]

    def _probe(self, eid: int, i: int, top: int) -> bool:
        """Place *eid* at the first free h_{i,1..top}."""
        for j in range(1, top + 1):
            if not self._claim(eid, self._slot(eid, i, j)):
                continue
            self._fill[i] += 1
            self._max_j[i] = max(self._max_j[i], j)
            self._where[eid] = (i + 1, j)
            return True
        return False

    def _greedy(self, eid: int, i: int) -> bool:
        return self._probe(eid, i, GREEDY_CAP * self._sizes[i])

    def _free(self, i: int) -> float:
        return 1 - self._fill[i] / (self._sizes[i] * self._WIDTH)

    def _f(self, eps: float) -> int:
        return math.ceil(self._c * min(math.log2(1 / eps) ** 2, self._log_inv))

    def _place(self, eid: int) -> bool:
        if self.batch == 0:
            if self._fill[0] < math.ceil(FILL_NEXT * self._sizes[0] * self._WIDTH):
                return self._greedy(eid, 0)
            self.batch = 1

        half = self._delta / 2
        while True:
            i, nxt = self.batch - 1, self.batch
            if nxt >= len(self._sizes):  # last array: nothing to spill into
                return self._greedy(eid, i)

            e1, e2 = self._free(i), self._free(nxt)
            if e1 <= half and e2 <= NEXT_FULL:  # batch done
                self.batch += 1
                continue
            if e1 <= half:
                return self._greedy(eid, nxt)
            if e2 <= NEXT_FULL:
                return self._greedy(eid, i)
            return self._probe(eid, i, self._f(e1)) or self._greedy(eid, nxt)

    def insert(self, eid: int) -> Insert:
        return Insert.PLACED if self._place(eid) else Insert.DROPPED

    def _scan_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """(array, probe) pairs in use, phi order; rebuilt when any max_j moves."""
        key = tuple(self._max_j)
        if key == self._pairs_key:
            return self._pairs

        order = sorted(
            (phi(i + 1, j), i, j) for i, top in enumerate(key) for j in range(1, top + 1)
        )
        arr = np.array([(i, j) for _, i, j in order], dtype=np.int64).reshape(-1, 2)
        self._pairs = (arr[:, 0], arr[:, 1])
        self._pairs_key = key
        return self._pairs

    def lookup(self, eid: int) -> Lookup:
        """Scan pairs in phi order; a miss cannot stop early, it scans them all."""
        arrays, probes = self._scan_pairs()
        if not arrays.size:
            return Lookup(False, 0, 0)

        keys = (
            (np.uint64(eid) << np.uint64(32))
            | ((arrays + 1).astype(np.uint64) << np.uint64(PAIR_I_SHIFT))
            | probes.astype(np.uint64)
        )
        offs = _splitmix64_np(keys) % self._sizes_np[arrays]
        slots = self._offs_np[arrays] + offs.astype(np.int64)

        hit = np.flatnonzero(self._slots[slots] == eid)
        n = int(hit[0]) + 1 if hit.size else int(slots.size)
        lines = np.unique((SHM_METADATA_SIZE + slots[:n] * ENTRY_BYTES) // LINE_BYTES)
        return Lookup(bool(hit.size), n, int(lines.size))


class HybridTable(ElasticTable):
    """Elastic batching over funnel-style buckets: one probe reads one cache line.

        |lead| A_1: b b b b ... | A_2: b b ... | ... |tail|   b = 8 entries, line-aligned

    Within an array, insert takes the first probed bucket with a free slot, so
    a bucket with an empty slot proves the key is not later in that array:
    a miss rules arrays out and stops once every array in use is ruled out.
    """

    _WIDTH = LINE_ENTRIES

    def _units(self, size: int) -> int:
        return (size - LINE_LEAD) // LINE_ENTRIES

    def bucket_base(self, unit: int) -> int:
        return LINE_LEAD + unit * LINE_ENTRIES

    def _claim(self, eid: int, unit: int) -> bool:
        base = self.bucket_base(unit)
        for slot in range(base, base + LINE_ENTRIES):
            if self._slots[slot] != EMPTY:
                continue
            self._slots[slot] = eid
            return True
        return False

    def lookup(self, eid: int) -> Lookup:
        """Buckets in phi order, skipping arrays an empty slot already ruled out."""
        arrays, probes = self._scan_pairs()
        live = sum(1 for top in self._max_j if top)
        ruled_out: set[int] = set()
        lines: set[int] = set()
        n = 0

        for i, j in zip(arrays.tolist(), probes.tolist(), strict=True):
            if i in ruled_out:
                continue
            n += 1
            base = self.bucket_base(self._slot(eid, i, j))
            lines.add(line_of(base))
            bucket = self._slots[base : base + LINE_ENTRIES].tolist()
            if eid in bucket:
                return Lookup(True, n, len(lines))
            if EMPTY not in bucket:
                continue
            ruled_out.add(i)
            if len(ruled_out) == live:
                break
        return Lookup(False, n, len(lines))


def make_table(kind: Kind, size: int, window: int, delta: float) -> _GreedyTable | ElasticTable:
    if kind is Kind.LINEAR:
        return LinearTable(size, window)
    if kind is Kind.ELASTIC:
        return ElasticTable(size, delta)
    if kind is Kind.HYBRID:
        return HybridTable(size, delta)
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


@dataclass(frozen=True)
class GrowthStats:
    resizes: int
    final_size: int
    final_load: float
    drop_events: int
    hit_lines: float  # on the final table: what the plateau's execs pay


def recommend_size(n: int, size: int, dropped: int, threshold: float, cap: int) -> int:
    """EdgeTracker.recommended_map_size with the load threshold as a parameter."""
    saturated = dropped > 0
    if not saturated and (n < MIN_EDGES or n / size < threshold):
        return 0

    # Smallest power of two >= HEADROOM x edges, clamped to [default, cap].
    needed = (n + dropped) * HEADROOM
    rec = 1 << (needed - 1).bit_length()
    rec = max(MAP_SIZE_DEFAULT, min(cap, rec))
    return rec if rec > size else 0


def _rebuild(kind: Kind, size: int, edges: list[int]):
    """Fresh table holding *edges*: a resized SHM segment starts empty."""
    table = make_table(kind, size, PROBE_MAX, MAX_DELTA)
    placed: list[int] = []
    dropped: list[int] = []
    for eid in edges:
        (placed if table.insert(eid) is Insert.PLACED else dropped).append(eid)
    return table, placed, dropped


def run_growth(
    kind: Kind,
    edges_total: int,
    threshold: float,
    check_every: int,
    sample: int,
    pool: RandPool,
    cap: int = MAP_SIZE_MAX,
) -> GrowthStats:
    """Discover *edges_total* edges, applying the resize policy every *check_every*."""
    size = MAP_SIZE_DEFAULT
    table, placed, pending = _rebuild(kind, size, [])
    drop_events = resizes = 0

    for k, eid in enumerate(unique_ids(pool, edges_total), 1):
        if table.insert(eid) is Insert.PLACED:
            placed.append(eid)
        else:
            pending.append(eid)
            drop_events += 1
        if k % check_every and k != edges_total:
            continue

        new = recommend_size(len(placed), size, len(pending), threshold, cap)
        if not new:
            continue
        size, resizes = new, resizes + 1
        table, placed, pending = _rebuild(kind, size, placed + pending)
        drop_events += len(pending)

    picks = pool.randrange_list(len(placed), min(sample, len(placed)))
    hit = statistics.fmean(table.lookup(placed[i]).lines for i in picks) if picks else 0.0
    return GrowthStats(resizes, size, len(placed) / size, drop_events, hit)


def _growth_main(args: argparse.Namespace) -> None:
    print("edges   thr  kind    resizes final_size load  drop_events hit_ln")
    for edges in (int(x) for x in args.edges.split(",")):
        for thr in (float(x) for x in args.thresholds.split(",")):
            for kind in Kind:
                runs = [
                    run_growth(
                        kind,
                        edges,
                        thr,
                        args.check_every,
                        args.sample,
                        RandPool(seed=args.seed + t),
                    )
                    for t in range(args.trials)
                ]
                print(
                    f"{edges:6d} {thr:4.2f} {kind.value:7} "
                    f"{statistics.fmean(r.resizes for r in runs):7.1f} "
                    f"{max(r.final_size for r in runs):10d} "
                    f"{statistics.fmean(r.final_load for r in runs):5.2f} "
                    f"{statistics.fmean(r.drop_events for r in runs):11.1f} "
                    f"{statistics.fmean(r.hit_lines for r in runs):6.2f}"
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
    ap.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.LOAD.value)
    ap.add_argument("--edges", default="6000,7000,20000,60000", help="growth: campaign sizes")
    ap.add_argument("--thresholds", default="0.7,0.9", help="growth: load triggers")
    ap.add_argument("--check-every", type=int, default=256, help="growth: edges per check")
    ap.add_argument("--sample", type=int, default=500, help="growth: hit lookups sampled")
    args = ap.parse_args()

    if args.mode == Mode.GROWTH.value:
        _growth_main(args)
        return

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
