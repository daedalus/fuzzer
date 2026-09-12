"""Host CPU data-cache hierarchy, read from sysfs.

Used for one informational startup line: how the coverage segment compares
to the cache levels it could live in.

What this is NOT for, stated up front because the obvious reading is wrong:
the level a segment "fits in" does not predict access cost. The edge table
is a sparsely-touched hash table, so only the lines an execution actually
reaches are ever resident, and the size of the allocation does not
determine that. Measured on this host, with the touched set held constant
at 32 KiB and only the segment size varied:

      4,096 entries     32 KiB segment  (fits L1d)   5.36 ns/fire
      8,192 entries     64 KiB segment  (fits L2)    5.28 ns/fire
     65,536 entries    512 KiB segment  (fits L2)    5.18 ns/fire
    262,144 entries      2 MiB segment  (fits L3)    5.21 ns/fire
  1,048,576 entries      8 MiB segment  (fits L3)    5.18 ns/fire
  4,194,304 entries     32 MiB segment  (fits L3)    5.25 ns/fire

A thousandfold range in segment size, two cache boundaries crossed, and no
trend -- the smallest segment measured fractionally slowest, which is
noise. Hold the segment at 8 MiB instead and vary the DISTINCT edges fired
and the cost does move, 5.18 ns/fire at 512 distinct to 5.98 at 524,288.

So residency tracks the working set, which is roughly one 64-byte line per
distinct edge (the hash scatters edges across the table, so two edges
rarely share a line until the table is dense). That is the quantity worth
reporting, and it is why the startup line quotes an edge capacity per level
rather than claiming the segment is fast because it is small.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_SYSFS_CPU0 = Path("/sys/devices/system/cpu/cpu0/cache")

#: Bytes pulled into cache per distinct edge touched. One coherency line,
#: because the table is hash-scattered: consecutive edge ids do not land in
#: consecutive slots, so a touched entry generally has the line to itself.
#: Optimistic once the table is dense (8 entries share a 64-byte line), which
#: makes the reported edge capacities a floor, not a promise.
DEFAULT_LINE_SIZE = 64


@dataclass(frozen=True)
class CacheLevel:
    """One data-visible cache level."""

    level: int
    size_bytes: int
    line_size: int
    #: How many logical CPUs share this level. L3 is typically shared, so its
    #: nominal size is not what a single fuzzer process gets to itself.
    shared_by: int

    @property
    def name(self) -> str:
        return "L1d" if self.level == 1 else f"L{self.level}"

    @property
    def shared(self) -> bool:
        return self.shared_by > 1


def _parse_size(raw: str) -> int | None:
    """Parse sysfs cache sizes: '48K', '2048K', '266240K', occasionally '8M'."""
    raw = raw.strip()
    if not raw:
        return None
    mult = 1
    if raw[-1] in "KkMmGg":
        mult = {"k": 1024, "m": 1024**2, "g": 1024**3}[raw[-1].lower()]
        raw = raw[:-1]
    try:
        return int(raw) * mult
    except ValueError:
        return None


def _count_cpu_list(raw: str) -> int:
    """Count CPUs in a sysfs cpu list ('0', '0-3', '0,2-4')."""
    total = 0
    for part in raw.strip().split(","):
        if not part:
            continue
        if "-" in part:
            try:
                lo, hi = (int(x) for x in part.split("-", 1))
            except ValueError:
                continue
            total += hi - lo + 1
        else:
            total += 1
    return max(total, 1)


def data_cache_levels(root: Path | None = None) -> list[CacheLevel]:
    """Data-visible cache levels for cpu0, ascending by level.

    Instruction caches are skipped: the coverage segment never occupies
    them, so reporting L1i alongside L1d would invite comparing a data
    footprint against a cache it cannot use. Unified levels are included,
    since data does live there.

    Returns an empty list when sysfs is unavailable (non-Linux, a container
    that does not mount it, an unusual topology). Callers must treat that as
    "unknown", never as "no cache".

    ``root`` overrides the sysfs cache directory. Only tests pass it, and
    they need to: whether instruction caches are skipped cannot be checked
    against the real host, because the keep-the-largest-per-level rule picks
    L1d anyway whenever L1d > L1i, which is the usual case. A host with a
    larger L1i would report the wrong cache and a host-based test would
    still pass.
    """
    base = root if root is not None else _SYSFS_CPU0
    if not base.is_dir():
        return []

    by_level: dict[int, CacheLevel] = {}
    try:
        indexes = sorted(base.glob("index*"))
    except OSError as e:  # noqa: BLE001
        log.debug("cache topology unreadable: %s", e)
        return []

    for idx in indexes:
        try:
            kind = (idx / "type").read_text().strip()
            if kind == "Instruction":
                continue
            level = int((idx / "level").read_text().strip())
            size = _parse_size((idx / "size").read_text())
            if size is None or size <= 0:
                continue
            line = _parse_size((idx / "coherency_line_size").read_text()) or DEFAULT_LINE_SIZE
            try:
                shared = _count_cpu_list((idx / "shared_cpu_list").read_text())
            except OSError:
                shared = 1
        except (OSError, ValueError) as e:  # noqa: PERF203
            log.debug("skipping %s: %s", idx, e)
            continue

        # A level can appear more than once (separate entries per CPU in some
        # topologies); keep the largest reported size for it.
        prev = by_level.get(level)
        if prev is None or size > prev.size_bytes:
            by_level[level] = CacheLevel(
                level=level, size_bytes=size, line_size=line, shared_by=shared
            )

    return [by_level[k] for k in sorted(by_level)]


def smallest_level_holding(nbytes: int, levels: list[CacheLevel]) -> CacheLevel | None:
    """The first level whose size is at least ``nbytes``, or None.

    None means the footprint exceeds every level, i.e. accesses to a fully
    touched footprint of that size reach memory.
    """
    for lvl in levels:
        if nbytes <= lvl.size_bytes:
            return lvl
    return None


def _fmt_bytes(n: int) -> str:
    for unit, div in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if n >= div:
            val = n / div
            return f"{val:.0f} {unit}" if val >= 10 or val == int(val) else f"{val:.1f} {unit}"
    return f"{n} B"


def describe_map_residency(
    segment_bytes: int,
    map_entries: int,
    levels: list[CacheLevel] | None = None,
) -> str | None:
    """One informational line about the coverage map against the cache levels.

    Deliberately phrased as capacity rather than speed. The segment's size
    decides which level could hold it if it were fully touched; what decides
    actual residency is how many distinct edges a target fires, roughly one
    line each. Quoting both lets an operator compare their observed edge
    count against the crossover instead of reading "fits L2" as "fast".

    Returns None when the topology is unknown, so the caller can omit the
    line rather than print a guess.
    """
    if levels is None:
        levels = data_cache_levels()
    if not levels:
        return None

    holding = smallest_level_holding(segment_bytes, levels)
    line = levels[0].line_size or DEFAULT_LINE_SIZE

    if holding is None:
        biggest = levels[-1]
        where = f"exceeds {biggest.name} ({_fmt_bytes(biggest.size_bytes)})"
    else:
        where = f"fits {holding.name} ({_fmt_bytes(holding.size_bytes)})"
        if holding.shared:
            where += f", shared by {holding.shared_by} CPUs"

    # Edge capacity per level: how many distinct edges stay resident at one
    # line each. Stop at the first level that holds the whole table -- every
    # larger level holds it too, and repeating the table size for each of
    # them says nothing.
    caps = []
    for lvl in levels:
        edges = lvl.size_bytes // line
        if edges >= map_entries:
            caps.append(f"{lvl.name} all {map_entries:,}")
            break
        caps.append(f"{lvl.name} {edges:,}")
    else:
        # No level holds the table; the last entry is the deepest capacity.
        caps[-1] += f" of {map_entries:,}"

    return (
        f"[*] Coverage map: {_fmt_bytes(segment_bytes)} segment, {where}; "
        f"edges resident at {line} B/edge: {', '.join(caps)}"
    )
