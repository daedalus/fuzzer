"""Checksum site locator and repair (TaintScope-style), ``--checksum-sites``.

TaintScope finds the checksum check with taint analysis, fuzzes past it,
then repairs the checksum in every mutant. This fuzzer has no taint engine,
so the location step is done on a valid seed instead: for each candidate
field, test whether it equals a known checksum of a candidate region. A hit
is a *site*: which bytes, which algorithm, which region.

    seed:  [ header | ........ body ........ | crc32(body) ]
                      ^------ region ------^   ^-- field

After a mutation, ``repair`` recomputes every site, so the target's
integrity check passes and the mutant reaches the code behind it.

``core/analyzers/analyzer_checksum_learner`` recovers *unknown* models but
patches only a trailing field; this module locates the field and the region.

Limits (all deliberate):
- Fixed algorithms only: CRC-32, Adler-32, CRC-16 (CCITT-false, XMODEM),
  Fletcher-16, 16-bit sum. 8-bit fields match by chance too often.
- 16-bit fields are tried only in the header window, at the tail, or where a
  cmplog operand equals the field bytes (the taint substitute). Random data
  yields a 16-bit false site in ~2-3% of seeds; 32-bit false sites: none seen.
- Regions: ``[start, field)`` with ``start`` in a few header offsets or just
  after a found field (PNG: ``prev_end + 4``), or ``(field, end]``.
- Seeds above ``LOCATE_MAX_LEN`` are not scanned.
- Sites are derived per seed and never persisted.
"""

from __future__ import annotations

import binascii
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import xxhash


class Algo(Enum):
    CRC32 = "crc32"
    ADLER32 = "adler32"
    CRC16 = "crc16"
    CRC16_XMODEM = "crc16_xmodem"
    FLETCHER16 = "fletcher16"
    SUM16 = "sum16"


class Endian(Enum):
    BIG = "big"
    LITTLE = "little"


class Span(Enum):
    PREFIX = "prefix"  # region = [start, field)
    SUFFIX = "suffix"  # region = (field, end of buffer]


class Anchor(Enum):
    HEAD = "head"  # field offset is fixed
    TAIL = "tail"  # field is the last bytes; follows length changes


WIDTH = {
    Algo.CRC32: 4,
    Algo.ADLER32: 4,
    Algo.CRC16: 2,
    Algo.CRC16_XMODEM: 2,
    Algo.FLETCHER16: 2,
    Algo.SUM16: 2,
}
# Specific first: on a tie the earlier algorithm owns the field.
_ALGOS_32 = (Algo.CRC32, Algo.ADLER32)
_ALGOS_16 = (Algo.CRC16, Algo.CRC16_XMODEM, Algo.FLETCHER16, Algo.SUM16)

LOCATE_MAX_LEN = 16384  # larger seeds are not scanned
HEAD_WINDOW = 32  # 16-bit fields are tried in the first bytes
MIN_REGION = 4  # a shorter region proves nothing
STATIC_STARTS = (0, 4, 8, 12, 16)  # region starts tried on every seed
NEXT_STARTS = (0, 4)  # after a found field: end, end + 4 (PNG length prefix)
MAX_PASSES = 4
MAX_SITES = 8
MAX_HINT_HITS = 4
MAX_HINTS = 64
MAX_SEEDS = 256
REPAIR_P = 0.9  # keep 10% broken so the check itself stays exercised

_FLETCHER_MOD = 255
_CRC16_CCITT_INIT = 0xFFFF
_ADLER_INIT = 1


@dataclass(frozen=True)
class Site:
    algo: Algo
    endian: Endian
    span: Span
    anchor: Anchor
    pos: int  # field offset at locate time (TAIL: re-derived per buffer)
    start: int  # PREFIX region start

    @property
    def width(self) -> int:
        return WIDTH[self.algo]


def _fletcher16(data: bytes) -> int:
    n = len(data)
    if n == 0:
        return 0
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    s1 = int(arr.sum()) % _FLETCHER_MOD
    s2 = int((arr * np.arange(n, 0, -1, dtype=np.int64)).sum()) % _FLETCHER_MOD
    return (s2 << 8) | s1


_FUNCS: dict[Algo, Callable[[bytes], int]] = {
    Algo.CRC32: zlib.crc32,
    Algo.ADLER32: zlib.adler32,
    Algo.CRC16: lambda d: binascii.crc_hqx(d, _CRC16_CCITT_INIT),
    Algo.CRC16_XMODEM: lambda d: binascii.crc_hqx(d, 0),
    Algo.FLETCHER16: _fletcher16,
    Algo.SUM16: lambda d: sum(d) & 0xFFFF,
}


def checksum(algo: Algo, data: bytes) -> int:
    return _FUNCS[algo](data)


def _field(data: bytes, pos: int, width: int, endian: Endian) -> int:
    return int.from_bytes(data[pos : pos + width], endian.value)


def _anchor(pos: int, width: int, n: int) -> Anchor:
    return Anchor.TAIL if pos + width == n else Anchor.HEAD


@dataclass(frozen=True)
class _Windows:
    """Per-seed precomputation shared by every scan: byte singles and the
    big/little-endian 32-bit value at each offset."""

    singles: list[bytes]
    big: list[int]
    little: list[int]


_BE_WEIGHTS = np.array([1 << 24, 1 << 16, 1 << 8, 1], dtype=np.uint64)


def _windows(data: bytes) -> _Windows:
    arr = np.frombuffer(data, dtype=np.uint8)
    win = np.lib.stride_tricks.sliding_window_view(arr, 4).astype(np.uint64)
    return _Windows(
        [data[i : i + 1] for i in range(len(data))],
        (win * _BE_WEIGHTS).sum(axis=1).tolist(),
        (win * _BE_WEIGHTS[::-1]).sum(axis=1).tolist(),
    )


def _scan32(data: bytes, w: _Windows, algo: Algo, start: int) -> list[tuple[int, Endian]]:
    """Fields equal to the checksum of ``[start, field)``, one running pass.

    The checksum of ``[start, p+1)`` extends that of ``[start, p)`` by one
    byte, so every field offset costs one C call instead of a rescan.
    """
    fn = zlib.crc32 if algo is Algo.CRC32 else zlib.adler32
    run = 0 if algo is Algo.CRC32 else _ADLER_INIT
    last = len(data) - 4
    if start > last:
        return []

    first = data[start]
    varied = False
    hits = []
    for p in range(start + 1, last + 1):
        run = fn(w.singles[p - 1], run)
        # crc32 of 4 x 0xFF is 0xFFFFFFFF: a constant region proves nothing.
        varied = varied or data[p - 1] != first
        if p - start < MIN_REGION or not varied:
            continue
        if w.big[p] == run:
            hits.append((p, Endian.BIG))
        elif w.little[p] == run:
            hits.append((p, Endian.LITTLE))
    return hits


def _positions16(data: bytes, hints: Iterable[bytes]) -> list[int]:
    """Where a 16-bit field is plausible: header window, tail, hinted bytes."""
    n = len(data)
    pos = set(range(min(HEAD_WINDOW, n - 1)))
    pos.add(n - 2)
    for op in hints:
        if len(op) != 2:
            continue
        at = data.find(op)
        for _ in range(MAX_HINT_HITS):
            if at < 0:
                break
            pos.add(at)
            at = data.find(op, at + 1)
    return sorted(p for p in pos if 0 <= p <= n - 2)


def _endian16(data: bytes, p: int, want: int) -> Endian | None:
    if _field(data, p, 2, Endian.BIG) == want:
        return Endian.BIG
    if _field(data, p, 2, Endian.LITTLE) == want:
        return Endian.LITTLE
    return None


def _region_ok(region: bytes) -> bool:
    """Long enough and not one repeated byte (sum16 of zeros is 0)."""
    return len(region) >= MIN_REGION and region.count(region[0]) != len(region)


def _hit16(data: bytes, p: int, algo: Algo, span: Span, start: int) -> Site | None:
    n = len(data)
    region = data[start:p] if span is Span.PREFIX else data[p + 2 :]
    if not _region_ok(region):
        return None
    endian = _endian16(data, p, checksum(algo, region))
    if endian is None:
        return None
    anchor = _anchor(p, 2, n) if span is Span.PREFIX else Anchor.HEAD
    return Site(algo, endian, span, anchor, p, start if span is Span.PREFIX else 0)


def _scan16(data: bytes, starts: Iterable[int], hints: Iterable[bytes]) -> list[Site]:
    out = []
    for p in _positions16(data, hints):
        for algo in _ALGOS_16:
            for start in starts:
                if start < p:
                    out.append(_hit16(data, p, algo, Span.PREFIX, start))
            out.append(_hit16(data, p, algo, Span.SUFFIX, 0))
    return [s for s in out if s is not None]


def _suffix32(data: bytes, p: int, algo: Algo) -> Site | None:
    region = data[p + 4 :]
    if not _region_ok(region):
        return None
    want = checksum(algo, region)
    for endian in Endian:
        if _field(data, p, 4, endian) == want:
            return Site(algo, endian, Span.SUFFIX, Anchor.HEAD, p, 0)
    return None


def _pass(data: bytes, w: _Windows, starts: set[int], hints: Iterable[bytes]) -> list[Site]:
    n = len(data)
    out = []
    for algo in _ALGOS_32:
        for start in sorted(starts):
            for p, endian in _scan32(data, w, algo, start):
                out.append(Site(algo, endian, Span.PREFIX, _anchor(p, 4, n), p, start))
        for p in range(min(HEAD_WINDOW, n - 3)):
            out.append(_suffix32(data, p, algo))
    out += _scan16(data, sorted(starts), hints)
    return [s for s in out if s is not None]


def _free(site: Site, taken: list[tuple[int, int]]) -> bool:
    a, b = site.pos, site.pos + site.width
    return all(b <= x or a >= y for x, y in taken)


def locate(data: bytes, hints: Iterable[bytes] = (), max_sites: int = MAX_SITES) -> list[Site]:
    """Checksum sites of a valid seed, earliest field first."""
    if len(data) < MIN_REGION + 2 or len(data) > LOCATE_MAX_LEN:
        return []

    hints = list(hints)
    w = _windows(data)
    starts = {s for s in STATIC_STARTS if s < len(data)}
    found: list[Site] = []
    taken: list[tuple[int, int]] = []
    for _ in range(MAX_PASSES):
        fresh = []
        for site in _pass(data, w, starts, hints):
            if len(found) + len(fresh) >= max_sites or not _free(site, taken):
                continue
            fresh.append(site)
            taken.append((site.pos, site.pos + site.width))
        if not fresh:
            break
        found += fresh
        for site in fresh:
            end = site.pos + site.width
            starts |= {end + d for d in NEXT_STARTS}
    return sorted(found, key=lambda s: s.pos)[:max_sites]


def _bounds(site: Site, n: int, parent_len: int) -> tuple[int, int, int] | None:
    """(field offset, region start, region end) in a buffer of length *n*."""
    w = site.width
    pos = n - w if site.anchor is Anchor.TAIL else site.pos
    if pos < 0 or pos + w > n:
        return None
    if site.span is Span.SUFFIX:
        return pos, pos + w, n
    if site.anchor is Anchor.HEAD and n != parent_len:
        return None  # an edit before the field may have moved it
    return pos, site.start, pos


def repair(buf: bytearray, sites: Iterable[Site], parent_len: int) -> int:
    """Recompute each site's field in place; return how many were written."""
    done = 0
    for site in sites:
        bounds = _bounds(site, len(buf), parent_len)
        if bounds is None:
            continue
        pos, a, b = bounds
        if a < 0 or b - a < MIN_REGION:
            continue
        value = checksum(site.algo, bytes(buf[a:b]))
        buf[pos : pos + site.width] = value.to_bytes(site.width, site.endian.value)
        done += 1
    return done


def hint_operands(pairs: Iterable[tuple[bytes, bytes]] | None) -> list[bytes]:
    """16-bit cmplog operands: a compared field that came from the input."""
    out: list[bytes] = []
    for pair in pairs or ():
        out += [op for op in pair if len(op) == 2]
        if len(out) >= MAX_HINTS:
            break
    return out[:MAX_HINTS]


class SiteBook:
    """Per-seed site cache plus the post-mutation repair entry point."""

    def __init__(self) -> None:
        self._by_seed: OrderedDict[int, list[Site]] = OrderedDict()
        self.located = 0
        self.repairs = 0

    def seeds_tracked(self) -> int:
        return len(self._by_seed)

    def sites_for(self, parent: bytes, hints: Iterable[bytes] = ()) -> list[Site]:
        key = xxhash.xxh3_64_intdigest(parent)
        cached = self._by_seed.get(key)
        if cached is not None:
            self._by_seed.move_to_end(key)
            return cached

        sites = locate(parent, hints=hints)
        self.located += len(sites)
        self._by_seed[key] = sites
        if len(self._by_seed) > MAX_SEEDS:
            self._by_seed.popitem(last=False)
        return sites

    def apply(self, parent: bytes, buf: bytearray, rng, hints_fn=None) -> int:
        """Repair *buf*, a mutant of *parent*, with probability ``REPAIR_P``."""
        key = xxhash.xxh3_64_intdigest(parent)
        hints = hints_fn() if hints_fn is not None and key not in self._by_seed else ()
        sites = self.sites_for(parent, hints)
        if not sites or rng.random() >= REPAIR_P:
            return 0

        done = repair(buf, sites, len(parent))
        self.repairs += done
        return done
