"""Merkle sites: locate hash links on a valid seed, rebuild them in mutants.

A *link* is a 32-byte field equal to ``H(input)``. Two shapes::

    LEAF  field = H(tag? region)           region = data[a:b]
    NODE  field = H(tag? data[a:a+32] || data[b:b+32])

    HEADER | chunk0 chunk1 | h(c0) h(c1) | h(h0||h1)      <- root
                              ^ LEAF        ^ NODE

``H`` is SHA-256 or SHA-256d (Bitcoin); ``tag`` is the RFC 6962 domain
byte (0x00 leaf, 0x01 node) or nothing. An odd node pairs with itself
(``a == b``, Bitcoin). A digest of the body at either end is a LEAF too.

Links form a DAG (a root may sit inside a region another link covers);
``order`` sorts them so inputs are rebuilt before their consumers.
Companion of ``checksum_sites``: no bespoke hashes, no length changes.
"""

import hashlib
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from operator import attrgetter

from fuzzer_tool.core.hashing import hash64

_SHA256 = hashlib.sha256  # the target's hash, not a corpus key (hash_data)


class Hash(Enum):
    SHA256 = "sha256"
    SHA256D = "sha256d"


class Tag(Enum):
    NONE = "none"
    RFC6962 = "rfc6962"  # 0x00 before leaf data, 0x01 before node children


class Kind(Enum):
    LEAF = "leaf"
    NODE = "node"


WIDTH = 32
LEAF_BYTE = b"\x00"
NODE_BYTE = b"\x01"
LOCATE_MAX_LEN = 16384  # larger seeds are not scanned
NODE_FINE_MAX = 4096  # node pairs tried at every offset up to here, else every 4th
NODE_COARSE_STEP = 4
MIN_LEN = 2 * WIDTH
MIN_DISTINCT = 16  # a digest has ~29 distinct bytes in 32; runs and padding fail
PREFIX_LEN = 8  # bytes of a field used as the lookup key
LEAF_LENS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
LEAF_ORIGINS = (0, 4, 8, 16, 32, 64, 80)  # chunk grid starts (80: Bitcoin header)
MAX_LINKS = 1024
MAX_SEEDS = 256
REPAIR_P = 0.9  # keep 10% broken so the check itself stays exercised

_HASHES = (Hash.SHA256, Hash.SHA256D)
_TAGS = (Tag.NONE, Tag.RFC6962)


@dataclass(frozen=True)
class Link:
    kind: Kind
    hash: Hash
    tag: Tag
    pos: int  # output field offset
    a: int  # LEAF: region start; NODE: left child offset
    b: int  # LEAF: region end; NODE: right child offset (== a: self-pair)


def _digest(algo: Hash, data: bytes) -> bytes:
    out = _SHA256(data).digest()
    return _SHA256(out).digest() if algo is Hash.SHA256D else out


def _prefix(kind: Kind, tag: Tag) -> bytes:
    if tag is Tag.NONE:
        return b""
    return LEAF_BYTE if kind is Kind.LEAF else NODE_BYTE


def _input(buf, link: Link) -> bytes:
    if link.kind is Kind.LEAF:
        return bytes(buf[link.a : link.b])

    return bytes(buf[link.a : link.a + WIDTH]) + bytes(buf[link.b : link.b + WIDTH])


def _spans(link: Link) -> tuple[tuple[int, int], ...]:
    """Byte ranges a link reads."""
    if link.kind is Kind.LEAF:
        return ((link.a, link.b),)

    return ((link.a, link.a + WIDTH), (link.b, link.b + WIDTH))


def _windows(data: bytes) -> dict[bytes, list[int]]:
    """Offsets of every 32-byte window, keyed by its first PREFIX_LEN bytes."""
    n = len(data)
    index: dict[bytes, list[int]] = {}
    for i in range(n - WIDTH + 1):
        index.setdefault(data[i : i + PREFIX_LEN], []).append(i)
    return index


def _hit(data: bytes, index, digest: bytes, taken: set[int], lo: int, hi: int) -> int | None:
    """Offset of a free window equal to *digest* outside [lo, hi), else None."""
    for pos in index.get(digest[:PREFIX_LEN], ()):
        if pos in taken or lo < pos + WIDTH and pos < hi:
            continue
        if data[pos : pos + WIDTH] != digest or len(set(digest)) < MIN_DISTINCT:
            continue
        return pos
    return None


def _regions(n: int) -> list[tuple[int, int]]:
    """Candidate leaf regions: the chunk grids, plus the body at either end."""
    out = [(0, n - WIDTH)]
    out += [(p + WIDTH, n) for p in LEAF_ORIGINS if p + WIDTH < n]
    for size in LEAF_LENS:
        for origin in LEAF_ORIGINS:
            out += [(s, s + size) for s in range(origin, n - size + 1, size)]
    return out


def _hits(data, index, taken, kind, pre, lo, hi):
    """Links whose field equals a variant digest of *pre* (the slow path)."""
    found = []
    for algo in _HASHES:
        for tag in _TAGS:
            digest = _digest(algo, _prefix(kind, tag) + pre)
            pos = _hit(data, index, digest, taken, lo, hi)
            if pos is None:
                continue
            taken.add(pos)
            found.append((algo, tag, pos))
    return found


def _maybe(keys, pre: bytes, tagged: bytes) -> bool:
    """True when some variant digest of *pre* has a field-prefix match.

    Four digests per candidate, one set probe each: the full variant loop
    runs only on a probe hit, which is rare.
    """
    plain = _SHA256(pre).digest()
    tag = _SHA256(tagged + pre).digest()
    return (
        plain[:PREFIX_LEN] in keys
        or tag[:PREFIX_LEN] in keys
        or _SHA256(plain).digest()[:PREFIX_LEN] in keys
        or _SHA256(tag).digest()[:PREFIX_LEN] in keys
    )


def _leaves(data: bytes, index, taken: set[int]) -> list[Link]:
    keys = index.keys()
    found: list[Link] = []
    for a, b in _regions(len(data)):
        region = data[a:b]
        if not _maybe(keys, region, LEAF_BYTE):
            continue
        for algo, tag, pos in _hits(data, index, taken, Kind.LEAF, region, a, b):
            found.append(Link(Kind.LEAF, algo, tag, pos, a, b))
    return found


def _nodes(data: bytes, index, taken: set[int]) -> list[Link]:
    n = len(data)
    keys = index.keys()
    step = 1 if n <= NODE_FINE_MAX else NODE_COARSE_STEP
    found: list[Link] = []
    for i in range(0, n - 2 * WIDTH + 1, step):
        left = data[i : i + WIDTH]
        for b in (i + WIDTH, i):  # adjacent siblings, then a self-pair
            pair = left + data[b : b + WIDTH]
            if not _maybe(keys, pair, NODE_BYTE):
                continue
            for algo, tag, pos in _hits(data, index, taken, Kind.NODE, pair, i, i + 2 * WIDTH):
                found.append(Link(Kind.NODE, algo, tag, pos, i, b))
    return found


def order(links: Iterable[Link]) -> list[Link]:
    """Links sorted so every input field is rebuilt before its consumer.

    Links on a dependency cycle are dropped (they cannot all hold).
    """
    links = sorted(links, key=attrgetter("pos"))
    starts = [ln.pos for ln in links]
    waiting = [0] * len(links)
    users: list[list[int]] = [[] for _ in links]
    for i in range(len(links)):
        deps = _producers(links, starts, i)
        waiting[i] = len(deps)
        for j in deps:
            users[j].append(i)

    ready = [i for i, w in enumerate(waiting) if w == 0]
    out: list[Link] = []
    while ready:
        i = ready.pop()
        out.append(links[i])
        for u in users[i]:
            waiting[u] -= 1
            if waiting[u] == 0:
                ready.append(u)
    return out


def _producers(links: list[Link], starts: list[int], i: int) -> set[int]:
    """Indices of links whose output field overlaps an input of links[i]."""
    deps: set[int] = set()
    for lo, hi in _spans(links[i]):
        j = max(bisect_right(starts, lo) - 1, 0)
        while j < len(links) and starts[j] < hi:
            if starts[j] + WIDTH > lo and j != i:
                deps.add(j)
            j += 1
    return deps


def locate(data: bytes, max_links: int | None = None) -> list[Link]:
    """Hash links of a valid seed, in rebuild order."""
    n = len(data)
    if n < MIN_LEN or n > LOCATE_MAX_LEN:
        return []

    cap = MAX_LINKS if max_links is None else max_links
    index = _windows(data)
    taken: set[int] = set()
    links = _leaves(data, index, taken)[:cap]
    links += _nodes(data, index, taken)[: max(cap - len(links), 0)]
    return order(links)


def repair(buf: bytearray, links: Iterable[Link], parent_len: int) -> int:
    """Rebuild each link's field in place; return how many fields changed."""
    if len(buf) != parent_len:
        return 0  # an edit may have moved a field

    changed = 0
    for ln in links:
        digest = _digest(ln.hash, _prefix(ln.kind, ln.tag) + _input(buf, ln))
        if buf[ln.pos : ln.pos + WIDTH] == digest:
            continue
        buf[ln.pos : ln.pos + WIDTH] = digest
        changed += 1
    return changed


class MerkleBook:
    """Per-seed link cache plus the post-mutation repair entry point."""

    def __init__(self) -> None:
        self._by_seed: OrderedDict[int, list[Link]] = OrderedDict()
        self.located = 0
        self.repairs = 0

    def seeds_tracked(self) -> int:
        return len(self._by_seed)

    def sites_for(self, parent: bytes) -> list[Link]:
        return self._links(hash64(parent), parent)

    def _links(self, key: int, parent: bytes) -> list[Link]:
        cached = self._by_seed.get(key)
        if cached is not None:
            self._by_seed.move_to_end(key)
            return cached

        links = locate(parent)
        self.located += len(links)
        self._by_seed[key] = links
        if len(self._by_seed) > MAX_SEEDS:
            self._by_seed.popitem(last=False)
        return links

    def apply(self, parent: bytes, buf: bytearray, rng) -> int:
        """Repair *buf*, a mutant of *parent*, with probability ``REPAIR_P``."""
        links = self._links(hash64(parent), parent)
        if not links or rng.random() >= REPAIR_P:
            return 0

        done = repair(buf, links, len(parent))
        self.repairs += done
        return done
