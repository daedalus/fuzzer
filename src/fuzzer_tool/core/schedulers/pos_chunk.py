"""PositionChunkScheduler: land mutations on container chunk headers.

A container's chunk header (length, type/fourcc, size) is where a mutation
changes how the rest of the file is framed. ``boundary`` guesses those spots
from byte statistics and ``field`` learns them slowly from coverage; the
format parsers the fuzzer already ships know them exactly. This arm parses a
seed once with them -- ``core/wfc_chunks.detect_chunks`` (isobmff, webp,
riff, gif, ogg, flv, nal, asf, mpegts, zip, webm) and PNG's
``parse_png_chunks`` -- and proposes bytes around each chunk's kind::

    PNG   ....len IHDR data crc | len IDAT data ... crc | len IEND crc
               [pad kind pad]    [pad kind pad]          [pad kind pad]

    span  = [k - HEADER_PAD, k + len(kind) + HEADER_PAD)    k = kind offset

Chunk objects do not carry their offsets, so each kind is located with
``data.find`` from the end of the previous one. ``HEADER_PAD`` (4) reaches
the length field before a PNG / ISO-BMFF type and the size field after a
RIFF fourcc. Kinds that are not literal bytes in the file (GIF node names,
Ogg role tuples) or shorter than ``MIN_KIND_LEN`` are not found and skipped.

Parsing is cached per seed (LRU of ``MAX_SEEDS``); at most ``MAX_SPANS``
spans are kept, evenly spaced over the file. Seeds over ``PARSE_CAP`` bytes
are not parsed (the parsers are Python; multi-MB media would stall a pick).

Passive: ``record()`` is a no-op and nothing is persisted. Declines (``None``,
which the arena turns into a uniform offset charged to this arm) on unknown
formats, parse failures, and with probability ``EPSILON``.

Tracker-style arm (``PositionArena._add_trackers``): joins the pool once any
seed parsed (``active()``), so a target with no container corpus never
fields a pure decliner.
"""

from __future__ import annotations

import logging
from array import array
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Sequence

import xxhash

from fuzzer_tool.core.mutations.png import parse_png_chunks
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.wfc_chunks import detect_chunks

log = logging.getLogger(__name__)

EPSILON = 0.1  # uniform escape (decline) probability
HEADER_PAD = 4  # bytes either side of a kind: length / size fields
MIN_KIND_LEN = 3  # shorter kinds match by accident
MAX_SEEDS = 256  # LRU bound on the per-seed span cache
MAX_SPANS = 512  # spans kept per seed
PARSE_CAP = 1 << 20  # larger seeds are not parsed


def _kinds(data: bytes) -> list[bytes]:
    """Top-level chunk kinds in file order; [] when no parser applies."""
    png = parse_png_chunks(data)
    if png:
        return [c.chunk_type for c in png]

    parsed = detect_chunks(data)
    if not parsed:
        return []

    fmt, chunks = parsed
    return [fmt.kind(c) for c in chunks]


def _locate(data: bytes, kinds: list[bytes]) -> list[tuple[int, int]]:
    """Header span of each kind found in order (see module docstring)."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    n = len(data)
    for kind in kinds:
        if len(kind) < MIN_KIND_LEN:
            continue

        k = data.find(kind, cursor)
        if k < 0:
            continue

        spans.append((max(0, k - HEADER_PAD), min(n, k + len(kind) + HEADER_PAD)))
        cursor = k + len(kind)
    return spans


class PositionChunkScheduler:
    """Propose a byte in a parsed chunk's header."""

    name = "chunk"

    def __init__(self, rng: RandPool) -> None:
        self._rng = rng
        self._cache: OrderedDict[int, tuple[array, array]] = OrderedDict()
        self._seen = False

    def active(self) -> bool:
        """Arena gate: some seed parsed into at least one span."""
        return self._seen

    # -- parsing --------------------------------------------------------------

    def _parse(self, data: bytes) -> tuple[array, array]:
        spans: list[tuple[int, int]] = []
        if len(data) <= PARSE_CAP:
            try:
                spans = _locate(data, _kinds(data))
            except Exception:  # third-party-shaped input: never raise on the hot path
                log.debug("chunk position: parse failed", exc_info=True)
                spans = []

        n = len(spans)
        if n > MAX_SPANS:
            spans = [spans[i * n // MAX_SPANS] for i in range(MAX_SPANS)]
        if spans:
            self._seen = True
        return array("I", (a for a, _ in spans)), array("I", (b for _, b in spans))

    def _spans(self, data: bytes) -> tuple[array, array]:
        key = xxhash.xxh3_64_intdigest(data)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit

        entry = self._parse(data)
        self._cache[key] = entry
        while len(self._cache) > MAX_SEEDS:
            self._cache.popitem(last=False)
        return entry

    # -- introspection (tests, stats) ----------------------------------------

    def spans(self, data: bytes) -> list[tuple[int, int]]:
        """Cached ``(start, end)`` spans for *data* (empty before the first parse)."""
        hit = self._cache.get(xxhash.xxh3_64_intdigest(data))
        return list(zip(hit[0], hit[1], strict=True)) if hit is not None else []

    def cached_seeds(self) -> int:
        return len(self._cache)

    # -- protocol -------------------------------------------------------------

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None

        starts, ends = self._spans(data)

        # Spans starting inside a shrunk buffer are a prefix.
        k = bisect_left(starts, buf_len)
        if k == 0:
            return None

        if self._rng.random() < EPSILON:
            return None

        i = self._rng.randint(0, k - 1)
        start = starts[i]
        return start + self._rng.randint(0, min(ends[i], buf_len) - start - 1)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """No-op: spans come from the format parsers, not outcomes."""
