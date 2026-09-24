"""Learned-adjacency WFC chunk reordering, generalized across container formats.

`core/wfc.py` already runs 1-D WFC over a hard-coded ``ConstraintSet``
(``png.py``, ``jpeg.py``, ``bmp.py`` only). This module is the piece
``docs/handover/handover_generators_2026-09-20.md`` (P2-1) found missing:
``AdjacencyTable.from_corpus`` has no production caller, and eleven formats
with an existing parse/serialize pair (riff, webp, isobmff, gif, ogg, flv,
asf, mpegts, webm, nal, zip) have no chunk-order table at all. All eleven
are covered here.

``ChunkFormat`` is the per-format adapter (parse/serialize/kind-extractor).
``WfcChunkTableStore`` learns one ``AdjacencyTable`` per format from admitted
corpus seeds (bigram counts over the format's own top-level kind sequence),
capped per the WFC cost-law learning
(``docs/learnings/2026-08-10-wfc-pixel-hang-cost-law.md``): a format's
vocabulary stops growing past ``MAX_TILES`` distinct kinds, and reordering
itself declines past ``MAX_CELLS`` chunks. ``wfc_reorder_chunks`` runs the
1-D collapse and maps cells back to the original chunk objects, cycling
when a kind is over-subscribed -- mirroring ``png.py::_wfc_reorder``.

Two modes:
  - ``"strict"``: collapse under the learned table as-is. WFC has no
    cardinality constraint, so a collapse can over-subscribe a kind and skip
    another; chunks are mapped back with skipped kinds placed at a table-legal
    slot, and a collapse that still leaves an adjacency the table has not
    observed is retried (``_COLLAPSE_ATTEMPTS``). Zero such adjacencies on the
    test fixtures across 500 seeds per format; a chunk with no legal slot in
    any attempt is kept, not dropped, so the guarantee is "in practice", not
    absolute.
  - ``"violate"``: copy the table, add exactly one adjacency pair that was
    *not* already present, and pin two adjacent cells so the collapse is
    forced to use it -- an ordering exactly one step outside anything the
    corpus has shown for this format. Falls back to ``"strict"`` if every
    ordered pair among the kinds present has already been observed.

Both modes fall back to a plain shuffle (respecting ``pin_first``/
``pin_last``) when there isn't enough learned data to reorder, or when the
WFC run can't find a valid ordering.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Callable

from fuzzer_tool.core.mutations.asf import (
    HEADER_OBJECT_GUID,
    parse_asf_objects,
    serialize_asf_objects,
)
from fuzzer_tool.core.mutations.flv import parse_flv, serialize_flv
from fuzzer_tool.core.mutations.gif import parse_gif, serialize_gif
from fuzzer_tool.core.mutations.isobmff import parse_boxes, serialize_boxes
from fuzzer_tool.core.mutations.mpegts import parse_ts_packets, serialize_ts_packets
from fuzzer_tool.core.mutations.nal import parse_nal_units, serialize_nal_units
from fuzzer_tool.core.mutations.ogg import parse_ogg_pages, serialize_ogg_pages
from fuzzer_tool.core.mutations.riff import parse_riff_chunks, serialize_riff
from fuzzer_tool.core.mutations.webp import parse_webp, serialize_webp
from fuzzer_tool.core.mutations.webm import Element, _encode_size_vint, parse_webm, serialize_webm
from fuzzer_tool.core.mutations.zip import ZipDoc, parse_zip, serialize_zip
from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase
from fuzzer_tool.core.wfc import AdjacencyTable, Tile, WaveGrid

# Per the cost-law learning: bound both the alphabet and the problem size a
# caller can feed this module, rather than relying solely on WaveGrid's own
# work_budget.
MAX_TILES = 64
STATE_VERSION = 1

log = logging.getLogger(__name__)
MAX_CELLS = 200_000


@dataclass
class ChunkFormat:
    """Adapter between a container format's own chunk model and WFC tiles.

    Attributes:
        name: Format name, used as the key into ``WfcChunkTableStore`` and
            matched against ``operator_registry.py``'s format sniffers.
        parse: ``data -> list[chunk] | None`` (falsy/None on parse failure).
        serialize: ``list[chunk] -> bytes``.
        kind: ``chunk -> bytes``, the tile identity (e.g. a PNG chunk type
            or an ISO-BMFF box's fourcc). Two chunks with the same kind are
            interchangeable for reordering purposes.
        pin_first: keep whatever kind occupied position 0 pinned there.
        pin_last: keep whatever kind occupied the last position pinned there.
    """

    name: str
    parse: Callable[[bytes], list[Any] | None]
    serialize: Callable[[list[Any]], bytes]
    kind: Callable[[Any], bytes]
    pin_first: bool = True
    pin_last: bool = False


def kind_sequence(fmt: ChunkFormat, data: bytes) -> list[bytes] | None:
    """Parse *data* with *fmt* and return its top-level kind sequence, or None."""
    chunks = fmt.parse(data)
    if not chunks:
        return None
    return [fmt.kind(c) for c in chunks]


class WfcChunkTableStore:
    """One learned ``AdjacencyTable`` per format name.

    Persisted through ``state_store`` (``Fuzzer._save_learned``) so a
    ``--resume`` keeps what earlier admissions taught.
    """

    def __init__(self) -> None:
        self._tables: dict[str, AdjacencyTable] = {}
        self._known: dict[str, set[bytes]] = {}

    def table_for(self, fmt_name: str) -> AdjacencyTable:
        table = self._tables.get(fmt_name)
        if table is None:
            table = AdjacencyTable()
            self._tables[fmt_name] = table
            self._known[fmt_name] = set()
        return table

    def observe(self, fmt: ChunkFormat, data: bytes) -> None:
        """Merge *data*'s kind sequence into *fmt*'s table, if it parses.

        Must never raise: called from the corpus-admission path, where a
        broken sniffer/parser must not be able to break a save.
        """
        try:
            kinds = kind_sequence(fmt, data)
        except Exception:
            return
        if not kinds or len(kinds) < 2:
            return
        table = self.table_for(fmt.name)
        known = self._known[fmt.name]
        for a, b in zip(kinds, kinds[1:]):
            if len(known) >= MAX_TILES and (a not in known or b not in known):
                continue
            table.add_forward(a, b)
            known.add(a)
            known.add(b)

    def to_dict(self) -> dict:
        return {
            "version": STATE_VERSION,
            "tables": {f: t.to_dict() for f, t in self._tables.items()},
        }

    def from_dict(self, data) -> None:
        """Replace every table with *data*'s; a malformed payload empties the store."""
        self._tables, self._known = {}, {}
        if not data:
            return
        try:
            if data.get("version") != STATE_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            tables = {str(f): AdjacencyTable.from_dict(t) for f, t in data["tables"].items()}
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            log.warning("wfc table state unreadable, starting fresh: %s", e)
            return

        self._tables = tables
        self._known = {f: set(t.to_dict()) for f, t in tables.items()}


def _table_pairs(table: AdjacencyTable, kinds: list[bytes]) -> set[tuple[bytes, bytes]]:
    """All observed forward pairs (a, b) among *kinds* in *table*."""
    return {
        (a, b)
        for a in kinds
        for b in kinds
        if a != b and table.compatible(a, b, "right")
    }


def _add_one_unobserved_pair(
    table: AdjacencyTable, kinds: list[bytes], rng
) -> tuple[AdjacencyTable, tuple[bytes, bytes] | None]:
    """Copy *table*, restricted to *kinds*, plus exactly one new forward pair.

    Returns the augmented table and the pair added, or (a table equivalent
    to the restriction of *table*, None) if every ordered pair among
    *kinds* is already observed.
    """
    observed = _table_pairs(table, kinds)
    candidates = [
        (a, b) for a in kinds for b in kinds if a != b and (a, b) not in observed
    ]
    new_table = AdjacencyTable()
    for a, b in observed:
        new_table.add_forward(a, b)
    if not candidates:
        return new_table, None
    a, b = candidates[rng.randint(0, len(candidates) - 1)]
    new_table.add_forward(a, b)
    return new_table, (a, b)


def _shuffle_fallback(fmt: ChunkFormat, chunks: list[Any], rng) -> list[Any]:
    """Shuffle *chunks*, respecting pin_first/pin_last, when WFC can't help."""
    if len(chunks) < 2:
        return list(chunks)
    body = list(chunks)
    first = body.pop(0) if fmt.pin_first and len(body) > 1 else None
    last = body.pop() if fmt.pin_last and len(body) > 1 else None
    rng.shuffle(body)
    result = []
    if first is not None:
        result.append(first)
    result.extend(body)
    if last is not None:
        result.append(last)
    return result


# Slots examined per leftover chunk. Bounded so a large input with many skipped
# chunks costs O(leftovers * _PLACEMENT_PROBES), not O(leftovers * cells).
_PLACEMENT_PROBES = 64


def _legal_slot(
    kinds: list[bytes],
    tail_kind: bytes | None,
    head_len: int,
    kind: bytes,
    taken: dict[int, Any],
    table: AdjacencyTable,
    rng,
) -> int | None:
    """A free slot where *kind* sits between two table-observed neighbours.

    Slot ``i`` is the gap before ``kinds[i]``; slot ``len(kinds)`` is the gap
    before the pinned tail. Examines a window of consecutive slots from a
    random start: every slot when the sequence is short (independent random
    draws would miss the one legal slot of a short sequence a third of the
    time), at most ``_PLACEMENT_PROBES`` when it is long.
    """
    n = len(kinds)
    span = n - head_len + 1
    start = rng.randint(0, span - 1)
    for step in range(min(_PLACEMENT_PROBES, span)):
        i = head_len + (start + step) % span
        if i in taken:
            continue
        prev = kinds[i - 1] if i > 0 else None
        nxt = kinds[i] if i < n else tail_kind
        if (prev is None or table.compatible(prev, kind, "right")) and (
            nxt is None or table.compatible(kind, nxt, "right")
        ):
            return i
    return None


def _place_leftovers(
    fmt: ChunkFormat,
    base: list[Any],
    head_len: int,
    tail: list[Any],
    leftovers: list[Any],
    table: AdjacencyTable,
    rng,
) -> list[Any]:
    """Return *base* with *leftovers* inserted, then *tail* appended.

    A leftover goes into a random legal slot (see ``_legal_slot``); at most
    one leftover per slot, so a leftover is never adjacent to another leftover
    and every checked pair is a real emitted pair. A chunk with no legal slot
    is appended just before *tail*: it is kept, at the cost of one adjacency
    the table has not observed (the caller retries the collapse when that
    happens).
    """
    if not leftovers:
        return base + tail
    kinds = [fmt.kind(c) for c in base]
    tail_kind = fmt.kind(tail[0]) if tail else None
    slots: dict[int, Any] = {}
    unplaced: list[Any] = []
    for chunk in leftovers:
        i = _legal_slot(kinds, tail_kind, head_len, fmt.kind(chunk), slots, table, rng)
        if i is None:
            unplaced.append(chunk)
        else:
            slots[i] = chunk
    out: list[Any] = []
    for i in range(len(base) + 1):
        if i in slots:
            out.append(slots[i])
        if i < len(base):
            out.append(base[i])
    out.extend(unplaced)
    out.extend(tail)
    return out


# Collapses attempted per call. WFC has no cardinality constraint, so a
# collapse can over-subscribe a kind: the surplus cells are skipped when the
# pool runs dry (which joins their neighbours) and the chunks of a skipped kind
# become leftovers. Either can leave an adjacency the table has not observed,
# and a fresh collapse usually avoids it, so retry rather than emit it.
_COLLAPSE_ATTEMPTS = 8


def _illegal_adjacencies(fmt: ChunkFormat, order: list[Any], table: AdjacencyTable) -> int:
    """Count consecutive chunk pairs in *order* that *table* has not observed."""
    return sum(
        1
        for a, b in zip(order, order[1:])
        if not table.compatible(fmt.kind(a), fmt.kind(b), "right")
    )


def _collapse_cells(
    fmt: ChunkFormat,
    chunks: list[Any],
    kinds_present: list[bytes],
    work_table: AdjacencyTable,
    violated_pair: tuple[bytes, bytes] | None,
    rng,
) -> list[bytes] | None:
    """One 1-D WFC collapse over ``len(chunks)`` cells; None if it fails.

    A fresh ``WaveGrid`` each call: ``run`` snapshots the grid it is handed as
    the state restarts return to, so a grid cannot be run twice.
    """
    tiles = [Tile(name=k) for k in kinds_present]
    wave = WaveGrid(tiles, work_table, width=len(chunks), height=1)

    def pin(cell: int, kind: bytes) -> None:
        kid = kinds_present.index(kind)
        for j in range(len(tiles)):
            wave.superpositions[cell][j] = j == kid

    if fmt.pin_first:
        pin(0, fmt.kind(chunks[0]))
    if fmt.pin_last:
        pin(len(chunks) - 1, fmt.kind(chunks[-1]))

    if violated_pair is not None:
        # The pair occupies cells pos and pos+1, both of which must exist and
        # neither of which may be a pinned cell.
        lo = 1 if fmt.pin_first else 0
        hi = len(chunks) - (3 if fmt.pin_last else 2)
        if hi >= lo:
            pos = rng.randint(lo, hi)
            pin(pos, violated_pair[0])
            pin(pos + 1, violated_pair[1])

    result = wave.run(seed=rng.randint(0, 2**31), max_restarts=3, ac3_budget=2000)
    if not result or not result[0] or any(c is None for c in result[0]):
        return None
    return result[0]


def _map_cells(
    fmt: ChunkFormat,
    chunks: list[Any],
    cells: list[bytes],
    table: AdjacencyTable,
    rng,
) -> list[Any]:
    """Turn collapsed cells back into an ordered chunk list."""
    # Pinned chunks are reserved by identity up front, not drawn from the
    # per-kind pools: a pool can be exhausted before its pinned cell is
    # reached, which used to let a pin silently move.
    head = chunks[:1] if fmt.pin_first else []
    tail = chunks[-1:] if fmt.pin_last else []
    by_kind: dict[bytes, list[Any]] = {}
    for c in chunks[len(head) : len(chunks) - len(tail)]:
        by_kind.setdefault(fmt.kind(c), []).append(c)

    placed = list(head)
    for kind_name in cells[len(head) : len(cells) - len(tail)]:
        pool = by_kind.get(kind_name)
        if pool:
            placed.append(pool.pop(0))
    # Chunks whose kind the collapse under-placed (an over-subscribed kind,
    # or a kind WFC skipped) are kept rather than dropped, each at a slot the
    # table allows if one turns up, and always ahead of the pinned-last chunk.
    leftovers = [c for pool in by_kind.values() for c in pool]
    return _place_leftovers(fmt, placed, len(head), tail, leftovers, table, rng)


def wfc_reorder_chunks(
    fmt: ChunkFormat,
    chunks: list[Any],
    table: AdjacencyTable,
    rng,
    mode: str = "strict",
    max_len: int | None = None,
) -> bytes:
    """Reorder *chunks* (as parsed by ``fmt.parse``) using a learned WFC table.

    Falls back to a pin-respecting shuffle when *table* has nothing useful
    for these kinds, when *chunks* is too large to afford (per the cost-law
    caller guard), or when the collapse doesn't converge.
    """

    def finish(order: list[Any]) -> bytes:
        out = fmt.serialize(order)
        return out[:max_len] if max_len is not None else out

    if len(chunks) < 3 or len(chunks) > MAX_CELLS:
        return finish(_shuffle_fallback(fmt, chunks, rng))

    kinds_present = list(dict.fromkeys(fmt.kind(c) for c in chunks))
    if len(kinds_present) < 2 or len(_table_pairs(table, kinds_present)) == 0:
        return finish(_shuffle_fallback(fmt, chunks, rng))

    work_table = table
    violated_pair: tuple[bytes, bytes] | None = None
    if mode == "violate":
        work_table, violated_pair = _add_one_unobserved_pair(table, kinds_present, rng)

    best: tuple[list[Any], int] | None = None
    for _ in range(_COLLAPSE_ATTEMPTS):
        cells = _collapse_cells(fmt, chunks, kinds_present, work_table, violated_pair, rng)
        if cells is None:
            break
        order = _map_cells(fmt, chunks, cells, work_table, rng)
        illegal = _illegal_adjacencies(fmt, order, work_table)
        if best is None or illegal < best[1]:
            best = (order, illegal)
        if illegal == 0:
            break
    if best is None:
        return finish(_shuffle_fallback(fmt, chunks, rng))
    return finish(best[0])


# ── Per-format adapters: isobmff, riff, webp, gif; then ogg, flv, nal, asf, mpegts, zip, webm below ──
#
# Sniffers mirror operator_registry.py's isobmff_chunk_mutate / riff_chunk_mutate /
# webp_chunk_mutate / gif_chunk_mutate _FORMAT_SNIFFERS entries verbatim (not
# imported from there, to keep this module import-independent of the operator
# registry until self-registration runs) -- riff and webp share the RIFF magic
# but are mutually exclusive on the WEBP form-type tag, same as those entries.

ISOBMFF_FORMAT = ChunkFormat(
    name="isobmff",
    parse=parse_boxes,
    serialize=serialize_boxes,
    kind=lambda b: b.box_type,
    pin_first=False,
    pin_last=False,
)

WEBP_FORMAT = ChunkFormat(
    name="webp",
    parse=parse_webp,
    serialize=serialize_webp,
    kind=lambda c: c.fourcc,
    pin_first=False,
    pin_last=False,
)

GIF_FORMAT = ChunkFormat(
    name="gif",
    parse=parse_gif,
    serialize=serialize_gif,
    kind=lambda n: n.kind.encode(),
    pin_first=True,  # "header" (magic) must stay first
    pin_last=True,  # "trailer" (0x3B) must stay last
)

# RIFF's serializer needs the container's form_type ("WAVE"/"AVI ", ...),
# which parse_riff_chunks returns alongside the chunk list rather than as
# part of any chunk -- so RIFF_FORMAT.parse drops it (for kind_sequence /
# table learning, which don't need it) and _try_riff below binds a
# form_type-aware serializer per call via dataclasses.replace.
RIFF_FORMAT = ChunkFormat(
    name="riff",
    parse=lambda d: (lambda r: r[1] if r else None)(parse_riff_chunks(d)),
    serialize=lambda chunks: serialize_riff(b"    ", chunks),  # replaced per-call
    kind=lambda c: c.fourcc,
    pin_first=False,
    pin_last=False,
)


def _sniff_isobmff(d: bytes) -> bool:
    return d[4:8] == b"ftyp"


def _sniff_webp(d: bytes) -> bool:
    return d[:4] == b"RIFF" and d[8:12] == b"WEBP"


def _sniff_riff(d: bytes) -> bool:
    return len(d) >= 12 and d[:4] == b"RIFF" and d[8:12] != b"WEBP"


def _sniff_gif(d: bytes) -> bool:
    return d[:3] == b"GIF"


def _try_isobmff(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    chunks = parse_boxes(data)
    return (ISOBMFF_FORMAT, chunks) if chunks else None


def _try_webp(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    chunks = parse_webp(data)
    return (WEBP_FORMAT, chunks) if chunks else None


def _try_gif(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    nodes = parse_gif(data)
    return (GIF_FORMAT, nodes) if nodes else None


def _try_riff(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    parsed = parse_riff_chunks(data)
    if not parsed:
        return None
    form_type, chunks = parsed
    if not chunks:
        return None
    bound = dataclasses.replace(
        RIFF_FORMAT, serialize=lambda cs, ft=form_type: serialize_riff(ft, cs)
    )
    return bound, chunks


# ── Rollout: ogg, flv, nal, asf, mpegts, zip, webm ──
#
# Formats whose serializer needs state the chunk list does not carry (FLV's
# header/trailing size, a ZIP's EOCD) follow the RIFF pattern: the shared
# ``*_FORMAT`` has a placeholder serializer good enough for table learning and
# the tests, and ``_try_*`` binds the real one per call with
# ``dataclasses.replace``.

# Ogg: a page's tile is (BOS/EOS role, logical-stream serial). Role alone would
# make every data page of a single-stream file identical, so reordering could
# not change anything; the serial is what makes multiplexed-stream interleaving
# (the order that matters to a demuxer) visible. The continuation bit is
# payload, not sequence, so it is masked out. BOS must open the file and EOS
# close it, as GIF's header and trailer do.
_OGG_ROLE_MASK = 0x06  # BOS (0x02) and EOS (0x04); 0x01 is "continued packet"


def _ogg_kind(page: Any) -> bytes:
    return bytes([page.header_type & _OGG_ROLE_MASK]) + page.serial_number.to_bytes(4, "little")


OGG_FORMAT = ChunkFormat(
    name="ogg",
    parse=parse_ogg_pages,
    serialize=serialize_ogg_pages,
    kind=_ogg_kind,
    pin_first=True,
    pin_last=True,
)

# FLV: a tag's tile is its type (script/video/audio). Nothing is pinned: the
# leading onMetaData script tag is convention, and moving it is a legitimate
# thing to try.
_FLV_STUB_HEADER = b"FLV\x01\x05\x00\x00\x00\x09"

FLV_FORMAT = ChunkFormat(
    name="flv",
    parse=lambda d: (lambda r: r[1] if r else None)(parse_flv(d)),
    serialize=lambda tags: serialize_flv(_FLV_STUB_HEADER, tags, None),  # replaced per-call
    kind=lambda t: bytes([t.tag_type & 0xFF]),
    pin_first=False,
    pin_last=False,
)

# NAL: a unit's tile is its nal_unit_type (SPS/PPS/IDR/slice/SEI ...), which
# is exactly the ordering a decoder's parameter-set handling depends on.
NAL_FORMAT = ChunkFormat(
    name="nal",
    parse=parse_nal_units,
    serialize=serialize_nal_units,
    kind=lambda u: bytes([u.unit_type]),
    pin_first=False,
    pin_last=False,
)

# ASF: a top-level object's tile is its GUID. The header object stays first:
# ``parse_asf_objects`` (and the operator's own sniffer) key on it, so moving
# it would make the output undetectable as the format it came from.
ASF_FORMAT = ChunkFormat(
    name="asf",
    parse=parse_asf_objects,
    serialize=serialize_asf_objects,
    kind=lambda o: o.guid,
    pin_first=True,
    pin_last=False,
)

# MPEG-TS: a packet's tile is its PID, so the table learns PAT/PMT/PES
# interleaving. Reordering keeps every packet, so the output stays 188-aligned.
MPEGTS_FORMAT = ChunkFormat(
    name="mpegts",
    parse=parse_ts_packets,
    serialize=serialize_ts_packets,
    kind=lambda p: p.pid.to_bytes(2, "big"),
    pin_first=False,
    pin_last=False,
)

# ZIP: an entry's tile is its role. Well-known ordering-sensitive names
# (`mimetype`, `[Content_Types].xml`, `META-INF/*`, manifests) are their own
# tiles; everything else groups by extension, and directories by trailing '/'.
# ``serialize_zip`` recomputes local offsets, CD size and counts from the new
# layout, so a reorder never leaves the central directory pointing at the wrong
# entry -- the consistency the handover flagged as the reason ZIP went last.
_ZIP_SPECIAL_NAMES = frozenset(
    {
        b"mimetype",
        b"[Content_Types].xml",
        b"_rels/.rels",
        b"AndroidManifest.xml",
        b"classes.dex",
        b"resources.arsc",
    }
)
_ZIP_EOCD_STUB = b"PK\x05\x06" + bytes(18)


def _zip_kind(entry: Any) -> bytes:
    name: bytes = entry.name
    if name.endswith(b"/"):
        return b"dir"
    if name in _ZIP_SPECIAL_NAMES or name.startswith(b"META-INF/"):
        return name
    leaf = name.rsplit(b"/", 1)[-1]
    return b"ext:" + (leaf.rsplit(b".", 1)[-1].lower() if b"." in leaf else b"")


ZIP_FORMAT = ChunkFormat(
    name="zip",
    parse=lambda d: (lambda doc: doc.entries if doc else None)(parse_zip(d)),
    serialize=lambda entries: serialize_zip(ZipDoc(entries, _ZIP_EOCD_STUB, b"")),  # per-call
    kind=_zip_kind,
    pin_first=False,
    pin_last=False,
)


def _sniff_ogg(d: bytes) -> bool:
    return d[:4] == b"OggS"


def _sniff_flv(d: bytes) -> bool:
    return d[:3] == b"FLV"


def _sniff_nal(d: bytes) -> bool:
    return d[:4] in (b"\x00\x00\x00\x01", b"\x00\x00\x01\x00") or d[:3] == b"\x00\x00\x01"


def _sniff_asf(d: bytes) -> bool:
    return d[:16] == HEADER_OBJECT_GUID


def _sniff_mpegts(d: bytes) -> bool:
    return len(d) >= 376 and d[0] == 0x47 and d[188] == 0x47


def _sniff_zip(d: bytes) -> bool:
    return d[:2] == b"PK"


# WebM (EBML): ``parse_webm`` returns exactly two top-level elements -- the
# EBML header and the Segment -- always, so the true top level has nothing
# to reorder (this was P2-1's "webm is the one still open" gap). The
# reorderable sequence one level down is the Segment's own children
# (SeekHead/Info/Tracks/Cues/Cluster*/Tags/...), whose relative order a
# streaming demuxer's linear scan actually depends on, unlike ISO-BMFF's
# free top level. A child's tile is its element ID; nothing is pinned: none
# of these are needed to still recognize the file as WebM (the EBML header
# and Segment ID, both outside this reorder's scope, already do that).
#
# Unlike a leaf format, "reorder just the children" has no bytes of its own
# to serialize to: the shared global's placeholder serializer (used by the
# generic per-format tests, which call ``fmt.serialize`` directly rather
# than through ``_try_webm``) has to wrap them in *some* valid EBML header
# and Segment to be self-sufficient -- mirroring FLV_FORMAT's stub header,
# not a full round-trip of the real file's own framing. ``_try_webm`` below
# binds the file's *real* EBML header and Segment for actual calls.
#
# The Segment's size is precomputed and written as a normal known-size vint
# (not the unknown-size/"streaming" marker some real WebM files use):
# ``_read_vint``/``_parse_element`` in this module only ever decode the
# single-byte pattern ``\xff`` as "unknown" (a length-1 vint whose value
# happens to equal that length's all-ones sentinel) -- an 8-byte ``\xff*8``
# fallback, which ``_encode_size_vint`` emits for a value too large to
# otherwise represent, is read back as that same 1-byte marker plus 7 stray
# data bytes, corrupting whatever follows. Not otherwise reachable at these
# payload sizes; found by giving this stub an unknown-size Segment first.
_WEBM_STUB_EBML_HEADER = Element(
    elem_id=0x1A45DFA3, id_raw=b"\x1a\x45\xdf\xa3", size_raw=b"\x80", size_val=0, data=b""
)


def _serialize_webm_stub(children: list[Any]) -> bytes:
    payload = b"".join(serialize_webm([c]) for c in children)
    segment = Element(
        elem_id=0x18538067,
        id_raw=b"\x18\x53\x80\x67",
        size_raw=_encode_size_vint(len(payload)),
        size_val=len(payload),
        data=b"",
        children=list(children),
    )
    return serialize_webm([_WEBM_STUB_EBML_HEADER, segment])


WEBM_FORMAT = ChunkFormat(
    name="webm",
    parse=lambda d: (lambda t: t[1].children if t and t[1].children else None)(parse_webm(d)),
    serialize=_serialize_webm_stub,  # replaced per-call by _try_webm
    kind=lambda el: el.elem_id.to_bytes(4, "big"),
    pin_first=False,
    pin_last=False,
)


def _sniff_webm(d: bytes) -> bool:
    return d[:4] == b"\x1a\x45\xdf\xa3"


def _try_ogg(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    pages = parse_ogg_pages(data)
    return (OGG_FORMAT, pages) if pages else None


def _try_flv(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    parsed = parse_flv(data)
    if not parsed:
        return None
    header, tags, trailing = parsed
    if not tags:
        return None
    bound = dataclasses.replace(
        FLV_FORMAT, serialize=lambda ts, h=header, tr=trailing: serialize_flv(h, ts, tr)
    )
    return bound, tags


def _try_nal(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    units = parse_nal_units(data)
    return (NAL_FORMAT, units) if units else None


def _try_asf(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    objs = parse_asf_objects(data)
    return (ASF_FORMAT, objs) if objs else None


def _try_mpegts(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    packets = parse_ts_packets(data)
    return (MPEGTS_FORMAT, packets) if packets else None


def _try_zip(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    doc = parse_zip(data)
    if doc is None or not doc.entries:
        return None
    bound = dataclasses.replace(
        ZIP_FORMAT,
        serialize=lambda es, d=doc: serialize_zip(ZipDoc(es, d.eocd_fixed, d.eocd_comment)),
    )
    return bound, doc.entries


def _try_webm(data: bytes) -> tuple[ChunkFormat, list[Any]] | None:
    top = parse_webm(data)
    if not top:
        return None
    ebml_header, segment = top
    if not segment.children:
        return None
    bound = dataclasses.replace(
        WEBM_FORMAT,
        serialize=lambda children, h=ebml_header, seg=segment: serialize_webm(
            [h, dataclasses.replace(seg, children=children)]
        ),
    )
    return bound, segment.children


# (format name, sniffer, parser) -- checked in this order; the sniffers are
# mutually exclusive (webp/riff split on the WEBP tag, isobmff/gif have
# disjoint magics) so at most one entry ever matches a given input.
_FORMATS: list[tuple[str, Callable[[bytes], bool], Callable[[bytes], Any]]] = [
    ("isobmff", _sniff_isobmff, _try_isobmff),
    ("webp", _sniff_webp, _try_webp),
    ("riff", _sniff_riff, _try_riff),
    ("gif", _sniff_gif, _try_gif),
    ("ogg", _sniff_ogg, _try_ogg),
    ("flv", _sniff_flv, _try_flv),
    ("nal", _sniff_nal, _try_nal),
    ("asf", _sniff_asf, _try_asf),
    ("mpegts", _sniff_mpegts, _try_mpegts),
    ("zip", _sniff_zip, _try_zip),
    ("webm", _sniff_webm, _try_webm),
]


# Fraction of applicable calls that use "violate" mode over "strict".
VIOLATE_RATE = 0.3


class WfcChunkMutator(MutatorBase):
    """``wfc_reorder_learned``: per-format learned-adjacency chunk reordering.

    Wires ``AdjacencyTable.from_corpus`` (P2-1 of
    ``docs/handover/handover_generators_2026-09-20.md``) to a production
    caller: a per-format table, learned from admitted corpus seeds via
    ``on_new_coverage``, applied to isobmff/webp/riff/gif/ogg/flv/nal/asf/
    mpegts/zip top-level (and webm's Segment-level) chunk
    sequences. Gated on ``--wfc`` (the same flag the PNG/JPEG/BMP WFC
    reorder ops already use) plus the format being sniffed, so it costs
    nothing when WFC mode is off.
    """

    name = "wfc_reorder_learned"
    category = "format"
    use_wfc: bool = False  # set by Fuzzer from --wfc; gates learning too

    def __init__(self) -> None:
        self.store = WfcChunkTableStore()

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        if not context.wfc_enabled or not data:
            return False
        return any(sniff(data) for _, sniff, _ in _FORMATS)

    def mutate(
        self,
        data: bytes,
        rng,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx,
    ) -> bytes | None:
        if not data:
            return None
        for fmt_name, sniff, try_parse in _FORMATS:
            if not sniff(data):
                continue
            parsed = try_parse(data)
            if parsed is None:
                return None
            fmt, chunks = parsed
            if len(chunks) < 3:
                return None
            table = self.store.table_for(fmt_name)
            mode = "violate" if rng.random() < VIOLATE_RATE else "strict"
            try:
                out = wfc_reorder_chunks(
                    fmt, chunks, table, rng, mode=mode, max_len=max_len or None
                )
            except Exception:
                return None
            return out if out != data else None
        return None

    def on_new_coverage(self, seed: bytes, new_edges: int) -> None:
        # Off means free: the NAL sniffer is a per-byte Python scan.
        if not self.use_wfc or not seed:
            return
        for fmt_name, sniff, try_parse in _FORMATS:
            if not sniff(seed):
                continue
            parsed = try_parse(seed)
            if parsed is not None:
                fmt, _chunks = parsed
                self.store.observe(fmt, seed)
            return


#: The registered instance; the fuzzer sets its flag and persists its store.
WFC_MUTATOR = WfcChunkMutator()


def _register() -> None:
    from fuzzer_tool.core.operator_registry import REGISTRY

    if WFC_MUTATOR.name not in REGISTRY.names():
        REGISTRY.register_mutator(WFC_MUTATOR)


_register()
