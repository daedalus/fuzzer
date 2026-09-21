"""Learned-adjacency WFC chunk reordering, generalized across container formats.

`core/wfc.py` already runs 1-D WFC over a hard-coded ``ConstraintSet``
(``png.py``, ``jpeg.py``, ``bmp.py`` only). This module is the piece
``docs/handover/handover_generators_2026-09-20.md`` (P2-1) found missing:
``AdjacencyTable.from_corpus`` has no production caller, and eleven formats
with an existing parse/serialize pair (riff, webp, isobmff, gif, ogg, flv,
asf, mpegts, webm, nal, zip) have no chunk-order table at all.

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
  - ``"strict"``: collapse under the learned table as-is. Never emits an
    adjacency the table hasn't observed.
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
from dataclasses import dataclass
from typing import Any, Callable

from fuzzer_tool.core.mutations.gif import parse_gif, serialize_gif
from fuzzer_tool.core.mutations.isobmff import parse_boxes, serialize_boxes
from fuzzer_tool.core.mutations.riff import parse_riff_chunks, serialize_riff
from fuzzer_tool.core.mutations.webp import parse_webp, serialize_webp
from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase
from fuzzer_tool.core.wfc import AdjacencyTable, Tile, WaveGrid

# Per the cost-law learning: bound both the alphabet and the problem size a
# caller can feed this module, rather than relying solely on WaveGrid's own
# work_budget.
MAX_TILES = 64
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

    Not persisted across process restarts (see the handover's "per-format
    table store ... persisted through state_store" proposal) -- this is the
    in-process half: a table built from whatever corpus admissions this run
    has already seen. Wiring survival across restarts was deferred; see the
    handover-fix commit message for the tradeoff.
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
    if len(chunks) < 3 or len(chunks) > MAX_CELLS:
        reordered = _shuffle_fallback(fmt, chunks, rng)
        out = fmt.serialize(reordered)
        return out[:max_len] if max_len is not None else out

    kinds_present = list(dict.fromkeys(fmt.kind(c) for c in chunks))
    if len(kinds_present) < 2 or len(_table_pairs(table, kinds_present)) == 0:
        reordered = _shuffle_fallback(fmt, chunks, rng)
        out = fmt.serialize(reordered)
        return out[:max_len] if max_len is not None else out

    work_table = table
    violated_pair: tuple[bytes, bytes] | None = None
    if mode == "violate":
        work_table, violated_pair = _add_one_unobserved_pair(table, kinds_present, rng)

    tiles = [Tile(name=k) for k in kinds_present]
    wave = WaveGrid(tiles, work_table, width=len(chunks), height=1)

    if fmt.pin_first:
        first_kind = fmt.kind(chunks[0])
        if first_kind in kinds_present:
            fid = kinds_present.index(first_kind)
            for j in range(len(tiles)):
                wave.superpositions[0][j] = j == fid
    if fmt.pin_last:
        last_kind = fmt.kind(chunks[-1])
        if last_kind in kinds_present:
            lid = kinds_present.index(last_kind)
            for j in range(len(tiles)):
                wave.superpositions[-1][j] = j == lid

    if violated_pair is not None:
        a, b = violated_pair
        lo = 1 if fmt.pin_first else 0
        hi = len(chunks) - (2 if fmt.pin_last else 1)
        if hi > lo:
            pos = rng.randint(lo, hi)
            aid = kinds_present.index(a)
            bid = kinds_present.index(b)
            for j in range(len(tiles)):
                wave.superpositions[pos][j] = j == aid
                wave.superpositions[pos + 1][j] = j == bid

    result = wave.run(seed=rng.randint(0, 2**31), max_restarts=3, ac3_budget=2000)

    if result is None or not result or any(c is None for c in result[0]):
        reordered = _shuffle_fallback(fmt, chunks, rng)
        out = fmt.serialize(reordered)
        return out[:max_len] if max_len is not None else out

    new_order = result[0]
    by_kind: dict[bytes, list[Any]] = {}
    for c in chunks:
        by_kind.setdefault(fmt.kind(c), []).append(c)

    reordered = []
    for kind_name in new_order:
        pool = by_kind.get(kind_name)
        if pool:
            reordered.append(pool.pop(0))
    # Leftover chunks whose kind the collapse under-placed (over-subscribed
    # kind, or a kind WFC dropped): append them rather than losing bytes.
    for leftover in by_kind.values():
        reordered.extend(leftover)

    out = fmt.serialize(reordered)
    return out[:max_len] if max_len is not None else out


# ── Per-format adapters (rollout order per the handover: isobmff, riff/webp, gif) ──
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


# (format name, sniffer, parser) -- checked in this order; the sniffers are
# mutually exclusive (webp/riff split on the WEBP tag, isobmff/gif have
# disjoint magics) so at most one entry ever matches a given input.
_FORMATS: list[tuple[str, Callable[[bytes], bool], Callable[[bytes], Any]]] = [
    ("isobmff", _sniff_isobmff, _try_isobmff),
    ("webp", _sniff_webp, _try_webp),
    ("riff", _sniff_riff, _try_riff),
    ("gif", _sniff_gif, _try_gif),
]

# Fraction of applicable calls that use "violate" mode over "strict".
VIOLATE_RATE = 0.3


class WfcChunkMutator(MutatorBase):
    """``wfc_reorder_learned``: per-format learned-adjacency chunk reordering.

    Wires ``AdjacencyTable.from_corpus`` (P2-1 of
    ``docs/handover/handover_generators_2026-09-20.md``) to a production
    caller: a per-format table, learned from admitted corpus seeds via
    ``on_new_coverage``, applied to isobmff/webp/riff/gif top-level chunk
    sequences. Gated on ``--wfc`` (the same flag the PNG/JPEG/BMP WFC
    reorder ops already use) plus the format being sniffed, so it costs
    nothing when WFC mode is off.
    """

    name = "wfc_reorder_learned"
    category = "format"

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
        if not seed:
            return
        for fmt_name, sniff, try_parse in _FORMATS:
            if not sniff(seed):
                continue
            parsed = try_parse(seed)
            if parsed is not None:
                fmt, _chunks = parsed
                self.store.observe(fmt, seed)
            return


def _register() -> None:
    from fuzzer_tool.core.operator_registry import REGISTRY

    m = WfcChunkMutator()
    if m.name not in REGISTRY.names():
        REGISTRY.register_mutator(m)


_register()
