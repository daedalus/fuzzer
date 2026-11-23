"""Atom-scoped MP4 mutation and cross-file atom transplant.

The byte-level operators hit one atom in thousands and break the size chain
of every ancestor when they change a length. Two operators fix that:

  mp4_atom_mutate      havoc on ONE atom's payload (stsd, stts, tkhd, ...),
                       fourcc kept, ancestors and stco/co64 resized
  mp4_atom_transplant  graft the same-type atom of a corpus donor into this
                       file: the donor supplies the payload, this file the
                       ancestor tree (and so the track type)

        ftyp moov                         moov
              trak          donor stsd -> trak
                mdia ...                    mdia ... stsd(donor)

Both resize through ``mp4_path.splice``. A repair draw keeps the atom's own
size honest half the time and restores the old, now lying, size the other
half; the ancestors stay consistent either way.
"""

from __future__ import annotations

from enum import IntEnum

from fuzzer_tool.core.mutations.mp4_codec_cfg import W1_EDGES, W2_EDGES, W4_EDGES
from fuzzer_tool.core.mutations.mp4_path import BOX_HDR, BoxHit, find_boxes, splice
from fuzzer_tool.core.rand_pool import RandPool

ATOM_TYPES = frozenset(
    {b"stsd", b"stsc", b"stts", b"ctts", b"stss", b"stsz", b"stz2", b"stco", b"co64"}
    | {b"dref", b"elst", b"tkhd", b"mdhd", b"mvhd", b"hdlr", b"vmhd", b"smhd", b"sdtp"}
    | {b"sbgp", b"sgpd", b"tfhd", b"trun", b"mehd", b"trex"}
)
INS_MAX = 16
DEL_MAX = 16
DUP_MAX = 16
WORD_WIDTHS = (2, 4)
WORD_EDGES = {2: W2_EDGES, 4: W4_EDGES}
BIT_COUNT = 8


class Havoc(IntEnum):
    """Payload edit; values are the RNG draw."""

    FLIP = 0
    BYTE = 1
    WORD = 2
    INSERT = 3
    DELETE = 4
    DUP = 5


class Repair(IntEnum):
    """Whether the atom's own size field follows its payload."""

    KEEP = 0
    FIX = 1


def _flip(p: bytearray, rng: RandPool) -> bool:
    if not p:
        return False
    p[rng.randint(0, len(p) - 1)] ^= 1 << rng.randint(0, BIT_COUNT - 1)
    return True


def _byte(p: bytearray, rng: RandPool) -> bool:
    if not p:
        return False
    i = rng.randint(0, len(p) - 1)
    p[i] = rng.choice(W1_EDGES)
    return True


def _word(p: bytearray, rng: RandPool) -> bool:
    width = rng.choice(WORD_WIDTHS)
    if len(p) < width:
        return False
    i = rng.randint(0, len(p) - width)
    p[i : i + width] = rng.choice(WORD_EDGES[width]).to_bytes(width, "big")
    return True


def _insert(p: bytearray, rng: RandPool) -> bool:
    i = rng.randint(0, len(p))
    k = rng.randint(1, INS_MAX)
    p[i:i] = rng.randbytes(k)
    return True


def _delete(p: bytearray, rng: RandPool) -> bool:
    if not p:
        return False
    i = rng.randint(0, len(p) - 1)
    k = rng.randint(1, min(DEL_MAX, len(p) - i))
    del p[i : i + k]
    return True


def _dup(p: bytearray, rng: RandPool) -> bool:
    if not p:
        return False
    i = rng.randint(0, len(p) - 1)
    k = rng.randint(1, min(DUP_MAX, len(p) - i))
    p[i + k : i + k] = p[i : i + k]
    return True


_HAVOC = {
    Havoc.FLIP: _flip,
    Havoc.BYTE: _byte,
    Havoc.WORD: _word,
    Havoc.INSERT: _insert,
    Havoc.DELETE: _delete,
    Havoc.DUP: _dup,
}


def _payload(data: bytes, hit: BoxHit) -> bytes:
    return data[hit.start + BOX_HDR : hit.start + hit.size]


def _graft(data: bytes, hit: BoxHit, payload: bytes, rng: RandPool) -> bytes | None:
    """splice, then leave the atom's old size in place on a KEEP draw."""
    mode = Repair(rng.randint(0, 1))
    out = splice(data, hit, payload)
    if out is None or mode is Repair.FIX:
        return out
    return data[hit.start : hit.start + 4].join((out[: hit.start], out[hit.start + 4 :]))


def mutate_mp4_atom(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Havoc one atom payload; None if declined.

    Draw order: atom index, havoc op, op draws, repair.
    """
    hits = find_boxes(data, ATOM_TYPES)
    if not hits:
        return None

    hit = hits[rng.randint(0, len(hits) - 1)]
    payload = bytearray(_payload(data, hit))
    if not _HAVOC[Havoc(rng.randint(0, len(Havoc) - 1))](payload, rng):
        return None

    out = _graft(data, hit, bytes(payload), rng)
    return None if out is None else out[:max_len]


def _type_of(data: bytes, hit: BoxHit) -> bytes:
    return data[hit.start + 4 : hit.start + 8]


def transplant_atom(
    data: bytes, donor: bytes, max_len: int = 65536, *, rng: RandPool
) -> bytes | None:
    """Replace one atom of *data* with a same-type atom of *donor*.

    Draw order: base atom among shared types, donor atom of that type, repair.
    """
    theirs: dict[bytes, list[BoxHit]] = {}
    for hit in find_boxes(donor, ATOM_TYPES):
        theirs.setdefault(_type_of(donor, hit), []).append(hit)
    mine = [h for h in find_boxes(data, ATOM_TYPES) if _type_of(data, h) in theirs]
    if not mine:
        return None

    base = mine[rng.randint(0, len(mine) - 1)]
    pool = theirs[_type_of(data, base)]
    src = pool[rng.randint(0, len(pool) - 1)]
    payload = _payload(donor, src)
    if payload == _payload(data, base):
        return None

    out = _graft(data, base, payload, rng)
    return None if out is None else out[:max_len]
