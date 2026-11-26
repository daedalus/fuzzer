"""Fixed-offset field overwrite shared by the AVI / WAV / FLV field mutators.

A *site* is one chunk payload plus the table of fields it holds; ``poke``
draws a site, a field, and writes an edge value in place (length kept):

  strh payload  [type|handler|...|scale|rate|...]   Field("rate", 24, 4)
                                          ^^^^ <- 0x80000000

Strided tables repeat their fields per entry (idx1: 16-byte records).
Edge values are ints (packed at the field width) or raw bytes (fourccs).
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from fuzzer_tool.core.mutations.mp4_codec_cfg import W1_EDGES, W2_EDGES, W3_EDGES, W4_EDGES
from fuzzer_tool.core.rand_pool import RandPool

W_EDGES = {1: W1_EDGES, 2: W2_EDGES, 3: W3_EDGES, 4: W4_EDGES}


class Endian(Enum):
    """Byte order of a format's integer fields."""

    LITTLE = "little"
    BIG = "big"


class Field(NamedTuple):
    """Field at *off* bytes after the payload start (negative: chunk header)."""

    name: str
    off: int
    width: int
    edges: tuple = ()  # () -> W_EDGES[width]


class Table(NamedTuple):
    """Fields of one chunk type; *stride* > 0 repeats them per entry."""

    fields: tuple[Field, ...]
    stride: int = 0


class Site(NamedTuple):
    """One payload: absolute offset, bytes present, its table."""

    base: int
    avail: int
    table: Table


def _fitting(fields: tuple[Field, ...], avail: int) -> list[Field]:
    return [f for f in fields if f.off + f.width <= avail]


def site(base: int, avail: int, table: Table) -> Site | None:
    """Site for a payload, or None when no field fits in it."""
    room = table.stride if table.stride else avail
    if table.stride and avail < table.stride:
        return None
    return Site(base, avail, table) if _fitting(table.fields, room) else None


def poke(data: bytes, sites: list[Site], endian: Endian, rng: RandPool) -> bytes | None:
    """Overwrite one field with an edge; None if no site or value unchanged.

    Draw order: site, entry (strided tables), field, edge.
    """
    if not sites:
        return None
    base, avail, table = sites[rng.randint(0, len(sites) - 1)]

    if table.stride:
        base += table.stride * rng.randint(0, avail // table.stride - 1)
        avail = table.stride

    fields = _fitting(table.fields, avail)
    f = fields[rng.randint(0, len(fields) - 1)]
    value = rng.choice(f.edges or W_EDGES[f.width])
    raw = value if isinstance(value, bytes) else value.to_bytes(f.width, endian.value)

    at = base + f.off
    if data[at : at + f.width] == raw:
        return None
    return b"".join((data[:at], raw, data[at + f.width :]))
