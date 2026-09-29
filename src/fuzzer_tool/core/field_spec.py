"""User-declared fixed-offset integer fields, as an adapter for ``failure_inducing``.

``core/failure_inducing.py::isolate`` is format-agnostic: it needs a row of
parameter values, a domain per parameter, and an oracle. What makes PNG IHDR
special (``covering_array_mutate.isolate_png_ihdr_failure``) is only that it
ships the three adapters -- decode bytes to a row, domains, encode a row back
to bytes. This module supplies them for any format whose interesting header
fields sit at fixed byte offsets (gzip/ZIP/ELF headers, length-prefixed
records with a fixed prologue, ad-hoc protocol frames), declared on the CLI
instead of coded.

Spec grammar (fields separated by ``,``)::

    NAME@OFFSET:SIZE[be|le][=V1|V2|...]

* ``OFFSET``: byte offset from the start of the input (decimal or ``0x``).
* ``SIZE``: 1..8 bytes.
* ``be`` / ``le``: byte order, default ``be``.
* ``=V1|V2``: explicit candidate values; default is the boundary set for
  the field width (0, 1, 2, signed max/min edges, unsigned max).

Example: ``magic@0:4,version@4:1,len@6:2le=0|1|65535``.

Fields must not overlap. Offsets are absolute: a crash input that carries an
inserted prefix relative to the format's usual layout needs a spec for *that*
layout.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from fuzzer_tool.core import failure_inducing

_FIELD_RE = re.compile(
    r"^(?P<name>[A-Za-z_][\w.-]*)@(?P<off>0x[0-9a-fA-F]+|\d+):(?P<size>\d+)"
    r"(?P<endian>be|le)?(?:=(?P<vals>.+))?$"
)


@dataclass(frozen=True)
class FieldDef:
    name: str
    offset: int
    size: int
    little: bool = False
    values: tuple[int, ...] = ()
    """Explicit candidate values; empty means use :func:`default_values`."""

    @property
    def end(self) -> int:
        return self.offset + self.size

    @property
    def max_value(self) -> int:
        return (1 << (8 * self.size)) - 1


def default_values(size: int) -> tuple[int, ...]:
    """Boundary set for an unsigned ``size``-byte field: 0/1/2, signed edges, max."""
    mx = (1 << (8 * size)) - 1
    smax = mx >> 1
    vals = {0, 1, smax - 1, smax, smax + 1, mx - 1, mx}
    if size > 1:
        vals.add(2)
    return tuple(sorted(v for v in vals if 0 <= v <= mx))


def parse_spec(spec: str) -> list[FieldDef]:
    """Parse the ``--isolate-fields`` grammar; raises ``ValueError`` on bad input."""
    fields: list[FieldDef] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        m = _FIELD_RE.match(raw)
        if m is None:
            raise ValueError(f"bad field spec {raw!r}; expected NAME@OFFSET:SIZE[be|le][=V|V...]")
        size = int(m["size"])
        if not 1 <= size <= 8:
            raise ValueError(f"field {m['name']!r}: size must be 1..8, got {size}")
        vals: tuple[int, ...] = ()
        if m["vals"]:
            try:
                vals = tuple(dict.fromkeys(int(v, 0) for v in m["vals"].split("|")))
            except ValueError as e:
                raise ValueError(f"field {m['name']!r}: bad value list {m['vals']!r}") from e
            mx = (1 << (8 * size)) - 1
            bad = [v for v in vals if not 0 <= v <= mx]
            if bad:
                raise ValueError(
                    f"field {m['name']!r}: value(s) {bad} out of range for {size} byte(s)"
                )
        fields.append(FieldDef(m["name"], int(m["off"], 0), size, m["endian"] == "le", vals))
    if not fields:
        raise ValueError("empty field spec")
    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise ValueError("duplicate field names in spec")
    ordered = sorted(fields, key=lambda f: f.offset)
    for a, b in zip(ordered, ordered[1:], strict=False):
        if b.offset < a.end:
            raise ValueError(f"fields {a.name!r} and {b.name!r} overlap")
    return fields


def read_row(data: bytes, fields: Iterable[FieldDef]) -> tuple[int, ...] | None:
    """Field values of *data*, or None if any field lies past the end."""
    row = []
    for f in fields:
        if f.end > len(data):
            return None
        row.append(int.from_bytes(data[f.offset : f.end], "little" if f.little else "big"))
    return tuple(row)


def apply_row(data: bytes, fields: Iterable[FieldDef], row: tuple[int, ...]) -> bytes:
    """Copy of *data* with each field overwritten by *row*'s value (inverse of ``read_row``)."""
    buf = bytearray(data)
    for f, v in zip(fields, row, strict=True):
        buf[f.offset : f.end] = v.to_bytes(f.size, "little" if f.little else "big")
    return bytes(buf)


def value_sets(fields: list[FieldDef], baseline: bytes | None = None) -> list[tuple[int, ...]]:
    """Per-field candidate values; a known-good *baseline*'s value is added when it fits.

    The baseline's value at the same offset is the best-attested passing value
    there is, so it makes a passing companion much more likely to exist.
    """
    base_row = read_row(baseline, fields) if baseline is not None else None
    out = []
    for i, f in enumerate(fields):
        vs = list(f.values or default_values(f.size))
        if base_row is not None and base_row[i] not in vs:
            vs.append(base_row[i])
        out.append(tuple(vs))
    return out


def isolate_fields_failure(
    data: bytes,
    fails: Callable[[bytes], bool],
    fields: list[FieldDef],
    *,
    baseline: bytes | None = None,
    rng: Any = None,
    **kw: Any,
) -> failure_inducing.FailureSchema | None:
    """Which declared *fields* (and values) of failing *data* cause *fails*?

    Returns None when *data* is too short to contain every field. Extra
    keyword arguments go to ``failure_inducing.isolate``.
    """
    row = read_row(data, fields)
    if row is None:
        return None

    def oracle(candidate: tuple[int, ...]) -> bool:
        return fails(apply_row(data, fields, candidate))

    return failure_inducing.isolate(row, value_sets(fields, baseline), oracle, rng=rng, **kw)


def format_fields_schema(schema: failure_inducing.FailureSchema, fields: list[FieldDef]) -> str:
    return failure_inducing.format_schema(schema, [f.name for f in fields])
