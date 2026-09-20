"""Crash explanation, static half: baseline pick and field annotation.

For a novel crash, record (1) the input it was mutated from and (2) the named
fields of the crashing input, marking those that differ from it::

    parent seed ──┐
                  ├─► levenshtein_align ─► per-byte changed/src ─┐
    crash input ──┤                                              ├─► rows
                  └─► field_map.map_fields ─► [FieldSpan] ───────┘

Nothing here executes the target. Whether a changed field is what triggers
the crash is the causal search's question (``core.root_cause`` ddmin), not
this module's: a row says "changed", never "causal".
"""

from __future__ import annotations

import re
from array import array
from bisect import bisect_left, bisect_right
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from fuzzer_tool.adapters.filesystem import hash_data, rehydrate_by_hash
from fuzzer_tool.core.crash_metadata import CrashMetadata
from fuzzer_tool.core.field_map import FieldKind, FieldSpan, map_fields, span_repr
from fuzzer_tool.core.similarity import levenshtein_align

MAX_EXPLAIN_BYTES = 1 << 20  # beyond this the alignment is not worth paying inline
MAX_DIFF_ROWS = 256  # deletion rows / unformatted diff runs kept
MAX_BASELINE_BYTES = 16  # bytes of a baseline value shown

DELETED_NAME = "deleted"
DIFF_NAME = "diff@0x{:x}"

_HASH_RE = re.compile(r"[0-9a-f]{16}\Z")  # hash_data() output
_NEAREST_RE = re.compile(r"seed_(\d+)\Z")  # find_nearest_corpus() label
_CHANGED_RUN_RE = re.compile(rb"\x01+")
_NO_SOURCE = -1  # crash byte has no counterpart in the baseline


class BaselineSource(Enum):
    PARENT = "parent"  # in-memory parent seed of this mutation
    DISK = "parent_disk"  # same, rehydrated by hash from the corpus
    NEAREST = "nearest"  # closest corpus seed; the parent is unknown
    NONE = "none"


class Baseline(NamedTuple):
    data: bytes | None
    source: BaselineSource
    hash: str


class ExplainFields(NamedTuple):
    fmt: str
    rows: list[dict[str, Any]]


class _Alignment(NamedTuple):
    changed: bytearray  # per crash byte: 1 if replaced or inserted
    src: array[int]  # per crash byte: baseline offset, or _NO_SOURCE if inserted
    cuts: list[tuple[int, bytes]]  # (crash offset, bytes deleted there), sorted


def _usable(base: bytes, crash: bytes, crash_hashes: set[str]) -> bool:
    """A baseline explains a crash only if it differs and is not itself known
    to crash: diffing a crash against a crash names nothing."""
    return base != crash and hash_data(base) not in crash_hashes


def _from_disk(parent_hash: str, corpus_dir: str | Path | None) -> bytes | None:
    # The hash may come from a sidecar on disk: only a well-formed one is
    # allowed to become a path component.
    if not corpus_dir or not _HASH_RE.match(parent_hash):
        return None
    return rehydrate_by_hash(parent_hash, Path(corpus_dir))


def _from_label(label: str, corpus: list[bytes]) -> bytes | None:
    m = _NEAREST_RE.match(label)
    if m is None:
        return None

    idx = int(m.group(1))
    return corpus[idx] if idx < len(corpus) else None


def pick_baseline(
    crash: bytes,
    *,
    parent: bytes | None,
    parent_hash: str,
    crash_hashes: set[str],
    corpus_dir: str | Path | None,
    corpus: list[bytes],
    nearest_label: str,
) -> Baseline:
    """Choose the input to diff *crash* against.

    Order: the parent seed in memory, the parent rehydrated from the corpus by
    hash, the nearest corpus seed. The parent is the exact pre-mutation input;
    the nearest seed is only a guess at it, so it comes last. Any candidate
    equal to the crash, or already known to crash, is skipped.
    """
    if parent and _usable(parent, crash, crash_hashes):
        return Baseline(parent, BaselineSource.PARENT, hash_data(parent))

    disk = _from_disk(parent_hash, corpus_dir)
    if disk and _usable(disk, crash, crash_hashes):
        return Baseline(disk, BaselineSource.DISK, hash_data(disk))

    near = _from_label(nearest_label, corpus)
    if near and _usable(near, crash, crash_hashes):
        return Baseline(near, BaselineSource.NEAREST, hash_data(near))

    return Baseline(None, BaselineSource.NONE, "")


def _align(base: bytes, crash: bytes) -> _Alignment | None:
    """Walk the edit script from *base* to *crash* in crash coordinates.

    ``levenshtein_align`` positions are baseline offsets; the crash index
    advances on match/replace/insert but not on delete. Returns None if the
    script does not account for exactly the crash's bytes.
    """
    changed = bytearray(len(crash))
    src = array("q", [_NO_SOURCE]) * len(crash)
    cuts: dict[int, bytearray] = {}

    at = 0
    for op, pos, _data in levenshtein_align(base, crash):
        if op == "delete":
            cuts.setdefault(at, bytearray()).append(base[pos])
            continue
        if at >= len(crash):
            return None

        if op != "insert":
            src[at] = pos
        if op != "match":
            changed[at] = 1
        at += 1

    if at != len(crash):
        return None
    return _Alignment(changed, src, [(c, bytes(b)) for c, b in sorted(cuts.items())])


def _shown(raw: bytes) -> str:
    text = raw[:MAX_BASELINE_BYTES].hex()
    return text + "…" if len(raw) > MAX_BASELINE_BYTES else text


def _row(crash: bytes, sp: FieldSpan, changed: bool | None, old: str | None) -> dict[str, Any]:
    return {
        "offset": sp.offset,
        "width": sp.width,
        "name": sp.name,
        "kind": sp.kind.value,
        "value": span_repr(crash, sp),
        "baseline": old,
        "changed": changed,
    }


def _old_value(base: bytes, sp: FieldSpan, src: array[int]) -> str | None:
    """Baseline text of *sp* when its bytes map one-to-one, else None."""
    end = sp.offset + sp.width
    first, last = src[sp.offset], src[end - 1]
    one_to_one = first != _NO_SOURCE and last - first == sp.width - 1
    if not one_to_one or _NO_SOURCE in src[sp.offset : end]:
        return None
    return span_repr(base, sp._replace(offset=first))


def _annotate(
    crash: bytes, base: bytes, spans: list[FieldSpan], al: _Alignment
) -> list[dict[str, Any]]:
    cut_at = [c for c, _lost in al.cuts]
    rows = []
    for sp in spans:
        end = sp.offset + sp.width
        # A deletion strictly inside a span alters it even when every
        # remaining byte matches.
        cut_inside = bisect_right(cut_at, sp.offset) < bisect_left(cut_at, end)
        changed = cut_inside or al.changed.find(1, sp.offset, end) >= 0

        old = None
        if changed and not cut_inside:
            old = _old_value(base, sp, al.src)
        rows.append(_row(crash, sp, changed, old))
    return rows


def _diff_spans(al: _Alignment) -> list[FieldSpan]:
    """Runs of changed bytes, for input whose format is unknown."""
    runs = _CHANGED_RUN_RE.finditer(bytes(al.changed))
    spans = [
        FieldSpan(m.start(), m.end() - m.start(), DIFF_NAME.format(m.start()), FieldKind.UNKNOWN)
        for m in runs
    ]
    return spans[:MAX_DIFF_ROWS]


def _deleted_rows(al: _Alignment) -> list[dict[str, Any]]:
    return [
        {
            "offset": cut,
            "width": 0,
            "name": DELETED_NAME,
            "kind": FieldKind.UNKNOWN.value,
            "value": "",
            "baseline": _shown(lost),
            "changed": True,
        }
        for cut, lost in al.cuts[:MAX_DIFF_ROWS]
    ]


def explain_fields(crash: bytes, base: bytes | None) -> ExplainFields:
    """Named fields of *crash*, each marked changed or not against *base*.

    Rows are ordered by offset. ``changed`` is None for every row when there
    is no usable baseline (or the alignment could not account for the input),
    so an unknown is never reported as "unchanged". Inputs over
    ``MAX_EXPLAIN_BYTES`` are not explained.
    """
    if len(crash) > MAX_EXPLAIN_BYTES or (base and len(base) > MAX_EXPLAIN_BYTES):
        return ExplainFields("", [])

    fm = map_fields(crash)
    unknown = ExplainFields(fm.fmt, [_row(crash, sp, None, None) for sp in fm.spans])
    if base is None:
        return unknown

    al = _align(base, crash)
    if al is None:
        return unknown

    spans = fm.spans or _diff_spans(al)
    rows = _annotate(crash, base, spans, al) + _deleted_rows(al)
    rows.sort(key=lambda r: (r["offset"], r["width"]))
    return ExplainFields(fm.fmt, rows)


def explain_static(
    meta: CrashMetadata,
    crash: bytes,
    *,
    parent: bytes | None,
    parent_hash: str,
    crash_hashes: set[str],
    corpus_dir: str | Path | None,
    corpus: list[bytes],
    nearest_label: str,
) -> None:
    """Fill the baseline and field-map parts of *meta* for a novel crash."""
    base = pick_baseline(
        crash,
        parent=parent,
        parent_hash=parent_hash,
        crash_hashes=crash_hashes,
        corpus_dir=corpus_dir,
        corpus=corpus,
        nearest_label=nearest_label,
    )
    result = explain_fields(crash, base.data)

    meta.baseline_source = base.source.value
    meta.baseline_hash = base.hash
    meta.field_format = result.fmt
    meta.fields = result.rows
