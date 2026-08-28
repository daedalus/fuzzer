"""Parse Angora-style structured track files into CondStmt objects.

When a target is instrumented with DFSan, Angora's ``fparser`` emits a
structured track file listing every comparison with taint-tagged input
offsets, operand values, and branch metadata. This module consumes that
format and produces ``CondStmt`` objects for the solver pipeline.

The parser is **optional and backward compatible**: when no track file is
present, ``conds_from_cmplog_pairs`` in ``cond_stmt.py`` builds equivalent
objects from existing cmplog data alone.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator

from fuzzer_tool.core.cond_stmt import CondStmt

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text format parser
# ---------------------------------------------------------------------------

# Angora text track format (one comparison per line, whitespace-separated):
#   cmpid context order arg1_hex arg2_hex size condition [pc_hex]
#
# Fields mirror ``cond_stmt.rs``'s ``Output`` struct.  ``context`` and
# ``order`` are currently unused but preserved for forward compatibility.


def _unhex(s: str) -> bytes:
    try:
        return bytes.fromhex(s)
    except ValueError:
        return b""


def parse_track_line(line: str, cmpid: int = 0) -> CondStmt | None:
    """Parse one Angora track line into a CondStmt.

    Returns ``None`` when the line is malformed.
    """
    parts = line.split()
    if len(parts) < 7:
        return None
    try:
        # cmpid context order arg1_hex arg2_hex size condition [pc_hex]
        parsed_cmpid = int(parts[0])
        op_a = _unhex(parts[3])
        op_b = _unhex(parts[4])
        width = int(parts[5])
        result = int(parts[6])
        pc = int(parts[7], 16) if len(parts) > 7 else None
    except (ValueError, IndexError):
        return None
    if not op_a and not op_b:
        return None
    return CondStmt.from_cmplog_pair(parsed_cmpid, op_a, op_b, width, result=result, pc=pc)


def iter_track_lines(path: str) -> Iterator[str]:
    """Yield non-empty lines from a track file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def parse_track_file(path: str) -> list[CondStmt]:
    """Parse an Angora-style text track file.

    Each non-empty line is one comparison.  Deduplicates by the CondStmt
    key and skips malformed lines.  Returns an empty list when the file
    is missing.
    """
    if not os.path.exists(path):
        return []
    out: list[CondStmt] = []
    seen: set[tuple] = set()
    cmpid = 0
    for line in iter_track_lines(path):
        c = parse_track_line(line, cmpid=cmpid)
        if c is None:
            continue
        k = c.key
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        cmpid += 1
    return out


# ---------------------------------------------------------------------------
# JSON format parser (Angora cond_stmt output)
# ---------------------------------------------------------------------------


def parse_track_json(path: str) -> list[CondStmt]:
    """Parse an Angora-style JSON track file.

    The JSON is an array of objects matching the fields accepted by
    ``CondStmt.from_track_record``.  Returns an empty list when the file
    is missing, empty, or malformed.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("Failed to parse track JSON %s: %s", path, exc)
        return []

    if not isinstance(data, list):
        return []

    out: list[CondStmt] = []
    for record in data:
        c = CondStmt.from_track_record(record)
        if c is not None:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Convenience: build from cmplog file text (no DFSan required)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Single-operand records (trace-div / trace-gep)
# ---------------------------------------------------------------------------

# Operands below this are not worth a pair. Every loop counter and every
# small struct offset is a GEP index; pairing them would flood the pool
# with values plain havoc reaches by itself.
OPERAND_MIN_VALUE = 256

# Widths the shim emits: trace-div4/8 and trace-gep (pointer-sized).
_OPERAND_WIDTHS = (4, 8)

# What makes each operand kind interesting, as the value to substitute in.
# A divisor wants 0 (division by zero) and 1 (degenerate quotient); an index
# wants 0 and the all-ones word (out of bounds in both directions).
_OPERAND_TARGETS: dict[str, tuple[bytes, ...]] = {}


def _operand_targets(kind: str, width: int) -> tuple[bytes, ...]:
    key = f"{kind}{width}"
    cached = _OPERAND_TARGETS.get(key)
    if cached is not None:
        return cached
    zero = (0).to_bytes(width, "little")
    out = (zero, (1).to_bytes(width, "little")) if kind == "DIV" else (zero, b"\xff" * width)
    _OPERAND_TARGETS[key] = out
    return out


def pairs_from_operand_records(lines: Iterable[str]) -> list[tuple[bytes, bytes]]:
    """Build input-to-state pairs from ``DIV``/``GEP`` shim records.

    The shim writes ``DIV <hex value> <width> 0x<pc>`` for every runtime
    divisor and ``GEP <hex index> <width> 0x<pc>`` for every runtime GEP
    index (``-fsanitize-coverage=trace-div,trace-gep``).

    These are single operands, not comparisons: there is no second operand
    to pair them with and no result to report, so they are deliberately not
    ``CMP`` records -- ``comparison_stats``/``comparison_walls`` count
    comparisons and a divisor would inflate both. Each operand becomes
    ``(observed, interesting)`` pairs instead, which the redqueen mutator
    already knows how to apply.
    """
    out: list[tuple[bytes, bytes]] = []
    for raw in lines:
        line = raw.strip()
        kind = line[:3]
        if kind not in ("DIV", "GEP") or not line[3:4].isspace():
            continue

        parts = line[4:].split()
        if len(parts) < 2:
            continue

        try:
            value = bytes.fromhex(parts[0])
            width = int(parts[1])
        except ValueError:
            continue

        if width not in _OPERAND_WIDTHS or len(value) != width:
            continue
        if int.from_bytes(value, "little") < OPERAND_MIN_VALUE:
            continue

        out.extend((value, target) for target in _operand_targets(kind, width))

    log.debug("pairs_from_operand_records: built %d pairs", len(out))
    return out


def conds_from_cmplog_text(lines: Iterable[str]) -> list[CondStmt]:
    """Build CondStmt objects from raw cmplog log lines.

    This consumes the same ``CMP <a> <b> <result> <width> [pc]`` lines
    the cmplog shim writes, so no separate track file is needed.
    """
    out: list[CondStmt] = []
    seen: set[tuple] = set()
    cmpid = 0
    for line in lines:
        line = line.strip()
        if not line.startswith("CMP "):
            continue
        parts = line[4:].split()
        if len(parts) < 2:
            continue
        try:
            op_a = bytes.fromhex(parts[0])
            op_b = bytes.fromhex(parts[1])
            result = int(parts[2]) if len(parts) > 2 else None
            width = int(parts[3]) if len(parts) > 3 else None
            pc = int(parts[4], 16) if len(parts) > 4 else None
        except (ValueError, IndexError):
            continue
        if not op_a and not op_b:
            continue
        c = CondStmt.from_cmplog_pair(cmpid, op_a, op_b, width, result=result, pc=pc)
        k = c.key
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        cmpid += 1
    log.debug("conds_from_cmplog_text: parsed %d CondStmt from input lines", len(out))
    return out
