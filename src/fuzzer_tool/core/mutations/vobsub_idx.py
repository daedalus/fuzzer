"""VobSub .idx stream-table mutator (FFmpeg ``libavformat/mpeg.c``).

A VobSub index is a small text file: a ``# VobSub index file`` header, one
``id: <tag>, index: N`` line per subtitle stream, and per-subtitle
``timestamp: H:MM:SS:ms, filepos: <hex>`` lines. FFmpeg's vobsub demuxer
creates a stream and a queue slot for every distinct ``id:`` tag with no
bound (CVE-2026-64830, fixed in dbd495f066a85ba96b17433f4306582aa37c3951:
a file declaring more distinct stream IDs than the fixed-size
``vobsub->q[]`` array writes past it via ``ff_subtitles_queue_insert``).

The text ops in ``text_codec`` / ``sql_text`` never scale the stream table,
so the trigger — many *distinct* ids — needs this op:

  inject:  splice 1..8 fresh ``id:`` lines (the CVE trigger; compounds per
           call past the 32-slot array)
  rewrite: re-tag one existing id with a fresh one
  filepos: one filepos to an edge (0, 2^32-1, ...)
  time:    one timestamp to an edge (0, max, out-of-range)

All edits keep the line grammar (``id: xx, index: n`` /
``timestamp: h:m:s:ms, filepos: %09x``) so the demuxer keeps parsing.
"""

from __future__ import annotations

import re

from fuzzer_tool.core.rand_pool import RandPool

HEADER = b"# VobSub index file"
ID_LINES = re.compile(rb"(?m)^id: (\S\S)(, index: (\d+))?$")
TS_LINES = re.compile(rb"(?m)^timestamp: (\d+:\d+:\d+:\d+), filepos: ([0-9a-fA-F]+)")

# Two-character tags too rare to appear in real language packs.
_ID_POOL = tuple(f"{c:02x}".encode() for c in range(0x60, 0x100))
FILEPOS_EDGES = (0, 1, 0xFFFF, 0x7FFFFFFF, 2**32 - 1)
TIME_EDGES = (
    b"0:00:00:000",
    b"9:59:59:999",
    b"99:59:59:999",
    b"9:99:99:999",
    b"0:00:00:0001",
)


def _used_ids(data: bytes) -> set[bytes]:
    return {m.group(1) for m in ID_LINES.finditer(data)}


def _fresh_ids(data: bytes, rng: RandPool, count: int) -> list[bytes]:
    """*count* unused two-character id tags."""
    used = _used_ids(data)
    free = [t for t in _ID_POOL if t not in used]
    return [free[rng.randint(0, len(free) - 1)] for _ in range(count)]


def _end_of_line(data: bytes, at: int) -> int:
    """Index just past the newline of the line containing *at*."""
    while at < len(data) and data[at] != 0x0A:
        at += 1
    return min(at + 1, len(data))


def sniff_vobsub_idx(data: bytes) -> bool:
    """A VobSub index file: the header line and at least one id line."""
    if not data.startswith(HEADER):
        return False
    return ID_LINES.search(data) is not None


def _inject_ids(data: bytes, rng: RandPool) -> bytes:
    """Splice 1..8 fresh id lines after the last existing id line."""
    tags = _fresh_ids(data, rng, 1 + rng.randint(0, 7))
    lines = b"".join(b"id: %s, index: 0\n" % t for t in tags)
    matches = list(ID_LINES.finditer(data))
    at = _end_of_line(data, matches[-1].end()) if matches else len(data)
    return data[:at] + lines + data[at:]


def _rewrite_id(data: bytes, rng: RandPool) -> bytes:
    """Re-tag one existing id line with a fresh tag (a new stream id)."""
    matches = list(ID_LINES.finditer(data))
    m = matches[rng.randint(0, len(matches) - 1)]
    tag = _fresh_ids(data, rng, 1)[0]
    return data[: m.start(1)] + tag + data[m.end(1) :]


def _rewrite_filepos(data: bytes, rng: RandPool) -> bytes:
    """One filepos to an edge value, spelled %09x."""
    matches = list(TS_LINES.finditer(data))
    m = matches[rng.randint(0, len(matches) - 1)]
    edge = FILEPOS_EDGES[rng.randint(0, len(FILEPOS_EDGES) - 1)]
    return data[: m.start(2)] + b"%09x" % edge + data[m.end(2) :]


def _rewrite_time(data: bytes, rng: RandPool) -> bytes:
    """One timestamp to an edge value."""
    matches = list(TS_LINES.finditer(data))
    m = matches[rng.randint(0, len(matches) - 1)]
    edge = TIME_EDGES[rng.randint(0, len(TIME_EDGES) - 1)]
    return data[: m.start(1)] + edge + data[m.end(1) :]


def mutate_vobsub_idx(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Edit one VobSub index entry; None if declined.

    Draw order: op (among the feasible), value(s).
    """
    if not sniff_vobsub_idx(data):
        return None
    ops = ["inject", "rewrite"]
    if TS_LINES.search(data):
        ops += ["filepos", "time"]
    op = ops[rng.randint(0, len(ops) - 1)]

    if op == "inject":
        out = _inject_ids(data, rng)
    elif op == "rewrite":
        out = _rewrite_id(data, rng)
    elif op == "filepos":
        out = _rewrite_filepos(data, rng)
    else:
        out = _rewrite_time(data, rng)
    return None if out == data else out[:max_len]
