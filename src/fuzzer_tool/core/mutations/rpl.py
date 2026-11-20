"""ARMovie / RPL mutator (FFmpeg ``rpl``: Escape 124/130 video, PCM/ADPCM audio).

Text header, one field per line (leading decimal is the value):

  0 "ARMovie"  1-3 title/copyright/author  4 video format  5-7 w/h/bpp
  8 fps (may be a fraction)  9 audio format  10 rate  11 channels  12 bits
  13 frames/chunk  14 last chunk index  15-16 chunk sizes  17 catalog offset
  18-19 sprite  [20 key frame list, only with video]
  catalog (at offset):  "<offset> , <video_size> ; <audio_size>" per chunk

FFmpeg 9.0.2 computes `offset + video_size` before its overflow check.
Header ints overflow read_int's error guard past 214748363. Rows are
found at the catalog offset (bytes between header and catalog are kept).
Repairs keep them reachable: a resized header shifts the offset, a video
toggle adds/drops the key frame list line, and dup/drop moves a count
that matched the rows.
"""

from __future__ import annotations

import re
from enum import IntEnum

from fuzzer_tool.core.mutations.frame_stream import FrameOp, reframe
from fuzzer_tool.core.rand_pool import RandPool

MAGIC = b"ARMovie\n"
HDR_LINES = 20  # +1 key frame list line when video_format != 0
SNIFF_WINDOW = 4096

VIDEO_LINE = 4
FPS_LINE = 8
AUDIO_LINE = 9
BITS_LINE = 12
COUNT_LINE = 14
CATALOG_LINE = 17
FIRST_INT_LINE = VIDEO_LINE
KEYFRAME_LINE = HDR_LINES  # only present with video
# read_int is int32: longer catalog offsets are garbage, and int() caps digits.
MAX_OFF_DIGITS = 10

VIDEO_FORMATS = (b"0", b"1", b"124", b"130", b"122", b"2147483647")
FPS_EDGES = (b"0", b"1", b"25", b"29.97", b"0.000000000000000001", b"9223372036854775807.99")
AUDIO_FORMATS = (b"0", b"1", b"2 adpcm", b"101", b"2147483647")
AUDIO_BITS = (b"0", b"4", b"8", b"8 unsigned", b"8 linear", b"16", b"2147483647")
# number_of_chunks == INT_MAX is rejected; it is incremented afterwards.
COUNT_EDGES = (b"0", b"1", b"2147483646", b"2147483647", b"4294967295")
# read_int flags an error once the running value exceeds 214748363.
INT_EDGES = (b"0", b"1", b"214748363", b"214748364", b"2147483647", b"4294967295", b"9" * 20)
ROW_EDGES = (
    b"0",
    b"-1",
    b"9223372036854775807",
    b"-9223372036854775808",
    b"9223372036854775808",
    b"4611686018427387904",
)

_DIGITS = b"0123456789"
_ROW = re.compile(rb"\s*(-?\d+)\s*,\s*(-?\d+)\s*;\s*(-?\d+)")
_ROW_FIELDS = 3


class Op(IntEnum):
    """RPL edit; values are the RNG draw."""

    VIDEO_FMT = 0
    FPS = 1
    AUDIO_FMT = 2
    AUDIO_BITS = 3
    CHUNK_COUNT = 4
    INT_FIELD = 5
    ROW_FIELD = 6
    DUP = 7
    DROP = 8
    SWAP = 9


_FRAME_OPS = {Op.DUP: FrameOp.DUP, Op.DROP: FrameOp.DROP, Op.SWAP: FrameOp.SWAP}
_LINE_OPS = {
    Op.VIDEO_FMT: (VIDEO_LINE, VIDEO_FORMATS),
    Op.FPS: (FPS_LINE, FPS_EDGES),
    Op.AUDIO_FMT: (AUDIO_LINE, AUDIO_FORMATS),
    Op.AUDIO_BITS: (BITS_LINE, AUDIO_BITS),
    Op.CHUNK_COUNT: (COUNT_LINE, COUNT_EDGES),
}


def sniff_rpl(data: bytes) -> bool:
    """Signature plus enough newlines for the 20-line header."""
    return data[:8] == MAGIC and data.count(b"\n", 0, SNIFF_WINDOW) >= HDR_LINES


def _lines(data: bytes) -> list[bytes]:
    """Split on LF only, keeping it; a trailing partial line is kept as is."""
    parts = data.split(b"\n")
    out = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def _lead_int(line: bytes) -> bytes:
    """Leading decimal digits; lstrip runs in C, ~2x a regex match."""
    return line[: len(line) - len(line.lstrip(_DIGITS))]


def _has_video(line: bytes) -> bool:
    return _lead_int(line).lstrip(b"0") != b""


def _lead_val(digits: bytes) -> int:
    """Value of a leading-digit span, -1 when empty or past int32 width.

    Leading zeros are free: "000123" is 123 (seed generators pad fields).
    """
    sig = digits.lstrip(b"0")
    if not digits or len(sig) > MAX_OFF_DIGITS:
        return -1
    return int(sig or b"0")


def _catalog_off(hdr: list[bytes]) -> int:
    """Catalog offset from the header, -1 when absent or out of int32 range."""
    return _lead_val(_lead_int(hdr[CATALOG_LINE]))


def parse_rpl(data: bytes) -> tuple[list[bytes], bytes, list[bytes]] | None:
    """(header lines, gap, catalog rows), or None if not RPL.

    Header is 20 lines, 21 when the video format is non-zero. Rows start at
    the catalog offset when it lies past the header (gap = bytes between);
    otherwise every line after the header is a row.
    """
    if not sniff_rpl(data):
        return None

    lines = _lines(data)
    hdr_n = HDR_LINES + _has_video(lines[VIDEO_LINE])
    if len(lines) < hdr_n or not lines[hdr_n - 1].endswith(b"\n"):
        return None

    hdr = lines[:hdr_n]
    start = sum(map(len, hdr))
    off = _catalog_off(hdr)
    if off == start or not start <= off <= len(data):
        return hdr, b"", lines[hdr_n:]
    return hdr, data[start:off], _lines(data[off:])


def _set_lead(hdr: list[bytes], idx: int, old: int, new: int) -> None:
    """Replace line *idx*'s leading int with *new* if it reads *old*.

    Zero padding is kept: "000046" -> "000051".
    """
    line = hdr[idx]
    digits = _lead_int(line)
    if _lead_val(digits) != old:
        return
    hdr[idx] = str(new).zfill(len(digits)).encode() + line[len(digits) :]


def _fix_catalog(hdr: list[bytes], old_len: int, gap: int) -> None:
    """Re-point the catalog at the rows if it pointed there before the edit."""
    # new = base + width, width = max(padded width, digits(new)); a carry
    # (99 -> 100) widens it at most once more.
    digits = _lead_int(hdr[CATALOG_LINE])
    base = sum(map(len, hdr)) - len(digits) + gap
    width = len(digits)
    while len(str(base + width)) > width:
        width = len(str(base + width))
    _set_lead(hdr, CATALOG_LINE, old_len + gap, base + width)


def _sync_keyframe(hdr: list[bytes]) -> None:
    """Add/drop the key frame list line so the layout matches the video flag."""
    want = _has_video(hdr[VIDEO_LINE])
    if want and len(hdr) == HDR_LINES:
        hdr.append(b"0\n")
    elif not want and len(hdr) > HDR_LINES:
        del hdr[KEYFRAME_LINE]


def _header_op(op: Op, hdr: list[bytes], rng: RandPool) -> int:
    """Rewrite one header line; returns its index."""
    if op in _LINE_OPS:
        idx, edges = _LINE_OPS[op]
        hdr[idx] = rng.choice(edges) + b"\n"
    else:
        idx = FIRST_INT_LINE + rng.randint(0, len(hdr) - FIRST_INT_LINE - 1)
        line = hdr[idx]
        hdr[idx] = rng.choice(INT_EDGES) + line[len(_lead_int(line)) :]

    if idx == VIDEO_LINE:
        _sync_keyframe(hdr)
    return idx


def _row_field(rows: list[bytes], rng: RandPool) -> list[bytes] | None:
    """Set one of offset / video_size / audio_size in one row to an i64 edge."""
    if not rows:
        return None
    i = rng.randint(0, len(rows) - 1)
    m = _ROW.match(rows[i])
    fields = list(m.groups()) if m else [b"0"] * _ROW_FIELDS
    fields[rng.randint(0, _ROW_FIELDS - 1)] = rng.choice(ROW_EDGES)

    out = list(rows)
    eol = b"\n" if rows[i].endswith(b"\n") else b""
    out[i] = b"%s , %s ; %s" % tuple(fields) + eol
    return out


def mutate_rpl(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Mutate one header field, catalog row field or row order; None if declined.

    Draw order: op, then the op's own draws (line / row index, field
    selector, then the value from its edge table).
    """
    parsed = parse_rpl(data)
    if parsed is None:
        return None
    hdr, gap, rows = parsed

    op = Op(rng.randint(0, len(Op) - 1))
    idx = -1
    old_len = 0
    if op in _FRAME_OPS:
        new_rows = reframe(rows, _FRAME_OPS[op], rng)
    elif op is Op.ROW_FIELD:
        new_rows = _row_field(rows, rng)
    else:
        old_len = sum(map(len, hdr))
        idx = _header_op(op, hdr, rng)
        new_rows = rows

    if new_rows is None:
        return None

    # The header stores the last chunk index (count - 1).
    recount = len(new_rows) != len(rows)
    if recount:
        old_len = sum(map(len, hdr))
        _set_lead(hdr, COUNT_LINE, len(rows) - 1, len(new_rows) - 1)
    if (recount or idx >= 0) and idx != CATALOG_LINE and sum(map(len, hdr)) != old_len:
        _fix_catalog(hdr, old_len, len(gap))

    out = b"".join(hdr) + gap + b"".join(new_rows)
    # A cut row would contradict the repaired count.
    if recount and len(out) > max_len:
        return None
    return out[:max_len]
