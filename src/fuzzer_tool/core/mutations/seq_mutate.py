"""Sequence-level mutation: reorder, duplicate, drop or splice whole messages.

A stateful target parses a stream of messages; byte edits rarely produce a
valid *sequence* (login after data, a repeated frame, a missing handshake).
This works on message boundaries only, so every message stays intact.

Boundary model, sniffed from the bytes (no flag, no seed metadata):

    length-prefixed   [len][payload][len][payload]...   u16/u32, BE/LE,
                      header excluded or included; must tile the buffer
    delimited         msg CRLFCRLF msg ...  /  line CRLF  /  line LF (text only: no NUL)
                      delimiter stays attached to the message before it

Fixed-stride records are left to ``chunk_shuffle``/``region_shuffle``.
"""

import struct

MIN_MSGS = 2
MIN_MSGS_WEAK = 3  # LF occurs in plenty of binary data

# (struct format, header width)
_LEN_FORMATS = ((">I", 4), ("<I", 4), (">H", 2), ("<H", 2))

# (delimiter, minimum pieces), strongest evidence first
_DELIMS = (
    (b"\r\n\r\n", MIN_MSGS),
    (b"\r\n", MIN_MSGS),
    (b"\n", MIN_MSGS_WEAK),
)

_MODES = ("swap", "dup", "drop", "splice")


def _walk_frames(data: bytes, fmt: str, width: int, inclusive: bool) -> list[bytes] | None:
    """Tile *data* with length-prefixed frames, or None if it does not tile."""
    out = []
    pos = 0
    payload = 0
    while pos < len(data):
        if pos + width > len(data):
            return None
        (n,) = struct.unpack_from(fmt, data, pos)
        if inclusive and n < width:
            return None
        end = pos + (n if inclusive else width + n)
        if end > len(data):
            return None
        payload += end - pos - width
        out.append(data[pos:end])
        pos = end
    if len(out) < MIN_MSGS or not payload:
        return None
    return out


def _split_delim(data: bytes, delim: bytes, need: int) -> list[bytes] | None:
    pieces = data.split(delim)
    tail = pieces.pop()
    out = [p + delim for p in pieces]
    if tail:
        out.append(tail)
    if len(out) < need or not data.replace(delim, b""):
        return None
    return out


def split_msgs(data: bytes) -> list[bytes] | None:
    """Split *data* into messages whose concatenation is *data*, or None."""
    if len(data) < 2 * MIN_MSGS:
        return None

    for inclusive in (False, True):
        for fmt, width in _LEN_FORMATS:
            msgs = _walk_frames(data, fmt, width, inclusive)
            if msgs:
                return msgs

    for delim, need in _DELIMS:
        if delim == b"\n" and b"\x00" in data:
            continue
        msgs = _split_delim(data, delim, need)
        if msgs:
            return msgs
    return None


def _delim_of(msgs: list[bytes]) -> bytes:
    """The delimiter every message ends with, or b'' for length framing."""
    for delim, _need in _DELIMS:
        if msgs[0].endswith(delim) and msgs[0] != delim:
            return delim
    return b""


def _swap(msgs, rng, _donor):
    n = len(msgs)
    i = rng.randint(0, n - 1)
    j = rng.randint(0, n - 1)
    if j == i:
        j = (i + 1) % n
    msgs[i], msgs[j] = msgs[j], msgs[i]
    return msgs


def _dup(msgs, rng, _donor):
    src = rng.randint(0, len(msgs) - 1)
    dst = rng.randint(0, len(msgs))
    msgs.insert(dst, msgs[src])
    return msgs


def _drop(msgs, rng, _donor):
    del msgs[rng.randint(0, len(msgs) - 1)]
    return msgs


def _splice(msgs, rng, donor):
    pool = split_msgs(donor) if donor else None
    if not pool:
        return None
    pick = rng.randint(0, len(pool) - 1)
    dst = rng.randint(0, len(msgs))
    msgs.insert(dst, pool[pick])
    return msgs


_APPLY = {"swap": _swap, "dup": _dup, "drop": _drop, "splice": _splice}


def seq_mutate(data: bytes, rng, max_len: int, donor: bytes | None = None) -> bytes | None:
    """Apply one sequence edit; None if *data* has no boundaries or nothing changes.

    The first mode is drawn; later ones are fallbacks, tried in order when a
    mode cannot apply (no donor), exceeds *max_len*, or leaves the bytes as
    they were (swapping equal messages).
    """
    msgs = split_msgs(data)
    if msgs is None:
        return None

    # An unterminated last line would fuse with its new neighbour; close it
    # for the edit and reopen it afterwards.
    delim = _delim_of(msgs)
    open_tail = bool(delim) and not msgs[-1].endswith(delim)
    if open_tail:
        msgs[-1] += delim

    start = rng.randint(0, len(_MODES) - 1)
    for k in range(len(_MODES)):
        mode = _MODES[(start + k) % len(_MODES)]
        out = _APPLY[mode](list(msgs), rng, donor)
        if not out:
            continue
        blob = b"".join(out)
        if open_tail and blob.endswith(delim):
            blob = blob[: -len(delim)]
        if blob != data and len(blob) <= max_len:
            return blob
    return None
