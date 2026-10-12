"""Text-format edge cases: line endings and embedded NUL bytes.

Line-oriented parsers disagree on what ends a line (CRLF, LF, CR, even LFCR),
and C-string code stops at the first NUL.  Each function here builds exactly
one such disagreement from the input.

Every function draws only through ``rng.randrange(n)`` so a scripted source
fixes the result.  Both return *data* itself, drawing nothing, when there is
nothing to act on.
"""

from __future__ import annotations

import re

CRLF, LF, CR, LFCR = b"\r\n", b"\n", b"\r", b"\n\r"
_ENDINGS = (CRLF, LF, CR, LFCR)
_SOURCES = (CRLF, LF, CR)  # LFCR is only ever a destination: it parses as LF then CR
_EOL = re.compile(rb"\r\n|\n|\r")  # CRLF first, so a lone CR is not part of a pair

_RUN = re.compile(rb"[\x20-\x7e]{4,}")  # printable ASCII, long enough to have an interior
_SCOPE_ALL, _SCOPE_ONE = 0, 1
_NUL_OVERWRITE, _NUL_INSERT = 0, 1


def line_ending_swap(data: bytes, rng) -> bytes:
    """Rewrite one kind of line ending as another, everywhere or at one spot.

    Picks a source ending among those present (CRLF, LF, lone CR), a different
    destination among CRLF, LF, CR and LFCR, and a scope: every occurrence, or a
    single one.  A single swap leaves mixed endings in one file, which is what
    a reader that sniffs the first line and trusts it gets wrong.  A CR that
    belongs to a CRLF is not a lone CR.

    Length changes when CRLF or LFCR is involved.  Returns *data* without
    drawing when it has no line ending.

    Example (LF to CRLF, one occurrence)::

        a\\nb\\nc  ->  a\\nb\\r\\nc

    Args:
        data: Input bytes.
        rng: Draw source with ``randrange``.

    Returns:
        Mutated bytes.
    """
    spans = [(m.start(), m.end(), m.group()) for m in _EOL.finditer(data)]
    present = [e for e in _SOURCES if any(tok == e for _, _, tok in spans)]
    if not present:
        return data
    src = present[rng.randrange(len(present))]
    dst_options = [e for e in _ENDINGS if e != src]
    dst = dst_options[rng.randrange(len(dst_options))]
    hits = [(s, e) for s, e, tok in spans if tok == src]
    if rng.randrange(2) == _SCOPE_ONE:
        hits = [hits[rng.randrange(len(hits))]]
    out = bytearray()
    prev = 0
    for start, end in hits:
        out += data[prev:start] + dst
        prev = end
    return bytes(out + data[prev:])


def nul_inject(data: bytes, rng) -> bytes:
    """Put a NUL inside a printable run: overwrite one byte or insert one.

    Finds runs of four or more printable ASCII bytes, picks one, then a byte
    strictly inside it (never the first, never the last), and either replaces
    that byte with 0x00 or inserts a 0x00 before it.  A C-string consumer
    sees a shorter string than the length field or the next parser does.

    Overwrite keeps the length, insert grows it by one.  Returns *data*
    without drawing when no run is long enough.

    Example (overwrite)::

        01 'hello' 02  ->  01 'he' 00 'lo' 02

    Args:
        data: Input bytes.
        rng: Draw source with ``randrange``.

    Returns:
        Mutated bytes.
    """
    runs = [(m.start(), m.end()) for m in _RUN.finditer(data)]
    if not runs:
        return data
    start, end = runs[rng.randrange(len(runs))]
    pos = start + 1 + rng.randrange(end - start - 2)
    if rng.randrange(2) == _NUL_INSERT:
        return data[:pos] + b"\x00" + data[pos:]
    return data[:pos] + b"\x00" + data[pos + 1 :]
