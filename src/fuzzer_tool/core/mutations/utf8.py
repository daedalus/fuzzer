"""Sequence-aware UTF-8 mutations.

The table already carries two UTF-8 operators and both only *add* bytes:
``utf8_widen`` rewrites one ASCII byte as a 2-byte overlong, ``utf8_insert``
splices a blob from ``_FUNNY_UNICODE`` in at a random offset. Neither decodes
the buffer, so the multi-byte sequences already in it are never the subject of
a mutation -- they are only ever collateral. Measured on a 87-byte mixed
JSON document (25% non-ASCII bytes), ``utf8_insert`` lands *inside* an
existing sequence 15.4% of the time, which corrupts the carrier as a side
effect of an operator whose stated job is a clean splice, and ``utf8_widen``
produced a +1 length delta on 2000 of 2000 draws -- the 2-byte overlong is
the only form it can reach.

This module takes the other half: find a well-formed multi-byte sequence,
decode it, and put back something that a decoder must reject for a *named*
reason. The six modes are one per rejection path, so a decoder's error
branches are reached deliberately rather than by luck:

    mode          what the decoder must complain about
    ------------  ------------------------------------------------
    truncate      sequence ends early (the classic read-past-end)
    orphan        continuation byte with no lead byte
    widen         overlong form, including the retired 5/6-byte ones
    surrogate     UTF-8-encoded surrogate half / CESU-8 pair
    boundary      code point on or past a range check
    cont_flip     continuation byte that is not one (resync)

The sequence is the one covering -- or nearest after -- the offset the
mutation loop already drew, so ``_last_mutation_offset`` stays a true
description of what was touched and the region liveness estimator gets an
honest observation. Nothing is found beyond ``SEQ_SEARCH_WINDOW`` bytes; the
operator declines instead, which is cheaper than a whole-buffer scan and
keeps the published offset inside one region.
"""

import re

# Lead bytes of well-formed multi-byte sequences. 0xC0/0xC1 are excluded
# because they can only begin an overlong form -- there is no code point
# under them to re-encode -- and 0xF5-0xFF can never begin a valid one.
_LEAD_BYTE = re.compile(rb"[\xc2-\xf4]")

_CONT_LO, _CONT_HI = 0x80, 0xBF
_MAX_SEQ_LEN = 4  # RFC 3629 ceiling for a *valid* sequence
_MAX_WIDTH = 6  # the original UTF-8 ceiling, which widen can still reach

# Equal to ``_REGION_MIN_LEN`` in services/operators.py, and for the same
# reason: the offset this operator is credited with feeds a per-region
# estimator, so a match this far away is still attributed to the right
# region. A wider window would buy a few more mutations at the cost of the
# attribution being a fiction.
SEQ_SEARCH_WINDOW = 512

# Bad candidates are skipped, not fatal, but a buffer of nothing but
# truncated leads must not turn the search into a scan.
SEQ_MAX_CANDIDATES = 8

# Code points sitting on a decoder's range checks: the last and first of
# each encoding width, then the top of the Unicode range and the first two
# values past it. The last two are not encodable at all under RFC 3629,
# which is the point -- 0x110000 is exactly F4 90 80 80, the encoding a
# bounds check on the *lead byte alone* lets through.
BOUNDARY_CPS = (0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x10FFFF, 0x110000, 0x1FFFFF)

# Both halves of the surrogate range, at both ends.
SURROGATE_CPS = (0xD800, 0xDBFF, 0xDC00, 0xDFFF)


def encode_width(cp: int, width: int) -> bytes:
    """Encode *cp* in exactly *width* UTF-8 bytes, valid or not.

    Deliberately unvalidated: at the minimal width it agrees with CPython's
    encoder, and at any wider width it emits the overlong form of the same
    code point. It will also encode surrogates and values past U+10FFFF,
    neither of which ``str.encode`` will produce, which is the only reason
    this exists instead of a call to it.

    Args:
        cp: Code point, 0 .. 0x7FFFFFFF.
        width: Encoded length in bytes, 1 .. 6.

    Returns:
        Exactly *width* bytes.
    """
    if width == 1:
        return bytes((cp & 0x7F,))

    # 110xxxxx / 1110xxxx / 11110xxx / 111110xx / 1111110x
    lead = ((0xFF << (8 - width)) & 0xFF) | (cp >> (6 * (width - 1)))
    tail = (0x80 | ((cp >> (6 * i)) & 0x3F) for i in range(width - 2, -1, -1))

    return bytes((lead, *tail))


def non_cont_byte(draw: int) -> int:
    """Map *draw* in [0, 0xBF] onto the 192 bytes that are not continuations.

    ``[0x00,0x7F] u [0xC0,0xFF]`` -- uniform, and by construction it can
    never return the one thing the mode exists to remove.
    """
    return draw if draw < _CONT_LO else draw + 0x40


def _seq_len(lead: int) -> int:
    """Byte length the *lead* byte declares (2, 3 or 4)."""
    if lead < 0xE0:
        return 2
    return 3 if lead < 0xF0 else 4


def find_seq(data: bytes, byte_idx: int) -> tuple[int, int] | None:
    """Locate a well-formed multi-byte sequence at or after *byte_idx*.

    If *byte_idx* lands inside a sequence, walks back to its lead byte, so
    the position the mutation loop drew selects the sequence covering it
    rather than the next one along.

    Returns:
        ``(offset, length)``, or None when the window holds no sequence
        that decodes strictly.
    """
    if not data:
        return None

    start = min(byte_idx, len(data) - 1)

    # At most three steps back: a continuation byte is never further than
    # that from its lead byte in a valid sequence.
    steps = 0
    while start > 0 and _CONT_LO <= data[start] <= _CONT_HI and steps < _MAX_SEQ_LEN - 1:
        start -= 1
        steps += 1

    stop = min(len(data), start + SEQ_SEARCH_WINDOW)

    for seen, match in enumerate(_LEAD_BYTE.finditer(data, start, stop)):
        if seen >= SEQ_MAX_CANDIDATES:
            return None

        off = match.start()
        length = _seq_len(data[off])

        # Strict decode is the authority on overlong, surrogate and
        # out-of-range forms; reimplementing those three checks here would
        # be a second opinion nobody asked for.
        try:
            data[off : off + length].decode("utf-8")
        except UnicodeDecodeError:
            continue

        return off, length

    return None


# ── modes: one per decoder rejection path ────────────────────────────
#
# Each takes the sequence, its code point and a draw source, and returns the
# bytes to put in its place. None means "not applicable here"; returning the
# sequence unchanged is caught by the caller.


def mode_truncate(seq: bytes, _cp: int, rng) -> bytes:
    """Drop trailing continuation bytes, leaving the sequence unfinished."""
    return seq[: len(seq) - rng.randint(1, len(seq) - 1)]


def mode_orphan(seq: bytes, _cp: int, _rng) -> bytes:
    """Drop the lead byte, leaving continuation bytes with nothing to continue."""
    return seq[1:]


def mode_widen(seq: bytes, cp: int, rng) -> bytes:
    """Re-encode the same code point in a wider, overlong form.

    Widths 5 and 6 are the pre-RFC-3629 forms. They are not reachable from
    any other operator, and a decoder that still accepts them decodes a
    code point no validator upstream of it ever saw.
    """
    return encode_width(cp, rng.randint(len(seq) + 1, _MAX_WIDTH))


def mode_surrogate(_seq: bytes, cp: int, rng) -> bytes:
    """Replace the sequence with UTF-8-encoded surrogate halves.

    An astral code point becomes its own CESU-8 pair -- the same character,
    spelled the way a UTF-16 round trip spells it -- so a decoder that
    accepts it silently agrees with a different reading of the input than
    the one the producer wrote. Anything else gets a lone half.
    """
    if cp < 0x10000:
        return encode_width(rng.choice(SURROGATE_CPS), 3)

    rest = cp - 0x10000
    hi = 0xD800 + (rest >> 10)
    lo = 0xDC00 + (rest & 0x3FF)

    return encode_width(hi, 3) + encode_width(lo, 3)


def mode_boundary(_seq: bytes, _cp: int, rng) -> bytes:
    """Substitute a code point that sits on a range check."""
    cp = rng.choice(BOUNDARY_CPS)
    width = 1 if cp < 0x80 else (2 if cp < 0x800 else (3 if cp < 0x10000 else 4))

    return encode_width(cp, width)


def mode_cont_flip(seq: bytes, _cp: int, rng) -> bytes:
    """Break one continuation byte without changing the length.

    The lead byte still promises N bytes and the decoder still has N bytes
    to read, so this reaches the mid-sequence resync path rather than the
    truncation one.
    """
    idx = rng.randint(1, len(seq) - 1)
    out = bytearray(seq)
    out[idx] = non_cont_byte(rng.randint(0, _CONT_HI))

    return bytes(out)


MODES = (
    mode_truncate,
    mode_orphan,
    mode_widen,
    mode_surrogate,
    mode_boundary,
    mode_cont_flip,
)


def seq_mutate(data: bytes, byte_idx: int, rng, max_len: int = 65536) -> bytes | None:
    """Break one well-formed UTF-8 sequence near *byte_idx*.

    Args:
        data: Input bytes.
        byte_idx: Offset the mutation loop drew; anchors the search.
        rng: Draw source (``RandPool``, or anything with the same API).
        max_len: Output ceiling.

    Returns:
        Mutated bytes, or None when there is nothing to work on. None means
        decline, and the caller records it as one: the alternative -- an
        approximate mutation somewhere else -- credits this operator for
        work it did not do, which is the failure ``_op_declined`` exists to
        stop.
    """
    found = find_seq(data, byte_idx)
    if found is None:
        return None

    off, length = found
    seq = data[off : off + length]
    mode = rng.choice(MODES)
    repl = mode(seq, ord(seq.decode("utf-8")), rng)

    if repl == seq:
        return None

    # Decline rather than clamp. Truncating the result would cut the
    # sequence this operator exists to have produced, and the caller cannot
    # tell that apart from a mutation -- the same reasoning `utf8_widen`
    # uses for its own +1 byte.
    if max_len and len(data) - length + len(repl) > max_len:
        return None

    return data[:off] + repl + data[off + length :]
