"""Token shuffle mutation operator.

Splits input on common delimiters and swaps two random tokens.
Effective for text protocols, config files, command lines.

Ported from honggfuzz mangle.c:mangle_TokenShuffle (line 1245).
"""

import random as _random

_DELIMS = b" \t\n\r,;:|/\\=&?"


def token_shuffle(data: bytes, rng=None) -> bytes:
    """Shuffle two random tokens delimited by common separators.

    Finds token boundaries by scanning for delimiter characters, then
    swaps two random tokens. Handles different-length tokens correctly
    by normalizing token spans to exclude delimiters and explicitly
    re-inserting delimiters after the swap.

    Args:
        data: Input bytes to mutate.
        rng: Random instance (default: module-level random).

    Returns:
        Mutated bytes with two tokens swapped, or original if <2 tokens found.
    """
    r = rng or _random
    n = len(data)
    if n < 4:
        return data

    token_starts = _token_starts(data, n)
    if len(token_starts) < 2:
        return data

    # Pick two random token indices
    idx1 = r.randint(0, len(token_starts) - 2)
    idx2 = r.randint(idx1 + 1, len(token_starts) - 1)

    start1, end1 = _token_bounds(token_starts, idx1, n)
    start2, end2 = _token_bounds(token_starts, idx2, n)

    # Extract token content (strip trailing delimiters)
    content1 = data[start1:end1].rstrip(_DELIMS)
    content2 = data[start2:end2].rstrip(_DELIMS)

    if len(content1) == 0 or len(content2) == 0 or len(content1) > 256 or len(content2) > 256:
        return data

    # Find the delimiter that follows each token's content
    delim1 = _trailing_delim(data, start1 + len(content1), end1)
    delim2 = _trailing_delim(data, start2 + len(content2), end2)

    # If neither delimiter exists, use a space as fallback
    if not delim1 and not delim2:
        delim1 = b" "

    # Rebuild: [Prefix][Token2 + delim1][Middle][Token1 + delim2][Suffix]
    prefix = data[:start1]
    suffix = data[end2:]

    # Middle section: bytes between token1's content end and token2's content start
    mid_start = start1 + len(content1) + (1 if delim1 else 0)
    mid_end = start2
    middle = data[mid_start:mid_end]

    # Build the result (an absent delimiter is b"" and joins to nothing)
    return b"".join((prefix, content2, delim1, middle, content1, delim2, suffix))


def _token_starts(data: bytes, n: int) -> list[int]:
    """Token start offsets: 0 plus each byte after a delimiter, capped at 64."""
    token_starts = [0]
    for i in range(n):
        if len(token_starts) >= 64:
            break
        if data[i] in _DELIMS and i + 1 < n:
            token_starts.append(i + 1)
    return token_starts


def _token_bounds(token_starts: list[int], idx: int, n: int) -> tuple[int, int]:
    """[start, end) of token *idx*; the last token runs to end of data."""
    end = token_starts[idx + 1] if idx + 1 < len(token_starts) else n
    return token_starts[idx], end


def _trailing_delim(data: bytes, content_end: int, end: int) -> bytes:
    """The one delimiter byte after a token's content, or b"" if none."""
    return data[content_end:end][:1] if content_end < end else b""
