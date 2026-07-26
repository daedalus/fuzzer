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
    if len(data) < 4:
        return data

    # Find token start positions (each token starts after a delimiter)
    token_starts = [0]
    for i in range(len(data)):
        if len(token_starts) >= 64:
            break
        if data[i] in _DELIMS and i + 1 < len(data):
            token_starts.append(i + 1)

    if len(token_starts) < 2:
        return data

    # Pick two random token indices
    idx1 = r.randint(0, len(token_starts) - 2)
    idx2 = r.randint(idx1 + 1, len(token_starts) - 1)

    start1 = token_starts[idx1]
    end1 = token_starts[idx1 + 1] if idx1 + 1 < len(token_starts) else len(data)
    start2 = token_starts[idx2]
    end2 = token_starts[idx2 + 1] if idx2 + 1 < len(token_starts) else len(data)

    # Extract token content (strip trailing delimiters)
    content1 = data[start1:end1].rstrip(_DELIMS)
    content2 = data[start2:end2].rstrip(_DELIMS)

    if len(content1) == 0 or len(content2) == 0 or len(content1) > 256 or len(content2) > 256:
        return data

    # Find the delimiter that follows each token's content
    delim1 = data[start1 + len(content1) : end1][:1] if start1 + len(content1) < end1 else b""
    delim2 = data[start2 + len(content2) : end2][:1] if start2 + len(content2) < end2 else b""

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

    # Build the result
    parts = [prefix, content2]
    if delim1:
        parts.append(delim1)
    parts.append(middle)
    parts.append(content1)
    if delim2:
        parts.append(delim2)
    parts.append(suffix)

    return b"".join(parts)
