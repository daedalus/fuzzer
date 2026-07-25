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
    swaps two random tokens. Handles different-length tokens correctly.

    Args:
        data: Input bytes to mutate.
        rng: Random instance (default: module-level random).

    Returns:
        Mutated bytes with two tokens swapped, or original if <2 tokens found.
    """
    r = rng or _random
    if len(data) < 4:
        return data

    # Find token start positions
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

    len1 = end1 - start1
    len2 = end2 - start2

    if len1 == 0 or len2 == 0 or len1 > 256 or len2 > 256:
        return data

    buf = bytearray(data)
    tmp1 = bytes(buf[start1:end1])
    tmp2 = bytes(buf[start2:end2])

    if len1 == len2:
        # Simple swap
        buf[start1:end1] = tmp2
        buf[start2:end2] = tmp1
    else:
        # Layout: [Prefix][Token1][Middle][Token2][Suffix]
        # Want:   [Prefix][Token2][Middle][Token1][Suffix]
        mid_len = start2 - end1

        # Move middle block first
        buf[start1 + len2 : start1 + len2 + mid_len] = buf[end1 : end1 + mid_len]
        # Write token2 at start1
        buf[start1 : start1 + len2] = tmp2
        # Write token1 after middle
        buf[start1 + len2 + mid_len : start1 + len2 + mid_len + len1] = tmp1

    return bytes(buf)
