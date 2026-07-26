"""Gradient CMP mutation operator.

Lightweight alternative to full RedQueen encoding — finds partial matches
of CMP feedback values in the input and applies gradient strategies to
the first differing byte. Operates directly on the input without the
full encoding pipeline.

Ported from honggfuzz mangle.c:mangle_GradientCmp (line 1328).
"""

import random as _random


def gradient_cmp(
    data: bytes,
    cmp_values: list[tuple[bytes, bytes]],
    rng=None,
) -> bytes:
    """Apply gradient-guided mutation using CMP feedback values.

    For each CMP value pair, searches the input for a partial match.
    When a partial match is found (some bytes match, first differing byte
    identified), applies one of 6 gradient strategies to the differing byte.

    Args:
        data: Input bytes to mutate.
        cmp_values: List of (operand_a, operand_b) pairs from cmplog tracing.
            These are the values the target compared against.
        rng: Random instance (default: module-level random).

    Returns:
        Mutated bytes, or original if no partial match found.
    """
    r = rng or _random
    if not data or not cmp_values:
        return data

    buf = bytearray(data)

    for cmp_a, cmp_b in cmp_values:
        if len(cmp_a) == 0 or len(cmp_a) > 32:
            continue

        # Try both operands as the target value
        for cmp_val in (cmp_a, cmp_b):
            if len(cmp_val) == 0:
                continue

            # Search for partial match in input
            for off in range(len(buf) - len(cmp_val) + 1):
                matches = 0
                first_diff = len(cmp_val)
                diff_mask = 0

                for i in range(len(cmp_val)):
                    if buf[off + i] == cmp_val[i]:
                        matches += 1
                    elif first_diff == len(cmp_val):
                        first_diff = i
                        diff_mask = buf[off + i] ^ cmp_val[i]

                # Partial match: some bytes match, first diff identified
                if 0 < matches < len(cmp_val) and first_diff < len(cmp_val):
                    target_off = off + first_diff
                    strategy = r.randint(0, 5)

                    if strategy == 0:
                        # Set to expected value
                        buf[target_off] = cmp_val[first_diff]
                    elif strategy == 1:
                        # Flip differing bits
                        buf[target_off] ^= diff_mask
                    elif strategy == 2:
                        # Increment toward target
                        if buf[target_off] < cmp_val[first_diff]:
                            buf[target_off] = min(buf[target_off] + 1, 255)
                        else:
                            buf[target_off] = max(buf[target_off] - 1, 0)
                    elif strategy == 3:
                        # Binary search toward target
                        buf[target_off] = (buf[target_off] + cmp_val[first_diff]) // 2
                    elif strategy == 4:
                        # Overwrite full comparison value
                        end = min(off + len(cmp_val), len(buf))
                        buf[off:end] = cmp_val[: end - off]
                        return bytes(buf)
                    else:
                        # Flip single bit
                        buf[target_off] ^= 1 << r.randint(0, 7)

                    return bytes(buf)

    # No partial match found — insert first CMP value at random position
    if cmp_values:
        cmp_val = cmp_values[r.randint(0, len(cmp_values) - 1)][0]
        if 0 < len(cmp_val) <= 32:
            pos = r.randint(0, len(buf))
            return bytes(buf[:pos]) + cmp_val + bytes(buf[pos:])

    return bytes(buf)
