"""Gradient CMP mutation operator.

Lightweight alternative to full RedQueen encoding — finds partial matches
of CMP feedback values in the input and applies gradient strategies to
the first differing byte. Operates directly on the input without the
full encoding pipeline.

Ported from honggfuzz mangle.c:mangle_GradientCmp (line 1328).
"""

import heapq
import random as _random
from itertools import groupby

# Last pool's byte -> ascending pair indices, keyed like scanner_for_pairs
# (owner identity + length).
_index_owner: object | None = None
_index_len = -1
_index: dict[int, list[int]] = {}


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
    buf_len = len(buf)

    # Positions of each byte value in the input, built once per call, so
    # candidate windows are found by lookup instead of scanning every
    # offset of every comparison value.
    pos_map = {}
    for idx, b in enumerate(buf):
        pos_map.setdefault(b, []).append(idx)

    # Only pairs sharing a byte with the input can partially match; visit
    # those in pool order (walking the whole pool was ~360 pairs per call).
    for p in _candidate_pairs(cmp_values, pos_map):
        cmp_a, cmp_b = cmp_values[p]
        if len(cmp_a) == 0 or len(cmp_a) > 32:
            continue

        # Try both operands as the target value
        for cmp_val in (cmp_a, cmp_b):
            n = len(cmp_val)
            # A 1-byte value can never partially match (the window is
            # either a full match or no match), and an oversized value
            # cannot fit in the input.
            if n < 2 or n > buf_len:
                continue

            hit = _partial_match(buf, pos_map, cmp_val)
            if hit is None:
                continue
            _apply_gradient(buf, hit[0], hit[1], cmp_val, r)
            return bytes(buf)

    # No partial match found — insert first CMP value at random position
    if cmp_values:
        cmp_val = cmp_values[r.randint(0, len(cmp_values) - 1)][0]
        if 0 < len(cmp_val) <= 32:
            pos = r.randint(0, len(buf))
            return bytes(buf[:pos]) + cmp_val + bytes(buf[pos:])

    return bytes(buf)


def _windows(pos_map: dict, cmp_val: bytes, buf_len: int) -> set[int]:
    """In-bounds offsets where at least one byte of *cmp_val* lines up with the input."""
    n = len(cmp_val)
    candidates = set()
    for i in range(n):
        for pos in pos_map.get(cmp_val[i], ()):
            off = pos - i
            if 0 <= off <= buf_len - n:
                candidates.add(off)
    return candidates


def _partial_match(buf: bytearray, pos_map: dict, cmp_val: bytes) -> tuple[int, int] | None:
    """Lowest-offset partial match of *cmp_val*: (offset, first differing index), or None."""
    n = len(cmp_val)

    # Candidate windows: offsets where at least one byte of the
    # comparison value occurs in the input.  Every partial match
    # must be among these, so zero-overlap values are skipped
    # entirely instead of scanning every window.
    candidates = _windows(pos_map, cmp_val, len(buf))
    if not candidates:
        return None

    # First partial match wins (offset ascending, as before).
    for off in sorted(candidates):
        # First differing byte; a window with none is a full
        # match — not a partial match.
        for i in range(n):
            if buf[off + i] != cmp_val[i]:
                return off, i
    return None


def _apply_gradient(buf: bytearray, off: int, first_diff: int, cmp_val: bytes, r) -> None:
    """Apply one of 6 random gradient strategies to the first differing byte, in place."""
    target_off = off + first_diff
    want = cmp_val[first_diff]
    diff_mask = buf[target_off] ^ want
    strategy = r.randint(0, 5)

    if strategy == 0:
        # Set to expected value
        buf[target_off] = want
    elif strategy == 1:
        # Flip differing bits
        buf[target_off] ^= diff_mask
    elif strategy == 2:
        # Increment toward target
        if buf[target_off] < want:
            buf[target_off] = min(buf[target_off] + 1, 255)
        else:
            buf[target_off] = max(buf[target_off] - 1, 0)
    elif strategy == 3:
        # Binary search toward target
        buf[target_off] = (buf[target_off] + want) // 2
    elif strategy == 4:
        # Overwrite full comparison value
        end = min(off + len(cmp_val), len(buf))
        buf[off:end] = cmp_val[: end - off]
    else:
        # Flip single bit
        buf[target_off] ^= 1 << r.randint(0, 7)


def _byte_index(cmp_values) -> dict[int, list[int]]:
    """Byte value -> ascending indices of pairs containing it (cached per pool)."""
    global _index_owner, _index_len, _index
    if _index_owner is cmp_values and _index_len == len(cmp_values):
        return _index
    index: dict[int, list[int]] = {}
    for p, (a, b) in enumerate(cmp_values):
        for byte in set(a) | set(b):
            index.setdefault(byte, []).append(p)
    _index_owner, _index_len, _index = cmp_values, len(cmp_values), index
    return index


def _candidate_pairs(cmp_values, pos_map: dict):
    """Ascending, distinct indices of pairs with a byte present in the input."""
    index = _byte_index(cmp_values)
    lists = [index[b] for b in pos_map if b in index]
    return (p for p, _ in groupby(heapq.merge(*lists)))
