"""Gradient-descent path-constraint solver.

Port of Angora's ``GdSearch`` (``fuzzer/src/search/gd.rs``). For each
cmplog pair, treats the comparison as an optimization problem: mutate input
bytes to minimize the byte-level distance between the input and the target
operand.

Without DFSan taint tracking, operates over candidate byte positions derived
from overlap between the current input and the target operand. Uses small
perturbations (±1, ±2, ±4, ±8, interesting values) and keeps improvements,
iterating for a bounded number of epochs.
"""

from __future__ import annotations

import logging

from fuzzer_tool.core.mutations import INTERESTING_8, INTERESTING_16, INTERESTING_32

log = logging.getLogger(__name__)

# Bounded descent to keep the operator fast.
_MAX_EPOCHS = 5
_MAX_STUCK = 2
# Perturbation ladder: small steps first, then larger.
_STEPS = (1, -1, 2, -2, 4, -4, 8, -8)


def _distance(a: bytes, b: bytes) -> int:
    """Hamming distance over the min-length prefix plus length delta."""
    n = min(len(a), len(b))
    dist = 0
    for i in range(n):
        dist += (a[i] ^ b[i]).bit_count()
    dist += abs(len(a) - len(b)) * 8
    return dist


def _interesting_for_width(width: int) -> list[int]:
    if width == 1:
        return list(INTERESTING_8)
    if width == 2:
        return list(INTERESTING_16)
    if width >= 4:
        return list(INTERESTING_32)
    return list(INTERESTING_8)


def _candidate_positions(buf: bytes, target: bytes, cap: int = 48) -> list[int]:
    """Return input positions that overlap with *target* bytes.

    Builds a value→positions map for the input, then finds offsets where at
    least one target byte occurs. Falls back to random positions if overlap
    is sparse.
    """
    if not buf or not target:
        return []

    pos_map: dict[int, list[int]] = {}
    for idx, b in enumerate(buf):
        pos_map.setdefault(b, []).append(idx)

    candidates: set[int] = set()
    for i, b in enumerate(target):
        for pos in pos_map.get(b, []):
            off = pos - i
            if 0 <= off < len(buf):
                candidates.add(off)

    if len(candidates) < cap // 2:
        import random

        sample = min(cap // 2, len(buf))
        candidates.update(random.sample(range(len(buf)), sample))

    return sorted(candidates)[:cap]


def gradient_descent(
    input_buf: bytes,
    cmp_pair: tuple[bytes, bytes],
    max_len: int = 0,
    max_epochs: int = _MAX_EPOCHS,
) -> bytes:
    """Optimize *input_buf* to match one operand of *cmp_pair*.

    Args:
        input_buf: Current fuzz input.
        cmp_pair: (operand_a, operand_b) from cmplog.
        max_len: Hard cap on output length (0 = no cap).
        max_epochs: Maximum descent epochs.

    Returns:
        Optimized input bytes, or the original if no improvement found.
    """
    op_a, op_b = cmp_pair
    if not input_buf or (not op_a and not op_b):
        return input_buf

    # Pick target operand: prefer the shorter one (more likely a constant).
    if len(op_b) <= len(op_a):
        target = op_b
        width = len(op_b)
    else:
        target = op_a
        width = len(op_a)

    if not target:
        return input_buf

    # Truncate to max_len before building candidate positions so all
    # candidates remain valid indices in the working buffer.
    buf = bytearray(input_buf[:max_len] if max_len else input_buf)
    if not buf:
        return input_buf

    candidates = _candidate_positions(bytes(buf), target)
    if not candidates:
        return input_buf

    best = bytearray(buf)
    best_score = _distance(bytes(best), target)
    if best_score == 0:
        return bytes(best)

    interesting = _interesting_for_width(width)
    stuck = 0

    for _ in range(max_epochs):
        improved = False

        # Gradient pass: try perturbations at each candidate position.
        for pos in candidates:
            if pos >= len(best):
                continue
            orig = best[pos]
            for delta in _STEPS:
                candidate = bytearray(best)
                candidate[pos] = max(0, min(255, orig + delta))
                score = _distance(bytes(candidate), target)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    if best_score == 0:
                        break
            if best_score == 0:
                break

        if not improved:
            stuck += 1
            if stuck >= _MAX_STUCK:
                break

            # Repick from interesting values to escape local minima.
            for pos in candidates:
                if pos >= len(best):
                    continue
                orig = best[pos]
                for v in interesting:
                    if 0 <= v <= 255 and v != orig:
                        candidate = bytearray(best)
                        candidate[pos] = v
                        score = _distance(bytes(candidate), target)
                        if score < best_score:
                            best = candidate
                            best_score = score
                            improved = True
                            if best_score == 0:
                                break
                if best_score == 0:
                    break

        if best_score == 0:
            break

    return bytes(best)
