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
    """Hamming distance over the min-length prefix plus length delta.

    Kept for the whole-buffer case (len(a) == len(b)); see
    ``_window_distance`` for the operand-matching objective. Scoring a
    large buffer against a short operand with this function is
    meaningless: it only ever inspects ``a[:len(b)]``, so bytes past the
    operand width cannot affect the result.
    """
    n = min(len(a), len(b))
    dist = 0
    for i in range(n):
        dist += (a[i] ^ b[i]).bit_count()
    dist += abs(len(a) - len(b)) * 8
    return dist


def _window_distance(buf: bytes, pos: int, target: bytes) -> int:
    """Hamming distance between ``buf[pos:pos+len(target)]`` and *target*.

    This is the objective the descent actually optimizes. The operand
    being matched is a few bytes wide and can sit anywhere in a
    multi-KB input, so the score has to be local to the position under
    consideration -- a whole-buffer comparison against a short operand
    is dominated by a constant length term and is blind to every byte
    past ``len(target)``, which made the descent unable to modify
    anything outside the first few bytes of the input.

    Positions where the window would run past the end of the buffer
    score the full window width (maximally bad) rather than being
    silently skipped, so they simply never win.
    """
    n = len(target)
    if pos < 0 or pos + n > len(buf):
        return n * 8
    dist = 0
    for i in range(n):
        dist += (buf[pos + i] ^ target[i]).bit_count()
    return dist


def _interesting_for_width(width: int) -> list[int]:
    if width == 1:
        return list(INTERESTING_8)
    if width == 2:
        return list(INTERESTING_16)
    if width >= 4:
        return list(INTERESTING_32)
    return list(INTERESTING_8)


def _candidate_positions(buf: bytes, target: bytes, rng=None, cap: int = 48) -> list[int]:
    """Return input positions that overlap with *target* bytes.

    Builds a value→positions map for the input, then finds offsets where at
    least one target byte occurs. Falls back to random positions if overlap
    is sparse.

    *rng* must be the fuzzer's seeded rand pool. It used to be omitted and
    the fallback drew from the ``random`` module's global state, which made
    every caller non-reproducible: pinning the campaign seed (``-s``) had
    no effect on the sampled offsets, so a seeded rerun took a different
    path. Measured before the fix: with the fuzzer rng pinned and only the
    global rng perturbed, six trials produced five distinct outputs. The
    parameter stays optional (falling back to the global module) only so
    direct callers in tests need not thread one through; every in-fuzzer
    call site passes it.
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
        sample = min(cap // 2, len(buf))
        if rng is not None:
            # randrange_list draws from the seeded pool, so a campaign
            # replayed with the same -s takes the same path here.
            candidates.update(rng.randrange_list(len(buf), sample))
        else:
            import random

            candidates.update(random.sample(range(len(buf)), sample))

    return sorted(candidates)[:cap]


def gradient_descent(
    input_buf: bytes,
    cmp_pair: tuple[bytes, bytes],
    max_len: int = 0,
    max_epochs: int = _MAX_EPOCHS,
    rng=None,
) -> bytes:
    """Optimize *input_buf* to match one operand of *cmp_pair*.

    Args:
        input_buf: Current fuzz input.
        cmp_pair: (operand_a, operand_b) from cmplog.
        max_len: Hard cap on output length (0 = no cap).
        max_epochs: Maximum descent epochs.
        rng: Seeded rand pool, threaded into candidate-site selection so
            a campaign replayed with the same -s takes the same path.

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

    candidates = _candidate_positions(bytes(buf), target, rng)
    if not candidates:
        return input_buf

    # Anchor the descent at the most promising site: the candidate
    # position whose window already best matches the target. Scoring is
    # local to this window (see _window_distance), so the descent can
    # reach an operand anywhere in the input rather than only the first
    # `width` bytes.
    site = min(candidates, key=lambda p: _window_distance(bytes(buf), p, target))

    best = bytearray(buf)
    best_score = _window_distance(bytes(best), site, target)
    if best_score == 0:
        return bytes(best)

    # Only the bytes inside the scored window can change the objective.
    window = [site + i for i in range(width) if site + i < len(best)]
    if not window:
        return bytes(best)

    interesting = _interesting_for_width(width)
    stuck = 0

    for _ in range(max_epochs):
        improved = False

        # Gradient pass: try perturbations at each byte of the window.
        for pos in window:
            orig = best[pos]
            for delta in _STEPS:
                candidate = bytearray(best)
                candidate[pos] = max(0, min(255, orig + delta))
                score = _window_distance(bytes(candidate), site, target)
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
            for pos in window:
                orig = best[pos]
                for v in interesting:
                    if 0 <= v <= 255 and v != orig:
                        candidate = bytearray(best)
                        candidate[pos] = v
                        score = _window_distance(bytes(candidate), site, target)
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
