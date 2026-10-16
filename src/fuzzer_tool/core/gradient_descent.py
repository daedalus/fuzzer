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
from enum import Enum

from fuzzer_tool.core.mutations import INTERESTING_8, INTERESTING_16, INTERESTING_32
from fuzzer_tool.core.rand_pool import get_default_rand_pool

log = logging.getLogger(__name__)

# Bounded descent to keep the operator fast.
_MAX_EPOCHS = 5
_MAX_STUCK = 2
# Perturbation ladder: small steps first, then larger.
#
# Must span the full byte. The objective is `_window_distance`, a *bitwise
# Hamming* distance, but these are *arithmetic* steps, and the two only
# line up where the step is a power of two that does not carry. With the
# ladder stopping at 8, the single-bit correction for bits 4-7 is
# unreachable at every one of the 256 byte values:
#
#     bit 0 (1)   100%      bit 4 (16)    0%
#     bit 1 (2)   100%      bit 5 (32)    0%
#     bit 2 (4)   100%      bit 6 (64)    0%
#     bit 3 (8)   100%      bit 7 (128)   0%
#
# so any operand byte differing in its high nibble -- about half of them --
# could not be fixed, and the descent stalled out two epochs later. Solve
# rate on a 4-byte operand planted in a 64-byte buffer, 400 trials:
#
#     ladder            two bytes matching     operand anywhere
#     +-1,2,4,8                        1.2%                1.2%
#     +-1..128                        85.8%               98.8%
#
# Raising _MAX_EPOCHS to 20 on top of that moves it by 0.4 points, so the
# step set was the entire constraint, not the budget.
_STEPS = (1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128)

# Popcount of every byte value, for the incremental window objective.
_POPCOUNT = bytes(bin(i).count("1") for i in range(256))

# trace-cmp integer widths; numeric scoring is meaningless for other lengths.
_INT_WIDTHS = frozenset((1, 2, 4, 8))
# Offsets around the constant. SanitizerCoverage hides the predicate, so
# cover all six: ==/!= flip at 0, </>= at -1/0, >/<= at 0/+1.
_BOUNDARY = (-1, 0, 1)
# trace-cmp logs operands little-endian; container fields are often BE.
_ORDERS = ("little", "big")


class Objective(Enum):
    """What the solver minimizes."""

    # Hamming distance to the operand bytes: memcmp/strcmp, unknown compares.
    BITWISE = "bitwise"
    # |int(window) - goal|, the branch distance: trace-cmp integer compares.
    NUMERIC = "numeric"


def pick_objective(pc: int | None) -> Objective:
    """trace-cmp records carry a PC; libc byte compares do not."""
    return Objective.BITWISE if pc is None else Objective.NUMERIC


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

    Finds offsets where at least one target byte occurs. Falls back to
    random positions if overlap is sparse.

    Only the target's own byte values can produce an overlap, and a target
    is at most eight bytes wide, so the scan looks for those values
    directly with ``bytes.find`` (memchr) instead of building a
    value→positions map over all 256 values of the whole buffer. Measured
    on a 64 KiB buffer with a 4-byte target: 5115 us -> 347 us.

    When more than *cap* candidates survive, they are subsampled at an
    even stride rather than truncated. ``sorted(candidates)[:cap]`` kept
    the *cap* smallest offsets, which confined the descent to the head of
    the file: measured on a 64 KiB buffer, all 48 returned candidates fell
    within the first 5.35% of it. That is the same failure the
    ``_window_distance`` docstring describes being fixed, reintroduced one
    layer up. An even stride is still deterministic, so a campaign
    replayed with the same ``-s`` takes the same path.

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

    n = len(buf)
    candidates: set[int] = set()
    for value in set(target):
        offsets = [i for i, tb in enumerate(target) if tb == value]
        needle = bytes((value,))
        pos = buf.find(needle)
        while pos >= 0:
            for i in offsets:
                off = pos - i
                if 0 <= off < n:
                    candidates.add(off)
            pos = buf.find(needle, pos + 1)

    if len(candidates) < cap // 2:
        sample = min(cap // 2, len(buf))
        if rng is not None:
            # randrange_list draws from the seeded pool, so a campaign
            # replayed with the same -s takes the same path here.
            candidates.update(rng.randrange_list(len(buf), sample))
        else:
            # Unique positions, as random.sample did before the pool migration.
            candidates.update(get_default_rand_pool().sample(len(buf), sample))

    ordered = sorted(candidates)
    if len(ordered) <= cap:
        return ordered
    stride = len(ordered) / cap
    return [ordered[int(i * stride)] for i in range(cap)]


def pick_target(cmp_pair: tuple[bytes, bytes]) -> bytes:
    """Return the operand to match: the shorter *non-empty* one.

    The shorter operand is more likely to be the constant side of the
    comparison (a magic number or length field) and is the one worth
    matching. The non-empty qualifier matters: a bare ``min`` by length
    selects a zero-length operand over a real one, and every caller then
    silently no-ops. Not currently reachable -- the cmplog parser splits
    on whitespace so a field cannot come back empty -- but the three
    searches shared this rule by copy rather than by import, and had
    drifted, so it lives here and mb_cbh imports it.
    """
    op_a, op_b = cmp_pair
    if not op_a:
        return op_b
    if not op_b:
        return op_a
    return op_b if len(op_b) <= len(op_a) else op_a


def gradient_descent(
    input_buf: bytes,
    cmp_pair: tuple[bytes, bytes],
    max_len: int = 0,
    max_epochs: int = _MAX_EPOCHS,
    rng=None,
    objective: Objective = Objective.BITWISE,
) -> bytes:
    """Optimize *input_buf* to match one operand of *cmp_pair*.

    Args:
        input_buf: Current fuzz input.
        cmp_pair: (operand_a, operand_b) from cmplog.
        max_len: Hard cap on output length (0 = no cap).
        max_epochs: Maximum descent epochs.
        rng: Seeded rand pool, threaded into candidate-site selection so
            a campaign replayed with the same -s takes the same path.
        objective: NUMERIC applies to 1/2/4/8-byte operands only; other
            widths fall back to BITWISE.

    Returns:
        Optimized input bytes, or the original if no improvement found.
    """
    op_a, op_b = cmp_pair
    if not input_buf or (not op_a and not op_b):
        return input_buf

    target = pick_target(cmp_pair)
    if not target:
        return input_buf
    width = len(target)

    # Truncate to max_len before building candidate positions so all
    # candidates remain valid indices in the working buffer.
    buf = bytearray(input_buf[:max_len] if max_len else input_buf)
    if not buf:
        return input_buf

    if objective is Objective.NUMERIC and width in _INT_WIDTHS and len(op_a) == len(op_b):
        source = op_a if target is op_b else op_b
        return _solve_numeric(buf, target, source, rng)

    candidates = _candidate_positions(bytes(buf), target, rng)
    if not candidates:
        return input_buf

    # Anchor the descent at the most promising site: the candidate
    # position whose window already best matches the target. Scoring is
    # local to this window (see _window_distance), so the descent can
    # reach an operand anywhere in the input rather than only the first
    # `width` bytes.
    frozen = bytes(buf)
    site = min(candidates, key=lambda p: _window_distance(frozen, p, target))

    best = bytearray(buf)
    best_score = _window_distance(frozen, site, target)
    if best_score == 0:
        return bytes(best)

    # A window that runs off the end scores n*8 whatever the bytes are, so
    # nothing can improve it. The incremental scoring below would happily
    # "improve" such a window, so bail before it can.
    if site + width > len(best):
        return bytes(best)

    _descend_window(best, site, target, best_score, max_epochs)
    return bytes(best)


def _solve_numeric(buf: bytearray, target: bytes, source: bytes, rng) -> bytes:
    """Write ``goal = target + off`` into the field holding *source*.

    Branch distance (Handwired's pre-ReLU ``a - b``): the compare reads an
    input-derived operand (*source*) against *target*. The solver never
    executes the target, so the minimizer of ``|field - goal|`` is ``goal``
    itself; the search is over *where* the field is: exact *source* hits in
    either byte order, else the window numerically closest to it.

    Example: limit 4096, BE field 4090 -> field becomes 4095/4096/4097.
    """
    width = len(target)
    if len(buf) < width:
        return bytes(buf)

    # Boundary draw first, so the draw order is fixed whatever the buffer.
    r = rng or get_default_rand_pool()
    goal = (int.from_bytes(target, "little") + r.choice(_BOUNDARY)) % (1 << 8 * width)

    frozen = bytes(buf)
    orders = _ORDERS[:1] if width == 1 else _ORDERS
    hits = [(p, o) for o in orders for p in _exact_sites(frozen, _encode(source, o))]
    if len(hits) > 1:
        # Small values (0, 1) recur; a fixed first hit biases to the head.
        site, order = r.choice(hits)
    elif hits:
        site, order = hits[0]
    else:
        site, order = _closest(frozen, source, orders, rng)

    want = goal.to_bytes(width, order)
    if site < 0 or frozen[site : site + width] == want:
        return frozen

    buf[site : site + width] = want
    return bytes(buf)


def _encode(source: bytes, order: str) -> bytes:
    """*source* is logged little-endian; re-encode it in *order*."""
    return source if order == "little" else source[::-1]


def _exact_sites(buf: bytes, probe: bytes) -> list[int]:
    """Every offset where *probe* occurs (overlapping)."""
    sites = []
    pos = buf.find(probe)
    while pos >= 0:
        sites.append(pos)
        pos = buf.find(probe, pos + 1)
    return sites


def _closest(buf: bytes, source: bytes, orders: tuple, rng) -> tuple[int, str]:
    """(site, order) of the candidate window numerically closest to *source*."""
    width = len(source)
    last = len(buf) - width
    value = int.from_bytes(source, "little")
    best = (1 << 8 * width, -1, orders[0])
    for order in orders:
        for p in _candidate_positions(buf, _encode(source, order), rng):
            if p > last:
                continue
            dist = abs(int.from_bytes(buf[p : p + width], order) - value)
            if dist < best[0]:
                best = (dist, p, order)
    return best[1], best[2]


def _descend_window(
    best: bytearray, site: int, target: bytes, best_score: int, max_epochs: int
) -> None:
    """Coordinate descent on ``best[site:site+len(target)]``, in place.

    Step pass first; after a stalled epoch, repick from interesting values
    to escape local minima. Stops at score 0 or _MAX_STUCK stalled epochs.
    """
    width = len(target)

    # Only the bytes inside the scored window can change the objective.
    window = [site + i for i in range(width)]

    interesting = _interesting_for_width(width)
    stuck = 0

    # The objective is separable: it is the sum, over the window, of
    # popcount(buf[site+k] ^ target[k]). Changing one byte moves exactly
    # one term, so a probe costs O(1) and needs no buffer at all. The
    # previous form copied the whole buffer twice per probe -- once for
    # the candidate and once to hand `bytes` to _window_distance -- to
    # evaluate a one-byte change inside a window of at most eight, which
    # made the cost of the descent scale with the input rather than with
    # the window (measured 0.17 ms at 256 B, 0.85 ms at 8 KiB).
    term = [_POPCOUNT[best[site + k] ^ target[k]] for k in range(width)]

    for _ in range(max_epochs):
        # Gradient pass: try perturbations at each byte of the window.
        prev = best_score
        best_score = _step_pass(best, window, target, term, best_score)
        improved = best_score < prev

        if not improved:
            stuck += 1
            if stuck >= _MAX_STUCK:
                break
            best_score = _repick_pass(best, window, target, term, best_score, interesting)

        if best_score == 0:
            break


def _step_pass(
    best: bytearray, window: list[int], target: bytes, term: list[int], best_score: int
) -> int:
    """One +/- _STEPS sweep over the window; returns the new score."""
    for k, pos in enumerate(window):
        orig = best[pos]
        tb = target[k]
        for delta in _STEPS:
            v = orig + delta
            if v < 0:
                v = 0
            elif v > 255:
                v = 255
            new_term = _POPCOUNT[v ^ tb]
            score = best_score - term[k] + new_term
            if score < best_score:
                best[pos] = v
                term[k] = new_term
                best_score = score
                if best_score == 0:
                    return 0
    return best_score


def _repick_pass(
    best: bytearray,
    window: list[int],
    target: bytes,
    term: list[int],
    best_score: int,
    interesting: list[int],
) -> int:
    """Try each interesting value at each window byte; returns the new score."""
    for k, pos in enumerate(window):
        orig = best[pos]
        tb = target[k]
        for v in interesting:
            if not (0 <= v <= 255 and v != orig):
                continue
            new_term = _POPCOUNT[v ^ tb]
            score = best_score - term[k] + new_term
            if score < best_score:
                best[pos] = v
                term[k] = new_term
                best_score = score
                if best_score == 0:
                    return 0
    return best_score
