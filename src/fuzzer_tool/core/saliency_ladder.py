"""Sign-directed byte ladder: NEUZZ's mutation walk as a single-shot operator.

NEUZZ (She et al., S&P'19) ranks input bytes by |d logit_edge / d byte| and walks
them outward in powers of two, nudging each tier's bytes along the gradient sign by
1..255, then spends a fixed share on block delete/insert at the hottest locations.
It enumerates every step as its own input; this operator draws *one* mutation from
the same space so the op arena can weigh it against the others::

    idx    byte offsets, most important first        (from PositionSaliencyScheduler)
    signs  +1 / -1: the direction that raises the target edge's logit

    tier t covers idx[lo:hi]: [0:2) [2:4) [4:8) ...   (NEUZZ's iteration_ranges)
    P(tier t) ~ 1 / (t + 1)                           the top bytes get the most draws
    step      log-uniform in 1..255                   small nudges are cheap to overshoot
    direction follows the sign with P_FOLLOW, else against it

    with P_INS_DEL: delete a block at / insert a block before idx[j]    (AFL havoc lengths)

Reimplemented from the algorithm; Neuzz++ (AGPL-3.0) code is not used. Pure: no fuzzer
imports. ``rng`` needs ``random()`` and ``randint(a, b)`` (inclusive), as ``RandPool`` and
``random.Random`` both have.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["P_FOLLOW", "P_INS_DEL", "choose_block_len", "saliency_ladder"]

#: Probability a tier moves along the gradient sign rather than against it.
P_FOLLOW = 0.75
#: Probability the draw is a block delete/insert instead of a byte nudge (NEUZZ: 20%).
P_INS_DEL = 0.2

_BLK_SMALL = 32
_BLK_MEDIUM = 128
_BLK_LARGE = 1500
_BLK_XL = 32768


def choose_block_len(limit: int, rng) -> int:
    """AFL-havoc style block length in ``[1, limit]`` (small, medium, large, rarely XL)."""
    if limit <= 1:
        return 1
    pick = rng.randint(0, 2)
    if pick == 0:
        lo, hi = 1, _BLK_SMALL
    elif pick == 1:
        lo, hi = _BLK_SMALL, _BLK_MEDIUM
    elif rng.randint(0, 9) != 0:
        lo, hi = _BLK_MEDIUM, _BLK_LARGE
    else:
        lo, hi = _BLK_LARGE, _BLK_XL
    if lo >= limit:
        lo = 1
    hi = min(hi, limit)
    return rng.randint(lo, max(lo, hi))


def _pick_tier(n: int, rng) -> tuple[int, int]:
    """(lo, hi) slice of the ranking for one tier, weight 1/(t+1), tiers of size 2, 2, 4, 8..."""
    bounds = [(0, min(2, n))]
    hi = 2
    while hi < n:
        bounds.append((hi, min(hi * 2, n)))
        hi *= 2
    weights = [1.0 / (t + 1) for t in range(len(bounds))]
    r = rng.random() * sum(weights)
    acc = 0.0
    for b, w in zip(bounds, weights, strict=True):
        acc += w
        if r <= acc:
            return b
    return bounds[-1]


def _step(rng) -> int:
    """Log-uniform step in 1..255: pick the bit length, then a value of that length."""
    b = rng.randint(0, 7)
    return rng.randint(1 << b, min(255, (1 << (b + 1)) - 1))


def saliency_ladder(
    data: bytes,
    idx: Sequence[int],
    signs: Sequence[int],
    rng,
    max_len: int = 0,
) -> bytes:
    """One NEUZZ-style mutation of *data*; the input itself when nothing can move.

    Args:
        data: Input bytes.
        idx: Offsets ranked by gradient magnitude, most important first. Offsets outside
            *data* are ignored.
        signs: Gradient sign per ranked offset (same length as *idx*).
        rng: Random source (``random()``, ``randint``).
        max_len: Cap on the result length; 0 = none.
    """
    n_data = len(data)
    pairs = [(int(i), int(s)) for i, s in zip(idx, signs, strict=False) if 0 <= int(i) < n_data]
    if not data or not pairs:
        return data

    lo, hi = _pick_tier(len(pairs), rng)

    if rng.random() < P_INS_DEL:
        loc = pairs[rng.randint(lo, hi - 1)][0]
        if rng.randint(0, 1) == 0:  # delete, never emptying the input
            cut = choose_block_len(n_data - loc, rng)
            if cut >= n_data:
                cut = n_data - 1
            if cut < 1:
                return data
            out = data[:loc] + data[loc + cut :]
        else:  # insert a copy of an existing block before loc
            cut = choose_block_len(max(1, (n_data - 1) // 2), rng)
            src = rng.randint(0, max(0, n_data - cut))
            out = data[:loc] + data[src : src + cut] + data[loc:]
        return out[:max_len] if max_len else out

    buf = bytearray(data)
    step = _step(rng)
    follow = rng.random() < P_FOLLOW
    for i, s in pairs[lo:hi]:
        delta = step if (s >= 0) == follow else -step
        buf[i] = min(255, max(0, buf[i] + delta))
    return bytes(buf)
