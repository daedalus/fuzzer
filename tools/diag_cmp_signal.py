#!/usr/bin/env python3
"""Is the cmplog operand-matching signal real, or is it a coin flip?

Port of the p-bit study's ``diag_grad.py`` discipline: before tuning a
mechanism, measure whether the mechanism has any signal at all. There, a
20-line diagnostic showed the per-site gradient pointed at the correct bit
50.8% of the time -- chance -- which retired the whole line of work that
had been built on top of it.

The three cmplog searches (``gradient_descent``, ``climb_hill``,
``magic_byte_search``) rest on two assumptions that have never been
measured separately:

1. **Site selection.** ``_candidate_positions`` derives offsets from
   byte-value overlap between the input and the operand, and the two
   descents then anchor on the argmin. If that pick is no better than
   chance, the descent optimizes an arbitrary window and the operator
   cannot solve the branch however well it converges. The module docstring
   already reports the extreme case (0/200 with no shared bytes) but not
   the general one.

2. **Ladder informativeness.** ``gradient_descent`` perturbs by arithmetic
   steps (+-1, +-2, +-4, +-8) against a *bitwise Hamming* objective. Those
   are different spaces: +1 on 0x7f flips eight bits. If the best ladder
   step is no better than a random byte value, the "gradient" is a name
   for a random search with extra bookkeeping.

Each diagnostic is scored against an explicit chance baseline, because a
number like "63% correct" means nothing until you know what a coin gets.

Usage::

    tools/diag_cmp_signal.py                 # all diagnostics
    tools/diag_cmp_signal.py --trials 2000
    tools/diag_cmp_signal.py --only site
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzzer_tool.core.gradient_descent import (  # noqa: E402
    _STEPS,
    _candidate_positions,
    _window_distance,
    gradient_descent,
)
from fuzzer_tool.core.mb_cbh import climb_hill, magic_byte_search  # noqa: E402
from fuzzer_tool.core.rand_pool import RandPool  # noqa: E402

OPERANDS = {
    2: b"\xca\xfe",
    4: b"\xde\xad\xbe\xef",
    8: b"\x89PNG\r\n\x1a\n",
}


def _planted(rng: random.Random, operand: bytes, buflen: int, n_matching: int) -> tuple[bytes, int]:
    """Buffer of *buflen* with *n_matching* operand bytes at a random offset."""
    buf = bytearray(rng.randrange(256) for _ in range(buflen))
    off = rng.randrange(0, buflen - len(operand))
    # Scrub incidental copies of the operand's bytes elsewhere so the only
    # planted signal is the one at `off`; otherwise the "true" site is
    # ambiguous and site accuracy is unmeasurable.
    for i in range(buflen):
        while buf[i] in operand:
            buf[i] = rng.randrange(256)
    for i in range(n_matching):
        buf[off + i] = operand[i]
    return bytes(buf), off


# ── 1. Site selection ──────────────────────────────────────────────────


def diag_site(trials: int, buflen: int) -> None:
    print("\n[1] SITE SELECTION -- does argmin over candidates find the planted offset?")
    print(f"    {trials} trials per cell, {buflen}-byte buffers\n")
    print(
        f"    {'operand':>8} {'planted':>8} {'accuracy':>9} {'chance':>8} {'lift':>7} {'in cand':>8}"
    )
    print("    " + "-" * 54)

    pool = RandPool()
    for width, operand in OPERANDS.items():
        for n_matching in (0, 1, 2):
            if n_matching > width:
                continue
            rng = random.Random(0xC0FFEE + width * 10 + n_matching)
            hits = 0
            present = 0
            n_cand_total = 0
            for _ in range(trials):
                buf, true_off = _planted(rng, operand, buflen, n_matching)
                cands = _candidate_positions(buf, operand, pool)
                cands = [p for p in cands if p + width <= len(buf)]
                if not cands:
                    continue
                n_cand_total += len(cands)
                if true_off in cands:
                    present += 1
                pick = min(cands, key=lambda p: _window_distance(buf, p, operand))
                hits += pick == true_off

            acc = hits / trials
            # Chance = picking uniformly among the candidates actually
            # offered, conditional on the true site being among them.
            mean_cands = n_cand_total / trials if trials else 1
            chance = (present / trials) / mean_cands if mean_cands else 0
            lift = acc / chance if chance else float("inf")
            print(
                f"    {width:>8} {n_matching:>8} {acc:>8.1%} {chance:>8.1%} "
                f"{lift:>6.1f}x {present / trials:>8.1%}"
            )
    print("\n    'in cand' is how often the true offset was even among the candidates;")
    print("    accuracy cannot exceed it, so a low value is a site-generation problem,")
    print("    not a ranking problem.")


# ── 2. Ladder informativeness ──────────────────────────────────────────


def _best_single_byte(buf: bytes, operand: bytes, values) -> int:
    """Lowest window distance reachable by rewriting one byte.

    *values* maps a position to the byte values that strategy would try.
    Passed explicitly rather than closed over so each strategy is a
    standalone, testable function.
    """
    best = _window_distance(buf, 0, operand)
    for pos in range(len(operand)):
        for v in values(pos, buf):
            c = bytearray(buf)
            c[pos] = v
            best = min(best, _window_distance(bytes(c), 0, operand))
    return best


def _ladder_values(pos: int, buf: bytes):
    return [max(0, min(255, buf[pos] + d)) for d in _STEPS]


def _exhaustive_values(_pos: int, _buf: bytes):
    return range(256)


def diag_ladder(trials: int) -> None:
    print("\n[2] LADDER -- do arithmetic steps beat random byte values on a Hamming objective?")
    print(f"    {trials} trials per cell\n")
    print(
        f"    {'operand':>8} {'start d':>8} {'ladder':>9} {'random':>9} "
        f"{'exhaustive':>11} {'ladder/exh':>11}"
    )
    print("    " + "-" * 62)

    for width, operand in OPERANDS.items():
        for start_d in (1, 4, 8):
            rng = random.Random(0xBEEF + width * 10 + start_d)

            def random_values(_pos, _buf, _rng=rng):
                # Same number of draws as the ladder, so the two strategies
                # are compared at equal cost per byte.
                return [_rng.randrange(256) for _ in _STEPS]

            ladder_gain = random_gain = exh_gain = 0
            for _ in range(trials):
                # Build a window at exactly `start_d` bits from the operand.
                w = bytearray(operand)
                for _ in range(start_d):
                    i = rng.randrange(width)
                    w[i] ^= 1 << rng.randrange(8)
                buf = bytes(w)
                base = _window_distance(buf, 0, operand)
                if base == 0:
                    continue

                ladder_gain += base - _best_single_byte(buf, operand, _ladder_values)
                random_gain += base - _best_single_byte(buf, operand, random_values)
                exh_gain += base - _best_single_byte(buf, operand, _exhaustive_values)

            denom = trials * start_d
            ratio = ladder_gain / exh_gain if exh_gain else 0
            print(
                f"    {width:>8} {start_d:>8} {ladder_gain / denom:>8.1%} "
                f"{random_gain / denom:>8.1%} {exh_gain / denom:>10.1%} {ratio:>10.1%}"
            )
    print("\n    Columns are the fraction of the starting Hamming distance removed by one")
    print("    best-of-strategy byte write. 'exhaustive' tries all 256 values and is the")
    print("    ceiling for a single-byte move; ladder/exh is how much of the available")
    print("    signal the arithmetic steps capture.")


# ── 3. End-to-end, site problem separated from descent problem ─────────


def diag_endtoend(trials: int, buflen: int) -> None:
    print("\n[3] END TO END -- solve rate at the planted offset vs anywhere")
    print(f"    {trials} trials per cell, {buflen}-byte buffers\n")
    print(f"    {'operator':>18} {'operand':>8} {'planted':>8} {'at site':>9} {'anywhere':>9}")
    print("    " + "-" * 56)

    ops = {
        "gradient_descent": lambda b, p, pool: gradient_descent(b, p, rng=pool),
        "climb_hill": lambda b, p, pool: climb_hill(b, p, pool),
        "climb_hill(sites=4)": lambda b, p, pool: climb_hill(b, p, pool, max_sites=4),
        "magic_byte_search": lambda b, p, pool: magic_byte_search(b, p, pool),
    }
    for name, fn in ops.items():
        for width, operand in OPERANDS.items():
            if width != 4:
                continue
            for n_matching in (0, 1, 2):
                rng = random.Random(0xD1A6 + n_matching)
                pool = RandPool()
                at_site = anywhere = 0
                for _ in range(trials):
                    buf, off = _planted(rng, operand, buflen, n_matching)
                    out = fn(buf, (operand, operand), pool)
                    at_site += out[off : off + width] == operand
                    anywhere += operand in out
                print(
                    f"    {name:>18} {width:>8} {n_matching:>8} "
                    f"{at_site / trials:>8.1%} {anywhere / trials:>8.1%}"
                )
    print("\n    A large gap between the two columns is the anchoring failure: the search")
    print("    converged, on the wrong window. Only 'at site' unlocks the branch.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--buflen", type=int, default=64)
    ap.add_argument("--only", choices=("site", "ladder", "e2e"), default=None)
    args = ap.parse_args()

    if args.only in (None, "site"):
        diag_site(args.trials, args.buflen)
    if args.only in (None, "ladder"):
        diag_ladder(args.trials)
    if args.only in (None, "e2e"):
        diag_endtoend(args.trials, args.buflen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
