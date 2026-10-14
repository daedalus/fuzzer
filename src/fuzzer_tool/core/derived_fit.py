"""Derived-compare fitters: operand maps cmplog logs but Redqueen cannot patch.

The compared operand ``v`` is computed from the input, not copied from it,
so there is nothing to substitute. Three map families are cheap to learn
from probes and to invert exactly::

    GF(2)-affine  v = A.x ^ c       parity, syndromes, CRC-like XOR maps
                  probe x0 ^ e_i -> column i of A; eliminate -> d, x = x0 ^ d
                  width + 4 execs (columns, 3 confirms)

    additive      v = v0 + sum w_i * dx_i  (mod 2^bits)
                  byte sums, weighted checksums over a run of bytes;
                  w_i come from the caller's +1 screen; greedy split of
                  t - v0 over the bytes, |w| descending; 0 execs

    byte map      v[j] = f_j(x[i_j])  one input byte per operand byte
                  rot13 / caesar / table lookups into strcmp, memcmp;
                  every unsolved position tries its next guess in the
                  same exec: <= 2 + 256 execs for the whole operand

Every fitter fails closed: a map outside its family returns None, and the
caller re-executes a candidate before trusting it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

BYTE_VALUES = 256
BYTE_MAX = 0xFF

# x -> operand from one run with the field set to x; None if not reached.
Observe = Callable[[int], int | None]

# {input offset: byte value} -> moving operand bytes; None if not reached.
ObserveBytes = Callable[[dict[int, int]], bytes | None]

# Masks XORed onto x0 to check a GF(2) fit away from the unit vectors.
_GF2_CONFIRMS = (0xFFFFFFFFFFFFFFFF, 0x5555555555555555, 0x3333333333333333)


# ── GF(2)-affine ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Gf2Affine:
    """``v(x) = v0 ^ XOR of cols[i] for each bit i set in x ^ x0``."""

    cols: tuple[int, ...]
    x0: int
    v0: int

    def __call__(self, x: int) -> int:
        v = self.v0
        d = x ^ self.x0
        for i, col in enumerate(self.cols):
            if d >> i & 1:
                v ^= col
        return v


def fit_gf2(observe: Observe, x0: int, width_bits: int) -> Gf2Affine | None:
    """GF(2)-affine map behind *observe* on a *width_bits* field; None if not one."""
    v0 = observe(x0)
    if v0 is None:
        return None

    cols: list[int] = []
    for i in range(width_bits):
        v = observe(x0 ^ (1 << i))
        if v is None:
            return None
        cols.append(v ^ v0)

    # Flat: no bit moves the operand.
    if not any(cols):
        return None

    model = Gf2Affine(tuple(cols), x0, v0)
    field = (1 << width_bits) - 1
    for mask in _GF2_CONFIRMS:
        x = x0 ^ (mask & field)
        if observe(x) != model(x):
            return None

    return model


def _top(v: int) -> int:
    return v.bit_length() - 1


def solve_gf2(model: Gf2Affine, target: int) -> int | None:
    """``x`` with ``model(x) == target``, changing as few columns as found; else None.

    XOR basis keyed by top bit; each basis vector carries the set of
    columns that sums to it, so reducing the target yields ``x ^ x0``.
    """
    basis: dict[int, tuple[int, int]] = {}
    for i, col in enumerate(model.cols):
        vec, combo = col, 1 << i
        while vec and _top(vec) in basis:
            bvec, bcombo = basis[_top(vec)]
            vec ^= bvec
            combo ^= bcombo
        if vec:
            basis[_top(vec)] = (vec, combo)

    rest, d = target ^ model.v0, 0
    while rest:
        top = _top(rest)
        if top not in basis:
            return None
        bvec, bcombo = basis[top]
        rest ^= bvec
        d ^= bcombo

    return model.x0 ^ d


# ── additive ─────────────────────────────────────────────────────────


def _toward_zero(num: int, den: int) -> int:
    q = abs(num) // abs(den)
    return q if (num < 0) == (den < 0) else -q


def _greedy(xs: Sequence[int], weights: Sequence[int], need: int) -> list[int] | None:
    """Split *need* over the bytes, largest |weight| first; None if a residue is left."""
    out = list(xs)
    order = sorted(range(len(xs)), key=lambda i: -abs(weights[i]))
    for i in order:
        w = weights[i]
        if need == 0:
            break
        if w == 0:
            continue

        step = max(-xs[i], min(BYTE_MAX - xs[i], _toward_zero(need, w)))
        out[i] = xs[i] + step
        need -= w * step

    return out if need == 0 else None


def solve_additive(
    xs: Sequence[int], weights: Sequence[int], v0: int, target: int, bits: int
) -> list[int] | None:
    """New byte values with ``v0 + sum w_i * (new_i - xs_i) == target (mod 2^bits)``.

    Example: 16 bytes summing to 0x0800, target 0x0A57 -> +0x257 spread over
    the bytes, each capped at 0xFF. None when no split fits the byte ranges.
    """
    mod = 1 << bits
    need = (target - v0) % mod
    # Both representatives: forward and backward through the modulus.
    for rep in (need, need - mod):
        out = _greedy(xs, weights, rep)
        if out is not None:
            return out

    return None


# ── byte map ─────────────────────────────────────────────────────────


def _guesses(x: int, v: int, t: int) -> list[int]:
    """Identity, then constant offset (``v = x + k``), then every byte."""
    first = [t, x + (t - v)]
    head = [g for n, g in enumerate(first) if 0 <= g <= BYTE_MAX and g not in first[:n]]
    return head + [g for g in range(BYTE_VALUES) if g not in head]


def invert_bytes(
    observe: ObserveBytes,
    links: dict[int, int],
    xs: dict[int, int],
    current: bytes,
    target: bytes,
    max_execs: int,
) -> dict[int, int] | None:
    """Input byte values making each linked operand byte equal *target*.

    ``links`` maps operand position j -> input offset i (the only input byte
    that moves v[j]); ``xs`` holds the input bytes, ``current`` the operand
    they give. Positions are independent, so one exec tests one guess for
    every unsolved position. None when a position runs out of guesses or
    the budget is spent.
    """
    todo = {j: _guesses(xs[i], current[j], target[j]) for j, i in links.items()}
    solved: dict[int, int] = {}
    tried = 0

    while todo:
        if tried >= max_execs or any(not g for g in todo.values()):
            return None

        trial = {links[j]: g[0] for j, g in todo.items()}
        out = observe({**solved, **trial})
        tried += 1
        if out is None:
            return None

        for j in list(todo):
            if j < len(out) and out[j] == target[j]:
                solved[links[j]] = trial[links[j]]
                del todo[j]
            else:
                todo[j].pop(0)

    return solved
