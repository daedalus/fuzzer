"""Truncated-output LCG state recovery by lattice reduction (P4-1).

``prng_state_recovery`` solves GF(2)-linear generators; an LCG's step
``x' = a*x + c mod m`` is multiplicative, so it is out of that module's
reach. Lattice reduction is the standard tool for it.

Model: the target sees ``y_i = (x_i >> shift) & (2^out_bits - 1)``.
Write ``x_i = h_i + z_i`` with ``h_i = y_i << shift`` known and
``0 <= z_i < 2^shift`` unknown. With ``A_i = a^i`` and ``C_i`` the
accumulated increment (both mod M):

    z_i ≡ A_i z_0 + d_i  (mod M),   d_i = A_i h_0 + C_i - h_i

so ``(z_0, z_1 - d_1, ..., z_{n-1} - d_{n-1})`` lies in the lattice

    row 0 : [1, A_1, A_2, ..., A_{n-1}]
    row i : M * e_i                      (i >= 1)

and is within ``2^(shift-1)`` per coordinate of the target
``t = (2^(shift-1), 2^(shift-1) - d_1, ...)``. LLL + Babai rounding finds it;
replaying the stream accepts or rejects the result, so float error in LLL
can cost a recovery but never produce a wrong one.

Modulus reduction: for ``m = 2^k`` the bits above ``shift + out_bits`` never
feed lower bits, so ``M = 2^(shift+out_bits)`` and the unobservable top bits
are dropped (MSVC ``rand``: 32-bit state, 31 bits matter).

State is a 1-tuple ``(x,)`` so the learner handles it like a linear state.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from fuzzer_tool.core.lattice import babai_round, lll_reduce

__all__ = [
    "LCG_FAMILIES",
    "LCGSpec",
    "confident_samples",
    "lcg_family",
    "min_samples",
    "output_word",
    "predict_words",
    "recover_state",
    "step_state",
    "verify_state",
    "walk_stream",
]

# Lattice dimension cap: extra samples beyond it only verify.
_MAX_DIM = 8
# Extra output word on top of min_samples, same margin as prng_state_recovery.
_CONFIRM_WORDS = 1


@dataclass(frozen=True)
class LCGSpec:
    """``x' = (a*x + c) mod m``; output ``(x >> shift) & (2^out_bits - 1)``."""

    name: str = field(compare=False)
    a: int = 0
    c: int = 0
    m: int = 1
    shift: int = 0
    out_bits: int = 32
    out_bytes: int = 4

    def __post_init__(self) -> None:
        if self.modulus > 1 << (self.shift + self.out_bits):
            raise ValueError(f"{self.name}: modulus bits above the output window are unobservable")

    @property
    def modulus(self) -> int:
        """Effective modulus: power-of-two moduli drop bits the output never sees."""
        window = 1 << (self.shift + self.out_bits)
        if self.m & (self.m - 1) == 0:
            return min(self.m, window)
        return self.m

    @property
    def state_bits(self) -> int:
        return (self.modulus - 1).bit_length()


LCG_FAMILIES: dict[str, LCGSpec] = {
    spec.name: spec
    for spec in (
        # java.util.Random.next(32)
        LCGSpec("java", a=0x5DEECE66D, c=0xB, m=1 << 48, shift=16, out_bits=32),
        # MSVC rand()
        LCGSpec("msvc", a=214013, c=2531011, m=1 << 32, shift=16, out_bits=15),
        # POSIX.1 example rand()
        LCGSpec("posix_rand", a=1103515245, c=12345, m=1 << 32, shift=16, out_bits=15),
        # C++ minstd_rand0 / minstd_rand: full-state output
        LCGSpec("minstd_rand0", a=16807, c=0, m=(1 << 31) - 1, shift=0, out_bits=31),
        LCGSpec("minstd_rand", a=48271, c=0, m=(1 << 31) - 1, shift=0, out_bits=31),
    )
}


def lcg_family(name: str) -> LCGSpec | None:
    """The shipped family called *name*, or None."""
    return LCG_FAMILIES.get(name)


def min_samples(spec: LCGSpec) -> int:
    """Outputs whose bits exceed the state bits, plus one for lattice slack.

    Measured (100 random states each): java recovers 100/100 from 2 outputs,
    msvc 52/100 from 2 and 100/100 from 3.
    """
    return math.ceil(spec.state_bits / spec.out_bits) + 1


def confident_samples(spec: LCGSpec) -> int:
    """One output word of false-positive margin over :func:`min_samples`."""
    return min_samples(spec) + _CONFIRM_WORDS


def step_state(state: Sequence[int], spec: LCGSpec) -> tuple[int]:
    return ((spec.a * state[0] + spec.c) % spec.modulus,)


def output_word(state: Sequence[int], spec: LCGSpec) -> int:
    return (state[0] >> spec.shift) & ((1 << spec.out_bits) - 1)


def walk_stream(state: Sequence[int], n: int, spec: LCGSpec) -> Iterator[tuple[tuple[int], int]]:
    """Yield ``(state, its output)`` for each of the next *n* steps."""
    a, c, mod = spec.a, spec.c, spec.modulus
    shift, mask = spec.shift, (1 << spec.out_bits) - 1
    x = state[0]
    for _ in range(n):
        x = (a * x + c) % mod
        yield (x,), (x >> shift) & mask


def predict_words(state: Sequence[int], n: int, spec: LCGSpec) -> list[int]:
    """The *n* outputs after *state*'s own output."""
    return [w for _, w in walk_stream(state, n, spec)]


def verify_state(state: Sequence[int], observed: Sequence[int], spec: LCGSpec) -> bool:
    """Replay from *state*; True when every observed word matches."""
    if not observed or output_word(state, spec) != observed[0]:
        return False
    return predict_words(state, len(observed) - 1, spec) == list(observed[1:])


def _lattice(ys: Sequence[int], spec: LCGSpec) -> tuple[list[list[int]], list[int], int]:
    """Basis, Babai target and h_0 for the observed words (see module docstring)."""
    mod, n = spec.modulus, len(ys)
    h = [y << spec.shift for y in ys]

    # A_i = a^i, C_i = c * (a^(i-1) + ... + 1), both mod M.
    mults, incs = [1], [0]
    for _ in range(1, n):
        mults.append(mults[-1] * spec.a % mod)
        incs.append((incs[-1] * spec.a + spec.c) % mod)
    d = [(mults[i] * h[0] + incs[i] - h[i]) % mod for i in range(n)]

    basis = [mults] + [[mod if j == i else 0 for j in range(n)] for i in range(1, n)]
    half = (1 << spec.shift) >> 1
    target = [half] + [half - d[i] for i in range(1, n)]
    return basis, target, h[0]


def recover_state(observed: Sequence[int], spec: LCGSpec) -> tuple[int] | None:
    """Recover the state whose own output is ``observed[0]``, verified on all of *observed*.

    Returns:
        ``(x,)``, or None when no state reproduces *observed*.

    Raises:
        ValueError: fewer than :func:`min_samples` words.
    """
    needed = min_samples(spec)
    if len(observed) < needed:
        raise ValueError(f"need >= {needed} consecutive outputs to pin {spec.name}")

    limit = 1 << spec.out_bits
    if any(not 0 <= y < limit for y in observed):
        return None

    # Full-state output: nothing to solve.
    if spec.shift == 0:
        state = (observed[0] % spec.modulus,)
        return state if verify_state(state, observed, spec) else None

    basis, target, h0 = _lattice(observed[:_MAX_DIM], spec)
    try:
        point = babai_round(lll_reduce(basis), target)
    except ValueError:
        return None

    state = ((h0 + point[0]) % spec.modulus,)
    return state if verify_state(state, observed, spec) else None
