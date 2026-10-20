"""Multiply-with-carry (MWC / FCSR) state recovery, known and unknown multiplier.

``prng_state_recovery`` solves GF(2)-linear generators and ``lcg_recovery``
truncated LCGs. A multiply-with-carry generator is neither: it is a
feedback-with-carry shift register (FCSR), the 2-adic analogue of an LFSR
(Arndt, *Matters Computational*, sec. 41.7, shows the 1-bit case
``a <- 2a mod c``).

Model, base ``b = 2^32``, state ``(x, c)`` with ``0 <= c < a``:

    output  x
    step    t = a*x + c;  x' = t mod b;  c' = t div b

Known multiplier: ``c0 = (x1 - a*x0) mod b`` and ``c0 < a`` pins the state
from two words; a third word confirms it.

Unknown multiplier: with ``z = c*b + x`` the step is ``z' = a*z mod (a*b-1)``,
so the output stream (word 0 first, LSB first inside a word) is the 2-adic
expansion of a fraction ``p/q`` with ``q = a*b - 1``. Rational approximation
(Klapper-Goresky) finds the shortest such ``p/q`` from ``N >= 2*log2(q) + 1``
bits, so five words give ``q``, hence ``a = (q + 1) / b``. When ``a*b - 1`` is
composite, ``p/q`` can reduce and ``q + 1`` is no longer a multiple of ``b``:
the result is then refused, never a wrong multiplier. Every result is replayed
against the observed words before use.

State is a 2-tuple ``(x, c)`` so the learner handles it like any other.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

__all__ = [
    "MIN_UNKNOWN_WORDS",
    "MWC_FAMILIES",
    "MWCSpec",
    "adic_synth",
    "confident_samples",
    "min_samples",
    "mwc_family",
    "output_word",
    "predict_words",
    "recover_state",
    "recover_unknown",
    "step_state",
    "verify_state",
    "walk_stream",
]

OUT_BITS = 32
OUT_BYTES = OUT_BITS // 8
_BASE = 1 << OUT_BITS
_WORD_MASK = _BASE - 1
# Known multiplier: x0, x1 pin c0; one more word confirms.
_MIN_KNOWN_WORDS = 2
_CONFIRM_WORDS = 1
# q = a*b - 1 has at most 2*OUT_BITS bits; rational approximation needs
# N >= 2*log2(q) + 1 bits.
_MAX_Q_BITS = 2 * OUT_BITS
MIN_UNKNOWN_WORDS = -(-(2 * _MAX_Q_BITS + 1) // OUT_BITS)
# Lattice combinations tried around the reduced basis when q must be odd.
_COMBO_RANGE = range(-3, 4)
_NAME_RE = re.compile(r"mwc32_([0-9]{1,10})")


@dataclass(frozen=True)
class MWCSpec:
    """``t = a*x + c``; ``x, c = t mod 2^32, t div 2^32``; output ``x``."""

    name: str = field(compare=False)
    a: int = 0
    out_bits: ClassVar[int] = OUT_BITS
    out_bytes: ClassVar[int] = OUT_BYTES

    def __post_init__(self) -> None:
        if not 0 < self.a < _BASE:
            raise ValueError(f"{self.name}: multiplier must be in (0, 2^32)")


def _named(a: int) -> MWCSpec:
    return MWCSpec(f"mwc32_{a}", a=a)


# Multipliers whose a*b - 1 is a safe prime (full-period l-sequence).
MWC_FAMILIES: dict[str, MWCSpec] = {
    spec.name: spec for spec in map(_named, (698769069, 4294584393, 4294883355))
}


def mwc_family(name: str) -> MWCSpec | None:
    """The family called *name*, or a spec rebuilt from ``mwc32_<a>``; None if invalid."""
    spec = MWC_FAMILIES.get(name)
    if spec is not None:
        return spec
    found = _NAME_RE.fullmatch(name)
    if found is None:
        return None
    a = int(found.group(1))
    return _named(a) if 0 < a < _BASE else None


def min_samples(spec: MWCSpec) -> int:
    return _MIN_KNOWN_WORDS


def confident_samples(spec: MWCSpec) -> int:
    """One word of false-positive margin (about 2^-32) over :func:`min_samples`."""
    return min_samples(spec) + _CONFIRM_WORDS


def step_state(state: Sequence[int], spec: MWCSpec) -> tuple[int, int]:
    carry, word = divmod(spec.a * state[0] + state[1], _BASE)
    return word, carry


def output_word(state: Sequence[int], spec: MWCSpec) -> int:
    return state[0]


def walk_stream(
    state: Sequence[int], n: int, spec: MWCSpec
) -> Iterator[tuple[tuple[int, int], int]]:
    """Yield ``(state, its output)`` for each of the next *n* steps."""
    a = spec.a
    x, c = state
    for _ in range(n):
        c, x = divmod(a * x + c, _BASE)
        yield (x, c), x


def predict_words(state: Sequence[int], n: int, spec: MWCSpec) -> list[int]:
    """The *n* outputs after *state*'s own output."""
    return [w for _, w in walk_stream(state, n, spec)]


def verify_state(state: Sequence[int], observed: Sequence[int], spec: MWCSpec) -> bool:
    """Replay from *state*; True when every observed word matches."""
    if not observed or output_word(state, spec) != observed[0]:
        return False
    return predict_words(state, len(observed) - 1, spec) == list(observed[1:])


def recover_state(observed: Sequence[int], spec: MWCSpec) -> tuple[int, int] | None:
    """State whose own output is ``observed[0]``, verified on all of *observed*.

    Returns:
        ``(x, c)``, or None when no reachable state reproduces *observed*.

    Raises:
        ValueError: fewer than :func:`min_samples` words.
    """
    needed = min_samples(spec)
    if len(observed) < needed:
        raise ValueError(f"need >= {needed} consecutive outputs to pin {spec.name}")
    if any(not 0 <= y <= _WORD_MASK for y in observed):
        return None

    x0, x1 = observed[0], observed[1]
    carry = (x1 - spec.a * x0) % _BASE
    # A reachable carry is < a: otherwise no earlier state produced x0.
    if carry >= spec.a:
        return None

    state = (x0, carry)
    return state if verify_state(state, observed, spec) else None


def adic_synth(s: int, n: int) -> tuple[int, int]:
    """Shortest ``(p, q)``, q odd and positive, with ``p/q = s`` 2-adically mod ``2^n``.

    ``s`` is the stream as an integer (first bit = bit 0). Gauss reduction of
    the lattice ``{(q, p) : p = q*s mod 2^n}``, then a small search around the
    reduced basis for the shortest vector with odd ``q`` (max-norm). Reliable
    once ``n >= 2*bitlen(max(|p|, q)) + 1``.
    """
    mod = 1 << n
    u, v = (1, s % mod), (0, mod)
    while True:
        nu = u[0] * u[0] + u[1] * u[1]
        nv = v[0] * v[0] + v[1] * v[1]
        if nu > nv:
            u, v, nu = v, u, nv
        # round(<u, v> / <u, u>): Gauss step.
        step = (2 * (u[0] * v[0] + u[1] * v[1]) + nu) // (2 * nu)
        if step == 0:
            break
        v = (v[0] - step * u[0], v[1] - step * u[1])

    best = None
    for i in _COMBO_RANGE:
        for j in _COMBO_RANGE:
            q, p = i * u[0] + j * v[0], i * u[1] + j * v[1]
            if not q & 1:
                continue
            size = max(abs(p), abs(q))
            if best is None or size < best[0]:
                best = (size, p, q)

    _, p, q = best
    return (-p, -q) if q < 0 else (p, q)


def recover_unknown(observed: Sequence[int]) -> tuple[MWCSpec, tuple[int, int]] | None:
    """Recover multiplier and state of an MWC from >= :data:`MIN_UNKNOWN_WORDS` words.

    Returns:
        ``(spec, state)`` verified on all of *observed*, or None.

    Raises:
        ValueError: fewer than :data:`MIN_UNKNOWN_WORDS` words.
    """
    if len(observed) < MIN_UNKNOWN_WORDS:
        raise ValueError(f"need >= {MIN_UNKNOWN_WORDS} consecutive outputs")
    if any(not 0 <= y <= _WORD_MASK for y in observed):
        return None

    packed = sum(y << (OUT_BITS * i) for i, y in enumerate(observed))
    _, q = adic_synth(packed, OUT_BITS * len(observed))

    # q = a*b - 1: reject anything else before spending a replay.
    a, rest = divmod(q + 1, _BASE)
    if rest or not 0 < a < _BASE:
        return None

    spec = _named(a)
    state = recover_state(observed, spec)
    return (spec, state) if state is not None else None
