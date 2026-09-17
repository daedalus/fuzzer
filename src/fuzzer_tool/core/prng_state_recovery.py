"""Predictable-PRNG state recovery for word-level GF(2)-linear generators.

Ports the methodology from Gegell's "Reversing Factorio's RNG"
(https://gegell.github.io/posts/factorio-rng, code at
https://github.com/Gegell/Factorio-RNG-Release) into this repo's existing
GF(2) toolkit, generalized past Factorio's specific ``taus88`` instance to
*any* generator whose step is linear over its state bits.

Factorio uses Boost's ``taus88``: three ``linear_feedback_shift_engine``
components XORed together::

    typedef xor_combine_engine<
        xor_combine_engine<
            linear_feedback_shift_engine<uint32_t, 32, 31, 13, 12>, 0,
            linear_feedback_shift_engine<uint32_t, 32, 29, 2, 4>, 0>, 0,
        linear_feedback_shift_engine<uint32_t, 32, 28, 3, 17>, 0> taus88;

That is one point in a much larger family. A generator's step qualifies
here whenever it is built from XOR, constant-mask AND, and shifts of its
own state words -- which covers combined Tausworthe/LFSR generators
(taus88, taus113 a.k.a. LFSR113, LFSR258), Marsaglia xorshift (which Brent
proved is an LFSR in disguise: "Note on Marsaglia's xorshift random number
generators", JSS 11, 2004), and classic single-register Galois/Fibonacci
LFSRs. See :data:`FAMILIES` for the ones shipped, and
:func:`combined_lfsr` / :func:`xorshift` / :func:`galois_lfsr` /
:func:`fib_lfsr` for building one that isn't.

How a generator is described
----------------------------
As a straight-line program over a register file of words: the state slots
first, then scratch. Five opcodes (:class:`Opcode`), every one of them
GF(2)-linear, so the *same* program runs two ways:

- on plain ints, to simulate the generator forward
  (:func:`step_state`, :func:`output_word`, :func:`predict_words`);
- on bitmask coefficient vectors -- one basis bit per unknown state bit,
  the representation :mod:`fuzzer_tool.core.gf2_common`'s bitmask-vector
  layer and :mod:`fuzzer_tool.core.xor_map_solver` already use -- to get
  the symbolic equations recovery needs.

One description, two evaluators, so the simulated stream and the recovered
state cannot drift apart: a family is either wrong in both or right in
both. Writing the step as a program rather than a parameter tuple is what
makes the coverage general -- any linear word map is expressible, because
a single (mask, shift) pair already carries an arbitrary input bit to an
arbitrary output bit and XOR sums them.

Recovery
--------
Each step is GF(2)-linear in the state bits, so the whole generator's
*state* (not just its output) evolves as ``s(t+1) = T @ s(t)`` over
``GF(2)^n``. Gegell's original approach builds that generation matrix with
sympy symbols and solves for the state with a manual RREF (see
``factorio_rng.py::build_generation_matrix`` and ``get_state`` in the
release repo). This module gets the same result without sympy, by reusing
:class:`fuzzer_tool.core.xor_map_solver.IncrementalXorMapSolver` outright:
"recover an unknown n-bit state from observed output bits" is the same
GF(2) elimination problem as "recover an unknown linear map from (input,
output) pairs" -- here each equation's known coefficient row plays the
role of the solver's ``input_bits``, and the observed output bit (bit 0
only; there is a single "output") plays the role of ``output_bits``. The
solver's full-rank determinacy gate then does exactly the job Gegell did
by hand: refuse a state unless the observations pin it uniquely.

How many outputs that takes is a property of the generator, not a
constant: :func:`min_samples` derives it per family by rank-probing the
symbolic equations (3 for taus88, 4 for taus113, 1 for xorshift32 -- whose
state *is* its output). :func:`confident_samples` adds one whole output
word on top as false-positive margin (see that function).

Why this matters for fuzzing/vuln-hunting: many C/C++ targets seed a weak
linear PRNG for session tokens, nonces, or "random" identifiers --
CWE-338. The Linux kernel's own ``prandom_u32`` was taus88 and is now
taus113; ``boost::random`` ships taus88; xorshift is everywhere. Given a
handful of consecutive outputs (leaked, side-channeled, or observed across
requests), this module recovers the internal state and predicts every
future output, without needing to reverse the target's binary the way the
original Factorio writeup did. That reversing step is now only needed to
say *which* family a target uses -- and even that can be answered by
trying the shipped ones and keeping whichever verifies, which is what
:class:`fuzzer_tool.core.prng_state_learner.PRNGStateLearner` does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

from fuzzer_tool.core.xor_map_solver import IncrementalXorMapSolver

__all__ = [
    "FAMILIES",
    "LFSR258",
    "LFSR258_PARAMS",
    "TAUS88",
    "TAUS88_PARAMS",
    "TAUS113",
    "TAUS113_PARAMS",
    "XORSHIFT32",
    "XORSHIFT64",
    "XORSHIFT128",
    "Instr",
    "LFSRParams",
    "LinearPRNG",
    "Opcode",
    "advance_state",
    "combined_lfsr",
    "confident_samples",
    "family",
    "fib_lfsr",
    "galois_lfsr",
    "min_samples",
    "output_word",
    "predict_next",
    "predict_words",
    "recover_state",
    "recover_taus88_state",
    "spec_from_params",
    "step_state",
    "structural_equations",
    "taus88_output",
    "taus88_step",
    "verify_recovery",
    "verify_state",
    "xorshift",
]

# Ceiling on the rank probe in min_samples(). A generator whose outputs
# have not pinned its state after this many words either never will (an
# output map that discards bits permanently) or is far outside anything
# the cmplog observation channel could feed, so raising beats looping.
_MAX_PROBE_WORDS = 64
# Extra observed words demanded on top of min_samples(), as
# false-positive margin. See confident_samples().
_CONFIRM_WORDS = 1


class Opcode(Enum):
    """The GF(2)-linear word operations a generator step is built from.

    ``AND`` is linear because its operand is a *constant* mask: it clears
    bits, it does not mix them. Shifts move bits without mixing them, and
    ``XOR`` is addition in GF(2). Nothing else is allowed, which is
    precisely what lets a program written with these be evaluated
    symbolically as well as concretely.
    """

    MOV = "mov"
    XOR = "xor"
    SHL = "shl"
    SHR = "shr"
    AND = "and"


@dataclass(frozen=True)
class Instr:
    """One instruction, ``dst = op(a, b)``, over the register file.

    ``b`` is a *register index* for ``XOR`` and an *immediate* otherwise
    (shift count for ``SHL``/``SHR``, bit mask for ``AND``); ``MOV``
    ignores it.
    """

    op: Opcode
    dst: int
    a: int
    b: int = 0


@dataclass(frozen=True)
class LinearPRNG:
    """A word-linear generator: its register widths and its two programs.

    Registers ``0 .. n_slots-1`` are the state words, in state-tuple
    order; the rest are scratch. ``step`` advances the state in place;
    ``out`` (which may be empty) leaves the output word in ``out_reg``.

    ``name`` is excluded from equality and hashing on purpose: identity
    here is the program, so two specs describing the same generator share
    one cache entry however they were labelled.
    """

    name: str = field(compare=False)
    widths: tuple[int, ...] = ()
    n_slots: int = 0
    step: tuple[Instr, ...] = ()
    out: tuple[Instr, ...] = ()
    out_reg: int = 0

    @property
    def state_bits(self) -> int:
        """Unknown bits recovery has to solve for."""
        return sum(self.widths[: self.n_slots])

    @property
    def out_bits(self) -> int:
        return self.widths[self.out_reg]

    @property
    def out_bytes(self) -> int:
        """Operand width this generator's draws show up at in cmplog."""
        return self.out_bits // 8


# ── Program construction ──────────────────────────────────────────────


def _word_mask(width: int) -> int:
    return (1 << width) - 1


class _Alloc:
    """Register allocator: state slots first, scratch after."""

    def __init__(self) -> None:
        self.widths: list[int] = []
        self.n_slots = 0

    def slot(self, width: int) -> int:
        """Allocate a state slot. All slots must precede all temps."""
        if self.n_slots != len(self.widths):
            raise ValueError("state slots must be allocated before temps")
        self.widths.append(width)
        self.n_slots += 1
        return len(self.widths) - 1

    def temp(self, width: int) -> int:
        self.widths.append(width)
        return len(self.widths) - 1


class _Block:
    """Emitter for one program: a step, or an output map."""

    def __init__(self, alloc: _Alloc) -> None:
        self._alloc = alloc
        self.code: list[Instr] = []

    def temp(self, width: int) -> int:
        return self._alloc.temp(width)

    def mov(self, dst: int, a: int) -> None:
        self.code.append(Instr(Opcode.MOV, dst, a))

    def xor(self, dst: int, a: int, b: int) -> None:
        self.code.append(Instr(Opcode.XOR, dst, a, b))

    def shl(self, dst: int, a: int, n: int) -> None:
        self.code.append(Instr(Opcode.SHL, dst, a, n))

    def shr(self, dst: int, a: int, n: int) -> None:
        self.code.append(Instr(Opcode.SHR, dst, a, n))

    def mask(self, dst: int, a: int, m: int) -> None:
        self.code.append(Instr(Opcode.AND, dst, a, m))


def _build(
    name: str, alloc: _Alloc, step: _Block, out: _Block, out_reg: int
) -> LinearPRNG:
    return LinearPRNG(
        name=name,
        widths=tuple(alloc.widths),
        n_slots=alloc.n_slots,
        step=tuple(step.code),
        out=tuple(out.code),
        out_reg=out_reg,
    )


def _xor_combine(out: _Block, slots: Sequence[int], width: int) -> int:
    """Emit ``acc = slots[0] ^ slots[1] ^ ...`` and return ``acc``."""
    if len(slots) == 1:
        return slots[0]

    acc = out.temp(width)
    out.mov(acc, slots[0])
    for slot in slots[1:]:
        out.xor(acc, acc, slot)
    return acc


# (word_size, lfsr_bits, feedback_tap, step_count) per component: the
# Boost linear_feedback_shift_engine<UIntType, w, k, q, s> arguments.
LFSRParams = tuple[int, int, int, int]


def _emit_lfs_engine(step: _Block, slot: int, params: LFSRParams) -> None:
    """One step of a single Boost ``linear_feedback_shift_engine``::

        b = (((v << q) ^ v) & word_mask) >> (k - s)
        v = ((v & (word_mask << (w - k))) << s) ^ b

    ``b`` reads the pre-step word, so it goes to a temp before ``slot``
    is overwritten.
    """
    w, k, q, s = params

    b = step.temp(w)
    step.shl(b, slot, q)
    step.xor(b, b, slot)
    step.shr(b, b, k - s)

    high = step.temp(w)
    step.mask(high, slot, (_word_mask(w) << (w - k)) & _word_mask(w))
    step.shl(high, high, s)
    step.xor(slot, high, b)


def combined_lfsr(name: str, params: Sequence[LFSRParams]) -> LinearPRNG:
    """XOR-combine of Boost ``linear_feedback_shift_engine`` components.

    The shape of taus88, taus113/LFSR113, LFSR258 and every other combined
    Tausworthe generator: each component advances independently and the
    output is the XOR of all of them.
    """
    alloc = _Alloc()
    step, out = _Block(alloc), _Block(alloc)

    slots = [alloc.slot(component[0]) for component in params]
    for slot, component in zip(slots, params, strict=True):
        _emit_lfs_engine(step, slot, component)

    out_reg = _xor_combine(out, slots, max(c[0] for c in params))
    return _build(name, alloc, step, out, out_reg)


def xorshift(name: str, width: int, shifts: Sequence[int]) -> LinearPRNG:
    """Single-word Marsaglia xorshift: ``x ^= x << a`` / ``x ^= x >> a``.

    Positive entries in *shifts* are left shifts, negative are right,
    applied in order. State is one word and the output is that word, so a
    single observed draw already *is* the state: the value here is in the
    shared prediction machinery, not in the solve.
    """
    alloc = _Alloc()
    step, out = _Block(alloc), _Block(alloc)

    slot = alloc.slot(width)
    tmp = step.temp(width)
    for shift in shifts:
        if shift >= 0:
            step.shl(tmp, slot, shift)
        else:
            step.shr(tmp, slot, -shift)
        step.xor(slot, slot, tmp)

    return _build(name, alloc, step, out, slot)


def _xorshift128(name: str) -> LinearPRNG:
    """Marsaglia's four-word xorshift128::

        t = x ^ (x << 11); x = y; y = z; z = w;
        w ^= (w >> 19) ^ t ^ (t >> 8); return w

    Unlike the combined-LFSR families this step is *not* block diagonal --
    the new ``w`` reads the old ``x`` -- which is the case a per-component
    parameter tuple cannot express and a program can.
    """
    alloc = _Alloc()
    step, out = _Block(alloc), _Block(alloc)
    x, y, z, w = (alloc.slot(32) for _ in range(4))

    t = step.temp(32)
    step.shl(t, x, 11)
    step.xor(t, t, x)

    fresh = step.temp(32)
    step.shr(fresh, w, 19)
    step.xor(fresh, fresh, w)
    step.xor(fresh, fresh, t)

    shifted = step.temp(32)
    step.shr(shifted, t, 8)
    step.xor(fresh, fresh, shifted)

    # The rotation happens only after both reads above, as in the C.
    step.mov(x, y)
    step.mov(y, z)
    step.mov(z, w)
    step.mov(w, fresh)

    return _build(name, alloc, step, out, w)


def galois_lfsr(name: str, width: int, poly: int) -> LinearPRNG:
    """Classic Galois (right-shifting) LFSR: ``x >>= 1; x ^= poly if x&1``.

    The conditional is linear: bit 0 of the pre-step word is XORed into
    every tap position of *poly*, one instruction per set tap.

    The whole register is taken as the output word. A hardware-style
    one-bit-per-step output stream is a different observation channel from
    the whole-word comparison operands cmplog gives us, and out of scope.
    """
    alloc = _Alloc()
    step, out = _Block(alloc), _Block(alloc)
    slot = alloc.slot(width)

    lsb = step.temp(width)
    step.mask(lsb, slot, 1)
    step.shr(slot, slot, 1)

    tap = step.temp(width)
    for bit in range(width):
        if (poly >> bit) & 1:
            step.shl(tap, lsb, bit)
            step.xor(slot, slot, tap)

    return _build(name, alloc, step, out, slot)


def fib_lfsr(name: str, width: int, taps: Sequence[int]) -> LinearPRNG:
    """Classic Fibonacci LFSR: XOR the *taps* into the vacated high bit.

    ``x = (x >> 1) ^ (XOR of bits x[t] for t in taps) << (width - 1)``.
    """
    alloc = _Alloc()
    step, out = _Block(alloc), _Block(alloc)
    slot = alloc.slot(width)

    feedback = step.temp(width)
    bit = step.temp(width)
    step.mask(feedback, slot, 1 << taps[0])
    step.shr(feedback, feedback, taps[0])
    for tap in taps[1:]:
        step.mask(bit, slot, 1 << tap)
        step.shr(bit, bit, tap)
        step.xor(feedback, feedback, bit)

    step.shl(feedback, feedback, width - 1)
    step.shr(slot, slot, 1)
    step.xor(slot, slot, feedback)

    return _build(name, alloc, step, out, slot)


# ── Shipped families ─────────────────────────────────────────────────

# L'Ecuyer 1996, "Maximally Equidistributed Combined Tausworthe
# Generators": Boost's taus88, GSL's taus/taus2, Linux prandom_u32 before
# 2013, and Factorio's generator.
TAUS88_PARAMS: tuple[LFSRParams, ...] = (
    (32, 31, 13, 12),
    (32, 29, 2, 4),
    (32, 28, 3, 17),
)
# L'Ecuyer 1999 erratum table (LFSR113): GSL taus113, Linux prandom_u32
# since 2013, ISPC's stdlib RNG. Read off the published
# TAUSWORTHE(s, q, k-s, word_mask << (w-k), s) expressions.
TAUS113_PARAMS: tuple[LFSRParams, ...] = (
    (32, 31, 6, 18),
    (32, 29, 2, 2),
    (32, 28, 13, 7),
    (32, 25, 3, 13),
)
# L'Ecuyer 1999, the 64-bit five-component generator. The component
# register widths 63/55/52/47/41 are the published period factorisation
# (2**63-1)(2**55-1)(2**52-1)(2**47-1)(2**41-1).
LFSR258_PARAMS: tuple[LFSRParams, ...] = (
    (64, 63, 1, 10),
    (64, 55, 24, 5),
    (64, 52, 3, 29),
    (64, 47, 5, 23),
    (64, 41, 3, 8),
)

TAUS88 = combined_lfsr("taus88", TAUS88_PARAMS)
TAUS113 = combined_lfsr("taus113", TAUS113_PARAMS)
LFSR258 = combined_lfsr("lfsr258", LFSR258_PARAMS)
# Marsaglia 2003, "Xorshift RNGs", the shift triples from pp. 4-5.
XORSHIFT32 = xorshift("xorshift32", 32, (13, -17, 5))
XORSHIFT64 = xorshift("xorshift64", 64, (13, -7, 17))
XORSHIFT128 = _xorshift128("xorshift128")

#: Generators addressable by name, and the default candidate set for
#: PRNGStateLearner. Ordered smallest state first, so a target using a
#: small generator is not charged for a large elimination before its own.
FAMILIES: dict[str, LinearPRNG] = {
    spec.name: spec
    for spec in (XORSHIFT32, TAUS88, TAUS113, XORSHIFT128, XORSHIFT64, LFSR258)
}


def family(name: str) -> LinearPRNG | None:
    """Look up a shipped family by name; ``None`` when unknown."""
    return FAMILIES.get(name)


# ── Concrete (integer) forward simulation ─────────────────────────────


def _read(regs: list[int], index: int, width: int) -> int:
    """Register *index* as a *width*-bit word (zero-extended, truncated)."""
    return regs[index] & _word_mask(width)


def _run(regs: list[int], code: Sequence[Instr], widths: Sequence[int]) -> None:
    """Execute *code* over the integer register file, in place."""
    for ins in code:
        width = widths[ins.dst]
        a = _read(regs, ins.a, width)

        if ins.op is Opcode.XOR:
            value = a ^ _read(regs, ins.b, width)
        elif ins.op is Opcode.SHL:
            value = a << ins.b
        elif ins.op is Opcode.SHR:
            value = a >> ins.b
        elif ins.op is Opcode.AND:
            value = a & ins.b
        else:
            value = a

        regs[ins.dst] = value & _word_mask(width)


def _regs_for(state: Sequence[int], spec: LinearPRNG) -> list[int]:
    return list(state) + [0] * (len(spec.widths) - spec.n_slots)


def step_state(state: Sequence[int], spec: LinearPRNG = TAUS88) -> tuple[int, ...]:
    """Advance *state* by one generator step."""
    regs = _regs_for(state, spec)
    _run(regs, spec.step, spec.widths)
    return tuple(regs[: spec.n_slots])


def output_word(state: Sequence[int], spec: LinearPRNG = TAUS88) -> int:
    """The generator's output word for the current *state*."""
    regs = _regs_for(state, spec)
    _run(regs, spec.out, spec.widths)
    return regs[spec.out_reg]


def advance_state(
    state: Sequence[int], n: int, spec: LinearPRNG = TAUS88
) -> tuple[int, ...]:
    """Step *state* forward *n* times."""
    current = tuple(state)
    for _ in range(n):
        current = step_state(current, spec)
    return current


def predict_words(
    state: Sequence[int], n: int = 1, spec: LinearPRNG = TAUS88
) -> list[int]:
    """Step *state* forward *n* times, returning the *n* outputs passed.

    Note this steps *before* recording, so it returns the outputs *after*
    ``state``'s own output -- it does not include ``output_word(state)``
    itself. A state returned by :func:`recover_state` already has
    ``output_word(state) == observed_words[0]`` with zero extra steps; use
    :func:`advance_state` for as many already-known words as you have
    before calling this, or :func:`verify_state` to check against a whole
    known sequence at once.
    """
    outs: list[int] = []
    current = tuple(state)
    for _ in range(n):
        current = step_state(current, spec)
        outs.append(output_word(current, spec))
    return outs


# ── Symbolic (GF(2) bitmask) simulation, for recovery ─────────────────
#
# A symbolic register is not a value but a list of `width` bitmasks over
# the unknown initial state's bits: bit j of vec[i] is set iff word bit i
# XORs in unknown bit j. Every opcode above is linear, so each one has an
# exact per-position analogue below.


def _sym_read(regs: list[list[int]], index: int, width: int) -> list[int]:
    vec = regs[index]
    if len(vec) >= width:
        return vec[:width]
    return vec + [0] * (width - len(vec))


def _run_sym(regs: list[list[int]], code: Sequence[Instr], widths: Sequence[int]) -> None:
    """Execute *code* over the symbolic register file, in place."""
    for ins in code:
        width = widths[ins.dst]
        a = _sym_read(regs, ins.a, width)

        if ins.op is Opcode.XOR:
            b = _sym_read(regs, ins.b, width)
            value = [a[i] ^ b[i] for i in range(width)]
        elif ins.op is Opcode.SHL:
            value = [0] * min(ins.b, width) + a[: max(0, width - ins.b)]
        elif ins.op is Opcode.SHR:
            value = a[ins.b :] + [0] * min(ins.b, width)
        elif ins.op is Opcode.AND:
            value = [a[i] if (ins.b >> i) & 1 else 0 for i in range(width)]
        else:
            value = list(a)

        regs[ins.dst] = value[:width]


def _sym_regs(spec: LinearPRNG) -> list[list[int]]:
    """Basis register file: state bit ``j`` depends only on itself."""
    regs: list[list[int]] = []
    offset = 0
    for index, width in enumerate(spec.widths):
        if index >= spec.n_slots:
            regs.append([0] * width)
            continue
        regs.append([1 << (offset + bit) for bit in range(width)])
        offset += width
    return regs


def _out_rows(regs: list[list[int]], spec: LinearPRNG) -> list[int]:
    """Coefficient rows of the output word, one per output bit.

    Runs on a copy: an output map may write scratch registers, and the
    caller still needs the symbolic state it passed in to step forward.
    """
    probe = [list(vec) for vec in regs]
    _run_sym(probe, spec.out, spec.widths)
    return probe[spec.out_reg]


def _left_nullspace(rows: Sequence[int]) -> list[int]:
    """Left null space of the map whose row ``i`` is ``rows[i]``.

    Each ``rows[i]`` is a coefficient mask (an equation row). Returns a
    list of masks over *row indices*: for every returned mask ``v``,
    ``XOR_{i where bit i of v is set} rows[i] == 0`` identically -- a
    linear relation that holds for *any* input to the map, not just the
    ones observed. Found with the standard "augment rows with a tracking
    identity column, eliminate, harvest every row that cancels to zero"
    trick: when a row reduces to the zero vector, its tracking column is
    by construction a combination of original rows that sums to zero.
    """
    pivots: dict[int, tuple[int, int]] = {}
    nulls: list[int] = []
    for i, value in enumerate(rows):
        v, t = value, 1 << i
        for p, (pv, pt) in pivots.items():
            if (v >> p) & 1:
                v ^= pv
                t ^= pt
        if v == 0:
            if t:
                nulls.append(t)
            continue
        pivot = (v & -v).bit_length() - 1
        pivots[pivot] = (v, t)
    return nulls


@lru_cache(maxsize=None)
def structural_equations(spec: LinearPRNG = TAUS88) -> tuple[int, ...]:
    """Free, observation-independent equations from the step's rank gap.

    A step that is not injective has an image smaller than the state
    space, and the bits of any state *in* that image satisfy linear
    relations that hold whatever the pre-step state was. Concretely for a
    ``linear_feedback_shift_engine``, per Gegell's writeup: "the least
    significant bits are linearly dependent on the higher significant
    bits" of the *same* post-step state, because the word size exceeds the
    register size (``k < w``) by ``w - k`` bits.

    Any state we are ever asked to recover (the one producing
    ``observed_words[0]``) is itself the result of at least one prior step
    of the real generator, so it already lies in that image and satisfies
    these relations -- for free, without spending an observed output on
    them. Folding them in is what turns taus88's raw 96-unknown system
    (which saturates at rank 92 from 3 outputs: an under-determined
    particular solution that reproduces the 3 fitting outputs and predicts
    the wrong future ones -- verified empirically) into a determined one.

    Derived from the whole-state step map rather than per component, so a
    generator whose components are not independent (xorshift128) needs no
    separate code path. A bijective step (every xorshift) simply yields
    none.

    Returns:
        Coefficient rows indexed by global state bit, each paired
        implicitly with right-hand side 0.
    """
    regs = _sym_regs(spec)
    _run_sym(regs, spec.step, spec.widths)

    rows: list[int] = []
    for slot in range(spec.n_slots):
        rows.extend(regs[slot])
    return tuple(_left_nullspace(rows))


def _seed_solver(spec: LinearPRNG) -> IncrementalXorMapSolver:
    solver = IncrementalXorMapSolver(spec.state_bits)
    for row in structural_equations(spec):
        solver.add_pair(row, 0)
    return solver


@lru_cache(maxsize=None)
def min_samples(spec: LinearPRNG = TAUS88) -> int:
    """Consecutive outputs needed to pin this generator's state uniquely.

    A property of the generator, so it is measured rather than assumed:
    output equations are added symbolically, one word at a time, until the
    elimination reaches full rank. Right-hand sides do not affect rank, so
    the probe runs once per family and is cached.

    Raises:
        ValueError: If the outputs never determine the state.
    """
    solver = _seed_solver(spec)
    regs = _sym_regs(spec)
    for count in range(1, _MAX_PROBE_WORDS + 1):
        for row in _out_rows(regs, spec):
            solver.add_pair(row, 0)
        if solver.is_determined:
            return count
        _run_sym(regs, spec.step, spec.widths)

    raise ValueError(f"{spec.name}: outputs never determine the state")


@lru_cache(maxsize=None)
def confident_samples(spec: LinearPRNG = TAUS88) -> int:
    """Outputs needed before a *verified* fit is worth trusting.

    :func:`min_samples` is the count that determines the state, and at
    exactly that count the system has almost no equations to spare, so
    unrelated constants fit: taus88's 3 outputs give 96 equations for 96
    unknowns but leave 8 free consistency bits, and random triples fit one
    in ~2**8 (measured: 74/20000). One whole extra output word takes the
    surplus to ``out_bits`` -- 40 bits for taus88, measured 0/20000 -- for
    the cost of one more observed operand, and a fuzzer's cmplog stream is
    full of unrelated constants.
    """
    return min_samples(spec) + _CONFIRM_WORDS


# ── Recovery ─────────────────────────────────────────────────────────


def _unpack(bits: int, spec: LinearPRNG) -> tuple[int, ...]:
    """Split a solved state bit-vector into per-slot words."""
    words: list[int] = []
    offset = 0
    for width in spec.widths[: spec.n_slots]:
        words.append((bits >> offset) & _word_mask(width))
        offset += width
    return tuple(words)


def recover_state(
    observed_words: Sequence[int], spec: LinearPRNG = TAUS88
) -> tuple[int, ...] | None:
    """Recover the internal state from consecutive outputs.

    Args:
        observed_words: Consecutive generator outputs, where
            ``observed_words[t]`` is the output of the state ``t`` steps
            after the (unknown) state being recovered.
            :func:`min_samples` words are the minimum; more are consistency
            checks, and :func:`confident_samples` is the count to trust.
        spec: Which generator. Defaults to taus88, the Factorio/Boost
            instance this module started from; see :data:`FAMILIES`.

    Returns:
        The recovered state, slot by slot, or ``None`` if the observations
        don't pin it down uniquely yet or are inconsistent with this
        generator at all.

    Raises:
        ValueError: If fewer than :func:`min_samples` words are given --
            the system cannot be determined from them, so a ``None``
            return would be indistinguishable from "wrong family".
    """
    needed = min_samples(spec)
    if len(observed_words) < needed:
        raise ValueError(
            f"need >= {needed} consecutive outputs to pin "
            f"{spec.name}'s {spec.state_bits}-bit state"
        )

    solver = _seed_solver(spec)
    regs = _sym_regs(spec)
    for word in observed_words:
        for bit, row in enumerate(_out_rows(regs, spec)):
            solver.add_pair(row, (word >> bit) & 1)
        _run_sym(regs, spec.step, spec.widths)

    if not solver.is_determined:
        return None

    solution, sat = solver.solve()
    if not sat or solution is None:
        return None

    bits = 0
    for index in solution[0]:
        bits |= 1 << index
    return _unpack(bits, spec)


def verify_state(
    state: Sequence[int], observed_words: Sequence[int], spec: LinearPRNG = TAUS88
) -> bool:
    """Replay from *state* and check every word in *observed_words* matches.

    Mirrors the fixed-point/witness-guard spirit of
    :func:`fuzzer_tool.core.xor_map_solver.verify_xor_model`: a recovered
    state is only trustworthy once it's checked against data that was not
    itself used to derive it (or, at minimum, reproduces everything it was
    derived from).
    """
    current = tuple(state)
    for word in observed_words:
        if output_word(current, spec) != word:
            return False
        current = step_state(current, spec)
    return True


# ── taus88-shaped façade ─────────────────────────────────────────────
#
# The original, parameter-tuple API, kept for callers that speak
# per-component (w, k, q, s) rather than specs.


@lru_cache(maxsize=None)
def spec_from_params(params: tuple[LFSRParams, ...] = TAUS88_PARAMS) -> LinearPRNG:
    """A combined-LFSR spec for per-component ``(w, k, q, s)`` params."""
    return combined_lfsr("combined_lfsr", tuple(params))


def taus88_step(
    state: tuple[int, int, int],
    params: tuple[LFSRParams, ...] = TAUS88_PARAMS,
) -> tuple[int, int, int]:
    """Advance a 3-component xor-combined LFSR state by one step."""
    a, b, c = step_state(state, spec_from_params(tuple(params)))
    return a, b, c


def taus88_output(state: tuple[int, int, int]) -> int:
    """The generator's output word for the current state (XOR-combine)."""
    a, b, c = state
    return a ^ b ^ c


def predict_next(
    state: tuple[int, int, int],
    n: int = 1,
    params: tuple[LFSRParams, ...] = TAUS88_PARAMS,
) -> list[int]:
    """Predict the *n* outputs after *state*'s own; see :func:`predict_words`."""
    return predict_words(state, n, spec_from_params(tuple(params)))


def recover_taus88_state(
    observed_words: Sequence[int],
    params: tuple[LFSRParams, ...] = TAUS88_PARAMS,
) -> tuple[int, int, int] | None:
    """Recover a 3-component combined-LFSR state; see :func:`recover_state`."""
    state = recover_state(observed_words, spec_from_params(tuple(params)))
    if state is None:
        return None
    a, b, c = state
    return a, b, c


def verify_recovery(
    state: tuple[int, int, int],
    observed_words: Sequence[int],
    params: tuple[LFSRParams, ...] = TAUS88_PARAMS,
) -> bool:
    """Replay a 3-component state; see :func:`verify_state`."""
    return verify_state(state, observed_words, spec_from_params(tuple(params)))
