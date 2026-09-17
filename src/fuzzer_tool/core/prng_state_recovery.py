"""Predictable-PRNG state recovery for ``xor_combine``-style LFSR generators.

Ports the methodology from Gegell's "Reversing Factorio's RNG"
(https://gegell.github.io/posts/factorio-rng, code at
https://github.com/Gegell/Factorio-RNG-Release) into this repo's existing
GF(2) toolkit, generalized beyond Factorio's specific ``taus88`` instance.

Factorio uses Boost's ``taus88``: three ``linear_feedback_shift_engine``
components XORed together, exactly the class this module targets::

    typedef xor_combine_engine<
        xor_combine_engine<
            linear_feedback_shift_engine<uint32_t, 32, 31, 13, 12>, 0,
            linear_feedback_shift_engine<uint32_t, 32, 29, 2, 4>, 0>, 0,
        linear_feedback_shift_engine<uint32_t, 32, 28, 3, 17>, 0> taus88;

Each component step is GF(2)-linear in its 32 state bits, and XOR-combining
components stays linear, so the whole generator's *state* (not just its
output) evolves as ``s(t+1) = T @ s(t)`` over ``GF(2)^n``. Gegell's original
approach builds that generation matrix with sympy symbols and solves for the
state with a manual RREF (see ``factorio_rng.py::build_generation_matrix``
and ``get_state`` in the release repo).

This module gets the same result without sympy, by reusing machinery already
in this repo:

- The symbolic step is done with plain Python ints as GF(2) bitmask vectors
  (one basis bit per unknown initial-state bit), the same representation
  :mod:`fuzzer_tool.core.gf2_common`'s bitmask-vector layer and
  :mod:`fuzzer_tool.core.xor_map_solver` already use.
- The actual linear-system solve reuses
  :class:`fuzzer_tool.core.xor_map_solver.IncrementalXorMapSolver` outright:
  "recover an unknown 96-bit initial state from observed output bits" is the
  same GF(2) elimination problem as "recover an unknown linear map from
  (input, output) pairs" — here each equation's known coefficient row plays
  the role of the solver's ``input_bits``, and the observed output bit (bit
  0 only; there is a single "output") plays the role of ``output_bits``. The
  solver's full-rank determinacy gate then does exactly the job Gegell did
  by hand: refuse a state unless the observations pin it uniquely.

Why this matters for fuzzing/vuln-hunting: many C/C++ targets seed a weak
combined-LFSR PRNG (taus88 and its relatives are common, e.g. via
``boost::random`` or hand-rolled xorshift/xor-combine variants) for session
tokens, nonces, or "random" identifiers -- CWE-338. Given a handful of
consecutive outputs (leaked, side-channeled, or observed across requests),
this module recovers the internal state and predicts all future outputs,
without needing to reverse the target's binary the way the original
Factorio writeup did (that reversing step told us *which* parameters to
plug in here -- see ``_TAUS88_PARAMS`` -- but the recovery technique itself
is generic over ``(w, k, q, s)``).
"""

from __future__ import annotations

from fuzzer_tool.core.xor_map_solver import IncrementalXorMapSolver

__all__ = [
    "LFSRParams",
    "TAUS88_PARAMS",
    "taus88_step",
    "taus88_output",
    "recover_taus88_state",
    "verify_recovery",
    "predict_next",
]

# (word_size, lfsr_bits, feedback_tap, step_count) per component, matching
# rngs.py::Taus88RNG / the Boost taus88 typedef.
LFSRParams = tuple[int, int, int, int]
TAUS88_PARAMS: tuple[LFSRParams, LFSRParams, LFSRParams] = (
    (32, 31, 13, 12),
    (32, 29, 2, 4),
    (32, 28, 3, 17),
)

_WORD_BITS = 32
_STATE_BITS = _WORD_BITS * 3  # 96: one taus88 component per 32-bit slot


# ── Concrete (integer) forward simulation ──────────────────────────────


def _lfsr_step_concrete(value: int, w: int, k: int, q: int, s: int) -> int:
    """One step of a single ``linear_feedback_shift_engine``, on a real int.

    Mirrors ``rngs.py::ExtractedRNG._single_LFSR`` / the Boost definition
    bit-for-bit.
    """
    word_mask = (1 << w) - 1
    b = (((value << q) ^ value) & word_mask) >> (k - s)
    mask = (word_mask << (w - k)) & word_mask
    return (((value & mask) << s) ^ b) & word_mask


def taus88_step(
    state: tuple[int, int, int],
    params: tuple[LFSRParams, LFSRParams, LFSRParams] = TAUS88_PARAMS,
) -> tuple[int, int, int]:
    """Advance a 3-component xor-combined LFSR state by one step."""
    a, b, c = state
    (w1, k1, q1, s1), (w2, k2, q2, s2), (w3, k3, q3, s3) = params
    return (
        _lfsr_step_concrete(a, w1, k1, q1, s1),
        _lfsr_step_concrete(b, w2, k2, q2, s2),
        _lfsr_step_concrete(c, w3, k3, q3, s3),
    )


def taus88_output(state: tuple[int, int, int]) -> int:
    """The generator's output word for the current state (XOR-combine)."""
    a, b, c = state
    return a ^ b ^ c


def predict_next(state: tuple[int, int, int], n: int = 1) -> list[int]:
    """Step *state* forward *n* times, returning the *n* predicted outputs.

    Note this steps *before* recording, so it returns the outputs *after*
    ``state``'s own output -- i.e. it does not include
    ``taus88_output(state)`` itself. A state returned by
    :func:`recover_taus88_state` already has ``taus88_output(state) ==
    observed_words[0]`` with zero extra steps; call ``taus88_step`` as many
    times as you have already-known words before calling this, or just use
    :func:`verify_recovery` to check against a whole known sequence at once.
    """
    outs: list[int] = []
    s = state
    for _ in range(n):
        s = taus88_step(s)
        outs.append(taus88_output(s))
    return outs


# ── Symbolic (GF(2) bitmask) forward simulation, for recovery ──────────


def _lfsr_step_symbolic(bits: list[int], w: int, k: int, q: int, s: int) -> list[int]:
    """Symbolic analogue of :func:`_lfsr_step_concrete`.

    ``bits[i]`` is not a 0/1 value but a bitmask over the *unknown* initial
    state's 96 basis bits: bit ``j`` of ``bits[i]`` is set iff output bit
    ``i`` depends (XORs in) unknown bit ``j``. Every operation below is the
    same shift/XOR/mask structure as the concrete step, just applied
    per-position to these coefficient vectors instead of to actual bits --
    valid because the LFSR step is GF(2)-linear.
    """
    shifted_left = [0] * q + bits[: w - q]
    tmp = [shifted_left[i] ^ bits[i] for i in range(w)]
    shift_b = k - s
    b = [tmp[i + shift_b] if i + shift_b < w else 0 for i in range(w)]
    masked = [bits[i] if i >= (w - k) else 0 for i in range(w)]
    shifted_masked = [0] * s + masked[: w - s]
    return [shifted_masked[i] ^ b[i] for i in range(w)]


def _initial_symbolic_state() -> list[list[int]]:
    """Basis state: component ``c``, bit ``i`` depends only on itself."""
    return [[1 << (c * _WORD_BITS + i) for i in range(_WORD_BITS)] for c in range(3)]


def _step_symbolic(
    state: list[list[int]],
    params: tuple[LFSRParams, LFSRParams, LFSRParams] = TAUS88_PARAMS,
) -> list[list[int]]:
    return [
        _lfsr_step_symbolic(state[c], w, k, q, s)
        for c, (w, k, q, s) in enumerate(params)
    ]


def _output_coefficients(state: list[list[int]]) -> list[int]:
    """32 coefficient rows: row ``i`` for output bit ``i`` (XOR of 3 comps)."""
    return [state[0][i] ^ state[1][i] ^ state[2][i] for i in range(_WORD_BITS)]


def _left_nullspace(rows: list[int]) -> list[int]:
    """Left null space of the map whose row ``i`` is ``rows[i]``.

    Each ``rows[i]`` is a coefficient mask (an output-equation row). Returns
    a list of masks over *row indices*: for every returned mask ``v``,
    ``XOR_{i where bit i of v is set} rows[i] == 0`` identically -- a linear
    relation that holds for *any* input to the map, not just the ones
    observed. Found with the standard "augment rows with a tracking
    identity column, eliminate, harvest every row that cancels to zero"
    trick: when a row reduces to the zero vector, its tracking column is by
    construction a combination of original rows that sums to zero.
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


def _structural_equations(
    params: tuple[LFSRParams, LFSRParams, LFSRParams] = TAUS88_PARAMS,
) -> list[int]:
    """Free, observation-independent equations from each component's rank gap.

    Each ``linear_feedback_shift_engine`` component maps its 32-bit input to
    a 32-bit output through a step of rank ``k < 32`` (``k`` = the LFSR's
    true bit width): the word size exceeds the register size, so the step
    is not injective and its *output* rows satisfy ``32 - k`` linear
    relations that hold for literally any input. Concretely, per Gegell's
    writeup: "the least significant bits are linearly dependent on the
    higher significant bits" of the *same* post-step state.

    Since any state we're ever asked to recover (``observed_words[0]``'s
    producing state) is itself the result of at least one prior step of the
    real generator, it already lies in that rank-``k`` image and satisfies
    these relations -- for free, without spending an extra observed output
    on them. Folding them in is exactly what turns the raw 96-unknown
    system (which saturates at rank 92, an under-determined particular
    solution that reproduces the 3 fitting outputs but predicts the wrong
    future ones -- verified empirically) into a fully determined one.

    Returns:
        A list of global (96-bit-indexed) coefficient rows, each paired
        implicitly with right-hand side 0.
    """
    equations: list[int] = []
    for c, (w, k, q, s) in enumerate(params):
        local_rows = _lfsr_step_symbolic([1 << i for i in range(w)], w, k, q, s)
        for local_null in _left_nullspace(local_rows):
            global_row = 0
            for j in range(w):
                if (local_null >> j) & 1:
                    global_row |= 1 << (c * _WORD_BITS + j)
            equations.append(global_row)
    return equations


# ── Recovery ─────────────────────────────────────────────────────────


def recover_taus88_state(
    observed_words: list[int],
    params: tuple[LFSRParams, LFSRParams, LFSRParams] = TAUS88_PARAMS,
) -> tuple[int, int, int] | None:
    """Recover the 96-bit combined-LFSR state from consecutive outputs.

    Args:
        observed_words: Consecutive 32-bit generator outputs,
            ``observed_words[t]`` = ``taus88_output`` of the state ``t``
            steps after the (unknown) state being recovered. Three values
            give exactly 96 equations for 96 unknowns -- the same count
            Gegell's writeup uses -- but more can be passed for
            over-determined confirmation.
        params: Per-component ``(w, k, q, s)``; defaults to Factorio's
            taus88. Pass different params to target another xor-combined
            LFSR generator (e.g. a different word/tap configuration found
            while reversing some other target).

    Returns:
        The recovered ``(a, b, c)`` state, or ``None`` if the observations
        don't pin it down uniquely yet (need more samples) or are
        inconsistent with an ``xor_combine`` LFSR of this shape at all.
    """
    if len(observed_words) < 3:
        raise ValueError("need >= 3 consecutive outputs to pin a 96-bit state")

    solver = IncrementalXorMapSolver(_STATE_BITS)
    for row in _structural_equations(params):
        solver.add_pair(row, 0)

    state = _initial_symbolic_state()
    for word in observed_words:
        for j, row in enumerate(_output_coefficients(state)):
            solver.add_pair(row, (word >> j) & 1)
        state = _step_symbolic(state, params)

    if not solver.is_determined:
        return None

    solution, sat = solver.solve()
    if not sat or solution is None:
        return None

    x = 0
    for i in solution[0]:
        x |= 1 << i
    word_mask = (1 << _WORD_BITS) - 1
    return x & word_mask, (x >> _WORD_BITS) & word_mask, (x >> (2 * _WORD_BITS)) & word_mask


def verify_recovery(
    state: tuple[int, int, int],
    observed_words: list[int],
    params: tuple[LFSRParams, LFSRParams, LFSRParams] = TAUS88_PARAMS,
) -> bool:
    """Replay from *state* and check every word in *observed_words* matches.

    Mirrors the fixed-point/witness-guard spirit of
    :func:`fuzzer_tool.core.xor_map_solver.verify_xor_model`: a recovered
    state is only trustworthy once it's checked against data that was not
    itself used to derive it (or, at minimum, reproduces everything it was
    derived from).
    """
    s = state
    for word in observed_words:
        if taus88_output(s) != word:
            return False
        s = taus88_step(s, params)
    return True
