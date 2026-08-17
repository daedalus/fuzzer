"""Incremental GF(2) linear-map recovery from (input, output) pairs.

Recovers an unknown map ``f: {0,1}^n -> {0,1}^m`` where each output bit is
a **binary linear form** over the input bits:

    output_bit_j = XOR_{i where w_{j,i}=1} input_bit_i

where ``w_{j,i}`` is a binary coefficient (the mask bit).  This is the
natural GF(2) generalization of XOR-bitmask: the coefficient vector for
each output bit is the "mask", and the output is the dot product
``mask_j . input`` over ``F2``.

Solved by elimination over ``F2``, not by SAT
--------------------------------------------
Every constraint this module builds has the form ``XOR_{i in S} w_i == c``
-- one *linear* equation over ``F2``.  Recovery is therefore a linear
system, and the coefficient matrix is the **same for every output bit**:
row ``k`` is the input bit-vector of pair ``k``, and only the right-hand
side changes with ``j``.  One incremental Gauss-Jordan elimination over
the augmented matrix ``[A | B]`` -- rows held as Python ints, RHS packed
so all ``m`` output bits ride along in a single word -- recovers the whole
map at once, in ``O(pairs * rank)`` word XORs.

This replaced one Z3 ``Solver`` per output bit.  Z3 was being asked to
search for something that is not a search problem, and paid the usual
price: at a 32-bit field the per-bit solve exceeded the 200 ms budget and
the whole 32-bit rung of the width ladder was unreachable -- documented as
a deliberate miss on cost.  It was never a cost problem, it was the wrong
algorithm.  Measured on the same box, same pair set (32x32, 64 pairs):
151 s for Z3 with the timeout lifted, 0.5 ms for elimination.

Elimination is also *complete* where the SAT path was not.  It has no
timeout and no ``unknown``: it returns a solution or a proof that none
exists, so a rejected model now means "the observations contradict a
linear map" rather than "Z3 ran out of time".

Determinacy gate
----------------
Speed exposed a second problem.  An underdetermined system (rank below the
field width) has many solutions, all of which reproduce every observed
pair by construction -- so verifying against the fitting pairs cannot
reject them.  :func:`recover_xor_model` requires the system to be of
**full rank**, i.e. the observations pin the map uniquely, before a
candidate is even offered to :func:`verify_xor_model`.  See
:meth:`IncrementalXorMapSolver.rank`.

Cost control
------------
- :func:`recover_xor_model` caps the pair set at :data:`_MAX_PAIRS` and
  the field width at :data:`_MAX_FIELD_BITS` (32).
- There is no solver timeout: elimination is polynomial and terminates.
  :data:`_SOLVER_TIMEOUT_MS` is retained only so that callers passing
  ``timeout_ms=`` keep working; it no longer gates anything.

Fixed-point witness guard
-------------------------
:func:`verify_xor_model` rejects candidate-pair combinations where the
candidate model predicts the same output for both inputs of the pair
(a fixed point *of this candidate*).  Such a pair does not discriminate
the candidate from a wrong model.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

# Retained for backward compatibility only: recovery no longer uses z3 and
# no longer has a timeout.  Kept so that callers doing
# ``IncrementalXorMapSolver(n, timeout_ms=...)`` or reading this constant
# do not break.
_SOLVER_TIMEOUT_MS: int = 200

# Bounds for the recovery entry points.
_MAX_FIELD_BITS: int = 32  # never attempt on fields wider than this
_MAX_PAIRS: int = 128  # cap pair set cost, mirroring CHECKSUM_PAIRS_MAX


# ── Public model ────────────────────────────────────────────────────────


class XorBitmaskModel(NamedTuple):
    """A recovered GF(2) linear checksum model.

    Each output bit ``j`` equals ``XOR_{i in masks[j]} input_bit_i``,
    where ``masks[j]`` is the list of input-bit indices with coefficient
    1 in the ``F2`` linear form for that output bit.

    Attributes:
        masks: ``masks[j]`` is the list of input-bit indices that XOR
            together to produce output bit ``j``.  Length equals
            ``out_bits``; each element is a non-empty sorted list.
        out_bits: Number of output bits (and therefore number of masks).
    """

    masks: tuple[tuple[int, ...], ...]
    out_bits: int

    @property
    def nbytes(self) -> int:
        """Width of the checksum field in bytes."""
        return self.out_bits // 8


# ── Module-level active-model plumbing ──────────────────────────────────

_active_xor_model: XorBitmaskModel | None = None
_xor_lock = threading.Lock()


def set_active_xor_model(model: XorBitmaskModel | None) -> None:
    """Set the active XOR-bitmask model (thread-safe)."""
    global _active_xor_model
    with _xor_lock:
        _active_xor_model = model


def get_active_xor_model() -> XorBitmaskModel | None:
    """Return the active XOR-bitmask model, or ``None``."""
    with _xor_lock:
        return _active_xor_model


def clear_active_xor_model() -> None:
    """Drop the active model (used by tests and state resets)."""
    set_active_xor_model(None)


# ── Solver ──────────────────────────────────────────────────────────────


class IncrementalXorMapSolver:
    """Incremental GF(2) elimination for an unknown linear map.

    Maintains the reduced row-echelon form of the linear system implied by
    the pairs added so far.  Each pair contributes **one** equation shared
    by every output bit, so a single elimination -- with the right-hand
    sides of all ``n_bits`` output bits packed into one integer -- solves
    the entire map.

    Constraint model
    ----------------
    For each output bit ``j`` and each pair ``(inp, out)``::

        XOR_{i where w_{j,i}=1} input_bit_i  ==  (out >> j) & 1

    where ``input_bit_i = (inp >> i) & 1``.  Read as a linear equation in
    the unknowns ``w_{j,*}``, the coefficient on ``w_{j,i}`` is
    ``input_bit_i`` -- so the equation's left-hand side is literally
    ``inp`` as a bit vector, and its right-hand side across all ``j`` is
    literally ``out``.  That is why one elimination serves every bit.

    Row representation
    ------------------
    ``self._rows`` maps a *pivot input-bit index* to ``(lhs, rhs)``:

    - ``lhs``: bitmask over input-bit indices (the equation's coefficients)
    - ``rhs``: bitmask over output-bit indices (the constant term for each
      output bit ``j``)

    The invariant is full Gauss-Jordan form: pivot bit ``p`` is set in row
    ``p`` and in no other row.  A single reduction pass over the rows
    therefore suffices, in any iteration order.

    Args:
        n_bits: Number of input bits (field width).
        timeout_ms: Accepted and stored for backward compatibility; it no
            longer has any effect.  Elimination is ``O(pairs * rank)`` and
            always terminates, so there is nothing to time out.

    Raises:
        ValueError: If ``n_bits`` is not positive.
    """

    def __init__(self, n_bits: int, *, timeout_ms: int = _SOLVER_TIMEOUT_MS) -> None:
        if n_bits <= 0:
            raise ValueError(f"n_bits must be positive, got {n_bits}")
        self._n_bits = n_bits
        self._timeout_ms = timeout_ms  # unused; see class docstring
        # pivot input-bit index -> (lhs mask over inputs, rhs mask over outputs)
        self._rows: dict[int, tuple[int, int]] = {}
        # Output bits whose system is provably inconsistent.  Tracked per bit
        # because a contradiction on one output bit says nothing about the
        # others: the rows are shared, the right-hand sides are not.
        self._unsat_bits: int = 0
        self._in_mask = (1 << n_bits) - 1
        self._pairs: list[tuple[int, int]] = []

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def rank(self) -> int:
        """Rank of the coefficient matrix over ``F2``.

        Equal to the number of pivots.  When this reaches ``n_bits`` the
        system is fully determined and :meth:`solve` returns the *unique*
        map consistent with the observations.  Below that, the returned map
        is one of ``2**(n_bits - rank)`` equally consistent solutions --
        every one of which reproduces all observed pairs, which is why
        callers must not treat agreement with the fitting pairs as evidence.
        """
        return len(self._rows)

    @property
    def is_determined(self) -> bool:
        """True when the observations pin the map uniquely (full rank)."""
        return len(self._rows) == self._n_bits

    def add_pair(self, input_bits: int, output_bits: int) -> None:
        """Append one ``(input, output)`` pair's equation.

        Both values are treated as unsigned bit-vectors of width
        ``n_bits``.  The equation is reduced against the existing rows and
        either becomes a new pivot row or -- if it reduces to ``0 == rhs``
        -- records a contradiction for each output bit set in ``rhs``.

        Args:
            input_bits: Observed input value.
            output_bits: Observed output value.
        """
        lhs = input_bits & self._in_mask
        rhs = output_bits & self._in_mask
        self._pairs.append((input_bits, output_bits))

        rows = self._rows
        # One pass is enough: Gauss-Jordan form means pivot p occurs only in
        # row p, so clearing it cannot reintroduce an already-cleared pivot.
        for p, (plhs, prhs) in rows.items():
            if (lhs >> p) & 1:
                lhs ^= plhs
                rhs ^= prhs

        if lhs == 0:
            # 0 == rhs.  Consistent exactly where rhs is 0; every set bit is
            # an output bit with no solution at all.
            self._unsat_bits |= rhs
            return

        pivot = (lhs & -lhs).bit_length() - 1  # lowest set bit
        # Restore Gauss-Jordan form: eliminate the new pivot from every
        # existing row.
        for q, (qlhs, qrhs) in rows.items():
            if (qlhs >> pivot) & 1:
                rows[q] = (qlhs ^ lhs, qrhs ^ rhs)
        rows[pivot] = (lhs, rhs)

    def add_pairs(self, pairs: list[tuple[int, int]]) -> None:
        """Append multiple pairs.

        Args:
            pairs: ``(input_bits, output_bits)`` observations.
        """
        for inp, out in pairs:
            self.add_pair(inp, out)

    def solve(self) -> tuple[list[list[int]] | None, bool]:
        """Solve for the GF(2) map given all pairs added so far.

        Free (non-pivot) variables are assigned 0, which yields the
        sparsest particular solution of the system.  When
        :attr:`is_determined` is true there are no free variables and the
        solution is unique.

        Returns:
            ``(solution, is_sat)`` where ``solution[j]`` is the sorted list
            of input-bit indices that XOR to produce output bit ``j``.
            ``solution`` is ``None`` when the system is inconsistent for at
            least one output bit.  Unlike the previous SAT-backed
            implementation there is no third, inconclusive outcome: a
            ``False`` here is a proof, not a timeout.
        """
        if self._unsat_bits:
            return None, False
        rows = self._rows
        # With free variables at 0, the value of pivot variable p for output
        # bit j is just bit j of row p's right-hand side.
        return [
            sorted(p for p, (_lhs, rhs) in rows.items() if (rhs >> j) & 1)
            for j in range(self._n_bits)
        ], True


# ── Checksum evaluation ─────────────────────────────────────────────────


def compute_xor_checksum(data: bytes, xmodel: XorBitmaskModel) -> int:
    """Compute the XOR-bitmask checksum of *data* under *xmodel*.

    Bit indices are LSB-first within each byte: input bit ``i`` is
    ``(data[i // 8] >> (i % 8)) & 1``.  Output bit ``j`` is the XOR of
    the input bits listed in ``xmodel.masks[j]``, placed at position ``j``
    in the returned integer.

    Args:
        data: Input bytes.
        xmodel: Recovered XOR-bitmask model.

    Returns:
        Checksum integer in the range ``[0, 2**xmodel.out_bits)``.
    """
    result = 0
    n_bytes = len(data)
    for j, bits in enumerate(xmodel.masks):
        bit = 0
        for i in bits:
            byte_idx = i // 8
            bit_idx = i % 8  # LSB-first within each byte
            if byte_idx < n_bytes:
                bit ^= (data[byte_idx] >> bit_idx) & 1
        result |= bit << j
    return result


# ── Recovery ────────────────────────────────────────────────────────────


def _extract_fixed_width(pairs: list[tuple[bytes, int]], width: int) -> list[tuple[int, int]]:
    """Extract fixed-width ``(input_int, output_int)`` pairs.

    The *output* (checksum value) is always used as-is, zero-padded to
    *width* bits.  The *input* is the first ``ceil(width/8)`` bytes of
    the data field, zero-padded when the data is shorter than the window.

    The window is packed **little-endian** so that bit ``i`` of
    ``input_int`` -- which is how :class:`IncrementalXorMapSolver` indexes
    it, via ``(input_bits >> i) & 1`` -- is the same bit that
    :func:`compute_xor_checksum` reads as ``data[i // 8] >> (i % 8)``.

    This packed big-endian originally, which byte-reverses the window
    relative to the evaluator. The two conventions coincide for a 1-byte
    window and diverge for every wider one, so the solver would fit a
    perfectly good mask set under its own indexing and
    :func:`verify_xor_model` would then evaluate it under the other and
    reject it. Every 16- and 32-bit candidate failed that way, making
    ``recover_xor_model``'s ``(8, 16, 32)`` width ladder effectively
    ``(8,)`` regardless of the input.

    Args:
        pairs: ``(data, checksum)`` observations.
        width: Field width in bits (must be a multiple of 8).

    Returns:
        List of ``(input_int, output_int)`` pairs, each fitting in
        *width* bits.
    """
    assert width % 8 == 0
    n_bytes = width // 8
    result: list[tuple[int, int]] = []
    for data, checksum in pairs:
        window = data[:n_bytes]
        input_int = int.from_bytes(window.ljust(n_bytes, b"\x00"), "little")
        output_int = checksum & ((1 << width) - 1)
        result.append((input_int, output_int))
    return result


def recover_xor_model(
    pairs: list[tuple[bytes, int]],
    min_matches: int = 4,
    max_pairs: int = _MAX_PAIRS,
    max_field_bits: int = _MAX_FIELD_BITS,
    require_determined: bool = True,
) -> XorBitmaskModel | None:
    """Attempt to recover a GF(2) linear checksum model.

    Tries 8/16/32-bit field widths in order.  For each width the pair set
    is capped at *max_pairs* and deduplicated; a candidate is accepted only
    when the system is fully determined (see *require_determined*) and
    :func:`verify_xor_model` reproduces at least *min_matches* distinct
    pairs.

    The full 8/16/32 ladder is live.  It previously stopped at 16: the
    per-bit SAT solve at a 32-bit field ran past the 200 ms budget, and the
    32-bit rung was recorded as a deliberate miss on cost.  Elimination over
    ``F2`` removed the cost -- a 32x32 recovery over 64 pairs is sub-
    millisecond -- so the rung is now reached on every call rather than
    abandoned to an offline pass.

    Why *require_determined* exists
    -------------------------------
    An underdetermined system reproduces **every** pair it was fitted on, by
    construction, for any choice of the free variables.  Verification against
    those same pairs therefore cannot reject it, and the wider the field the
    more pairs are needed to pin it down -- so simply reaching the 32-bit
    rung would have accepted spurious 32-bit models on evidence that only
    ever supported a narrower one.  Requiring full rank makes acceptance mean
    "the observations admit exactly one linear map", which is the claim the
    caller actually needs.  Pass ``False`` for the old behaviour, e.g. in an
    offline pass where a guessed model is better than none.

    Args:
        pairs: ``(data, checksum)`` observations.
        min_matches: Minimum distinct pairs a candidate must reproduce.
        max_pairs: Maximum pairs fed to the solver.
        max_field_bits: Maximum field width to attempt.
        require_determined: Reject candidates whose system is not full rank.

    Returns:
        A verified :class:`XorBitmaskModel`, or ``None``.
    """
    if len(pairs) < 2:
        return None

    # Filter out empty-data pairs — they carry no input-bit information.
    usable = [p for p in pairs if p[0]]
    if len(usable) < 2:
        return None

    # Deduplicate and cap once: the pair set does not depend on the width.
    unique = list({(d, c) for d, c in usable})[:max_pairs]

    for width in (8, 16, 32):
        if width > max_field_bits:
            break
        int_pairs = _extract_fixed_width(unique, width)
        masks, is_sat, determined = _solve_xor_map(int_pairs, width)
        if not is_sat or masks is None:
            # Inconsistent at this width. Not a timeout — a proof that no
            # linear map over this window explains the observations. A wider
            # window can still succeed (it sees input bits this one missed),
            # so keep climbing.
            continue
        if require_determined and not determined:
            continue
        model = XorBitmaskModel(masks=tuple(tuple(m) for m in masks), out_bits=width)
        if verify_xor_model(model, usable, min_matches=min_matches):
            return model
    return None


def _solve_xor_map(
    pairs: list[tuple[int, int]], n_bits: int
) -> tuple[list[list[int]] | None, bool, bool]:
    """Run the incremental solver over *pairs*.

    Creates a fresh :class:`IncrementalXorMapSolver` per call so cost is
    bounded to this recovery attempt only.

    Returns:
        ``(masks, is_sat, is_determined)``.  ``is_determined`` is true when
        the coefficient matrix reached full rank, i.e. ``masks`` is the only
        map consistent with *pairs* rather than one of many.
    """
    solver = IncrementalXorMapSolver(n_bits)
    solver.add_pairs(pairs)
    masks, is_sat = solver.solve()
    return masks, is_sat, solver.is_determined


def verify_xor_model(
    model: XorBitmaskModel,
    pairs: list[tuple[bytes, int]],
    min_matches: int = 4,
) -> bool:
    """True when *model* reproduces the checksums of enough distinct pairs.

    Two guards beyond the raw count:

    1. Matched pairs must carry at least two *distinct* checksum values, so
       a degenerate all-zero-mask model (output always 0) cannot pass by
       matching a run of zero-checksum observations.
    2. The verification rejects candidate-pair combinations where the
       candidate model predicts the same output for both inputs of the pair
       (a fixed point *of this candidate*).  Such a pair does not
       discriminate the candidate from a wrong model.
    """
    required = max(2, min(min_matches, len(pairs)))
    matched: set[int] = set()
    matches = 0
    for data, checksum in pairs:
        predicted = compute_xor_checksum(data, model)
        if predicted == checksum:
            matches += 1
            matched.add(checksum)
            if matches >= required and len(matched) >= 2:
                return True
    return False


# ── Serialization ───────────────────────────────────────────────────────


def xor_model_to_dict(model: XorBitmaskModel | None) -> dict[str, object] | None:
    """Serialize *model* for ``state.json`` persistence."""
    if model is None:
        return None
    return {
        "kind": "xor_bitmask",
        "masks": model.masks,
        "out_bits": model.out_bits,
    }


def xor_model_from_dict(data: dict[str, object] | None) -> XorBitmaskModel | None:
    """Rebuild a model from :func:`xor_model_to_dict` output.

    Returns ``None`` for missing or malformed input — a corrupt state file
    must not take down the fuzz run.
    """
    if not data or data.get("kind") != "xor_bitmask":
        return None
    try:
        raw_masks = data["masks"]
        if not isinstance(raw_masks, list | tuple):
            return None
        masks = tuple(tuple(m) for m in raw_masks)
        out_bits = int(data["out_bits"])
        return XorBitmaskModel(masks=masks, out_bits=out_bits)
    except (TypeError, KeyError, ValueError):
        return None
