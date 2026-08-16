"""Incremental GF(2) linear-map recovery from (input, output) pairs.

Recovers an unknown map ``f: {0,1}^n -> {0,1}^m`` where each output bit is
a **binary linear form** over the input bits:

    output_bit_j = XOR_{i where w_{j,i}=1} input_bit_i

where ``w_{j,i}`` is a binary coefficient (the mask bit).  This is the
natural GF(2) generalization of XOR-bitmask: the coefficient vector for
each output bit is the "mask", and the output is the dot product
``mask_j . input`` over ``F2``.

Uses one Z3 ``Solver`` per output bit so Z3 retains learned clauses
across :meth:`IncrementalXorMapSolver.add_pair` calls.

**No Z3 dependency at import time.**  ``import z3`` is deferred to the
first :meth:`IncrementalXorMapSolver.solve` call and gated by
:func:`_z3_available`, matching the convention in
:mod:`fuzzer_tool.core.smt_solver`.

Cost control
------------
- Per-bit-solver timeout of :data:`_SOLVER_TIMEOUT_MS` (200 ms).
- :func:`recover_xor_model` caps the pair set at :data:`_MAX_PAIRS` and
  the field width at :data:`_MAX_FIELD_BITS` (32).

Fixed-point witness guard
-------------------------
:func:`verify_xor_model` rejects candidate-pair combinations where the
candidate model predicts the same output for both inputs of the pair
(a fixed point *of this candidate*).  Such a pair does not discriminate
the candidate from a wrong model.
"""

from __future__ import annotations

import threading
from functools import reduce
from typing import NamedTuple

# z3 is optional; import is deferred to solve().
_z3_available = None  # tri-state: None=unchecked, True, False

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
    """Incremental Z3-backed solver for a GF(2) linear map.

    One ``Solver`` per output bit; Z3 retains learned clauses across
    :meth:`add_pair` calls so solve cost amortizes over the run.  The Z3
    ``Bool`` variables for each ``(output_bit, input_bit)`` coefficient
    are created once in :meth:`_build_solvers` and reused for every
    subsequent constraint.

    Constraint model
    ----------------
    For each output bit ``j`` and each pair ``(inp, out)``::

        XOR_{i where w_{j,i}=1} input_bit_i  ==  (out >> j) & 1

    where ``w_{j,i}`` is the Bool variable for coefficient ``(j, i)``
    and ``input_bit_i = (inp >> i) & 1``.  This is the ``F2`` dot
    product ``w_j . inp_bitvector``.

    Args:
        n_bits: Number of input bits (field width).
        timeout_ms: Per-bit solver timeout.  Defaults to
            :data:`_SOLVER_TIMEOUT_MS`.

    Raises:
        ValueError: If ``n_bits`` is not positive.
    """

    def __init__(self, n_bits: int, *, timeout_ms: int = _SOLVER_TIMEOUT_MS) -> None:
        if n_bits <= 0:
            raise ValueError(f"n_bits must be positive, got {n_bits}")
        self._n_bits = n_bits
        self._timeout_ms = timeout_ms
        # Per-output-bit solvers and their coefficient Bool vars.
        self._solvers: list[object] = []  # list[z3.Solver] once z3 is imported
        self._wvars: list[list[object]] = []  # wvars[j][i] = z3.Bool for coeff (j, i)
        self._ready = False
        self._pairs: list[tuple[int, int]] = []

    def add_pair(self, input_bits: int, output_bits: int) -> None:
        """Append one ``(input, output)`` pair's constraints.

        Both values are treated as unsigned bit-vectors of width ``n_bits``.
        The pair's constraints are appended to every per-bit solver so the
        next :meth:`solve` call considers all evidence seen so far.

        Args:
            input_bits: Observed input value.
            output_bits: Observed output value.
        """
        self._pairs.append((input_bits, output_bits))
        self._add_pair_constraints(input_bits, output_bits)

    def add_pairs(self, pairs: list[tuple[int, int]]) -> None:
        """Append multiple pairs.

        Args:
            pairs: ``(input_bits, output_bits)`` observations.
        """
        for inp, out in pairs:
            self.add_pair(inp, out)

    def solve(self) -> tuple[list[list[int]] | None, bool]:
        """Solve for the GF(2) map given all pairs added so far.

        Returns:
            ``(solution, is_sat)`` where ``solution[j]`` is the list of
            input-bit indices that XOR to produce output bit ``j``.
            ``solution`` is ``None`` when Z3 reports UNSAT, times out, or
            is otherwise unable to produce a definitive answer.
        """
        z3 = self._import_z3()
        if z3 is None:
            return None, False

        if not self._ready:
            self._build_solvers(z3)

        solutions: list[list[int]] = []
        for bit_idx, bit_solver in enumerate(self._solvers):
            # Use per-solver timeout instead of the global `z3.set_param`,
            # which would leak into every subsequent Z3 operation in this
            # process and flake later tests.
            bit_solver.set("timeout", self._timeout_ms)
            result = bit_solver.check()
            # Z3 returns 'unknown' on timeout; treat as inconclusive.
            if result != z3.sat:
                return None, False
            model = bit_solver.model()
            bits = [
                i
                for i in range(self._n_bits)
                if z3.is_true(model.evaluate(self._wvars[bit_idx][i]))
            ]
            solutions.append(bits)
        return solutions, True

    # ── Private ──────────────────────────────────────────────────────────

    def _import_z3(self):  # type: ignore[return]
        """Import z3 on first use; return None when unavailable."""
        global _z3_available
        if _z3_available is None:
            try:
                import z3  # noqa: F401

                _z3_available = True
            except ImportError:
                _z3_available = False
        if not _z3_available:
            return None
        import z3

        from fuzzer_tool.core.z3_lifecycle import guard_z3_shutdown

        guard_z3_shutdown()
        return z3

    def _build_solvers(self, z3) -> None:
        """Create per-bit solvers with fresh coefficient Bool vars."""
        self._solvers = []
        self._wvars = []
        for j in range(self._n_bits):
            s = z3.Solver()
            wvars = [z3.Bool(f"w_{j}_{i}") for i in range(self._n_bits)]
            self._solvers.append(s)
            self._wvars.append(wvars)
        # Mark ready before replaying pairs so _add_pair_constraints doesn't
        # recurse back into _build_solvers.
        self._ready = True
        for inp, out in self._pairs:
            self._add_pair_constraints(inp, out)

    def _add_pair_constraints(self, input_bits: int, output_bits: int) -> None:
        """Append F2 linear constraints for one pair to each per-bit solver.

        For output bit ``j`` the constraint is::

            XOR_{i where w_{j,i}=1} input_bit_i  ==  (output_bits >> j) & 1

        where ``input_bit_i = (input_bits >> i) & 1`` and ``w_{j,i}`` is
        the Z3 Bool created in :meth:`_build_solvers`.  Bits where
        ``input_bit_i == 0`` contribute nothing to the XOR and are
        dropped from the expression for that pair.
        """
        z3 = self._import_z3()
        if z3 is None:
            return

        if not self._ready:
            self._build_solvers(z3)

        for j in range(self._n_bits):
            out_bit = (output_bits >> j) & 1
            wvars = self._wvars[j]
            # Select only the input bits that are set for this pair; their
            # coefficients are the wvars.  XOR of selected wvars equals out_bit.
            active = [wvars[i] for i in range(self._n_bits) if (input_bits >> i) & 1]
            if not active:
                # No input bits set: XOR of empty set is 0.
                if out_bit:
                    self._solvers[j].add(z3.BoolVal(False))  # unsatisfiable
                # else: 0 == 0, no constraint needed.
            else:
                xor_expr = reduce(z3.Xor, active)
                if out_bit:
                    self._solvers[j].add(xor_expr)
                else:
                    self._solvers[j].add(z3.Not(xor_expr))


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
) -> XorBitmaskModel | None:
    """Attempt to recover a GF(2) linear checksum model.

    Tries 8/16/32-bit field widths in order.  For each width the pair set
    is capped at *max_pairs* and deduplicated; a candidate is accepted only
    when :func:`verify_xor_model` reproduces at least *min_matches*
    distinct pairs.

    In practice the ladder reaches 16 bits, not 32. Cost per output bit
    grows with the number of input bits, and at 32x32 the per-bit solve
    needs roughly 600 ms against :data:`_SOLVER_TIMEOUT_MS`'s 200 ms budget
    -- measured at ~22 s total to recover a known 32-bit model with the
    timeout lifted, against ~0.3 s to give up at the default. That is the
    right trade for a solver on the fuzzing hot path, so the 32-bit rung is
    left as a deliberate miss rather than widened; a target with a genuine
    32-bit XOR checksum needs an offline recovery pass, not a bigger
    in-loop budget.

    Args:
        pairs: ``(data, checksum)`` observations.
        min_matches: Minimum distinct pairs a candidate must reproduce.
        max_pairs: Maximum pairs fed to the solver.
        max_field_bits: Maximum field width to attempt.

    Returns:
        A verified :class:`XorBitmaskModel`, or ``None``.
    """
    if len(pairs) < 2:
        return None

    # Filter out empty-data pairs — they carry no input-bit information.
    usable = [p for p in pairs if p[0]]
    if len(usable) < 2:
        return None

    for width in (8, 16, 32):
        if width > max_field_bits:
            break
        # Deduplicate and cap.
        unique = list({(d, c) for d, c in usable})[:max_pairs]
        int_pairs = _extract_fixed_width(unique, width)
        masks, is_sat = _solve_xor_map(int_pairs, width)
        if not is_sat or masks is None:
            continue
        model = XorBitmaskModel(masks=tuple(tuple(m) for m in masks), out_bits=width)
        if verify_xor_model(model, usable, min_matches=min_matches):
            return model
    return None


def _solve_xor_map(
    pairs: list[tuple[int, int]], n_bits: int
) -> tuple[list[list[int]] | None, bool]:
    """Run the incremental solver over *pairs* and return ``(masks, is_sat)``.

    Creates a fresh :class:`IncrementalXorMapSolver` per call so cost is
    bounded to this recovery attempt only.
    """
    solver = IncrementalXorMapSolver(n_bits)
    solver.add_pairs(pairs)
    return solver.solve()


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
