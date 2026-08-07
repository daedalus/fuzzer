"""Path-condition negation: solve for inputs that flip a recorded branch.

This is the piece that makes the z3 dependency earn its keep.
``ConcolicTrace.solve`` pins every input byte to a concrete literal — the
overridden window to the target operand, everything else to its original
value — so the constraint system has exactly one model and produces the same
bytes as ``data.replace(op_a, op_b)``. No search happens.

Here the operand window is left **symbolic** and the *negated branch
predicate* is asserted over it. The cmplog shim already records what each
comparison did (``result``: -1 for a<b, 0 for a==b, 1 for a>b), so for a
comparison that went one way we can ask z3 for an input that makes it go the
other way:

    observed a <  b   ->   solve for  a >= b
    observed a >  b   ->   solve for  a <= b
    observed a == b   ->   solve for  a !=  b

That is the difference between replaying a comparison already observed and
inverting one not yet satisfied — the standard concolic loop (QSYM, SymCC,
Driller) and the reason path negation reaches branches mutation cannot.

Scope and honest limits:

- Only comparisons whose operand appears verbatim in the input can be
  mapped back to input offsets. This is the same input-to-state assumption
  redqueen makes; operands that are computed rather than copied are skipped.
- A single predicate is negated per query. Full path-prefix conjunction
  (asserting all *preceding* conditions hold while the last flips) needs an
  ordered trace, which the shim's dedup-by-pair log does not preserve.
  Negating one condition still reaches the sibling branch whenever the
  predicate is input-determined, which is the common case.
- Cryptographic relations (hashes, HMAC) are not invertible and are not
  attempted.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 200
"""Per-query solver budget. Higher than the cmplog fast paths use, since a
genuine search costs more than a byte substitution, but still small enough
that a stuck query cannot stall the fuzzing loop."""

MAX_INPUT_BYTES = 1 << 16
"""Inputs above this are skipped: the model allocates a BitVec per byte."""

MAX_OVERLAP = 8
"""Cap on overlapping branches folded into one conjunctive query.

Solve cost grows with the constraint count, and a hot comparison site can
record many branches over the same bytes. Beyond this many the extra
constraints rarely change the answer but reliably cost milliseconds, so the
widest few are kept and the rest dropped."""

MAX_WIDTH = 8
"""Widest comparison handled. The shim logs up to 64 bytes for memcmp-style
calls, but those are equality-only and better served by the existing token
substitution than by an integer relation."""


def _z3():
    try:
        import z3
    except ImportError:
        return None
    return z3


class BranchRecord:
    """One observed comparison, and which way it went."""

    __slots__ = ("op_a", "op_b", "result", "width", "pc")

    def __init__(self, op_a: bytes, op_b: bytes, result: int, width: int, pc: int | None):
        self.op_a = op_a
        self.op_b = op_b
        self.result = result
        self.width = width
        self.pc = pc

    @property
    def key(self) -> tuple[int | None, bytes, bytes, int]:
        """Identity for frontier bookkeeping: site plus observed direction."""
        return (self.pc, self.op_a, self.op_b, self.result)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"BranchRecord(pc={self.pc}, a={self.op_a.hex()}, "
            f"b={self.op_b.hex()}, result={self.result}, width={self.width})"
        )


class PathConstraintSolver:
    """Solves for inputs that take the opposite side of a recorded branch.

    Tracks which (site, direction) pairs have already been attempted so the
    frontier shrinks as branches are flipped, rather than re-solving the same
    condition every iteration.
    """

    def __init__(self, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self.timeout_ms = timeout_ms
        self._attempted: set[tuple] = set()
        self.queries = 0
        self.solved = 0
        self.unsat = 0
        self.skipped_unmapped = 0
        self.direct_solves = 0
        self.z3_solves = 0

    # ── frontier ───────────────────────────────────────────────────────

    def frontier(self, records: list[BranchRecord], input_data: bytes) -> list[BranchRecord]:
        """Records not yet attempted whose operand is locatable in the input.

        Ordered widest-first: a wider comparison constrains more bytes, so
        flipping it tends to move the input further than flipping a single
        byte check.
        """
        out = []
        for rec in records:
            if rec.key in self._attempted:
                continue
            if not (0 < rec.width <= MAX_WIDTH):
                continue
            if self._locate(rec, input_data) is None:
                continue
            out.append(rec)
        out.sort(key=lambda r: r.width, reverse=True)
        return out

    def _locate(self, rec: BranchRecord, data: bytes) -> tuple[int, int] | None:
        """Find the operand window in *data*.

        Returns ``(offset, other_value)`` where *other_value* is the operand
        that is **not** input-derived — the constant side of the comparison.
        Returns None when neither operand appears in the input, which means
        the comparison is on computed data this approach cannot steer.
        """
        w = rec.width
        a, b = rec.op_a[:w], rec.op_b[:w]
        if len(a) < w or len(b) < w:
            return None
        idx = data.find(a)
        if idx != -1:
            return idx, int.from_bytes(b, "little")
        idx = data.find(b)
        if idx != -1:
            # Operands are swapped relative to the recorded comparison, so
            # the observed result must be read in the opposite sense too.
            return idx, int.from_bytes(a, "little")
        return None

    def _effective_result(self, rec: BranchRecord, data: bytes) -> int:
        """Observed result oriented so it describes (input_side, other)."""
        w = rec.width
        if data.find(rec.op_a[:w]) != -1:
            return rec.result
        return -rec.result  # matched op_b: comparison reads the other way

    # ── solving ────────────────────────────────────────────────────────

    @staticmethod
    def _direct_solve(observed: int, original: int, other: int, width: int) -> int | None:
        """Closed-form solution for a single unsigned predicate.

        A lone comparison against a constant needs no search: the extremal
        satisfying value can be written down. Calling z3 for this costs
        ~3.4ms per query — several times the whole per-execution budget at
        800 eps — purely in Python API overhead, for a constraint with an
        obvious answer. z3 is reserved for the conjunctive case below, where
        overlapping byte windows make the system genuinely coupled.

        Returns the new value, or None when the predicate is unsatisfiable
        (or satisfiable only by the value already present).
        """
        limit = (1 << (width * 8)) - 1
        if observed < 0:  # was a < b -> want a >= b
            candidate = other
            if candidate == original:
                candidate = other + 1
            return candidate if candidate <= limit else None
        if observed > 0:  # was a > b -> want a <= b
            candidate = other
            if candidate == original:
                candidate = other - 1
            return candidate if candidate >= 0 else None
        # was a == b -> want a != b
        candidate = (other + 1) & limit
        if candidate == original:
            candidate = (other + 2) & limit
        return candidate

    def _overlapping(
        self, rec: BranchRecord, others: list[BranchRecord], data: bytes
    ) -> list[tuple[BranchRecord, int, int]]:
        """Other mapped branches whose byte window intersects *rec*'s.

        Only these can be invalidated by the mutation, so only these need to
        enter the constraint system.
        """
        located = self._locate(rec, data)
        if located is None:
            return []
        start, _ = located
        end = start + rec.width
        out = []
        for other in others:
            if other is rec or other.key == rec.key:
                continue
            other_located = self._locate(other, data)
            if other_located is None:
                continue
            o_start, o_value = other_located
            if o_start < end and start < o_start + other.width:
                out.append((other, o_start, o_value))
        # Widest first, then bounded: the widest windows constrain the most
        # bytes and so are the ones most likely to be broken by the mutation.
        out.sort(key=lambda t: t[0].width, reverse=True)
        return out[:MAX_OVERLAP]

    def negate(
        self,
        rec: BranchRecord,
        input_data: bytes,
        others: list[BranchRecord] | None = None,
    ) -> bytes | None:
        """Solve for an input that flips *rec*, or None.

        The operand window is symbolic and everything else stays at its
        original value, so the result is a minimal mutation that satisfies a
        genuinely searched constraint.

        When *others* is given, branches whose byte windows overlap *rec*'s
        are asserted to keep their observed relation, so flipping one gate
        does not silently break another that shares bytes with it. That
        coupling is what actually requires a solver.
        """
        if not input_data or len(input_data) > MAX_INPUT_BYTES:
            return None

        located = self._locate(rec, input_data)
        if located is None:
            self.skipped_unmapped += 1
            return None
        offset, other = located
        width = rec.width
        if offset + width > len(input_data):
            return None

        self._attempted.add(rec.key)
        self.queries += 1

        observed = self._effective_result(rec, input_data)
        original = int.from_bytes(input_data[offset : offset + width], "little")
        overlaps = self._overlapping(rec, others, input_data) if others else []

        if not overlaps:
            value = self._direct_solve(observed, original, other, width)
            if value is None:
                self.unsat += 1
                return None
            out = bytearray(input_data)
            out[offset : offset + width] = value.to_bytes(width, "little")
            self.solved += 1
            self.direct_solves += 1
            return bytes(out)

        return self._solve_z3(input_data, offset, width, observed, original, other, overlaps)

    def _solve_z3(
        self,
        input_data: bytes,
        offset: int,
        width: int,
        observed: int,
        original: int,
        other: int,
        overlaps: list[tuple[BranchRecord, int, int]],
    ) -> bytes | None:
        """Conjunctive solve: flip one predicate, preserve the overlapping ones."""
        z3 = _z3()
        if z3 is None:
            return None

        # Per-solver timeout. z3.set_param("timeout", ...) sets a *global*
        # parameter that leaks into every other solver in the process.
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)

        # One symbolic byte per position touched by any participating window.
        lo = min([offset] + [s for _, s, _ in overlaps])
        hi = max([offset + width] + [s + r.width for r, s, _ in overlaps])
        hi = min(hi, len(input_data))
        sym = {i: z3.BitVec(f"in{i}", 8) for i in range(lo, hi)}

        def window(start: int, size: int):
            parts = [sym[i] for i in range(start, start + size) if i in sym]
            if len(parts) != size:
                return None
            return z3.Concat(*reversed(parts)) if size > 1 else parts[0]

        target = window(offset, width)
        if target is None:
            return None
        bits = width * 8
        other_bv = z3.BitVecVal(other, bits)

        if observed < 0:
            solver.add(z3.UGE(target, other_bv))
        elif observed > 0:
            solver.add(z3.ULE(target, other_bv))
        else:
            solver.add(target != other_bv)
        solver.add(target != z3.BitVecVal(original, bits))

        # Preserve every overlapping branch's observed relation.
        for rec_o, start_o, value_o in overlaps:
            expr = window(start_o, rec_o.width)
            if expr is None:
                continue
            const = z3.BitVecVal(value_o, rec_o.width * 8)
            observed_o = self._effective_result(rec_o, input_data)
            if observed_o < 0:
                solver.add(z3.ULT(expr, const))
            elif observed_o > 0:
                solver.add(z3.UGT(expr, const))
            else:
                solver.add(expr == const)

        try:
            status = solver.check()
        except z3.Z3Exception as exc:  # pragma: no cover - solver internals
            log.debug("path negation solver error: %s", exc)
            return None

        if status != z3.sat:
            self.unsat += 1
            return None

        model = solver.model()
        out = bytearray(input_data)
        for i, var in sym.items():
            out[i] = model.eval(var, model_completion=True).as_long() & 0xFF
        self.solved += 1
        self.z3_solves += 1
        return bytes(out)

    def solve_first(self, records: list[BranchRecord], input_data: bytes) -> bytes | None:
        """Negate the first branch in the frontier that yields a solution.

        Passes the full record list so overlapping branches are preserved
        rather than clobbered by the mutation.
        """
        for rec in self.frontier(records, input_data):
            result = self.negate(rec, input_data, others=records)
            if result is not None and result != input_data:
                return result
        return None

    # ── bookkeeping ────────────────────────────────────────────────────

    def reset_frontier(self) -> None:
        """Forget which branches were attempted (e.g. on a new seed)."""
        self._attempted.clear()

    def stats(self) -> dict:
        return {
            "queries": self.queries,
            "solved": self.solved,
            "unsat": self.unsat,
            "skipped_unmapped": self.skipped_unmapped,
            "direct_solves": self.direct_solves,
            "z3_solves": self.z3_solves,
            "attempted_branches": len(self._attempted),
            "solve_rate": (self.solved / self.queries) if self.queries else 0.0,
        }


def records_from_collector(collector) -> list[BranchRecord]:
    """Build BranchRecords from a CmplogCollector's captured outcomes."""
    if collector is None or not hasattr(collector, "branch_records"):
        return []
    return [
        BranchRecord(a, b, result, width, pc)
        for a, b, result, width, pc in collector.branch_records()
    ]
