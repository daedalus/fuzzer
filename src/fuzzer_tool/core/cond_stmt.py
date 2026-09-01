"""First-class comparison-statement model.

Ported from Angora's ``cond_stmt`` module. Each comparison observed by
cmplog becomes a ``CondStmt`` object carrying:

- The raw operand pair plus comparison metadata (result, width, PC).
- Taint-tagged input byte ranges (``offsets`` / ``offsets_opt``).
- Constant operand values (``variables``).
- Search state (``CondState``) so the solver avoids re-solving satisfied
  branches.
- Quality flags (``is_desirable``, ``is_consistent``) used by schedulers.

The model is **backward compatible**: ``CondStmt.from_cmplog_pair`` builds a
minimal object from existing cmplog data without any new instrumentation.
When Angora-style structured track files are available,
``CondStmt.from_track_record`` consumes them directly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search state
# ---------------------------------------------------------------------------


class CondState(Enum):
    """Lifecycle of a comparison branch in the solver frontier."""

    ONE_BYTE = "one_byte"
    """Width-1 comparison: trivially solved by byte substitution."""
    UNSOLVED = "unsolved"
    """Observed but not yet attempted."""
    SOLVED = "solved"
    """Solver found a satisfying input mutation."""
    UNSOLVABLE = "unsolvable"
    """Constraint is unsatisfiable or too costly."""
    TIMEOUT = "timeout"
    """Solver query exceeded its time budget."""


# ---------------------------------------------------------------------------
# Base record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CondStmtBase:
    """Immutable identity for one comparison branch.

    The ``key`` tuple is used for deduplication and frontier tracking.
    """

    cmpid: int
    """Monotonically increasing comparison ID."""
    op_a: bytes
    """Left operand observed by the shim."""
    op_b: bytes
    """Right operand observed by the shim."""
    width: int
    """Operand width in bytes (1, 2, 4, or 8)."""
    result: int
    """Observed outcome: -1 (a<b), 0 (a==b), 1 (a>b)."""
    pc: int | None
    """Program counter of the comparison site, when available."""

    @property
    def key(self) -> tuple[int, bytes, bytes, int]:
        return (self.cmpid, self.op_a, self.op_b, self.width)

    def __str__(self) -> str:
        pc = f" pc={self.pc:#x}" if self.pc is not None else ""
        return (
            f"CondStmtBase(id={self.cmpid}, a={self.op_a.hex()}, "
            f"b={self.op_b.hex()}, w={self.width}, r={self.result}{pc})"
        )


# ---------------------------------------------------------------------------
# Rich comparison statement
# ---------------------------------------------------------------------------


@dataclass
class CondStmt:
    """Mutable comparison statement with taint and solver metadata."""

    base: CondStmtBase
    offsets: tuple[int, ...] = field(default_factory=tuple)
    """Input byte offsets taint-tagged to this comparison."""
    offsets_opt: tuple[int, ...] = field(default_factory=tuple)
    """Optional/uncertain taint offsets."""
    variables: tuple[bytes, ...] = field(default_factory=tuple)
    """Constant operand values derived from the comparison."""
    speed: int = 0
    """Observation frequency across runs."""
    is_desirable: bool = True
    """Branch quality flag: True when flipping this branch is likely useful."""
    is_consistent: bool = True
    """True when the branch behaves consistently across inputs."""
    state: CondState = CondState.UNSOLVED
    """Current solver frontier state."""
    linear: bool = False
    """True when the branch is linear in the input bytes."""
    fuzz_times: int = 0
    """How many times the solver has attempted this branch."""
    num_minimal_optima: int = 0
    """Local-minima count observed during gradient descent."""

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_cmplog_pair(
        cls,
        cmpid: int,
        op_a: bytes,
        op_b: bytes,
        width: int,
        result: int = 0,
        pc: int | None = None,
    ) -> CondStmt:
        """Build a minimal CondStmt from raw cmplog data.

        No DFSan/taint tracking is required — offsets and variables are left
        empty and can be filled later by ``update_from_input`` or a track
        parser.
        """
        base = CondStmtBase(cmpid=cmpid, op_a=op_a, op_b=op_b, width=width, result=result, pc=pc)
        return cls(base=base)

    @classmethod
    def from_track_record(cls, record: dict) -> CondStmt | None:
        """Build a CondStmt from an Angora-style structured track record.

        The record dict is produced by ``TrackParser`` and carries the same
        fields as Angora's ``CondStmt`` JSON/text representation. Returns
        ``None`` when the record is malformed.
        """
        try:
            base = CondStmtBase(
                cmpid=int(record["cmpid"]),
                op_a=bytes(record["arg1"]),
                op_b=bytes(record["arg2"]),
                width=int(record["size"]),
                result=int(record.get("condition", 0)),
                pc=int(record["pc"]) if record.get("pc") is not None else None,
            )
        except (KeyError, ValueError, TypeError):
            return None
        return cls(
            base=base,
            offsets=tuple(record.get("offsets", [])),
            offsets_opt=tuple(record.get("offsets_opt", [])),
            variables=tuple(bytes(v) for v in record.get("variables", [])),
            speed=int(record.get("speed", 0)),
            is_desirable=bool(record.get("is_desirable", True)),
            is_consistent=bool(record.get("is_consistent", True)),
            state=CondState(record.get("state", "unsolved")),
            linear=bool(record.get("linear", False)),
            fuzz_times=int(record.get("fuzz_times", 0)),
            num_minimal_optima=int(record.get("num_minimal_optima", 0)),
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update_from_input(self, data: bytes) -> None:
        """Populate taint offsets by scanning the input for operand matches.

        For each byte position where the input overlaps with either operand,
        record the offset. This is a coarse proxy for DFSan taint labels when
        no structured track file is available.
        """
        offs: set[int] = set()
        for op in (self.base.op_a, self.base.op_b):
            if not op:
                continue
            start = 0
            while True:
                idx = data.find(op, start)
                if idx == -1:
                    break
                for i in range(len(op)):
                    offs.add(idx + i)
                start = idx + 1
        self.offsets = tuple(sorted(offs))

    def mark_solved(self) -> None:
        self.state = CondState.SOLVED
        self.fuzz_times += 1

    def mark_unsolvable(self) -> None:
        self.state = CondState.UNSOLVABLE
        self.fuzz_times += 1

    def mark_timeout(self) -> None:
        self.state = CondState.TIMEOUT
        self.fuzz_times += 1

    def bump_minima(self) -> None:
        self.num_minimal_optima += 1

    @property
    def key(self) -> tuple[int, bytes, bytes, int]:
        return self.base.key


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def filter_cond_list(
    conds: Iterable[CondStmt],
    *,
    drop_unsolvable: bool = True,
    drop_timeout: bool = True,
    drop_one_byte: bool = False,
    max_speed: int | None = None,
) -> list[CondStmt]:
    """Filter a comparison list to the solver-relevant subset.

    - Deduplicates by ``CondStmt.key`` (first occurrence wins).
    - Optionally drops ``UNSOLVABLE`` and ``TIMEOUT`` branches.
    - Optionally drops ``ONE_BYTE`` branches (they are trivially solvable
      by byte substitution and do not need gradient descent).
    - Optionally caps by ``speed`` to focus on hot branches.
    """
    seen: set[tuple] = set()
    out: list[CondStmt] = []
    for c in conds:
        k = c.key
        if k in seen:
            continue
        seen.add(k)
        if drop_unsolvable and c.state is CondState.UNSOLVABLE:
            continue
        if drop_timeout and c.state is CondState.TIMEOUT:
            continue
        if drop_one_byte and c.state is CondState.ONE_BYTE:
            continue
        if max_speed is not None and c.speed < max_speed:
            continue
        out.append(c)
    return out


def conds_from_cmplog_pairs(
    pairs: Sequence[tuple[bytes, bytes]],
    base_cmpid: int = 0,
    pair_meta: dict[tuple[bytes, bytes], tuple[int, int]] | None = None,
    pair_pc: dict[tuple[bytes, bytes], int] | None = None,
) -> list[CondStmt]:
    """Build CondStmt objects from a flat cmplog pair list.

    This is the backward-compatible bridge: existing cmplog infrastructure
    produces ``(op_a, op_b)`` tuples; this function wraps each in a
    ``CondStmt`` with optional result/width/PC metadata.

    Args:
        pairs: Sequence of ``(operand_a, operand_b)`` tuples.
        base_cmpid: Starting comparison ID.
        pair_meta: Optional mapping ``pair -> (result, width)``.
        pair_pc: Optional mapping ``pair -> pc``.

    Returns:
        List of ``CondStmt`` objects, one per unique pair.
    """
    pair_meta = pair_meta or {}
    pair_pc = pair_pc or {}
    out: list[CondStmt] = []
    seen: set[tuple[bytes, bytes]] = set()
    cmpid = base_cmpid
    for op_a, op_b in pairs:
        pair = (op_a, op_b)
        if pair in seen:
            continue
        seen.add(pair)
        result, width = pair_meta.get(pair, (0, max(len(op_a), len(op_b))))
        pc = pair_pc.get(pair)
        c = CondStmt.from_cmplog_pair(cmpid, op_a, op_b, width, result=result, pc=pc)
        out.append(c)
        cmpid += 1
    return out


# ---------------------------------------------------------------------------
# Minimax comparison-wall solver (Phase 3)
# ---------------------------------------------------------------------------


def _check_comparison_satisfied(cond: CondStmt, data: bytes) -> bool:
    """Check if a comparison is satisfied by the given input.

    Simplified heuristic: looks for the expected operand at taint offsets.
    A full implementation would use the SMT solver.
    """
    if not cond.base.op_a or not cond.base.op_b:
        return False
    for offset in cond.offsets:
        if offset + cond.base.width <= len(data):
            chunk = data[offset : offset + cond.base.width]
            if chunk == cond.base.op_a or chunk == cond.base.op_b:
                return True
    return False


def solve_comparison_wall_minimax(
    wall: list[CondStmt],
    max_depth: int = 10,
    branching_limit: int = 16,
) -> tuple[list[bytes], float]:
    """Solve a comparison wall as a minimax game.

    A "comparison wall" is a sequence of comparison statements that must all
    be satisfied to flip a branch. The fuzzer (maximizer) chooses byte
    mutations; the target (minimizer) resists by selecting which comparison
    to check next. The minimax value is the minimum number of mutations
    needed to satisfy all comparisons in the worst case.

    Args:
        wall: List of CondStmt objects forming the comparison wall.
        max_depth: Maximum search depth for alpha-beta pruning.
        branching_limit: Maximum number of candidate mutations per node.

    Returns:
        Tuple of (mutation_sequence, minimax_value).
    """
    if not wall:
        return [], 0.0

    def evaluate_state(states: dict) -> float:
        """Evaluate the minimax value: count unsolved comparisons (lower is better)."""
        return sum(1 for s in states.values() if s is CondState.UNSOLVED)

    def generate_mutations(cond: CondStmt, current_input: bytes) -> list[bytes]:
        """Generate candidate mutations for a comparison."""
        candidates = []
        for offset in cond.offsets:
            if cond.base.width == 1 and offset < len(current_input):
                target_byte = cond.base.op_b[0] if cond.base.op_b else 0
                mutated = bytearray(current_input)
                mutated[offset] = target_byte
                candidates.append(bytes(mutated))
            elif offset < len(current_input):
                for delta in (-1, 1, -2, 2):
                    mutated = bytearray(current_input)
                    mutated[offset] = max(0, min(255, current_input[offset] + delta))
                    candidates.append(bytes(mutated))
        return candidates[:branching_limit]

    def alpha_beta(
        states: dict,
        current_input: bytes,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> tuple[float, list[bytes]]:
        """Alpha-beta minimax search over comparison-wall mutations."""
        unsolved = [c for c in wall if states.get(c.key) is CondState.UNSOLVED]
        if depth <= 0 or not unsolved:
            return evaluate_state(states), []

        if maximizing:
            best_value = float("inf")
            best_seq: list[bytes] = []
            next_cond = unsolved[0]
            candidates = generate_mutations(next_cond, current_input)
            if not candidates:
                return evaluate_state(states), []
            for candidate in candidates:
                new_states = dict(states)
                if _check_comparison_satisfied(next_cond, candidate):
                    new_states[next_cond.key] = CondState.SOLVED
                val, seq = alpha_beta(new_states, candidate, depth - 1, alpha, beta, False)
                if val < best_value:
                    best_value = val
                    best_seq = [candidate] + seq
                    alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best_value, best_seq
        else:
            worst_value = -float("inf")
            worst_seq: list[bytes] = []
            for _cond in unsolved:
                val, seq = alpha_beta(states, current_input, depth - 1, alpha, beta, True)
                if val > worst_value:
                    worst_value = val
                    worst_seq = seq
                    beta = min(beta, val)
                if beta <= alpha:
                    break
            return worst_value, worst_seq

    initial_states = {c.key: c.state for c in wall}
    width = max((c.base.width for c in wall), default=1)
    initial_input = b"\x00" * width
    return alpha_beta(initial_states, initial_input, max_depth, -float("inf"), float("inf"), True)
