"""Path negation locates each branch record once per solve_first call.

solve_first walked the frontier and, per candidate, _overlapping re-ran
_locate (a ``bytes.find`` over the whole input) on every record:
frontier x records x len(input) per round. On ffmpeg under --hail-mary that
was 54-76% of wall time (py-spy, 2026-10-09). The fix locates every record
once and reads the windows from there; the answers must not change.
"""

import pytest

from fuzzer_tool.core.path_constraints import BranchRecord, PathConstraintSolver
from fuzzer_tool.core.rand_pool import RandPool

pytest.importorskip("z3")

INPUT_LEN = 256
N_RECORDS = 60
N_CASES = 12
MAX_ROUNDS = 25
WIDTHS = (1, 2, 4, 8, 16)  # 16 > MAX_WIDTH: never in the frontier, still an overlap
RESULTS = (-1, 0, 1)


class _QuadraticSolver(PathConstraintSolver):
    """Oracle: frontier/negate/_overlapping/solve_first as before the fix."""

    def frontier(self, records, input_data):
        out = []
        for rec in records:
            if rec.key in self._attempted:
                continue
            if not (0 < rec.width <= 8):
                continue
            if self._locate(rec, input_data) is None:
                continue
            out.append(rec)
        out.sort(key=lambda r: r.width, reverse=True)
        return out

    def _overlapping(self, rec, others, data):
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
        out.sort(key=lambda t: t[0].width, reverse=True)
        return out[:8]

    def negate(self, rec, input_data, others=None):
        if not input_data or len(input_data) > (1 << 16):
            return None
        located = self._locate(rec, input_data)
        if located is None:
            self.skipped_unmapped += 1
            return None
        offset, other = located
        width = rec.width
        if offset + width > len(input_data):
            return None
        self._attempted[rec.key] = None
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

    def solve_first(self, records, input_data):
        for rec in self.frontier(records, input_data):
            result = self.negate(rec, input_data, others=records)
            if result is not None and result != input_data:
                return result
        return None


def _case(seed: int) -> tuple[bytes, list[BranchRecord]]:
    """Input plus records: windows of it (either operand), absent ones, dups."""
    rng = RandPool(seed)
    data = rng.randbytes(INPUT_LEN)
    recs = []
    for i in range(N_RECORDS):
        width = rng.choice(WIDTHS)
        start = rng.randrange(INPUT_LEN - width)
        window = data[start : start + width]
        other = rng.randbytes(width)
        kind = rng.randrange(4)
        if kind == 0:
            op_a, op_b = window, other  # input on the left
        elif kind == 1:
            op_a, op_b = other, window  # input on the right: inverted sense
        elif kind == 2:
            op_a, op_b = rng.randbytes(width), rng.randbytes(width)  # likely absent
        else:
            op_a, op_b = window, window  # equality already satisfied
        recs.append(BranchRecord(op_a, op_b, rng.choice(RESULTS), width, pc=i % 7))
    # Same key twice, and the same object twice: both are skipped as "self".
    recs.append(BranchRecord(recs[0].op_a, recs[0].op_b, recs[0].result, recs[0].width, recs[0].pc))
    recs.append(recs[1])
    return data, recs


def _recorder(solver: PathConstraintSolver, queries: list) -> None:
    """Replace z3 with a deterministic stand-in that logs each query.

    z3 models are not reproducible across solver instances (the oracle
    disagreed with itself), so the comparison is on what the fix changes:
    which branch is queried and with exactly which overlapping windows.
    """

    def solve(data, offset, width, observed, original, other, overlaps):
        queries.append(
            (offset, width, observed, original, other, [(r.key, s, v) for r, s, v in overlaps])
        )
        if len(queries) % 2:
            return None  # odd queries fail: the frontier walk continues
        out = bytearray(data)
        out[offset] ^= 0xFF
        return bytes(out)

    solver._solve_z3 = solve


def _trace(solver: PathConstraintSolver, data: bytes, recs: list) -> tuple[list, list, dict]:
    """solve_first until the frontier is spent, chaining each answer."""
    queries: list = []
    _recorder(solver, queries)
    outs = []
    for _ in range(MAX_ROUNDS):
        out = solver.solve_first(recs, data)
        outs.append(out)
        if out is None:
            break
        data = out
    return outs, queries, solver.stats()


class TestLocateOnceEquivalence:
    def test_regression_oracle_matches_itself(self):
        """Control (Hard Rule 46): two oracle runs agree, or the check is void."""
        for seed in range(N_CASES):
            data, recs = _case(seed)
            assert _trace(_QuadraticSolver(), data, recs) == _trace(_QuadraticSolver(), data, recs)

    def test_regression_same_answers_as_quadratic(self):
        for seed in range(N_CASES):
            data, recs = _case(seed)
            assert _trace(PathConstraintSolver(), data, recs) == _trace(
                _QuadraticSolver(), data, recs
            ), f"seed {seed}"

    def test_regression_cases_reach_z3_and_direct(self):
        """The equivalence must cover both solve paths, not just one."""
        z3_total = direct_total = 0
        for seed in range(N_CASES):
            data, recs = _case(seed)
            _, queries, stats = _trace(PathConstraintSolver(), data, recs)
            z3_total += len(queries)
            direct_total += stats["direct_solves"]
        assert z3_total > 0
        assert direct_total > 0

    def test_regression_adversarial_all_windows_overlap(self):
        """Every record on one byte run: max overlap fan-in, MAX_OVERLAP cut."""
        data = b"\x10" * 32
        recs = [
            BranchRecord(b"\x10" * w, bytes([0x20 + i]) * w, (-1, 0, 1)[i % 3], w, pc=i)
            for i, w in enumerate((1, 2, 4, 8) * 5)
        ]
        assert _trace(PathConstraintSolver(), data, recs) == _trace(_QuadraticSolver(), data, recs)


class TestLocateOnceCost:
    def test_regression_locate_linear_in_records(self, monkeypatch):
        """Falsification: with every negation failing, the whole frontier is
        walked. Quadratic code calls _locate frontier x records times."""
        data, recs = _case(0)
        solver = PathConstraintSolver()
        calls = 0
        real = solver._locate

        def counting(rec, d):
            nonlocal calls
            calls += 1
            return real(rec, d)

        monkeypatch.setattr(solver, "_locate", counting)
        monkeypatch.setattr(solver, "_direct_solve", lambda *a: None)
        monkeypatch.setattr(solver, "_solve_z3", lambda *a: None)

        frontier_size = len(PathConstraintSolver().frontier(recs, data))
        assert frontier_size > 1
        assert solver.solve_first(recs, data) is None
        # One locate per record for the map, one per attempted candidate.
        assert calls <= len(recs) + frontier_size


class TestWindowIndex:
    """_Windows.touching against a brute-force scan of every window."""

    @staticmethod
    def _brute(located, start, end):
        return [t for t in located if t[1] < end and start < t[1] + t[0].width]

    @staticmethod
    def _located(seed):
        rng = RandPool(seed)
        return [
            (BranchRecord(b"", b"", 0, rng.choice(WIDTHS), i), rng.randrange(INPUT_LEN), i)
            for i in range(N_RECORDS)
        ]

    def test_regression_matches_brute_force(self):
        from fuzzer_tool.core.path_constraints import _Windows

        for seed in range(N_CASES):
            located = self._located(seed)
            index = _Windows(located)
            for start in range(INPUT_LEN):
                for width in WIDTHS:
                    assert index.touching(start, start + width) == self._brute(
                        located, start, start + width
                    )

    def test_regression_adjacent_windows_do_not_touch(self):
        """Adversarial: [0,4) and [4,8) share an edge, not a byte."""
        from fuzzer_tool.core.path_constraints import _Windows

        left = (BranchRecord(b"", b"", 0, 4, 1), 0, 0)
        right = (BranchRecord(b"", b"", 0, 4, 2), 4, 0)
        index = _Windows([left, right])
        assert index.touching(0, 4) == [left]
        assert index.touching(4, 8) == [right]
        assert index.touching(3, 5) == [left, right]

    def test_regression_empty_index(self):
        from fuzzer_tool.core.path_constraints import _Windows

        assert _Windows([]).touching(0, 8) == []
