"""z3 runs under 30 ms per query and 10 ms per fuzz round.

ffmpeg --hail-mary spent 15-30% of a round in z3 (path negation + concolic)
with 50-200 ms per-query timeouts and no per-round cap. Every z3 check now
goes through core.z3_budget: each query is capped at QUERY_MAX_MS, and
inside a fuzz round all queries share ROUND_BUDGET_MS.
"""

import pytest

from fuzzer_tool.core import z3_budget
from fuzzer_tool.core.path_constraints import BranchRecord, PathConstraintSolver
from fuzzer_tool.core.z3_budget import QUERY_MAX_MS, ROUND_BUDGET_MS, Z3Budget

z3 = pytest.importorskip("z3")


class _FakeClock:
    """Scripted perf_counter: each check() costs exactly *step_ms*."""

    def __init__(self, step_ms: float):
        self.now = 0.0
        self.step = step_ms / 1000

    def __call__(self) -> float:
        return self.now


class _FakeSolver:
    """Records the timeout it was given; advances the clock per check()."""

    def __init__(self, clock: _FakeClock):
        self.clock = clock
        self.timeouts: list[int] = []
        self.checks = 0

    def set(self, key, value):
        assert key == "timeout"
        self.timeouts.append(value)

    def check(self):
        self.checks += 1
        self.clock.now += self.clock.step
        return z3.sat


class _SpyBudget(Z3Budget):
    """Real budget that records each requested timeout."""

    def __init__(self):
        super().__init__()
        self.calls: list[int] = []

    def check(self, solver, requested_ms):
        self.calls.append(requested_ms)
        return super().check(solver, requested_ms)


@pytest.fixture
def fresh_budget(monkeypatch):
    """Isolated module budget; restored after the test (Hard Rule 21)."""
    budget = _SpyBudget()
    monkeypatch.setattr(z3_budget, "BUDGET", budget)
    return budget


class TestQueryCap:
    def test_regression_query_capped_outside_round(self):
        clock = _FakeClock(1)
        solver = _FakeSolver(clock)
        assert Z3Budget(clock=clock).check(solver, 200) == z3.sat
        assert solver.timeouts == [QUERY_MAX_MS]

    def test_regression_smaller_request_kept(self):
        clock = _FakeClock(1)
        solver = _FakeSolver(clock)
        Z3Budget(clock=clock).check(solver, QUERY_MAX_MS - 1)
        assert solver.timeouts == [QUERY_MAX_MS - 1]

    def test_regression_no_round_never_exhausts(self):
        """Outside a round only the per-query cap applies: no starvation."""
        clock = _FakeClock(ROUND_BUDGET_MS)
        solver = _FakeSolver(clock)
        budget = Z3Budget(clock=clock)
        for _ in range(5):
            assert budget.check(solver, 200) == z3.sat
        assert solver.checks == 5


class TestRoundBudget:
    def test_regression_round_budget_shared_and_spent(self):
        step = ROUND_BUDGET_MS / 4
        clock = _FakeClock(step)
        solver = _FakeSolver(clock)
        budget = Z3Budget(clock=clock)
        budget.open_round()
        results = [budget.check(solver, 200) for _ in range(6)]
        assert results[:4] == [z3.sat] * 4
        assert results[4:] == [None, None]  # spent: solver never called
        assert solver.checks == 4
        # Each grant is what was left, never above the per-query cap.
        expected = [min(QUERY_MAX_MS, int(ROUND_BUDGET_MS - i * step)) for i in range(4)]
        assert solver.timeouts == expected
        assert budget.spent()

    def test_regression_close_round_restores_query_cap(self):
        clock = _FakeClock(ROUND_BUDGET_MS)
        solver = _FakeSolver(clock)
        budget = Z3Budget(clock=clock)
        budget.open_round()
        budget.check(solver, 200)
        assert budget.check(solver, 200) is None
        budget.close_round()
        assert not budget.spent()
        assert budget.check(solver, 200) == z3.sat

    def test_regression_open_round_refills(self):
        clock = _FakeClock(ROUND_BUDGET_MS)
        solver = _FakeSolver(clock)
        budget = Z3Budget(clock=clock)
        budget.open_round()
        budget.check(solver, 200)
        budget.open_round()
        assert budget.check(solver, 200) == z3.sat

    def test_regression_failing_check_still_charged(self):
        """Adversarial: a raising check() must still consume budget."""
        clock = _FakeClock(ROUND_BUDGET_MS)

        class Boom(_FakeSolver):
            def check(self):
                self.clock.now += self.clock.step
                raise z3.Z3Exception("boom")

        budget = Z3Budget(clock=clock)
        budget.open_round()
        with pytest.raises(z3.Z3Exception):
            budget.check(Boom(clock), 200)
        assert budget.spent()


def _coupled_case():
    """Two overlapping windows: negation needs the conjunctive z3 path."""
    data = (0x00001000).to_bytes(4, "little") + b"TAIL"
    flip = BranchRecord((0x1000).to_bytes(2, "little"), (0x2000).to_bytes(2, "little"), -1, 2, 1)
    keep = BranchRecord(
        (0x00001000).to_bytes(4, "little"), (0xFFFFFFFF).to_bytes(4, "little"), -1, 4, 2
    )
    return data, [flip, keep]


class TestWiring:
    def test_regression_path_negation_uses_budget(self, fresh_budget):
        calls = fresh_budget.calls
        data, recs = _coupled_case()
        assert PathConstraintSolver().solve_first(recs, data) is not None
        assert calls

    def test_regression_spent_round_keeps_frontier(self, fresh_budget):
        """A spent budget skips z3 without marking branches attempted."""
        data, recs = _coupled_case()
        solver = PathConstraintSolver()
        fresh_budget.open_round()
        fresh_budget._remaining_ms = 0.0
        assert solver.solve_first(recs, data) is None
        assert len(solver.frontier(recs, data)) == len(PathConstraintSolver().frontier(recs, data))
        fresh_budget.close_round()
        assert solver.solve_first(recs, data) is not None

    def test_regression_concolic_uses_budget_not_global(self, fresh_budget):
        from fuzzer_tool.core.smt_solver import ConcolicTrace

        calls = fresh_budget.calls
        before = z3.get_param("timeout")
        trace = ConcolicTrace()
        trace.set_input(b"HEADabcdTAIL")
        trace.add_entry(b"abcd", b"wxyz", 4)
        assert trace.solve(timeout_ms=50) == b"HEADwxyzTAIL"
        assert calls == [50]
        assert z3.get_param("timeout") == before  # no process-global leak

    def test_regression_structural_and_field_use_budget(self, fresh_budget):
        from fuzzer_tool.core.field_constraints import CONSTANT, Field, solve_coupled
        from fuzzer_tool.core.structural_constraints import solve_coupled_sections

        calls = fresh_budget.calls
        solve_coupled_sections(2, 4, 1 << 12)
        fields = [Field(CONSTANT, offset=0, width=2, value=0)]
        solve_coupled(fields, b"\x00\x00", [])
        assert len(calls) == 2


class TestRoundScope:
    def test_regression_fuzz_round_opens_and_closes(self, fresh_budget):
        from types import SimpleNamespace

        from fuzzer_tool.services.fuzz_round import FuzzRound

        seen = []
        stub = SimpleNamespace(_steps=lambda: seen.append(fresh_budget._remaining_ms) or True)
        assert FuzzRound.run(stub) is True
        assert seen == [float(ROUND_BUDGET_MS)]
        assert fresh_budget._remaining_ms is None

    def test_regression_fuzz_round_closes_on_error(self, fresh_budget):
        """Adversarial: a raising round must not leave the budget armed."""
        from types import SimpleNamespace

        from fuzzer_tool.services.fuzz_round import FuzzRound

        def boom():
            raise RuntimeError("round failed")

        with pytest.raises(RuntimeError):
            FuzzRound.run(SimpleNamespace(_steps=boom))
        assert fresh_budget._remaining_ms is None


class TestConfigurable:
    """--smt-query-cap / --smt-round-budget (ms) set the process budget."""

    def test_regression_configure_sets_caps(self):
        clock = _FakeClock(2)
        solver = _FakeSolver(clock)
        budget = Z3Budget(clock=clock)
        budget.configure(query_cap_ms=7, round_budget_ms=3)
        assert budget.check(solver, 200) == z3.sat
        budget.open_round()
        results = [budget.check(solver, 200) for _ in range(3)]
        assert solver.timeouts == [7, 3, 1]  # cap, then what the round has left
        assert results == [z3.sat, z3.sat, None]

    @pytest.mark.parametrize("bad", [(0, 10), (30, 0), (-1, 10)])
    def test_regression_configure_rejects_non_positive(self, bad):
        """Adversarial: a zero cap would silently disable every z3 call."""
        with pytest.raises(ValueError):
            Z3Budget().configure(*bad)

    @pytest.mark.parametrize("text", ["0", "-5", "abc", "1.5"])
    def test_regression_cli_rejects_bad_ms(self, text):
        import argparse

        from fuzzer_tool.cli.commands import _positive_ms_arg

        with pytest.raises(argparse.ArgumentTypeError):
            _positive_ms_arg(text)

    def test_regression_cli_flags_and_defaults(self):
        import ast
        import inspect

        from fuzzer_tool.cli import commands
        from fuzzer_tool.services.fuzzer import Fuzzer
        from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

        dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
        assert {"smt_query_cap", "smt_round_budget"} <= dests
        params = inspect.signature(Fuzzer.__init__).parameters
        assert params["smt_query_cap"].default == QUERY_MAX_MS == 30
        assert params["smt_round_budget"].default == ROUND_BUDGET_MS == 10

    def test_regression_fuzzer_configures_budget(self, fresh_budget, tmp_path):
        from unittest.mock import patch

        from fuzzer_tool.services.fuzzer import Fuzzer

        with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
            Fuzzer(
                target="/bin/true",
                corpus_dir=str(tmp_path / "corpus"),
                crashes_dir=str(tmp_path / "crashes"),
                max_len=256,
                timeout=1,
                mutations_per_input=2,
                smt_query_cap=7,
                smt_round_budget=3,
            )
        assert fresh_budget.grant(200) == 7
        fresh_budget.open_round()
        assert fresh_budget.grant(200) == 3
