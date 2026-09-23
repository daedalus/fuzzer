"""Regression: ``fuzzer-tool fuzz -n`` counts iterations, not executions.

The ``-n/--iterations`` budget bounds the main-loop count, but each iteration
runs many target executions (the mutation budget: probes, deterministic
stages, havoc rounds, trim re-runs, dedup re-rolls).  A hail-mary campaign
run with ``-n 2500`` actually executed ~392k target execs, and the tool help
for ``edge_diagnostic.py matrix --hail-mary --iters`` promised "executions".
The fix adds a real execution budget: ``Fuzzer.run(max_execs=.N)`` and the
fuzz CLI ``--max-execs`` flag stop the campaign once ``exec_count`` crosses
the budget, so the tool can honor "run this many executions" instead of
misreporting iterations as executions.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests import test_regression_end_of_run_persistence as end_of_run


class TestRunHonorsExecBudget:
    def test_run_stops_when_exec_count_crosses_budget(self, tmp_root):
        (tmp_root / "corpus" / "seed_a").write_bytes(b"AAAA")
        (tmp_root / "corpus" / "seed_b").write_bytes(b"BBBB")
        f = end_of_run._make_fuzzer(
            tmp_root,
            corpus_dir=str(tmp_root / "corpus"),
            quiet_stats=True,
        )

        budget = 5
        with patch.object(f, "_run_target", return_value=(0, "")):
            f.run(iterations=10_000, max_execs=budget)

        # The budget is checked once per main-loop iteration, so the final
        # exec_count may overshoot by at most what one fuzz_one costs plus
        # the initial seed-as-is pass (one exec per corpus seed).
        seed_pass = 2
        assert seed_pass <= f.exec_count <= budget + seed_pass

    def test_iterations_still_bounds_a_finite_run(self, tmp_root):
        f = end_of_run._make_fuzzer(tmp_root)
        with patch.object(f, "_run_target", return_value=(0, "")):
            f.run(iterations=3)  # no exec budget: falls back to iterations
        assert f.exec_count >= 1


class TestCliBudgetWiring:
    def test_run_receives_arg_max_execs(self, monkeypatch, tmp_path):
        from fuzzer_tool.cli import commands

        captured = {}

        def fake_fuzzer(**kwargs):
            return SimpleNamespace(run=lambda **rk: captured.update(rk))

        monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake_fuzzer)
        args = _default_fuzz_args(tmp_path)
        args.max_execs = 1234
        commands.cmd_fuzz(args)
        assert captured.get("max_execs") == 1234


def _default_fuzz_args(tmp: Path):
    from tests import test_commands_extended

    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp)
    args.max_execs = 0
    return args
