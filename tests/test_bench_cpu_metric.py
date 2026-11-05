"""Bench harness rows carry child CPU time and can be scored on it (§4.1)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

import bench_paired  # noqa: E402
import bench_replicated  # noqa: E402
from child_rusage import TimedRun  # noqa: E402

LOG = "Edges discovered: 42\nCorpus: 7\nCrashes: 0\nExecutions: 100\n"


def _fake(cpu_user: float, cpu_sys: float):
    def run(cmd, timeout, **_kw):
        return TimedRun(rc=0, log=LOG, wall=2.0, cpu_user=cpu_user, cpu_sys=cpu_sys)

    return run


def _row(seed: int, cpu: float) -> dict:
    return {"target": "t.so", "seed": seed, "cpu_user": cpu, "coverage_attached": True}


class TestPairedRows:
    def test_run_cell_records_cpu(self, monkeypatch):
        monkeypatch.setattr(bench_paired, "run_timed", _fake(1.25, 0.5))
        arm = next(iter(bench_paired.ARMS))

        row = bench_paired.run_cell(arm, "t.so", "", 1, 100, 60)

        assert row["cpu_user"] == 1.25
        assert row["cpu_sys"] == 0.5
        assert row["secs"] == 2.0

    def test_compare_on_cpu_user(self):
        base = [_row(s, 1.0) for s in range(6)]
        test = [_row(s, 1.0 + 0.1 * (s + 1)) for s in range(6)]

        r = bench_paired.compare(base, test, metric="cpu_user")

        # Test used more CPU on every cell: all "wins" in the raw sign.
        assert r["wins"] == len(base)
        assert r["median_delta"] > 0

    def test_cpu_control_against_itself(self):
        """Rule 46: the same rows against themselves must show no effect."""
        base = [_row(s, 1.0 + 0.01 * s) for s in range(6)]

        r = bench_paired.compare(base, [dict(x) for x in base], metric="cpu_user")

        assert r["wins"] == r["losses"] == 0
        assert r["mcnemar_p"] == 1.0

    def test_metric_flag_accepts_cpu_user(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["bench_paired", "analyse", "x.json", "--metric", "cpu_user"]
        )
        seen = {}
        monkeypatch.setattr(bench_paired, "cmd_analyse", lambda a: seen.update(m=a.metric) or 0)

        assert bench_paired.main() == 0
        assert seen["m"] == "cpu_user"

    def test_regression_old_rows_without_cpu_are_skipped(self):
        """Adversarial: rows recorded before the cpu fields existed must not crash."""
        base = [{"target": "t.so", "seed": 1, "edges": 3, "coverage_attached": True}]

        r = bench_paired.compare(base, base, metric="cpu_user")

        assert r["cells"] == 0


class TestReplicatedRows:
    def test_run_one_records_cpu(self, monkeypatch):
        monkeypatch.setattr(bench_replicated, "run_timed", _fake(3.0, 0.25))

        r = bench_replicated.run_one("src", "t.so", "", 1, 100, 60)

        assert r["cpu_user"] == 3.0
        assert r["cpu_sys"] == 0.25
        assert r["edges"] == 42

    def test_summarise_cpu_metric(self, capsys):
        rows = [
            {"target": "t.so", "seed": 1, "arm": "a", "cpu_user": 1.0, "edges": 5},
            {"target": "t.so", "seed": 1, "arm": "b", "cpu_user": 1.5, "edges": 5},
        ]

        bench_replicated.summarise(rows, "a", "b", metric="cpu_user")

        assert "+0.5" in capsys.readouterr().out

    def test_regression_old_rows_without_cpu_are_skipped(self, capsys):
        """Adversarial: rows recorded before §4.1 lack cpu_user; skip, don't crash."""
        rows = [
            {"target": "t.so", "seed": 1, "arm": "a", "edges": 5},
            {"target": "t.so", "seed": 1, "arm": "b", "edges": 6},
        ]

        bench_replicated.summarise(rows, "a", "b", metric="cpu_user")

        assert "wins" not in capsys.readouterr().out
