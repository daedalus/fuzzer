"""Bench metric ``kills``: mutant kills per cell (docs/action_plan_mutant_kill_metric.md P5)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

import bench_paired  # noqa: E402
from bench_paired import _PLAIN_METRICS, _metric_arg, compare  # noqa: E402
from child_rusage import TimedRun  # noqa: E402

from fuzzer_tool.core.exact_tests import mcnemar_exact  # noqa: E402
from fuzzer_tool.services.mutant_kill import KillScore  # noqa: E402

_ITERS = 10
_SCORE = KillScore(
    kills=7,
    hangs=2,
    flaky=1,
    inputs=5,
    clean_inputs=4,
    killed=frozenset(range(7)),
    hung=frozenset(),
)


def _rows(arm: str, kills: list[int]) -> list[dict]:
    return [
        {
            "arm": arm,
            "target": "t.so",
            "seed": s,
            "rep": 0,
            "iters": _ITERS,
            "coverage_attached": True,
            "kills": k,
        }
        for s, k in enumerate(kills)
    ]


class TestMetric:
    def test_registered(self) -> None:
        assert "kills" in _PLAIN_METRICS
        assert _metric_arg("kills") == "kills"

    def test_control_self_compare_is_null(self) -> None:
        """Rule 46: the same rows against themselves give no evidence at all."""
        rows = _rows("a", [3, 5, 8, 2, 9])
        got = compare(rows, rows, "kills")
        assert (got["wins"], got["losses"]) == (0, 0)
        assert got["mcnemar_p"] == mcnemar_exact(0, 0) == 1.0

    def test_higher_kills_win(self) -> None:
        base = _rows("a", [3, 5, 8, 2, 9])
        test = _rows("b", [4, 6, 9, 3, 9])
        got = compare(base, test, "kills")
        assert (got["wins"], got["losses"], got["ties"]) == (4, 0, 1)
        assert got["mcnemar_p"] == mcnemar_exact(4, 0)

    def test_rows_without_kills_are_skipped_not_zero(self) -> None:
        base = _rows("a", [3, 5])
        test = [{k: v for k, v in r.items() if k != "kills"} for r in _rows("b", [9, 9])]
        got = compare(base, test, "kills")
        assert got["wins"] + got["losses"] + got["ties"] == 0


class _FakeScorer:
    """Records when and on what it was asked to score."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.seen: list[bytes] = []

    def score(self, corpus: Path) -> KillScore:
        self.calls.append("score")
        self.seen = sorted(p.read_bytes() for p in (corpus / "seeds").iterdir())
        return _SCORE


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """run_cell with the campaign replaced: one seed file, a fake timed run."""
    calls: list[str] = []
    workdirs: list[Path] = []

    def seed(target: str, workdir: Path) -> None:
        (workdir / "seeds").mkdir()
        (workdir / "seeds" / "s0").write_bytes(b"corpus-input")
        workdirs.append(workdir)

    def timed(cmd, timeout, **kw) -> TimedRun:
        calls.append("run")
        return TimedRun(rc=0, log="", wall=1.5, cpu_user=1.0, cpu_sys=0.1)

    monkeypatch.setattr(bench_paired, "_seed_workdir", seed)
    monkeypatch.setattr(bench_paired, "run_timed", timed)
    monkeypatch.setattr(bench_paired, "_parse", lambda log: {"execs": _ITERS, "edges": 1})
    monkeypatch.setattr(bench_paired, "_first_crash_execs", lambda crashes: {})
    monkeypatch.setattr(bench_paired, "_real_edges", lambda workdir: None)
    monkeypatch.setitem(bench_paired.ARMS, "kills-test-arm", [])
    return calls, workdirs


class TestRunCell:
    def test_scored_after_run_before_cleanup(self, stubbed) -> None:
        calls, workdirs = stubbed
        scorer = _FakeScorer(calls)
        row = bench_paired.run_cell("kills-test-arm", "t.so", "", 1, _ITERS, 60, scorer=scorer)

        assert calls == ["run", "score"]
        assert scorer.seen == [b"corpus-input"]
        assert not workdirs[0].exists()
        assert (row["kills"], row["kill_hangs"], row["kill_flaky"], row["clean_inputs"]) == (
            _SCORE.kills,
            _SCORE.hangs,
            _SCORE.flaky,
            _SCORE.clean_inputs,
        )

    def test_scoring_time_not_in_secs(self, stubbed) -> None:
        calls, _ = stubbed
        row = bench_paired.run_cell(
            "kills-test-arm", "t.so", "", 1, _ITERS, 60, scorer=_FakeScorer(calls)
        )
        assert row["secs"] == 1.5

    def test_no_scorer_leaves_keys_absent(self, stubbed) -> None:
        """Absent, never 0: analyse skips the cell rather than scoring zero kills."""
        row = bench_paired.run_cell("kills-test-arm", "t.so", "", 1, _ITERS, 60)
        assert "kills" not in row

    def test_scorer_failure_still_cleans_workdir(self, stubbed) -> None:
        calls, workdirs = stubbed

        class Boom(_FakeScorer):
            def score(self, corpus: Path) -> KillScore:
                raise RuntimeError("driver failed")

        with pytest.raises(RuntimeError):
            bench_paired.run_cell("kills-test-arm", "t.so", "", 1, _ITERS, 60, scorer=Boom(calls))
        assert not workdirs[0].exists()


class TestRunArgs:
    def _args(self, **kw) -> object:
        import argparse

        return argparse.Namespace(kill_set=None, kill_jobs=1, **kw)

    def test_no_kill_set_no_scorer(self) -> None:
        assert bench_paired._open_scorer(self._args()) is None

    def test_unbuilt_kill_set_exits_2_before_lock(self, tmp_path, monkeypatch, capsys) -> None:
        engaged: list[bool] = []
        monkeypatch.setattr(bench_paired.bench_lock, "engage", lambda flag: engaged.append(flag))
        args = self._args()
        args.kill_set = str(tmp_path / "never-built")
        assert bench_paired.cmd_run(args) == 2
        assert engaged == []
        assert "--kill-set" in capsys.readouterr().err

    def test_parser_accepts_kill_flags(self, monkeypatch) -> None:
        seen: list[object] = []
        monkeypatch.setattr(bench_paired, "cmd_run", lambda a: seen.append(a) or 0)
        monkeypatch.setattr(
            sys, "argv", ["bench_paired", "run", "--kill-set", "K", "--kill-jobs", "3"]
        )
        bench_paired.main()
        (args,) = seen
        assert (args.kill_set, args.kill_jobs) == ("K", 3)
