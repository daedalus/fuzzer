"""bench_paired: a campaign that stops short of --max-execs is not a result.

PR #244 review: rc 0 before ``iters`` execs was checkpointed like a full
campaign, a resume skipped it, and the pairing scored it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

import bench_paired  # noqa: E402

ITERS = 10_000
TARGET = "/t/target.so"
MATRIX = [(TARGET, "", 1)]


def _summary(execs: int) -> str:
    return f"  Executions:        {execs:,}\n  Edges discovered: 42\n"


@pytest.fixture
def runs(monkeypatch, tmp_path):
    """Scripted campaign logs, one per call; records the call count."""
    script: list[str] = []
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        log = script[calls["n"]]
        calls["n"] += 1
        return subprocess.CompletedProcess(cmd, 0, log, "")

    monkeypatch.setattr(bench_paired.subprocess, "run", fake_run)
    monkeypatch.setattr(bench_paired, "RESULTS", tmp_path)
    return script, calls


def _arm(tmp_path: Path) -> list:
    args = argparse.Namespace(set="s", restart=False, reps=1, iters=ITERS, timeout=10)
    bench_paired._run_arm(args, "baseline", MATRIX, 0, 1)
    out = tmp_path / "s_baseline.json"
    return json.loads(out.read_text()) if out.exists() else []


def test_regression_early_exit_not_checkpointed(runs, tmp_path) -> None:
    script, _ = runs
    script.append(_summary(ITERS // 2))

    assert _arm(tmp_path) == []

    # The log survives for diagnosis.
    kept = list((tmp_path / "s_baseline_early").glob("*.log"))
    assert [p.read_text() for p in kept] == [script[0]]


def test_regression_early_exit_rerun_on_resume(runs, tmp_path) -> None:
    script, calls = runs
    script += [_summary(ITERS // 2), _summary(ITERS)]

    _arm(tmp_path)
    rows = _arm(tmp_path)

    assert calls["n"] == 2
    assert [r["execs"] for r in rows] == [ITERS]


def test_full_campaign_recorded_and_skipped(runs, tmp_path) -> None:
    # Falsification: a complete cell is still checkpointed once.
    script, calls = runs
    script.append(_summary(ITERS))

    _arm(tmp_path)
    rows = _arm(tmp_path)

    assert calls["n"] == 1
    assert [r["execs"] for r in rows] == [ITERS]
    assert "log" not in rows[0]
    assert not (tmp_path / "s_baseline_early").exists()


@pytest.mark.parametrize(
    ("log", "complete"),
    [
        ("", False),  # died before the summary
        (_summary(ITERS - 1), False),
        (_summary(ITERS + 37), True),  # batch overshoot
    ],
)
def test_exec_count_edges(runs, tmp_path, log, complete) -> None:
    # Adversarial: missing summary, off-by-one, overshoot.
    script, _ = runs
    script.append(log)

    assert bool(_arm(tmp_path)) is complete
