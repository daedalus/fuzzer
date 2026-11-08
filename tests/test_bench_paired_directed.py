"""bench_paired: directed-metric arms and the time-to-exposure (tte) metric.

A directed A/B is read on how soon the target bug fires, not on edges:
``tte_<kind>`` is the first ``exec_count`` among a cell's crash records of
that sanitizer kind, censored at ``iters + 1`` when it never fired.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

import bench_paired  # noqa: E402
import eval_set  # noqa: E402
from bench_paired import ARM_BASELINES, ARMS, DIRECTED_ARMS  # noqa: E402
from child_rusage import TimedRun  # noqa: E402

from fuzzer_tool.cli import commands  # noqa: E402

ITERS = 100
KIND = "bad-free"
METRIC = f"tte_{KIND}"


def _parse(monkeypatch, flags: list[str]):
    """Run the real ``main()`` parser and return the Namespace it built."""
    captured: dict[str, object] = {}

    def _spy(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *flags])
    assert commands.main() == 0
    return captured["args"]


def _row(seed: int, tte: dict[str, int]) -> dict:
    return {
        "arm": "a",
        "target": "/t.so",
        "seed": seed,
        "rep": 0,
        "iters": ITERS,
        "coverage_attached": True,
        "tte": tte,
    }


# ── arms ────────────────────────────────────────────────────────────────


def test_arms_registered_against_harmonic():
    assert ARM_BASELINES["dist-reach"] == "dist-harmonic"
    assert ARM_BASELINES["dist-harmonic-ctl"] == "dist-harmonic"
    assert set(DIRECTED_ARMS) <= set(ARMS)


def test_only_the_metric_changes(monkeypatch):
    base = vars(_parse(monkeypatch, ARMS["dist-harmonic"]))
    test = vars(_parse(monkeypatch, ARMS["dist-reach"]))
    changed = {k for k in base if k != "func" and base[k] != test[k]}
    assert changed == {"distance_metric"}, changed


def test_control_is_harmonic_again():
    assert ARMS["dist-harmonic-ctl"] == ARMS["dist-harmonic"]
    assert "dist-harmonic-ctl" not in DIRECTED_ARMS


def test_directed_sets_carry_a_target():
    for name in ("fuzzgoat_dir279", "fuzzgoat_dir258"):
        (target, flags), *_ = eval_set.TARGET_SETS[name]
        assert "--target-functions" in flags.split()
        assert "--schedule" in flags.split()


# ── tte recording ───────────────────────────────────────────────────────


def _fake_run(records: list[dict]):
    """A campaign that writes *records* into the --crashes dir it was given."""
    seen: dict[str, object] = {}

    def run(cmd, timeout, **kw):
        crashes = Path(cmd[cmd.index("--crashes") + 1])
        seen["crashes"] = crashes
        seen["workdir"] = Path(cmd[cmd.index("-d") + 1])
        crashes.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(records):
            (crashes / f"crash_{i}.json").write_text(json.dumps(rec))
        log = f"Executions:        {ITERS}\nEdges discovered:  5\n"
        return TimedRun(rc=0, log=log, wall=0.0, cpu_user=0.0, cpu_sys=0.0)

    return run, seen


def test_cell_records_first_exec_per_kind(monkeypatch):
    run, seen = _fake_run(
        [
            {"error_type": KIND, "exec_count": 70},
            {"error_type": KIND, "exec_count": 40},
            {"error_type": "heap-use-after-free", "exec_count": 9},
        ]
    )
    monkeypatch.setattr(bench_paired, "run_timed", run)
    row = bench_paired.run_cell("dist-harmonic", "/t.so", "", 1, ITERS, 10)
    assert row["tte"] == {KIND: 40, "heap-use-after-free": 9}


def test_crashes_dir_is_per_cell_and_cleaned(monkeypatch):
    # A shared ~/fuzzing/<target>/crashes re-syncs earlier cells' signatures
    # and suppresses this cell's "novel" saves: the tte would be wrong.
    run, seen = _fake_run([])
    monkeypatch.setattr(bench_paired, "run_timed", run)
    bench_paired.run_cell("dist-harmonic", "/t.so", "", 1, ITERS, 10)
    assert seen["crashes"].parent == seen["workdir"]
    assert not seen["workdir"].exists()


def test_adversarial_record_without_exec_count_ignored(monkeypatch):
    run, _ = _fake_run([{"error_type": KIND}, {"exec_count": 3}])
    monkeypatch.setattr(bench_paired, "run_timed", run)
    assert bench_paired.run_cell("dist-harmonic", "/t.so", "", 1, ITERS, 10)["tte"] == {}


# ── tte scoring ─────────────────────────────────────────────────────────


def test_lower_tte_wins_and_censoring_counts():
    base = [_row(0, {KIND: 50}), _row(1, {}), _row(2, {KIND: 10})]
    test = [_row(0, {KIND: 20}), _row(1, {KIND: 90}), _row(2, {})]
    r = bench_paired.compare(base, test, METRIC)
    # seed 0: 20 < 50 win; seed 1: 90 < censored 101 win; seed 2: censored loss.
    assert (r["wins"], r["losses"], r["ties"]) == (2, 1, 0)


def test_both_censored_is_a_tie():
    r = bench_paired.compare([_row(0, {})], [_row(0, {})], METRIC)
    assert (r["wins"], r["losses"], r["ties"]) == (0, 0, 1)


def test_falsification_control_against_itself_is_all_ties():
    rows = [_row(s, {KIND: 10 * s + 5}) for s in range(5)]
    r = bench_paired.compare(rows, [dict(x) for x in rows], METRIC)
    assert r["ties"] == len(rows) and r["mcnemar_p"] == 1.0


def test_rows_without_tte_are_not_scored():
    # Rows recorded before tte existed cannot be read as censored.
    legacy = {k: v for k, v in _row(0, {}).items() if k != "tte"}
    r = bench_paired.compare([legacy], [_row(0, {KIND: 1})], METRIC)
    assert r["cells"] == 0


@pytest.mark.parametrize("metric", ["tte_bad-free", "edges"])
def test_analyse_accepts_metric(metric):
    assert bench_paired._metric_arg(metric) == metric


def test_analyse_rejects_unknown_metric():
    with pytest.raises(Exception, match="metric"):
        bench_paired._metric_arg("bogus")
