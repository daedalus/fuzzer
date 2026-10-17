"""Regression: three places a smallest-of-many p or a best-of-many pick was read raw.

* ``bench_paired analyse`` printed one McNemar p per arm (and per arm x
  target) uncorrected: with 30 rows ~1.5 read p < .05 by chance.
* ``minimax_arm`` picked and scored the arm on the same seeds, so the
  printed regret carried the winner's own noise (winner's curse).
* one homogeneity rejection set SUBCRITICAL, re-tested every tick for the
  campaign: alpha per tick guarantees false escalations on a long run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import _add_holm, _target_results, cmd_analyse, holdout_minimax  # noqa: E402

from fuzzer_tool.core.analyzers.analyzer_coverage_regime import CoverageRegimeDetector
from fuzzer_tool.core.analyzers.analyzer_critical_slowing import (
    CoverageHomogeneityDetector,
    CriticalSlowingDown,
)
from fuzzer_tool.core.percolation import CoverageRegime


def _rows(arm: str, target: str, edges: list[int]) -> list[dict]:
    return [
        {"arm": arm, "target": target, "seed": s, "rep": 0, "edges": e, "coverage_attached": True}
        for s, e in enumerate(edges)
    ]


# ── Gap 1: Holm over the printed family ─────────────────────────────────


def test_regression_holm_matches_hand_example():
    # holm() docstring example: [.04, .01, .03] -> [.06, .03, .06].
    rows = [{"mcnemar_p": p} for p in (0.04, 0.01, 0.03)]

    _add_holm(rows)

    assert [round(r["holm_p"], 6) for r in rows] == [0.06, 0.03, 0.06]


def test_holm_single_row_is_unadjusted():
    # Falsification: a family of one carries no multiplicity.
    rows = [{"mcnemar_p": 0.02}]

    _add_holm(rows)

    assert rows[0]["holm_p"] == 0.02


def test_holm_empty_family():
    # Adversarial: no rows must not raise.
    rows: list[dict] = []
    _add_holm(rows)
    assert rows == []


def test_regression_by_target_family_spans_arms_and_targets():
    base = _rows("base", "T1", [1] * 8) + _rows("base", "T2", [1] * 8)
    loaded = {
        "A": _rows("A", "T1", [2] * 8) + _rows("A", "T2", [1] * 8),
        "B": _rows("B", "T1", [2] * 8) + _rows("B", "T2", [0] * 8),
    }

    results = _target_results(base, loaded, "edges")
    _add_holm([r for _, _, r in results])

    # 4 rows: the smallest raw p is scaled by the whole family, not per arm.
    assert len(results) == 4
    raw = sorted(r["mcnemar_p"] for _, _, r in results)
    smallest = min(results, key=lambda t: t[2]["mcnemar_p"])[2]
    assert smallest["holm_p"] == min(1.0, len(results) * raw[0])


def test_regression_analyse_prints_holm(tmp_path, capsys):
    paths = []
    rows = {
        "base": _rows("base", "T1", [1] * 8),
        "A": _rows("A", "T1", [2] * 8),
    }
    for arm, r in rows.items():
        p = tmp_path / f"{arm}.json"
        p.write_text(json.dumps(r))
        paths.append(str(p))
    args = argparse.Namespace(
        files=paths, baseline="base", metric="edges", by_target=True, risk_matrix=False
    )

    assert cmd_analyse(args) == 0

    out = capsys.readouterr().out
    assert out.count("Holm") >= 2  # pooled and per-target headers


# ── Gap 2: minimax regret on held-out seeds ─────────────────────────────


def test_regression_holdout_exposes_winners_curse():
    # Pure noise: A and B alternate wins seed by seed. Each fold's minimax
    # pick looks perfect on its own seeds (regret 0) and loses every
    # held-out seed (regret 1).
    loaded = {
        "A": _rows("A", "T1", [2, 1, 2, 1]),
        "B": _rows("B", "T1", [1, 2, 1, 2]),
    }

    arm, in_sample, held = holdout_minimax(loaded)

    assert arm in loaded
    assert in_sample == 0.5  # full-data tie
    assert held == 1.0


def test_holdout_dominant_arm_keeps_its_regret():
    # Falsification: an arm that wins every seed is not a curse artefact.
    loaded = {
        "A": _rows("A", "T1", [5, 5, 5, 5]),
        "B": _rows("B", "T1", [1, 1, 1, 1]),
    }

    arm, in_sample, held = holdout_minimax(loaded)

    assert (arm, in_sample, held) == ("A", 0.0, 0.0)


def test_holdout_single_seed_has_no_estimate():
    # Adversarial: one seed cannot be split.
    loaded = {"A": _rows("A", "T1", [5]), "B": _rows("B", "T1", [1])}

    arm, _in_sample, held = holdout_minimax(loaded)

    assert arm == "A"
    assert held is None


def test_holdout_empty():
    assert holdout_minimax({}) == ("", 0.0, None)


# ── Gap 3: homogeneity escalation needs a disjoint confirming window ────

_CLUSTERED = {
    "homogeneous": False,
    "chi2": 50.0,
    "p_value": 0.001,
    "cramers_v": 0.4,
    "total_edges": 200,
}
_UNIFORM = dict(_CLUSTERED, homogeneous=True, p_value=0.5)


def _regime() -> tuple[CoverageRegimeDetector, int]:
    hom = CoverageHomogeneityDetector()
    d = CoverageRegimeDetector(csd=CriticalSlowingDown(), homogeneity=hom, stall_threshold=10**6)
    return d, hom.window_size


def _feed(d: CoverageRegimeDetector, results: list[dict]) -> CoverageRegime:
    regime = CoverageRegime.SUPERCRITICAL
    for i, r in enumerate(results):
        regime = d.observe(2.0, 0, r, execs_since_edge=5, exec_count=100 * (i + 1))
    return regime


def test_regression_single_rejection_does_not_escalate():
    d, _w = _regime()

    assert _feed(d, [_CLUSTERED]) is not CoverageRegime.SUBCRITICAL


def test_rejection_confirmed_by_disjoint_window_escalates():
    # Rejected now and one full window ago: two windows sharing no tick.
    d, w = _regime()

    regime = _feed(d, [_CLUSTERED] + [_UNIFORM] * (w - 1) + [_CLUSTERED])

    assert regime is CoverageRegime.SUBCRITICAL
    assert "biased exploration" in d.reason


def test_overlapping_rejections_do_not_escalate():
    # Adversarial: one burst stays in the sliding window for w ticks and
    # rejects on each; those w rejections share data and confirm nothing.
    d, w = _regime()

    regime = _feed(d, [_CLUSTERED] * w)

    assert regime is not CoverageRegime.SUBCRITICAL


def test_reset_forgets_pending_rejection():
    d, w = _regime()
    _feed(d, [_CLUSTERED] + [_UNIFORM] * (w - 1))

    d.reset()

    assert _feed(d, [_CLUSTERED]) is not CoverageRegime.SUBCRITICAL
