"""Regression: bench_paired ``analyse --risk-matrix`` minimax pick (minimax Phase 2).

``EloTracker.select_minimax_scheduler`` had no caller and the risk matrix
printed by ``--risk-matrix`` fed nothing. Three defects made the matrix
unusable for a minimax pick even once wired:

* worst case over *seeds*: with two arms a per-seed regret is 0, 0.5 or 1,
  so the max is 1.0 for almost every arm and the pick degenerates to list
  order. The adversary is the target, not seed noise: mean over seeds,
  max over targets.
* no data read as 0.0 regret, so an arm never run on a target looked
  perfect there -- the least favourable reading for a minimax pick.
* the baseline arm was popped before the matrix was built.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import cmd_analyse, compute_risk_matrix, minimax_arm  # noqa: E402


def _rows(arm: str, target: str, edges: list[int]) -> list[dict]:
    return [
        {"arm": arm, "target": target, "seed": s, "rep": 0, "edges": e, "coverage_attached": True}
        for s, e in enumerate(edges)
    ]


def _loaded() -> dict[str, list[dict]]:
    # B iterated first. Per seed, A wins T1 every time and loses T2 on 2/3
    # seeds; B loses T1 every time. Worst case over seeds is 1.0 for both.
    return {
        "B": _rows("B", "T1", [1, 1, 1]) + _rows("B", "T2", [9, 9, 1]),
        "A": _rows("A", "T1", [5, 5, 5]) + _rows("A", "T2", [1, 1, 5]),
    }


def test_regression_mean_over_seeds_not_worst_seed():
    risk = compute_risk_matrix(_loaded())

    assert risk["A"] == {"T1": 0.0, "T2": 2 / 3}
    assert risk["B"] == {"T1": 1.0, "T2": 1 / 3}
    assert minimax_arm(risk) == ("A", 2 / 3)


def test_minimax_control_single_target():
    # Control: on one target minimax is plain argmin regret.
    risk = {"A": {"T": 0.4}, "B": {"T": 0.2}}

    assert minimax_arm(risk) == ("B", 0.2)


def test_adversarial_missing_target_is_worst_case():
    # C only ran on T1 (and won there): no evidence on T2 must not read as
    # zero regret, or C beats A on a target it never saw.
    loaded = _loaded()
    loaded["C"] = _rows("C", "T1", [9, 9, 9])

    risk = compute_risk_matrix(loaded)

    assert risk["C"]["T2"] == 1.0
    assert minimax_arm(risk)[0] != "C"


def test_adversarial_empty():
    assert compute_risk_matrix({}) == {}
    assert minimax_arm({}) == ("", 0.0)


def test_regression_analyse_includes_baseline(tmp_path, capsys):
    paths = []
    for arm, rows in _loaded().items():
        p = tmp_path / f"{arm}.json"
        p.write_text(json.dumps(rows))
        paths.append(str(p))
    args = argparse.Namespace(
        files=paths, baseline="A", metric="edges", by_target=False, risk_matrix=True
    )

    assert cmd_analyse(args) == 0

    out = capsys.readouterr().out
    assert "minimax-robust arm: A" in out
