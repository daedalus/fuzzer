"""Wiring of core/edge_stats.py: EdgeTracker API and the run summary."""

import io
import types
from contextlib import redirect_stdout

import pytest

from fuzzer_tool.core.edge_stats import BOOT_RESAMPLES, KEXEC, rarefy
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.stats import StatsReporter
from tests.support.scripted_rng import ScriptedRng

SEEDS = [{1, 2, 3}, {2, 3}, {3, 4}, {5}]
HEAD, TAIL, BREAK, POINTS = 20, 0, 6, 12


def _tracker() -> EdgeTracker:
    et = EdgeTracker()
    for i, edges in enumerate(SEEDS):
        et.record_edges(f"s{i}", edges)
    for i in range(POINTS):
        et._coverage_execs.append(i * KEXEC)
        et._coverage_edges.append(HEAD * min(i, BREAK) + TAIL * max(i - BREAK, 0))
    return et


def _summary(et: EdgeTracker, rng) -> str:
    reporter = StatsReporter.__new__(StatsReporter)
    buf = io.StringIO()
    with redirect_stdout(buf):
        reporter._print_summary_edge_stats(types.SimpleNamespace(_edge_tracker=et, _rng=rng))
    return buf.getvalue()


def _all_seeds_draws() -> list[list[int]]:
    return [list(range(len(SEEDS)))] * BOOT_RESAMPLES


def test_tracker_methods_delegate():
    et = _tracker()
    owners = [et.edge_owner_count(e) for e in sorted(et.cumulative_edges)]

    assert et.hit_profile().edges == len(et.cumulative_edges)
    assert et.coverage_plateau().exec_at == BREAK * KEXEC
    assert et.rarefied_edges(2) == pytest.approx(rarefy(owners, len(SEEDS), [2])[0])
    assert et.seed_robustness(ScriptedRng(randrange_lists=_all_seeds_draws())) == (1.0, 1.0)


def test_summary_prints_edge_stats():
    out = _summary(_tracker(), ScriptedRng(randrange_lists=_all_seeds_draws()))

    assert "Hit share:" in out
    assert f"Plateau:           from exec {BREAK * KEXEC:,}" in out
    assert "Half corpus:" in out
    assert "Seed resample:     100-100% of edges survive" in out


def test_summary_silent_on_empty_tracker():
    # Adversarial: nothing recorded must print nothing and draw nothing.
    assert _summary(EdgeTracker(), ScriptedRng()) == ""


def test_summary_skips_plateau_on_speedup():
    # Adversarial: a faster tail is not a plateau.
    et = _tracker()
    for i in range(POINTS):
        et._coverage_edges[i] = i if i < BREAK else BREAK + HEAD * (i - BREAK)
    out = _summary(et, ScriptedRng(randrange_lists=_all_seeds_draws()))
    assert "Plateau:" not in out


def test_robustness_skips_over_budget(monkeypatch):
    # Adversarial: a corpus above the incidence budget skips the bootstrap.
    from fuzzer_tool.core import edge_tracker as mod

    monkeypatch.setattr(mod, "BOOT_NNZ_BUDGET", 1)
    assert _tracker().seed_robustness(ScriptedRng()) is None
