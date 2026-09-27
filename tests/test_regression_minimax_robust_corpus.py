"""Regression: minimax-robust corpus selection (minimax Phase 5).

``_minimax_pick`` scored a seed's loss as ``|seed ∩ covered|`` (its size),
not the edges only it covers, so it could not tell a backup of a
high-unique seed from a useless duplicate. ``minimax_robust_pruning``
bounded its robustness pass by ``len(selected) < target_count`` -- a seed
count against an edge count -- and padded the corpus with seeds that
protect nothing. Both functions had no caller; now wired behind
``fuzz --minimax-select`` and ``minimize --minimax-robust``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fuzzer_tool.cli import commands
from fuzzer_tool.core.rate_distortion import RateDistortionCorpus
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.minimize import PruneMode
from tests.test_corpus_minimization import MockFuzzer, _cm_seed_key

BIG = set(range(1, 11))
SMALL = {11, 12}


def _max_unique_loss(seed_edges: dict[str, set[int]], keys: list[str]) -> int:
    """Largest edge count lost by dropping one of *keys* (independent oracle)."""
    worst = 0
    for k in keys:
        others: set[int] = set()
        for o in keys:
            if o != k:
                others |= seed_edges[o]
        worst = max(worst, len(seed_edges[k] - others))
    return worst


def test_regression_minimax_pick_uses_unique_loss():
    # Falsification: B2 (backup of the 2-edge seed) is iterated first. The
    # old size-based loss tied A2 and B2 at 10 and kept B2; the unique-loss
    # criterion must back up A (10 unique edges) instead.
    seed_edges = {"A": BIG, "B": SMALL, "B2": set(SMALL), "A2": set(BIG)}
    rd = RateDistortionCorpus()

    picked = rd.minimax_robust_corpus_admission(seed_edges, max_seeds=3)

    assert picked == ["A", "B", "A2"]
    assert _max_unique_loss(seed_edges, picked) == len(SMALL)


def test_regression_pruning_adds_no_useless_seed():
    # Z only touches the shared edge 5: it protects nothing. The old
    # seed-vs-edge bound (2 < 5 edges) padded it in anyway.
    seed_edges = {"A": {1, 2, 5}, "B": {3, 4, 5}, "Z": {5}}
    rd = RateDistortionCorpus()

    kept, frac = rd.minimax_robust_pruning(seed_edges, target_fraction=1.0)

    assert kept == ["A", "B"]
    assert frac == 1.0


def test_pruning_backs_up_worst_seed():
    seed_edges = {"A": BIG, "B": SMALL, "A2": set(BIG)}
    rd = RateDistortionCorpus()

    kept, _ = rd.minimax_robust_pruning(seed_edges, target_fraction=1.0)

    assert "A2" in kept
    assert _max_unique_loss(seed_edges, kept) < _max_unique_loss(seed_edges, ["A", "B"])


def test_admission_respects_preselected():
    # A is already kept (mandatory): the one free slot must back it up,
    # not re-cover B's edges.
    seed_edges = {"A": BIG, "B": SMALL, "A2": set(BIG), "B2": set(SMALL)}
    rd = RateDistortionCorpus()

    picked = rd.minimax_robust_corpus_admission(seed_edges, max_seeds=1, preselected=["A", "B"])

    assert picked == ["A2"]


def test_adversarial_identical_seeds_stop_at_one_backup():
    # Once no seed owns an edge alone, more copies protect nothing.
    seed_edges = {f"s{i}": set(BIG) for i in range(10)}
    rd = RateDistortionCorpus()

    picked = rd.minimax_robust_corpus_admission(seed_edges, max_seeds=5)

    assert len(picked) == 2


def test_adversarial_degenerate_inputs():
    rd = RateDistortionCorpus()

    assert rd.minimax_robust_corpus_admission({}, max_seeds=3) == []
    assert rd.minimax_robust_corpus_admission({"A": BIG}, max_seeds=0) == []
    assert rd.minimax_robust_corpus_admission({"E": set()}, max_seeds=3) == []
    assert rd.minimax_robust_pruning({"E": set()}) == ([], 1.0)


def _meta(coverage_edges: int) -> dict:
    return {
        "fuzz_count": 1,
        "coverage_edges": coverage_edges,
        "added_at": 100.0,
        "edge_bitmap": bytearray(0),
        "redqueen_offsets": [],
        "momentum": 0.0,
        "lineage_depth": 0,
        "hamming_distance": 0,
    }


def _backup_corpus(tmp_path: Path, minimax: bool) -> tuple[MockFuzzer, bytes, bytes, bytes]:
    """A (10 unique edges), its backup A2, and X (one shared edge, top score)."""
    f = MockFuzzer(tmp_path)
    f.max_corpus = 2
    f._use_minimax_select = minimax

    a, a2, x = b"A" * 40, b"a" * 40, b"X" * 40
    f.corpus = [a, a2, x]
    f.seed_meta = {a: _meta(10), a2: _meta(1), x: _meta(50)}
    et = f._edge_tracker
    et.record_edges(_cm_seed_key(a), set(BIG))
    et.record_edges(_cm_seed_key(a2), set(BIG))
    et.record_edges(_cm_seed_key(x), {1})
    return f, a, a2, x


def test_minimax_select_keeps_backup(tmp_path):
    f, a, a2, _x = _backup_corpus(tmp_path, minimax=True)

    CorpusManager(f).auto_minimize_corpus()

    assert a in f.corpus and a2 in f.corpus


def test_minimax_select_control_off(tmp_path):
    # Control: without the flag top-K keeps X, so the test above measures
    # the flag, not the fixture.
    f, _a, _a2, x = _backup_corpus(tmp_path, minimax=False)

    CorpusManager(f).auto_minimize_corpus()

    assert x in f.corpus


def test_minimize_cli_maps_minimax_flag(monkeypatch, tmp_path):
    target = tmp_path / "t"
    target.write_bytes(b"\x7fELF")
    target.chmod(0o755)
    captured: dict = {}
    monkeypatch.setattr(
        "fuzzer_tool.services.minimize.minimize_corpus",
        lambda **kw: captured.update(kw) or (1, 0),
    )
    argv = ["fuzzer-tool", "minimize", str(target), "-d", str(tmp_path), "-c", "--minimax-robust"]
    monkeypatch.setattr(sys, "argv", argv)

    commands.main()

    assert captured["prune"] is PruneMode.MINIMAX_ROBUST
