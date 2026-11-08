"""Bench arms and metric for the allocation-size A/B (`--alloc-sizes`).

The arm adds synthetic edges (bit 31 set), so its raw edge count is not
comparable with the baseline's; ``real_edges`` counts real ids only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import ARM_BASELINES, ARMS, _metric_arg, _real_edges  # noqa: E402

from fuzzer_tool.cli import commands  # noqa: E402
from fuzzer_tool.core.state_store import StateStore  # noqa: E402

SYNTH_BIT = 1 << 31


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


def _save(corpus: Path, edges: list[int]) -> None:
    store = StateStore(corpus)
    store.set("edge_tracker", {"cumulative_edges": sorted(edges)})
    store.save()


def test_falsify_arm_sets_only_alloc_sizes(monkeypatch):
    assert ARM_BASELINES["alloc-sizes"] == "baseline"
    assert _parse(monkeypatch, ARMS["alloc-sizes"]).alloc_sizes is True
    assert not _parse(monkeypatch, ARMS["baseline"]).alloc_sizes


def test_adversarial_control_is_baseline_exactly():
    """A/A control (Rule 46): any extra flag would make it a second arm."""
    assert ARMS["alloc-sizes-ctl"] == ARMS["baseline"]
    assert ARM_BASELINES["alloc-sizes-ctl"] == "baseline"


def test_real_edges_drops_synthetic_ids(tmp_path):
    """Falsification: bit-31 ids (alloc / COMPCOV marks) are not real edges."""
    real = [1, 7, 0x00FFFFFF]
    _save(tmp_path, real + [SYNTH_BIT | 5, SYNTH_BIT | 0x1234])
    assert _real_edges(tmp_path) == len(real)


def test_real_edges_equals_edges_without_synthetics(tmp_path):
    """Control: a state with no synthetic ids counts every edge."""
    edges = list(range(1, 50))
    _save(tmp_path, edges)
    assert _real_edges(tmp_path) == len(edges)


def test_adversarial_missing_state_is_none(tmp_path):
    """No state file: not a zero score, a missing value."""
    assert _real_edges(tmp_path) is None


def test_metric_accepted():
    assert _metric_arg("real_edges") == "real_edges"
