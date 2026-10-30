"""SARIF export hardening (PR #236 review): hostile crash dirs, bounded memory."""

from __future__ import annotations

import json
import os
import threading
import weakref
from argparse import Namespace

import pytest

from fuzzer_tool.adapters.filesystem import load_crash_sidecars
from fuzzer_tool.cli.commands import cmd_sarif
from fuzzer_tool.core import sarif
from fuzzer_tool.core.sarif import build_sarif
from fuzzer_tool.services.sarif_export import export_sarif

LOAD_TIMEOUT_S = 5.0
NESTING = 200_000  # far past the C JSON decoder's recursion limit


class _Side(dict):
    """Weak-referenceable sidecar, so a test can count live ones."""


def _side(cluster: str, tier: str, name: str) -> _Side:
    return _Side(exploitability=tier, cluster_id=cluster, crash_input=name)


def _write(d, base: str, cluster: str = "deadbeef") -> None:
    (d / f"{base}.json").write_text(json.dumps(_side(cluster, "HIGH", "")))


def _old_pick(sidecars: list[dict], k: int) -> list[dict]:
    """Frozen pre-change selection of build_sarif (the oracle)."""
    seen: dict[str, dict] = {}
    for side in sorted(sidecars, key=sarif._rank):
        key = sarif._str(side, "cluster_id") or str(len(seen))
        seen.setdefault(key, side)
    return list(seen.values())[:k]


def _inputs(doc: dict) -> list[str]:
    return [r["properties"]["crash_input"] for r in doc["runs"][0]["results"]]


def _mixed() -> list[dict]:
    """Repeats, a late better-tier repeat after eviction, and a cluster-less crash."""
    spec = [
        ("aa", "LOW"), ("bb", "MEDIUM"), ("cc", "LOW"), ("aa", "MEDIUM"),
        ("dd", "HIGH"), ("ee", "LOW"), ("cc", "CRITICAL"), ("", "HIGH"),
        ("ff", "BOGUS"), ("bb", "LOW"), ("gg", "CRITICAL"), ("ee", "HIGH"),
    ]  # fmt: skip
    return [_side(c, t, f"crash_{i}.bin") for i, (c, t) in enumerate(spec)]


def test_regression_fifo_sidecar_skipped(tmp_path):
    """A FIFO named crash_*.json must not block the loader."""
    _write(tmp_path, "crash_1_deadbeef_asan_uaf")
    fifo = tmp_path / "crash_2_cafebabe_asan_x.json"
    os.mkfifo(fifo)

    out: list[list[dict]] = []
    t = threading.Thread(target=lambda: out.append(load_crash_sidecars(tmp_path)), daemon=True)
    t.start()
    t.join(LOAD_TIMEOUT_S)
    try:
        assert not t.is_alive(), "loader blocked on a FIFO sidecar"
        assert [s["crash_input"] for s in out[0]] == ["crash_1_deadbeef_asan_uaf.bin"]
    finally:
        # Unblock a stuck reader so the thread exits.
        if t.is_alive():
            fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            os.close(fd)
            t.join(LOAD_TIMEOUT_S)


def test_regression_nested_sidecar_skipped(tmp_path):
    """Deep JSON nesting (RecursionError) skips one sidecar, not the export."""
    _write(tmp_path, "crash_1_deadbeef_asan_uaf")
    (tmp_path / "crash_2_cafebabe_asan_x.json").write_text("[" * NESTING)
    out = tmp_path / "o.sarif"
    assert export_sarif(tmp_path, out, "") == 1


def test_regression_out_overwrites_sidecar(tmp_path):
    """Output path naming an input sidecar is rejected; the sidecar survives."""
    _write(tmp_path, "crash_1_deadbeef_asan_uaf")
    target = tmp_path / "crash_1_deadbeef_asan_uaf.json"
    before = target.read_text()
    with pytest.raises(ValueError):
        export_sarif(tmp_path, target, "")
    assert target.read_text() == before


def test_regression_cli_rejects_sidecar_out(tmp_path, capsys):
    _write(tmp_path, "crash_1_deadbeef_asan_uaf")
    out = tmp_path / "crash_9_00000000_asan_x.json"
    args = Namespace(crashes_dir=str(tmp_path), output=str(out), src_root="")
    assert cmd_sarif(args) == 1
    assert not out.exists()
    assert "sidecar" in capsys.readouterr().err


def test_out_beside_crashes_still_allowed(tmp_path):
    """Adversarial: a non-sidecar name in the crash dir is fine."""
    _write(tmp_path, "crash_1_deadbeef_asan_uaf")
    assert export_sarif(tmp_path, tmp_path / "fuzz.sarif", "") == 1


def test_oracle_control():
    """Rule 46: the oracle must agree with itself."""
    a = _old_pick(_mixed(), 3)
    assert [s["crash_input"] for s in a] == [s["crash_input"] for s in _old_pick(_mixed(), 3)]
    assert len(a) == 3


@pytest.mark.parametrize("k", [1, 2, 3, 5, 100])
def test_selection_matches_oracle(monkeypatch, k):
    """Bounded selection keeps the pre-change result set and order."""
    monkeypatch.setattr(sarif, "MAX_RESULTS", k)
    want = [s["crash_input"] for s in _old_pick(_mixed(), k)]
    assert want
    assert _inputs(build_sarif(iter(_mixed()), "")) == want


def test_regression_sarif_memory_bounded(monkeypatch):
    """Sidecars past the cap are released while streaming."""
    k, n = 4, 400
    monkeypatch.setattr(sarif, "MAX_RESULTS", k)
    live = [0]
    peak = [0]

    def gen():
        for i in range(n):
            peak[0] = max(peak[0], live[0])
            s = _side(f"{i:08x}", "HIGH", f"crash_{i}.bin")
            live[0] += 1
            weakref.finalize(s, lambda: live.__setitem__(0, live[0] - 1))
            yield s

    doc = build_sarif(gen(), "")
    assert _inputs(doc) == [f"crash_{i}.bin" for i in range(k)]
    assert peak[0] <= 2 * k + 1
