"""Flaky crashes: replay reproducibility lands in the sidecar and SARIF.

``--replay-n`` scored each crash but kept the rate in memory (report only);
a crash that reproduces 1 in 3 looked like any other on disk and in code
scanning. The rate and a ``flaky`` flag are now written once replays finish.
"""

from __future__ import annotations

import json

from fuzzer_tool.adapters.filesystem import load_crash_sidecars
from fuzzer_tool.core.sarif import build_sarif
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.stats_reporter import run_crash_replays
from tests.test_regression_crash_triage_cost import _Fuzzer

_ASAN = (
    "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010\n"
    "    #0 0x401111 in json_parse_ex\n"
)
_CLEAN = (0, "")
_CRASH = (1, _ASAN)


def _saved(tmp_path, monkeypatch):
    f = _Fuzzer(tmp_path)
    f._crash_files = {}
    cm = CorpusManager(f)
    monkeypatch.setattr(cm, "_novel_crash_meta", lambda *a, **k: None)
    assert cm.save_crash(b"in", 1, _ASAN)
    return f


def _replay(f, outcomes, n: int) -> dict[str, list[int]]:
    script = iter(outcomes)
    replays = {sig: [] for sig in f._crash_files}
    for _ in range(len(outcomes)):
        run_crash_replays(
            f.crashes_dir, "t", 1.0, replays, n, lambda d: "", 1e9, f._crash_files,
            run_fn=lambda data: next(script),
        )  # fmt: skip
    return replays


def _side(f) -> dict:
    (side,) = f.crashes_dir.glob("*.json")
    return json.loads(side.read_text())


def test_regression_flaky_rate_in_sidecar(tmp_path, monkeypatch) -> None:
    f = _saved(tmp_path, monkeypatch)
    _replay(f, [_CRASH, _CLEAN, _CRASH], 3)

    assert _side(f)["repro"] == {"crashed": 2, "runs": 3, "rate": 2 / 3, "flaky": True}
    (txt,) = f.crashes_dir.glob("*.txt")
    assert "repro_rate:    2/3 (flaky)" in txt.read_text()


def test_flaky_in_sarif(tmp_path, monkeypatch) -> None:
    f = _saved(tmp_path, monkeypatch)
    _replay(f, [_CLEAN, _CRASH], 2)

    (res,) = build_sarif(load_crash_sidecars(f.crashes_dir), "")["runs"][0]["results"]
    assert res["properties"]["flaky"] is True
    assert res["properties"]["repro_rate"] == 0.5


def test_stable_crash_not_flaky(tmp_path, monkeypatch) -> None:
    # Falsification: always reproducing is not flaky.
    f = _saved(tmp_path, monkeypatch)
    _replay(f, [_CRASH, _CRASH], 2)

    assert _side(f)["repro"]["flaky"] is False
    (res,) = build_sarif(load_crash_sidecars(f.crashes_dir), "")["runs"][0]["results"]
    assert res["properties"]["flaky"] is False


def test_unfinished_replays_not_recorded(tmp_path, monkeypatch) -> None:
    # Edge: 1 of 3 replays run: no verdict yet.
    f = _saved(tmp_path, monkeypatch)
    _replay(f, [_CLEAN], 3)

    assert "repro" not in _side(f)
    (res,) = build_sarif(load_crash_sidecars(f.crashes_dir), "")["runs"][0]["results"]
    assert "flaky" not in res["properties"]


def test_missing_sidecar_not_created(tmp_path, monkeypatch) -> None:
    # Adversarial: the .json was removed; replays must not invent one.
    f = _saved(tmp_path, monkeypatch)
    for p in f.crashes_dir.glob("*.json"):
        p.unlink()
    _replay(f, [_CRASH], 1)

    assert not list(f.crashes_dir.glob("*.json"))


def test_corrupt_sidecar_left_alone(tmp_path, monkeypatch) -> None:
    # Adversarial: unreadable JSON is not overwritten.
    f = _saved(tmp_path, monkeypatch)
    (side,) = f.crashes_dir.glob("*.json")
    side.write_text("{broken")
    _replay(f, [_CRASH], 1)

    assert side.read_text() == "{broken"
