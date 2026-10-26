"""Regression: crash replays bypassed the campaign's execution path.

``run_crash_replays`` always spawned ``target`` with the input on stdin, no
environment and no arguments: a ``.so`` target cannot be exec'd (every replay
scored as not reproduced) and ``@@`` file targets never read the input.
"""

from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest

from fuzzer_tool.services.stats import StatsReporter
from fuzzer_tool.services.stats_reporter import ReplayOutcome

SIG = "SIG"
BASE = "crash_1_c_sig_e"
INPUT = b"CRASH-INPUT"


class _Runner:
    """The fuzzer's TargetRunner, scripted to crash."""

    def __init__(self) -> None:
        self.seen: list[bytes] = []

    def run_target(self, data: bytes) -> tuple[int, str]:
        self.seen.append(data)
        return -int(signal.SIGSEGV), ""


@pytest.fixture
def fuzzer(tmp_path):
    crashes = tmp_path / "crashes"
    crashes.mkdir()
    (crashes / f"{BASE}.bin").write_bytes(INPUT)
    return SimpleNamespace(
        crashes_dir=crashes,
        target=str(tmp_path / "target.so"),
        timeout=1.0,
        _crash_replays={SIG: []},
        replay_n=1,
        _seed_key=lambda d: d,
        _crash_files={SIG: BASE},
        _runner=_Runner(),
    )


def test_regression_replay_uses_campaign_runner(fuzzer, monkeypatch) -> None:
    def _spawn(*_a, **_kw):
        raise AssertionError("replay spawned the target directly")

    monkeypatch.setattr("fuzzer_tool.adapters.process.run_target_stdin", _spawn)
    StatsReporter.run_crash_replays(SimpleNamespace(f=fuzzer), budget_ms=10_000)

    assert fuzzer._runner.seen == [INPUT]
    assert fuzzer._crash_replays[SIG] == [ReplayOutcome.CRASHED]


def test_replay_clean_run_not_reproduced(fuzzer) -> None:
    # Falsification: the runner's verdict decides, a clean exit is CLEAN.
    fuzzer._runner.run_target = lambda data: (0, "")
    StatsReporter.run_crash_replays(SimpleNamespace(f=fuzzer), budget_ms=10_000)

    assert fuzzer._crash_replays[SIG] == [ReplayOutcome.CLEAN]
