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


# ── Same crash, not just a crash ─────────────────────────────────────────

ASAN_UAF = (
    "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n"
    "    #0 0x401111 in parse_chunk\n"
)


def _replay_as(fuzzer, sig: str, rc: int, stderr: str) -> list[int]:
    fuzzer._crash_replays = {sig: []}
    fuzzer._crash_files = {sig: BASE}
    fuzzer._runner.run_target = lambda data: (rc, stderr)
    StatsReporter.run_crash_replays(SimpleNamespace(f=fuzzer), budget_ms=10_000)
    return fuzzer._crash_replays[sig]


def test_regression_replay_other_bug_diverged(fuzzer) -> None:
    # Saved as a UAF; the replay hits a plain SIGSEGV: not reproduced.
    sig = "AddressSanitizer:heap-use-after-free@parse_chunk"

    assert _replay_as(fuzzer, sig, -int(signal.SIGSEGV), "") == [ReplayOutcome.DIVERGED]


def test_replay_same_bug_class_reproduced(fuzzer) -> None:
    # Falsification: same sanitizer:type, frames may differ.
    sig = "AddressSanitizer:heap-use-after-free@other_frame"

    assert _replay_as(fuzzer, sig, 1, ASAN_UAF) == [ReplayOutcome.CRASHED]


def test_replay_signal_mode_forms_match(fuzzer) -> None:
    # Edge: saved in-process as signal:11, replayed as 128+11.
    assert _replay_as(fuzzer, "signal:11@0x0", 128 + int(signal.SIGSEGV), "") == [
        ReplayOutcome.CRASHED
    ]


def test_replay_legacy_key_not_compared(fuzzer) -> None:
    # Edge: content-hash keys from older states carry no bug class.
    assert _replay_as(fuzzer, "a3f09c2b11d4e5f6", 1, ASAN_UAF) == [ReplayOutcome.CRASHED]


def test_report_counts_diverged_replays() -> None:
    from fuzzer_tool.services.report import _crash_reproducibility

    f = SimpleNamespace(
        _crash_replays={"signal:11": [ReplayOutcome.CRASHED, ReplayOutcome.DIVERGED]},
        replay_n=2,
    )
    text = _crash_reproducibility(f)

    assert "signal:11: 50% (2 replays, 1 as another bug)" in text
