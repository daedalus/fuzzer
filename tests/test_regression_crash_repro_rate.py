"""Regression: crash reproducibility counted ``rc >= 0`` as "reproduced".

``run_target_stdin`` returns ``-WTERMSIG`` for a signal death, so a SIGSEGV
that reproduces scored 0% and a clean ``exit 0`` scored 100%. Replays now
record a crash verdict and every consumer reads it through ``repro_rate``.
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest

from fuzzer_tool.services.stats_reporter import ReplayOutcome, repro_rate, run_crash_replays

SIG = "SIG"
ASAN_STDERR = "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n"


def _replay(tmp_path: Path, monkeypatch, rc: int, stderr: str) -> list[int]:
    """Replay one crash file once against a runner scripted to (rc, stderr)."""
    crashes = tmp_path / "crashes"
    crashes.mkdir()
    (crashes / "crash_1_c_sig_e.bin").write_bytes(b"X")
    monkeypatch.setattr(
        "fuzzer_tool.adapters.process.run_target_stdin",
        lambda *a, **kw: (rc, stderr, 1),
    )

    replays: dict[str, list[int]] = {SIG: []}
    run_crash_replays(
        crashes,
        "/bin/true",
        1.0,
        replays,
        replay_n=1,
        seed_key_fn=lambda d: d,
        budget_ms=10_000,
        crash_files={SIG: "crash_1_c_sig_e"},
    )
    return replays[SIG]


def test_regression_signal_crash_counts_as_reproduced(tmp_path, monkeypatch):
    replays = _replay(tmp_path, monkeypatch, -int(signal.SIGSEGV), "")
    assert repro_rate(replays) == 1.0


def test_regression_clean_exit_is_not_reproduced(tmp_path, monkeypatch):
    replays = _replay(tmp_path, monkeypatch, 0, "")
    assert repro_rate(replays) == 0.0


def test_asan_exit_counts_as_reproduced(tmp_path, monkeypatch):
    replays = _replay(tmp_path, monkeypatch, 1, ASAN_STDERR)
    assert repro_rate(replays) == 1.0


@pytest.mark.parametrize("rc", [-1, -2, 1])
def test_falsify_timeout_sentinel_or_plain_error(tmp_path, monkeypatch, rc):
    """Timeout, infra failure and a bare non-zero exit are not crashes."""
    replays = _replay(tmp_path, monkeypatch, rc, "")
    assert repro_rate(replays) == 0.0


def test_adversarial_target_prints_segfault_but_exits_zero(tmp_path, monkeypatch):
    """A target claiming a crash on stderr while exiting 0 did not crash."""
    replays = _replay(tmp_path, monkeypatch, 0, "Segmentation fault\n")
    assert repro_rate(replays) == 0.0


def test_missing_file_is_not_reproduced():
    assert repro_rate([ReplayOutcome.MISSING, ReplayOutcome.CRASHED]) == 0.5


def test_empty_replays_rate_is_zero():
    assert repro_rate([]) == 0.0


def test_regression_report_reads_verdicts():
    """The report used ``r >= 0``: a CLEAN replay (0) scored as reproduced."""
    from types import SimpleNamespace

    from fuzzer_tool.services.report import _crash_reproducibility

    f = SimpleNamespace(
        replay_n=2, _crash_replays={SIG: [ReplayOutcome.CLEAN, ReplayOutcome.CRASHED]}
    )
    expected = f"{repro_rate(f._crash_replays[SIG]):.0%}"
    assert f"Overall repro rate:   {expected}" in _crash_reproducibility(f)
