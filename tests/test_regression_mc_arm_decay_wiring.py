"""``--mc-arm-decay`` reaches MonteCarloScheduler.arm_decay.

Only ``--mc-decay-interval`` was wired: the decay factor stayed at the
constructor default (0.999 per 100 records, a ~69k-record half-life), so
the bandit's forgetting could not be tuned from the CLI.
"""

import inspect

import pytest

from fuzzer_tool.core.schedulers import MonteCarloScheduler
from tests.test_regression_resume_state import _fuzzer


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")
    monkeypatch.setattr("fuzzer_tool.core.elf.detect_ctx_bits", lambda _t: 4)


def _cli_kwargs(monkeypatch, tmp_path, *flags):
    """Fuzzer kwargs ``fuzzer-tool fuzz`` builds from *flags*."""
    from fuzzer_tool.cli import commands

    seen = {}

    class _Stop(Exception):
        pass

    def fake_fuzzer(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(commands, "Fuzzer", fake_fuzzer)
    target = tmp_path / "t"
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    argv = ["fuzzer-tool", "fuzz", str(target), "-d", str(tmp_path / "c"), *flags]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(_Stop):
        commands.main()
    return seen


def test_regression_cli_flag_reaches_fuzzer(monkeypatch, tmp_path):
    kw = _cli_kwargs(monkeypatch, tmp_path, "--mc-bandit", "--mc-arm-decay", "0.95")

    assert kw["mc_arm_decay"] == 0.95


def test_regression_fuzzer_builds_scheduler_with_it(tmp_path):
    f = _fuzzer(tmp_path, mc_bandit=True, mc_arm_decay=0.95)

    assert f.mc.arm_decay == 0.95


def test_default_matches_the_scheduler(monkeypatch, tmp_path):
    """Falsification: omitting the flag keeps the scheduler's own default."""
    default = inspect.signature(MonteCarloScheduler).parameters["arm_decay"].default

    kw = _cli_kwargs(monkeypatch, tmp_path, "--mc-bandit")

    assert kw["mc_arm_decay"] == default


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan")])
def test_out_of_range_decay_is_rejected(bad):
    """Adversarial: > 1 grows evidence away from the prior; 0 or NaN erases it."""
    with pytest.raises(ValueError, match="arm_decay"):
        MonteCarloScheduler(arm_decay=bad)
