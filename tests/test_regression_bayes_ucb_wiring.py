"""Regression: ``op_bayes_ucb`` shipped uncabled.

No flag, no ballot entry, no ``_register_arms`` call: the module and its
tests existed, but no campaign could select it. ``--bayes-ucb`` now wires it
like every other operator scheduler.
"""

import pytest

from fuzzer_tool.core.schedulers import BayesUCBScheduler
from fuzzer_tool.services.fuzzer import _OPERATOR_STRATEGY_NAMES
from fuzzer_tool.services.operators import operator_strategy_pool
from tests.test_regression_resume_state import _fuzzer


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")
    monkeypatch.setattr("fuzzer_tool.core.elf.detect_ctx_bits", lambda _t: 4)


def test_regression_flag_builds_and_arms(tmp_path):
    f = _fuzzer(tmp_path, bayes_ucb=True)

    assert isinstance(f._bayes_ucb, BayesUCBScheduler)
    assert f._track_op_effect
    assert f._bayes_ucb.bandit_stats()["bayes_ucb_arms"] > 0
    assert "bayes_ucb" in operator_strategy_pool(f)


def test_off_by_default(tmp_path):
    """Falsification: the flag, not construction, is what enables it."""
    f = _fuzzer(tmp_path)

    assert f._bayes_ucb is None
    assert "bayes_ucb" not in operator_strategy_pool(f)


def test_ballot_name_registered():
    assert "bayes_ucb" in _OPERATOR_STRATEGY_NAMES


def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path):
    """Adversarial: a parsed flag the CLI forgets to forward is silently off."""
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
    argv = ["fuzzer-tool", "fuzz", str(target), "-d", str(tmp_path / "c"), "--bayes-ucb"]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(_Stop):
        commands.main()

    assert seen["bayes_ucb"] is True
