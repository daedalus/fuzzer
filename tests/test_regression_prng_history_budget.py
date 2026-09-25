"""Regression: the PRNG learner's replay history grew with comparison volume.

``_run_history`` capped inputs (256) but not what each keeps: one record per
comparison site the input hit, with every value seen there. On ffmpeg
(non-ASAN) that was ~1.9 MB per input and +337 MB between 1.5k and 3.5k
execs. A total budget (one unit per site, one per value) now bounds it; the
input being recorded is never the one evicted.
"""

from fuzzer_tool.core.analyzers import analyzer_prng_state_learner as mod
from tests.test_prng_state_learner import PAYLOAD, ZERO_FIELD, _conds, _learner, _stream

_SITES = 64
_INPUTS = 50
_BUDGET = 500


def _units(learner) -> int:
    return sum(1 + len(e.values) for h in learner._run_history.values() for e in h.values())


def _wide_drain(i: int) -> list:
    """One execution comparing _SITES different PCs, one value each."""
    return [c for s in range(_SITES) for c in _conds([0x1000 + i * _SITES + s], pc=0x5000 + s)]


def test_regression_prng_history_budget(monkeypatch):
    """Falsification: many site-heavy inputs stay within the budget."""
    monkeypatch.setattr(mod, "_RUN_HISTORY_BUDGET", _BUDGET)
    learner = _learner()

    for i in range(_INPUTS):
        learner.f._cmplog.last_conds = _wide_drain(i)
        learner.observe_execution(ZERO_FIELD + f"input-{i}".encode())

    assert _units(learner) <= _BUDGET


def test_replayed_input_survives_tiny_budget(monkeypatch):
    """Adversarial: a budget below one input's cost keeps the current input,
    so a replay still exposes the varying site."""
    monkeypatch.setattr(mod, "_RUN_HISTORY_BUDGET", 1)
    words = _stream(8)
    learner = _learner()

    learner.f._cmplog.last_conds = _conds(words[:4], pc=0x1000)
    learner.observe_execution(PAYLOAD)
    learner.f._cmplog.last_conds = _conds(words[4:8], pc=0x1000)
    learner.observe_execution(PAYLOAD)

    assert list(learner._run_history) == [hash(PAYLOAD)]
    assert (0x1000, 4) in learner._varied_sites()
