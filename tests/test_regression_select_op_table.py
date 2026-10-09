"""select_op dispatches the resolved strategy through a table, not an elif chain.

Each Elo-forced strategy must reach its own scheduler; a strategy whose
scheduler is missing, or that names nothing, falls back to the RNG.
"""

from types import SimpleNamespace

import pytest

from fuzzer_tool.services import operators as operators_mod
from fuzzer_tool.services.operators import OperatorEngine
from tests.test_regression_scheduler_fallback_precedence import (
    _FakeFuzzer,
    _RecordingRandPool,
)

_OPS = ["bit_flip", "byte_flip"]


def _forced(strategy: str) -> _FakeFuzzer:
    """Fake fuzzer whose Elo always picks *strategy*."""
    f = _FakeFuzzer()
    f._use_elo = True
    f._elo = SimpleNamespace(select_strategy=lambda available: strategy)
    return f


def _engine(f: _FakeFuzzer) -> tuple[OperatorEngine, _RecordingRandPool]:
    rng = _RecordingRandPool()
    f._rng = rng
    return OperatorEngine(f), rng


def _enable(f: _FakeFuzzer, name: str):
    """Enable *name*; op_credit's ballot entry also asks ``available()``."""
    fake = f.enable(name)
    fake.available = lambda: True
    return fake


def test_table_covers_plain_schedulers():
    """Every plain (ops)->op scheduler in the fake is in the dispatch table."""
    special = {"mopt", "contextual", "c2ucb", "op_firefly"}
    expected = set(_FakeFuzzer._SCHEDULER_ATTRS) - special
    assert expected <= set(operators_mod._PLAIN_STRATEGIES)


@pytest.mark.parametrize("name", sorted(operators_mod._PLAIN_STRATEGIES))
def test_plain_strategy_reaches_scheduler(name):
    """Falsification: forcing *name* consults exactly its scheduler."""
    f = _forced(name)
    fake = _enable(f, name)
    # A second scheduler on the ballot so Elo has two strategies to pick from.
    other = "exp3" if name != "exp3" else "exp4"
    decoy = _enable(f, other)
    eng, rng = _engine(f)

    assert eng.select_op(_OPS) == f"op_{name}"
    assert (fake.calls, decoy.calls, rng.choices) == (1, 0, [])
    assert f._last_mopt_particles == [None]


def test_missing_scheduler_falls_back_to_rng():
    """Adversarial: Elo names a strategy whose object is gone -> RNG, no crash."""
    f = _forced("fewa")
    # Three on the ballot: with one left Elo short-circuits to available[0].
    for name in ("fewa", "exp3", "exp4"):
        f.enable(name)
    f._fewa = None
    eng, rng = _engine(f)

    assert eng.select_op(_OPS) == _OPS[0]
    assert rng.choices == ["random"]
    assert f._last_mopt_particles == [None]


def test_unknown_strategy_falls_back_to_rng():
    """Adversarial: a name outside the table and the specials -> RNG."""
    f = _forced("no_such_strategy")
    f.enable("exp3")
    f.enable("exp4")
    eng, rng = _engine(f)

    assert eng.select_op(_OPS) == _OPS[0]
    assert rng.choices == ["random"]


def _count_pool(monkeypatch, fuzzer) -> dict:
    """Count operator_strategy_pool() builds for *fuzzer* only."""
    calls = {"n": 0}
    real = operators_mod.operator_strategy_pool

    def counting(f):
        calls["n"] += f is fuzzer
        return real(f)

    monkeypatch.setattr(operators_mod, "operator_strategy_pool", counting)
    return calls


@pytest.mark.parametrize("elo", [True, False])
def test_pool_built_once_per_exec(monkeypatch, elo):
    """Falsification: n mutations in one exec build the ballot once."""
    f = _forced("moss") if elo else _FakeFuzzer()
    _enable(f, "moss")
    _enable(f, "exp3")
    calls = _count_pool(monkeypatch, f)
    eng, _ = _engine(f)

    picks = [eng.select_op(_OPS) for _ in range(8)]
    assert calls["n"] == 1
    assert set(picks) == {"op_moss" if elo else "op_exp3"}

    # Exec boundary: mutate() clears the per-exec strategy cache.
    f._meta_strategy_cached = None
    eng.select_op(_OPS)
    assert calls["n"] == 2


def test_poisoned_cache_rebuilds_pool(monkeypatch):
    """Adversarial: a cached strategy the pool was not built for is a miss."""
    f = _forced("moss")
    moss = _enable(f, "moss")
    _enable(f, "exp3")
    calls = _count_pool(monkeypatch, f)
    eng, _ = _engine(f)

    eng.select_op(_OPS)
    f._meta_strategy_cached = "not_a_strategy"
    assert eng.select_op(_OPS) == "op_moss"
    assert (calls["n"], moss.calls) == (2, 2)


def test_empty_pool_never_caches(monkeypatch):
    """Adversarial: nothing enabled -> RNG each call, pool rebuilt each call."""
    f = _FakeFuzzer()
    calls = _count_pool(monkeypatch, f)
    eng, rng = _engine(f)

    assert [eng.select_op(_OPS) for _ in range(3)] == [_OPS[0]] * 3
    assert (calls["n"], rng.choices) == (3, ["random"] * 3)
