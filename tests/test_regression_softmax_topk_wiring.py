"""Regression: ``--softmax`` / ``--topk`` select, learn, and attribute.

Both were constructed, on the Elo ballot and in ``operator_strategy_pool``,
but had no ``select_op`` branch (a pick fell through to the uniform
terminal), no ``record`` fan-out, no ``_register_arms`` call and no
``_track_op_effect`` term. An Elo pick of either ran the uniform terminal
under its name. Like fewa, neither is in ``_FALLBACK_PRECEDENCE``: both
are reached only through Elo.
"""

from __future__ import annotations

from collections import Counter

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import SoftmaxScheduler, TopKScheduler
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer

_OPS = ["bit_flip", "byte_flip", "arith_inc"]
_BEST = "arith_inc"

#: ballot name -> (attr, factory, Fuzzer kwargs)
_CASES = {
    "softmax": ("_softmax", lambda: SoftmaxScheduler(tau=0.05, rng=RandPool(3)), {"softmax": True}),
    "topk": ("_topk", lambda: TopKScheduler(k=1, rng=RandPool(3)), {"use_topk": True}),
}


class _Elo:
    """Elects *name* whenever it is on the ballot."""

    def __init__(self, name):
        self._name = name

    def select_strategy(self, available):
        assert self._name in available
        return self._name


def _with(name):
    attr, factory, _ = _CASES[name]
    f = make_minimal_fuzzer(seed=3)
    setattr(f, f"_use{attr}", True)
    setattr(f, attr, factory())
    # A second ballot entry so Elo actually votes.
    f._use_ducb, f._ducb = True, object()
    f._use_elo, f._elo = True, _Elo(name)
    return f, getattr(f, attr)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_regression_softmax_topk_elo_elects(name):
    f, _ = _with(name)
    OperatorEngine(f).select_op(_OPS)
    assert f._op_selector == name


@pytest.mark.parametrize("name", sorted(_CASES))
def test_regression_softmax_topk_choices_follow_estimates(name):
    # FALSIFICATION: the uniform terminal splits ~1/3 each.
    f, sched = _with(name)
    for _ in range(50):
        for op in _OPS:
            sched.record(op, op == _BEST)
    engine = OperatorEngine(f)
    picks = Counter(engine.select_op(_OPS) for _ in range(300))
    assert picks[_BEST] > 250, picks


@pytest.mark.parametrize("name", sorted(_CASES))
def test_regression_softmax_topk_fuzzer_wiring(name, tmp_path):
    # Constructed alone: arms registered, attribution on, rounds recorded.
    from fuzzer_tool.services.fuzzer import Fuzzer

    attr, _, kwargs = _CASES[name]
    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir()
    crashes.mkdir()
    f = Fuzzer(
        target="/bin/true",
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=256,
        timeout=1,
        **kwargs,
    )
    sched = getattr(f, attr)
    assert f._track_op_effect, f"--{name} alone leaves per-op attribution off"
    assert sched.bandit_stats(), "arms never registered"

    for i in range(10):
        f.fuzz_one(bytes([65 + i]) * 16)
    # Off-policy sample means: every round is recorded, whoever selected.
    assert sum(c for _, c in sched.bandit_stats().values()) > 0, "never recorded"
