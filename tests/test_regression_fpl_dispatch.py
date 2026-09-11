"""Regression: ``--fpl`` selects operators.

FPLScheduler was constructed by ``--fpl`` (and by ``--elo all``), had its
arms registered, and was fed ``record()`` on every outcome -- but
``OperatorEngine.select_op`` had no ``fpl`` branch in either of its two
dispatch chains, and ``fpl`` was not on the ballot Elo picks from. So with
``--fpl`` alone selection fell through to whatever else was enabled (or to
the random terminal), and with ``--elo`` it could never be elected. The
flag's help text, "Enable Follow Perturbed Leader operator scheduling", was
true of nothing the fuzzer did.

It was invisible for the same reason cmaes's missing ballot entry was: each
piece looked right in isolation. The construction was there, the reward
wiring was there, the Elo opponent list even named it. Only the dispatch
was missing, and nothing asserted that an enabled scheduler ever chooses.
"""

from __future__ import annotations

from collections import Counter

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import FPLScheduler
from fuzzer_tool.services.operators import OperatorEngine, operator_strategy_pool
from tests.support.operator_env import make_minimal_fuzzer

_OPS = ["bit_flip", "byte_flip", "arith_inc"]


def _with_fpl(**flags):
    f = make_minimal_fuzzer(seed=3)
    f._use_fpl = True
    f._fpl = FPLScheduler(rng=RandPool(3))
    for op in _OPS:
        f._fpl.init_arm(op)
    for name, value in flags.items():
        setattr(f, name, value)
    return f


def test_fpl_alone_selects():
    f = _with_fpl()
    engine = OperatorEngine(f)
    engine.select_op(_OPS)
    assert f._op_selector == "fpl"


def test_fpl_choices_follow_its_own_estimates():
    """Not merely "fpl was named": the choices are FPL's. After one arm is
    credited heavily and the others fail, FPL exploits it -- the random
    terminal it used to fall through to would split ~1/3 each."""
    f = _with_fpl()
    engine = OperatorEngine(f)
    for op in _OPS:  # FPL opens unpulled arms first
        f._fpl.record(op, op == "arith_inc")
    for _ in range(50):
        f._fpl.record("arith_inc", True)
        f._fpl.record("bit_flip", False)
        f._fpl.record("byte_flip", False)
    picks = Counter(engine.select_op(_OPS) for _ in range(300))
    assert picks["arith_inc"] > 250, picks


def test_fpl_is_on_the_elo_ballot():
    f = _with_fpl()
    assert "fpl" in operator_strategy_pool(f)


def test_elo_can_elect_fpl():
    class _Elo:
        def select_strategy(self, available):
            assert "fpl" in available
            return "fpl"

    f = _with_fpl(_use_elo=True, _elo=_Elo(), _use_ducb=True, _ducb=object())
    OperatorEngine(f).select_op(_OPS)
    assert f._meta_strategy == "fpl"
    assert f._op_selector == "fpl"
