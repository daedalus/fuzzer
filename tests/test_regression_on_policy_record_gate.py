"""Regression: on-policy schedulers learn only from rounds they selected.

``Fuzzer.fuzz_one`` fans each round's operator outcomes out to every
enabled scheduler. For most of them that is sound -- a sample mean or a Beta
posterior does not care who pulled the arm -- but three have updates that
are only valid for their own draws:

- Exp3 importance-weights each reward by 1/p_i, where p_i must be the
  probability Exp3 drew i with. Fed another scheduler's draw it divided by
  the ``_last_probs`` of the last round Exp3 itself had selected, which
  could be thousands of rounds stale.
- CMA-ES closes a generation after ``generation_size`` records; counting
  every scheduler's records closed it after roughly generation_size / N of
  its own evaluations, and credited its current candidate whenever another
  scheduler picked the op that candidate had last drawn.
- MOpt credits the particle that drew an op. It was already gated to its
  own rounds under Elo, but with Elo off it recorded every round with
  ``particle_id=None``, which spreads the outcome over all particles.

The gate is ``Fuzzer._op_selector``, set by ``select_op`` to whichever
scheduler chose (and cleared at the top of every ``mutate``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer

_TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")
requires_test_target = pytest.mark.skipif(
    not Path(_TARGET).exists(), reason="targets/test_target not built"
)

_ROUNDS = 40


def _build(**kwargs):
    tmp = tempfile.TemporaryDirectory()
    corpus = Path(tmp.name) / "corpus"
    crashes = Path(tmp.name) / "crashes"
    corpus.mkdir()
    crashes.mkdir()
    f = Fuzzer(
        target=_TARGET,
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        use_coverage=True,
        **kwargs,
    )
    f._test_tmp = tmp
    return f


def _count_records(f, attr):
    sched = getattr(f, attr)
    calls = []
    real = sched.record

    def spy(*args, **kwargs):
        calls.append(f._op_selector)
        return real(*args, **kwargs)

    sched.record = spy
    return calls


def _run(f):
    selectors = []
    for i in range(_ROUNDS):
        f.fuzz_one(bytes([65 + (i % 26)]) * 16)
        selectors.append(f._op_selector)
    return selectors


@requires_test_target
@pytest.mark.parametrize(
    ("kwargs", "attr", "name", "winner"),
    [
        # bandit precedes exp3 and cmaes; replicator precedes mopt.
        ({"mc_bandit": True, "exp3": True}, "_exp3", "exp3", "bandit"),
        ({"mc_bandit": True, "cmaes": True}, "_cmaes", "cmaes", "bandit"),
        ({"replicator": True, "mopt": True}, "_mopt", "mopt", "replicator"),
    ],
)
def test_not_fed_another_schedulers_rounds(kwargs, attr, name, winner):
    f = _build(**kwargs)
    calls = _count_records(f, attr)
    selectors = _run(f)
    assert winner in selectors, f"premise: {winner} never selected ({set(selectors)})"
    assert name not in selectors
    assert calls == [], f"{name} was fed {len(calls)} records from rounds it did not select"


@requires_test_target
@pytest.mark.parametrize(
    ("kwargs", "attr", "name"),
    [
        ({"exp3": True}, "_exp3", "exp3"),
        ({"cmaes": True}, "_cmaes", "cmaes"),
        ({"mopt": True}, "_mopt", "mopt"),
    ],
)
def test_still_learns_from_its_own_rounds(kwargs, attr, name):
    """Falsification for the above: the gate must not starve it entirely."""
    f = _build(**kwargs)
    calls = _count_records(f, attr)
    _run(f)
    assert calls, f"{name} selected but received no records"
    assert set(calls) == {name}


@requires_test_target
def test_off_policy_tolerant_schedulers_still_see_every_round():
    """The gate is for on-policy learners only: a Beta/mean learner keeps
    learning from rounds another scheduler selected."""
    f = _build(mc_bandit=True, hierarchical_bandit=True)
    calls = _count_records(f, "_hierarchical")
    selectors = _run(f)
    assert "hierarchical" not in selectors
    assert calls, "hierarchical stopped receiving other schedulers' rounds"
