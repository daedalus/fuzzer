"""Regression: every learning operator scheduler turns on per-op attribution.

``Fuzzer._track_op_effect`` gates the per-operator buffer-change check that
decides which operators in a round actually changed the input. With it off,
``effective`` is None and every operator in the round -- including the ones
that changed nothing -- is credited with the round's outcome. The flag is
an ``or`` over a hand-written list of schedulers, and that list has lost
schedulers before: cmaes (see the comment at the list), and kl_ducb and
kl_swucb, which were absent, so ``--kl-ducb`` or ``--kl-swucb`` alone ran
with attribution off.

The test is driven off the ballot names in services.fuzzer, so a scheduler
added to the ballot and forgotten here fails by name.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.services.fuzzer import _OPERATOR_STRATEGY_NAMES, Fuzzer

_TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")
requires_test_target = pytest.mark.skipif(
    not Path(_TARGET).exists(), reason="targets/test_target not built"
)

#: Ballot name -> Fuzzer kwargs that enable it.
_KWARGS = {
    "replicator": {"replicator": True},
    "bandit": {"mc_bandit": True},
    "mopt": {"mopt": True},
    "cem": {"mc_cem": True, "mc_bandit": True},
    "exp3": {"exp3": True},
    "eps_greedy": {"eps_greedy": True},
    "hierarchical": {"hierarchical_bandit": True},
    "gp_ucb": {"gp_ucb": True},
    "contextual": {"contextual": True},
    "cmaes": {"cmaes": True},
    "ducb": {"ducb": True},
    "swucb": {"swucb": True},
    "kl_ducb": {"kl_ducb": True},
    "kl_swucb": {"kl_swucb": True},
    "cucb": {"cucb": True},
    "cusum_ucb": {"cusum_ucb": True},
    "c2ucb": {"c2ucb": True},
    "fpl": {"fpl": True},
    "invasion": {"invasion": True, "mc_bandit": True},
}

#: On the ballot but learns nothing, so attribution has no consumer.
_NON_LEARNING = {"round_robin"}


def test_every_ballot_name_is_mapped():
    missing = set(_OPERATOR_STRATEGY_NAMES) - set(_KWARGS) - _NON_LEARNING
    assert not missing, f"ballot schedulers with no kwargs mapping here: {sorted(missing)}"


@requires_test_target
@pytest.mark.parametrize("name", sorted(set(_OPERATOR_STRATEGY_NAMES) - _NON_LEARNING))
def test_enabling_it_alone_turns_attribution_on(name):
    with tempfile.TemporaryDirectory() as tmp:
        corpus, crashes = Path(tmp) / "c", Path(tmp) / "k"
        corpus.mkdir()
        crashes.mkdir()
        f = Fuzzer(
            target=_TARGET,
            corpus_dir=str(corpus),
            crashes_dir=str(crashes),
            max_len=4096,
            **_KWARGS[name],
        )
        assert f._track_op_effect, f"--{name} alone leaves per-op attribution off"
