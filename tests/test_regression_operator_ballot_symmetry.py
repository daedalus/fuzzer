"""Regression: the Elo opponents are exactly the ballot select_op offered.

The operator meta-strategy has two sides. ``OperatorEngine.select_op`` builds
a ballot and asks Elo to pick from it; ``Fuzzer._record_operator_strategy_matches``
then plays the picked strategy against the others. Both sides used to keep
their own hand-written list, and the lists had drifted apart in both
directions:

- ``kl_ducb`` and ``kl_swucb`` were offered but never played as opponents,
  so their ratings moved only on the games they were selected for (the
  same asymmetry already fixed once for cmaes, see
  ``test_regression_cmaes_elo_ballot``).
- ``fpl`` was played as an opponent but never offered: a phantom that
  collected ratings for a scheduler nothing could select.

The fix is one function, ``operators.operator_strategy_pool``, read by both
sides. These tests assert the property rather than the list, so the next
scheduler added cannot reintroduce the drift on either side.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import BALLOT_SCHEDULERS, make_minimal_fuzzer


class _Pick:
    """A scheduler stand-in whose select_op returns a candidate op."""

    def select_op(self, ops, *args):
        return ops[0]


class _Mopt(_Pick):
    def select_op(self, ops, *args):
        return ops[0], 0


class _Elo:
    """Records the ballot it was offered and every match it was asked to rate."""

    def __init__(self, pick):
        self.pick = pick
        self.ballots: list[list[str]] = []
        self.matches: list[tuple[str, str]] = []

    def select_strategy(self, available):
        self.ballots.append(list(available))
        return self.pick if self.pick in available else available[0]

    def record_strategy_match(self, a, b, score):
        self.matches.append((a, b))


def _fuzzer_with(enabled: list[str], pick: str):
    f = make_minimal_fuzzer(seed=1)
    for name in enabled:
        setattr(f, f"_use_{name}", True)
        setattr(f, f"_{name}", _Mopt() if name == "mopt" else _Pick())
    f._use_elo = True
    f._elo = _Elo(pick)
    return f


# Every scheduler the shared test surface knows about, minus the ones that
# are not operator schedulers (tang schedules seeds).
_OPERATOR_SCHEDULERS = sorted(set(BALLOT_SCHEDULERS) - {"tang"})


@pytest.mark.parametrize("name", _OPERATOR_SCHEDULERS)
def test_opponents_equal_the_offered_ballot(name):
    """Enable *name* plus one other; the recorded opponents of the selected
    strategy must be exactly the ballot minus itself -- no more, no fewer."""
    # Two others, so the ballot has at least two entries (and Elo is asked)
    # whether or not *name* itself makes it onto the ballot.
    others = [o for o in ("ducb", "swucb", "hierarchical") if o != name][:2]
    f = _fuzzer_with([name, *others], pick=others[0])

    OperatorEngine(f).select_op(["bit_flip", "byte_flip"])
    offered = f._elo.ballots[-1]
    Fuzzer._record_operator_strategy_matches(f, 0.0)

    opponents = {b for a, b in f._elo.matches if a == f._meta_strategy}
    assert opponents == set(offered) - {f._meta_strategy}, (
        f"offered {offered}, but {f._meta_strategy!r} was played against {sorted(opponents)}"
    )


@pytest.mark.parametrize("name", ["kl_ducb", "kl_swucb"])
def test_kl_schedulers_play_the_games_they_are_not_selected_for(name):
    """The concrete defect: offered, selected against, and never an opponent."""
    f = _fuzzer_with([name, "ducb"], pick="ducb")
    OperatorEngine(f).select_op(["bit_flip"])
    assert name in f._elo.ballots[-1]
    Fuzzer._record_operator_strategy_matches(f, 0.0)
    assert ("ducb", name) in f._elo.matches


def test_no_opponent_is_off_the_ballot():
    """With everything the surface knows enabled, the opponent set is a
    subset of what select_op offered: nothing plays that cannot be picked."""
    f = _fuzzer_with(_OPERATOR_SCHEDULERS, pick="ducb")
    OperatorEngine(f).select_op(["bit_flip"])
    offered = set(f._elo.ballots[-1])
    Fuzzer._record_operator_strategy_matches(f, 0.0)
    phantoms = {b for _, b in f._elo.matches} - offered
    assert not phantoms, f"played as opponents but never offered: {sorted(phantoms)}"


def test_both_sides_read_one_function():
    """The property above holds because there is a single list; pin that the
    recorder's opponents are that function's output, not a copy of it."""
    from fuzzer_tool.services.operators import operator_strategy_pool

    f = _fuzzer_with(["exp3", "hierarchical"], pick="exp3")
    f._meta_strategy = "exp3"
    f._meta_strategy_used = {"exp3"}
    Fuzzer._record_operator_strategy_matches(f, 1.0)
    assert {b for _, b in f._elo.matches} == set(operator_strategy_pool(f)) - {"exp3"}
