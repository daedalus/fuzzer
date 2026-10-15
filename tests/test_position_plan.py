"""Position plan: proposer gates decided once per mutate(), not per operator call.

`select_position` re-tested ~10 feature gates (getattr/isinstance) on every
operator call. `_pos_plan()` binds the enabled proposers once per round;
`_plan_position()` must return exactly what `select_position` returns,
draw for draw, for every gate combination.
"""

from __future__ import annotations

import itertools

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

SEED = 1234
DRAWS = 64
BUF_LEN = 512
GATES = ("te", "mi", "sens", "crash", "field")


class _Proposer:
    """Draws from the shared RNG so evaluation order shows up in the output."""

    def __init__(self, rng: RandPool, offset: int):
        self._rng = rng
        self._offset = offset

    def pick(self, n: int) -> int:
        return (self._rng.randint(0, n - 1) + self._offset) % n


class _Fuzzer:
    """Mock exposing every source select_position consults."""

    def __init__(self, enabled: set[str]):
        self.max_len = 1 << 20
        self._rng = RandPool(seed=SEED)
        self.seed_meta = {}
        self._use_region_profile = False
        self._use_transfer_entropy = "te" in enabled
        self._te = _Proposer(self._rng, 1)
        self._use_mi = "mi" in enabled
        self._mi = _Proposer(self._rng, 2)
        self._use_sensitivity = "sens" in enabled
        self._sensitivity = _Proposer(self._rng, 3)
        self._crash_mi = _Proposer(self._rng, 4) if "crash" in enabled else None
        if self._crash_mi:
            self._crash_mi.total_execs = 10
            self._crash_mi.min_observations = 1
            self._crash_mi.weighted_position = self._crash_mi.pick
        self._format_learner = _Proposer(self._rng, 5) if "field" in enabled else None
        if self._format_learner:
            self._format_learner.weighted_position = lambda _d, n: self._format_learner.pick(n)
        self._mi.weighted_position = self._mi.pick
        self._sensitivity.get_weighted_position = lambda _d, n: self._sensitivity.pick(n)

    def _get_te_weighted_position(self, n: int) -> int:
        return self._te.pick(n)

    def _get_phase_weighted_position(self, _n: int, _stride) -> None:
        return None


def _old(enabled: set[str]) -> list[int]:
    f = _Fuzzer(enabled)
    eng = OperatorEngine(f)
    buf = bytearray(BUF_LEN)
    return [eng.select_position(buf, bytes(buf)) for _ in range(DRAWS)]


def _new(enabled: set[str]) -> list[int]:
    f = _Fuzzer(enabled)
    eng = OperatorEngine(f)
    buf = bytearray(BUF_LEN)
    plan = eng._pos_plan()
    return [eng._plan_position(plan, buf, bytes(buf)) for _ in range(DRAWS)]


COMBOS = [set(c) for r in range(len(GATES) + 1) for c in itertools.combinations(GATES, r)]


@pytest.mark.parametrize("enabled", COMBOS, ids=lambda e: "+".join(sorted(e)) or "none")
def test_control_old_matches_itself(enabled):
    """Control (Hard Rule 46): the oracle is deterministic under a fixed seed."""
    assert _old(enabled) == _old(enabled)


@pytest.mark.parametrize("enabled", COMBOS, ids=lambda e: "+".join(sorted(e)) or "none")
def test_plan_matches_select_position(enabled):
    """Falsification: same positions, same RNG draws, every gate combination."""
    assert _new(enabled) == _old(enabled)


def test_empty_buffer_is_zero():
    """Edge: an empty buffer has one position, 0, and draws nothing."""
    f = _Fuzzer(set(GATES))
    eng = OperatorEngine(f)
    assert eng._plan_position(eng._pos_plan(), bytearray(), b"") == 0


def test_plan_ignores_gate_flipped_after_build():
    """Adversarial: the plan is a snapshot; mutate() rebuilds it per round."""
    f = _Fuzzer({"mi"})
    eng = OperatorEngine(f)
    plan = eng._pos_plan()
    f._use_mi = False
    buf = bytearray(BUF_LEN)
    got = eng._plan_position(plan, buf, bytes(buf))
    assert got == _old({"mi"})[0]
    assert eng._pos_plan() == ()
