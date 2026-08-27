"""Hard Rule 16: schedulers draw randomness from ``RandPool``, not ``random``.

Two defects are pinned here.

**Rule 16.** ``DUCBScheduler``, ``SWUCBScheduler`` and ``CUCBScheduler`` broke
ties and opened unpulled arms with the module-level ``random``. The symptom is
not a wrong distribution -- it is that ``--seed`` stops determining the
campaign, so a crash found under one of these schedulers cannot be replayed.
``CMAESScheduler`` already takes ``rng: RandPool | None`` and is the
convention these follow. The six older schedulers still use the module-level
``random`` and are deliberately *not* covered: ``tests/support/bandit_env.py``
seeds the global PRNG precisely because of that, and converting them is a
separate change with its own harness impact.

**Use-before-assignment.**

``Fuzzer.__init__`` passed ``rng=self._rand_pool`` to ``CMAESScheduler``
roughly 9,000 characters before ``self._rand_pool`` was assigned, so
``--cma-es`` raised ``AttributeError`` at construction. The comment on that
argument explains why the pool has to be threaded in (Hard Rule 16: an
unseeded scheduler means a crash found under it cannot be replayed from
``--seed``); nothing checked that it existed yet.

Position, not presence, is what is asserted -- a source-order check rather
than a construction test, because constructing a real ``Fuzzer`` needs a
target binary and the defect is purely one of ordering.
"""

import ast
import inspect
import pathlib
import re

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.cucb import CUCBScheduler
from fuzzer_tool.core.schedulers.ducb import DUCBScheduler
from fuzzer_tool.core.schedulers.swucb import SWUCBScheduler

SCHEDULERS_DIR = (
    pathlib.Path(__file__).parent.parent / "src" / "fuzzer_tool" / "core" / "schedulers"
)

#: Schedulers added with the RandPool convention in force from the start.
RULE_16_SCHEDULERS = {
    "ducb.py": DUCBScheduler,
    "swucb.py": SWUCBScheduler,
    "cucb.py": CUCBScheduler,
}

#: Enough arms that a tie-break or unpulled-arm draw is overwhelmingly
#: unlikely to coincide across two independent PRNG streams.
ARMS = [f"op_{i:02d}" for i in range(24)]

#: Long enough that every arm leaves the unpulled state and the index, not the
#: initialisation draw, is doing the selecting.
PULLS = 200


def _source(name: str) -> str:
    return (SCHEDULERS_DIR / name).read_text()


def _settle(scheduler) -> None:
    settle = getattr(scheduler, "settle_round", None)
    if settle is not None:
        settle()


class TestNoModuleRandom:
    """Static: the module-level PRNG must not appear in these sources."""

    @pytest.mark.parametrize("name", sorted(RULE_16_SCHEDULERS))
    def test_does_not_import_random(self, name):
        tree = ast.parse(_source(name))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert "random" not in imported, (
            f"{name} imports the module-level `random`; Hard Rule 16 requires RandPool"
        )

    @pytest.mark.parametrize("name", sorted(RULE_16_SCHEDULERS))
    def test_no_bare_random_calls(self, name):
        hits = re.findall(r"(?<![\w.])random\.\w+", _source(name))
        assert not hits, f"{name} calls {sorted(set(hits))} instead of drawing from RandPool"

    @pytest.mark.parametrize("name,cls", sorted(RULE_16_SCHEDULERS.items()))
    def test_accepts_injected_rng(self, name, cls):
        params = inspect.signature(cls.__init__).parameters
        assert "rng" in params, f"{cls.__name__} takes no rng= argument (see cmaes.py)"
        assert params["rng"].default is None, (
            f"{cls.__name__}'s rng default must be None so the fuzzer's shared pool wins"
        )


class TestSeedDeterminism:
    """Behavioural: same injected seed, same selection sequence."""

    @pytest.mark.parametrize("cls", sorted(RULE_16_SCHEDULERS.values(), key=lambda c: c.__name__))
    def test_same_seed_same_sequence(self, cls):
        def drive(seed):
            s = cls(rng=RandPool(seed=seed))
            for a in ARMS:
                s.init_arm(a)
            picks = []
            for i in range(PULLS):
                op = s.select_op(ARMS)
                picks.append(op)
                s.record(op, success=(i % 7 == 0))
                _settle(s)
            return picks

        assert drive(4242) == drive(4242)

    @pytest.mark.parametrize("cls", sorted(RULE_16_SCHEDULERS.values(), key=lambda c: c.__name__))
    def test_different_seed_diverges(self, cls):
        """Falsification: a scheduler ignoring its rng passes the test above
        trivially by being deterministic. It must also respond to the seed."""

        def drive(seed):
            s = cls(rng=RandPool(seed=seed))
            for a in ARMS:
                s.init_arm(a)
            picks = []
            for _ in range(len(ARMS)):
                op = s.select_op(ARMS)
                picks.append(op)
                s.record(op, success=False)
                _settle(s)
            return picks

        # The opening pulls are pure unpulled-arm draws, so the orders differ
        # unless the rng is being ignored.
        assert drive(1) != drive(2)


class TestGlobalRandomIsolation:
    """Adversarial: perturbing the global PRNG must not move these schedulers."""

    @pytest.mark.parametrize("cls", sorted(RULE_16_SCHEDULERS.values(), key=lambda c: c.__name__))
    def test_global_seed_does_not_leak_in(self, cls):
        import random as _global_random

        def drive(global_seed):
            _global_random.seed(global_seed)
            s = cls(rng=RandPool(seed=99))
            for a in ARMS:
                s.init_arm(a)
            picks = []
            for i in range(PULLS):
                # Churn the global stream between selections: a scheduler
                # still touching it will desynchronise.
                _global_random.random()
                op = s.select_op(ARMS)
                picks.append(op)
                s.record(op, success=(i % 5 == 0))
                _settle(s)
            return picks

        assert drive(1) == drive(987654321)


class TestRandPoolAvailableBeforeSchedulers:
    """``rng=self._rand_pool`` read before the assignment that creates it."""

    def test_rand_pool_assigned_before_first_use(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        src = inspect.getsource(Fuzzer.__init__)
        assign = src.find("self._rand_pool = RandPool")
        assert assign != -1, "Fuzzer.__init__ no longer constructs its RandPool"

        first_use = min(
            (m.start() for m in re.finditer(r"rng=self\._rand_pool", src)),
            default=None,
        )
        if first_use is None:
            pytest.skip("no scheduler currently receives the shared pool")
        assert assign < first_use, (
            "self._rand_pool is read at char "
            f"{first_use} but only assigned at char {assign}: "
            "constructing that scheduler raises AttributeError"
        )
