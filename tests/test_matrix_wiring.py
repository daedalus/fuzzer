"""Wiring of the matrix arms: ballot membership, gate abstention, seed-picker hook."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.edge_matrix import MatrixSubstrate
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_credit import OpCreditScheduler
from fuzzer_tool.core.schedulers.seed_residual import ResidualSeedScheduler
from fuzzer_tool.services.operators import operator_strategy_pool
from fuzzer_tool.services.seed_picker import SeedPicker
from tests.support.operator_env import install_scheduler_surface


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")


class _Tracker:
    def __init__(self, profiles):
        self.seed_hit_counts = profiles
        self.seed_edges = {k: set(v) for k, v in profiles.items()}
        self.cumulative_edges = set().union(*self.seed_edges.values())


PROFILES = {"a": {1: 1, 2: 1}, "b": {1: 1, 3: 1}, "c": {1: 1, 4: 1}}


def _fitted():
    sub = MatrixSubstrate(target="/x")
    sub.maybe_refit(_Tracker(PROFILES), 0)
    return sub


def _fuzzer_with_op_credit(sub):
    f = SimpleNamespace(mc=None)
    install_scheduler_surface(f)
    f._use_op_credit = True
    f._op_credit = OpCreditScheduler(RandPool(seed=1), sub)
    return f


class TestBallot:
    def test_op_credit_is_on_the_ballot_while_the_gate_is_open(self):
        assert "op_credit" in operator_strategy_pool(_fuzzer_with_op_credit(_fitted()))

    def test_op_credit_leaves_the_ballot_when_ids_are_unstable(self):
        # F1: under per-process ids every phantom id is a "discovery".
        sub = _fitted()
        f = _fuzzer_with_op_credit(sub)
        sub.set_stability(0.007)
        assert "op_credit" not in operator_strategy_pool(f)
        sub.set_stability(1.0)
        assert "op_credit" in operator_strategy_pool(f)

    def test_flag_off_means_absent(self):
        f = SimpleNamespace(mc=None)
        install_scheduler_surface(f)
        assert "op_credit" not in operator_strategy_pool(f)


class TestSeedPickerHook:
    def _picker(self, sub):
        f = SimpleNamespace(corpus=[b"a", b"b", b"c"], _seed_key=lambda s: s.decode())
        f._seed_residual = ResidualSeedScheduler(RandPool(seed=1), sub)
        p = SeedPicker.__new__(SeedPicker)
        p.f = f
        return p

    def test_returns_a_corpus_member_when_fitted(self):
        p = self._picker(_fitted())
        assert p._pick_residual_seed() in p.f.corpus

    def test_abstains_when_the_gate_is_closed(self):
        sub = _fitted()
        sub.set_stability(0.5)
        assert self._picker(sub)._pick_residual_seed() is None

    def test_abstains_before_any_fold_exists(self):
        assert self._picker(MatrixSubstrate(target="/x"))._pick_residual_seed() is None

    def test_absent_arm_declines(self):
        p = self._picker(_fitted())
        p.f._seed_residual = None
        assert p._pick_residual_seed() is None


_TARGET = Path(__file__).resolve().parent.parent / "targets" / "test_target"


def _instrumented() -> bool:
    if not _TARGET.exists():
        return False
    from fuzzer_tool.core.elf import sancov_guard_status

    return sancov_guard_status(str(_TARGET)) == "present"


@pytest.mark.skipif(not _instrumented(), reason="targets/test_target not built with --clang-scov")
def test_real_fuzzer_folds_classes_and_op_credit_learns(tmp_path, monkeypatch):
    """End to end on a compiler-instrumented target: the fold forms from the tracker,
    duplicate edges collapse, op_credit is elected, credit is in class units, and the
    seed arm has scored (no monkeypatched gate: the real ELF check says "present")."""
    monkeypatch.undo()  # this test wants the real sancov_guard_status, not the autouse stub
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir()
    crashes.mkdir()
    f = Fuzzer(
        target=str(_TARGET),
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        use_coverage=True,
        seed_residual=True,
        op_credit=True,
        elo=True,
        mc_bandit=True,
    )
    f._matrix_substrate.refit_interval = 20
    for i in range(64):
        f.fuzz_one(b"CRASH" + bytes([65 + i % 26]) + bytes([i % 251]) * (i % 40))
    sub = f._matrix_substrate
    assert sub.trusted and sub.fold is not None
    assert sub.fold.n_classes < sub.fold.n_edges  # duplicate-profile edges folded (F10)
    stats = f._seed_residual.stats()
    assert stats["fitted"] and "falsification" in stats
    assert "op_credit" in operator_strategy_pool(f)
    credit = {op: f._op_credit.credit(op) for op in f._op_credit._found}
    assert credit and all(v <= len(f._op_credit._found[op]) for op, v in credit.items())
