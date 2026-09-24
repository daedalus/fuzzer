"""Resume state for op_credit, PLL, burn-front and the WFC chunk tables.

Each learner used to live in process memory only, so ``--resume`` restarted
it cold. Every block here checks a round trip (falsification: the restored
object behaves like the original) and a malformed payload (adversarial: the
loader starts fresh instead of raising).
"""

import math

import pytest

from fuzzer_tool.core.analyzers.analyzer_pll import PLLMonitor, Series, Stall
from fuzzer_tool.core.edge_matrix import MatrixSubstrate
from fuzzer_tool.core.pll import PhaseLockedLoop
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_credit import OpCreditScheduler
from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.schedulers.pos_burn_front import BurnFrontPositionScheduler
from fuzzer_tool.core.wfc_chunks import (
    ISOBMFF_FORMAT,
    WFC_MUTATOR,
    WfcChunkMutator,
    WfcChunkTableStore,
)
from tests.test_wfc_chunks import isobmff_sample

_TARGET = "targets/test_target"
_MALFORMED = (None, 7, "x", [], {"version": -1}, {"version": 1, "found": 3, "pulls": "no"})
_OPS = ["flip", "splice", "havoc"]
_SEED = bytes(range(64))
_PERIOD = 20.0


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")
    monkeypatch.setattr("fuzzer_tool.core.elf.detect_ctx_bits", lambda _t: 4)


@pytest.fixture(autouse=True)
def _restore_wfc_singleton():
    """WFC_MUTATOR is process-global: put its flag and tables back."""
    use_wfc, store = WFC_MUTATOR.use_wfc, WFC_MUTATOR.store
    yield
    WFC_MUTATOR.use_wfc, WFC_MUTATOR.store = use_wfc, store


def _fuzzer(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir(parents=True)
    crashes.mkdir(parents=True)
    return Fuzzer(
        target=_TARGET, corpus_dir=str(corpus), crashes_dir=str(crashes), max_len=4096, **kw
    )


def _sine(n: int) -> list[float]:
    return [100.0 + 10.0 * math.cos(2.0 * math.pi * t / _PERIOD) for t in range(n)]


# ── op_credit ─────────────────────────────────────────────────────────


def _credit_arm(seed: int = 1) -> OpCreditScheduler:
    arm = OpCreditScheduler(RandPool(seed=seed), MatrixSubstrate())
    for op in _OPS:
        arm.init_arm(op)
    return arm


class TestOpCredit:
    def test_roundtrip_keeps_found_and_pulls(self):
        a = _credit_arm()
        a.observe_new_edges("flip", {10, 11})
        a.observe_new_edges("splice", {20})
        for op in ("flip", "flip", "havoc"):
            a.record(op, True)

        b = _credit_arm()
        b.from_dict(a.to_dict())

        assert {o: b.credit(o) for o in _OPS} == {o: a.credit(o) for o in _OPS}
        assert b.bandit_stats()["op_credit_pulls"] == a.bandit_stats()["op_credit_pulls"]

    def test_restored_arm_selects_like_the_original(self):
        a = _credit_arm()
        a.observe_new_edges("splice", {20, 21, 22})
        a.record("flip", False)
        b = _credit_arm(seed=9)
        b.from_dict(a.to_dict())
        b._rng = RandPool(seed=3)
        a._rng = RandPool(seed=3)

        assert [b.select_op(_OPS) for _ in range(20)] == [a.select_op(_OPS) for _ in range(20)]

    @pytest.mark.parametrize("bad", _MALFORMED)
    def test_malformed_payload_starts_fresh(self, bad):
        arm = _credit_arm()
        arm.from_dict(bad)
        assert all(arm.credit(o) == 0 for o in _OPS)
        assert arm.bandit_stats()["op_credit_pulls"] == dict.fromkeys(_OPS, 0.0)


# ── PLL ───────────────────────────────────────────────────────────────


class TestPhaseLockedLoop:
    def test_roundtrip_continues_the_same_trajectory(self):
        a = PhaseLockedLoop.from_period(_PERIOD, kp=0.01, min_lock_ticks=5)
        xs = _sine(400)
        for x in xs[:200]:
            a.step(x)

        b = PhaseLockedLoop.from_dict(a.to_dict())

        assert [b.step(x) for x in xs[200:]] == [a.step(x) for x in xs[200:]]

    @pytest.mark.parametrize("bad", [None, 7, {}, {"center_freq": 0.9}, {"center_freq": "x"}])
    def test_malformed_payload_raises_value_error(self, bad):
        with pytest.raises(ValueError):
            PhaseLockedLoop.from_dict(bad)


class TestPLLMonitor:
    def _fed(self) -> PLLMonitor:
        m = PLLMonitor(warmup=64)
        for x in _sine(300):
            m.push(Series.EXEC_TIME, x)
        m.flush(1000, Stall.NO)
        for x in _sine(30):
            m.push(Series.DISCOVERY, x)  # stays pending: below warm-up
        return m

    def test_roundtrip_keeps_loop_counters_and_pending(self):
        a = self._fed()
        assert a.bootstrap_period(Series.EXEC_TIME) is not None  # a live loop is saved
        b = PLLMonitor(warmup=64)
        b.load(a.save())

        for s in Series:
            assert b.summary(s) == a.summary(s)
            assert b.state(s) == a.state(s)
            assert b.pending(s) == a.pending(s)
        assert list(b.transitions) == list(a.transitions)

        tail = _sine(120)
        for x in tail:
            a.push(Series.EXEC_TIME, x)
            b.push(Series.EXEC_TIME, x)
        assert b.flush(2000, Stall.YES) == a.flush(2000, Stall.YES)

    @pytest.mark.parametrize("bad", [None, 7, {}, {"version": 1, "tracks": 3}])
    def test_malformed_payload_starts_fresh(self, bad):
        m = PLLMonitor(warmup=64)
        m.load(bad)
        assert all(m.state(s) is None and m.pending(s) == 0 for s in Series)

    def test_resume_restores_saved_monitor(self, tmp_path):
        from fuzzer_tool.services.fuzzer import Fuzzer

        f = _fuzzer(tmp_path, pll=True)
        f._pll = self._fed()
        f._save_learned()
        f._state_store.save()

        g = Fuzzer(
            target=_TARGET,
            corpus_dir=f.corpus_dir,
            crashes_dir=str(tmp_path / "k"),
            max_len=4096,
            pll=True,
            resume=True,
        )

        assert g._pll.state(Series.EXEC_TIME) == self._fed().state(Series.EXEC_TIME)


# ── burn front ────────────────────────────────────────────────────────


def _burnt() -> BurnFrontPositionScheduler:
    bf = BurnFrontPositionScheduler(RandPool(seed=1))
    bf.record(_SEED, [5, 40], Outcome.GAIN)
    bf.record(b"other seed", [2], Outcome.GAIN, weight=2.0)
    for _ in range(40):
        bf.propose(_SEED, len(_SEED))
    return bf


class TestBurnFront:
    def test_roundtrip_proposes_the_same_offsets(self):
        a = _burnt()
        b = BurnFrontPositionScheduler(RandPool(seed=1))
        b.from_dict(a.to_dict())
        a._rng, b._rng = RandPool(seed=4), RandPool(seed=4)

        assert b.seed_count() == a.seed_count()
        assert b.hot_bins(_SEED) == a.hot_bins(_SEED)
        assert [b.propose(_SEED, 64) for _ in range(50)] == [
            a.propose(_SEED, 64) for _ in range(50)
        ]

    def test_roundtrip_keeps_lru_order(self):
        a = _burnt()
        b = BurnFrontPositionScheduler(RandPool(seed=1))
        b.from_dict(a.to_dict())
        assert list(b.to_dict()["fronts"]) == list(a.to_dict()["fronts"])

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            7,
            {},
            {"version": 1, "fronts": [(1, 0, {}, {}, 0)]},
            {"version": 1, "fronts": {1: (0, {}, {}, 0)}},
        ],
    )
    def test_malformed_payload_starts_fresh(self, bad):
        bf = _burnt()
        bf.from_dict(bad)
        assert bf.seed_count() == 0


# ── WFC ───────────────────────────────────────────────────────────────


def _trained_store() -> WfcChunkTableStore:
    store = WfcChunkTableStore()
    store.observe(ISOBMFF_FORMAT, isobmff_sample())
    return store


class TestWfcTables:
    def test_roundtrip_keeps_every_learned_pair(self):
        a = _trained_store()
        b = WfcChunkTableStore()
        b.from_dict(a.to_dict())
        assert b.to_dict() == a.to_dict()
        assert b.to_dict()["tables"]["isobmff"]

    @pytest.mark.parametrize("bad", [None, 7, {}, {"version": 1, "tables": {"isobmff": 3}}])
    def test_malformed_payload_starts_fresh(self, bad):
        store = _trained_store()
        store.from_dict(bad)
        assert store.to_dict()["tables"] == {}

    def test_off_does_not_learn(self):
        m = WfcChunkMutator()
        m.on_new_coverage(isobmff_sample(), 3)
        assert m.store.to_dict()["tables"] == {}

    def test_on_learns(self):
        m = WfcChunkMutator()
        m.use_wfc = True
        m.on_new_coverage(isobmff_sample(), 3)
        assert m.store.to_dict()["tables"]["isobmff"]

    def test_fuzzer_flag_reaches_the_registry_mutator(self, tmp_path):
        _fuzzer(tmp_path / "on", wfc=True)
        assert WFC_MUTATOR.use_wfc
        _fuzzer(tmp_path / "off")
        assert not WFC_MUTATOR.use_wfc

    def test_admission_teaches_the_table_only_with_wfc(self, tmp_path):
        off = _fuzzer(tmp_path / "off")
        WFC_MUTATOR.store = WfcChunkTableStore()
        off.save_to_corpus(isobmff_sample())
        assert WFC_MUTATOR.store.to_dict()["tables"] == {}

        on = _fuzzer(tmp_path / "on", wfc=True)
        on.save_to_corpus(isobmff_sample())
        assert WFC_MUTATOR.store.to_dict()["tables"]["isobmff"]


# ── fuzzer wiring ─────────────────────────────────────────────────────


class TestFuzzerResume:
    def test_learned_state_survives_resume(self, tmp_path):
        f = _fuzzer(tmp_path, op_credit=True, burn_front=True, wfc=True)
        f._op_credit.observe_new_edges("flip", {1, 2})
        f._op_credit.record("flip", True)
        f._burn_front.record(_SEED, [3], Outcome.GAIN)
        WFC_MUTATOR.store = _trained_store()
        f._save_learned()
        saved_tables = WFC_MUTATOR.store.to_dict()

        g = _fuzzer(tmp_path / "g", op_credit=True, burn_front=True, wfc=True, resume=True)
        g._state_store = f._state_store
        WFC_MUTATOR.store = WfcChunkTableStore()
        g._load_learned()

        assert g._op_credit.to_dict() == f._op_credit.to_dict()
        assert g._burn_front.to_dict() == f._burn_front.to_dict()
        assert WFC_MUTATOR.store.to_dict() == saved_tables

    def test_fresh_run_ignores_stored_state(self, tmp_path):
        f = _fuzzer(tmp_path, op_credit=True, wfc=True)
        f._op_credit.observe_new_edges("flip", {1, 2})
        WFC_MUTATOR.store = _trained_store()
        f._save_learned()

        g = _fuzzer(tmp_path / "g", op_credit=True, wfc=True)
        g._state_store = f._state_store
        g._load_learned()

        assert g._op_credit.credit("flip") == 0
        assert WFC_MUTATOR.store.to_dict()["tables"] == {}
