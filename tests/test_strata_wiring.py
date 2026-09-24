"""Wiring of the strata arms: construction, ledger feed, ballot, picker hook, CLI."""

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.edge_ledger import EdgeLedger, Trust
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_strata import OpStrataScheduler
from fuzzer_tool.core.schedulers.seed_strata import Guard, StrataSeedScheduler
from fuzzer_tool.services.operators import operator_strategy_pool
from fuzzer_tool.services.seed_picker import SeedPicker
from tests.support.operator_env import install_scheduler_surface

_TARGET = "targets/test_target"


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")
    monkeypatch.setattr("fuzzer_tool.core.elf.detect_ctx_bits", lambda _t: 4)


def _fuzzer(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir(parents=True)
    crashes.mkdir(parents=True)
    return Fuzzer(
        target=_TARGET, corpus_dir=str(corpus), crashes_dir=str(crashes), max_len=4096, **kw
    )


def _e(fam, tag):
    return (fam << 4) | tag


class TestConstruction:
    def test_off_by_default(self, tmp_path):
        f = _fuzzer(tmp_path)
        assert f._edge_ledger is None
        assert f._seed_strata is None
        assert f._op_strata is None

    def test_strata_builds_ledger_and_arm(self, tmp_path):
        f = _fuzzer(tmp_path, strata=True)
        assert isinstance(f._edge_ledger, EdgeLedger)
        assert f._edge_ledger.shift == 4
        assert isinstance(f._seed_strata, StrataSeedScheduler)
        assert f._seed_strata.ledger is f._edge_ledger
        assert f._op_strata is None

    def test_op_strata_shares_the_ledger(self, tmp_path):
        f = _fuzzer(tmp_path, strata=True, op_strata=True)
        assert isinstance(f._op_strata, OpStrataScheduler)
        assert f._use_op_strata

    def test_op_strata_alone_builds_ledger(self, tmp_path):
        f = _fuzzer(tmp_path, op_strata=True)
        assert isinstance(f._edge_ledger, EdgeLedger)
        assert f._seed_strata is None


class TestLedgerFeed:
    def test_observe_feeds_ledger_and_credits_pick(self, tmp_path):
        f = _fuzzer(tmp_path, strata=True)
        f._strata_observe(b"a", {_e(0, 0), _e(1, 0)})
        f._strata_observe(b"b", {_e(0, 0), _e(2, 0)})
        f._strata_observe(b"c", {_e(0, 0), _e(3, 0)})
        assert f._edge_ledger.n_seeds == 3
        arm = f._seed_strata
        key = arm.select_key(f._strata_bytes)
        phi = arm.last_phi
        f._strata_observe(f._strata_bytes[key], {_e(phi, 5)})
        assert arm.posterior(phi) == (2.0, 1.0)

    def test_stability_sets_trust(self, tmp_path):
        f = _fuzzer(tmp_path, strata=True)
        f._strata_set_stability(1.0)
        assert f._edge_ledger.trust is Trust.STABLE
        f._strata_set_stability(0.2)
        assert f._edge_ledger.trust is Trust.UNSTABLE

    def test_stratum_follows_strata_pick_else_rarest(self, tmp_path):
        f = _fuzzer(tmp_path, strata=True, op_strata=True)
        for k, fam in ((b"a", 1), (b"b", 2), (b"c", 3)):
            f._strata_observe(k, {_e(0, 0), _e(fam, 0)})
        f._strata_observe(b"a", {_e(4, 0)})
        assert f._strata_stratum(b"a") == 1  # rarest of {0, 1, 4}: owners 3, 1, 1
        key = f._seed_strata.select_key(f._strata_bytes)
        assert f._strata_stratum(f._strata_bytes[key]) == f._seed_strata.last_phi

    def test_noop_when_off(self, tmp_path):
        f = _fuzzer(tmp_path)
        f._strata_observe(b"a", {1})
        assert f._strata_stratum(b"a") is None


class TestBallot:
    def test_op_strata_on_ballot(self):
        f = SimpleNamespace(mc=None)
        install_scheduler_surface(f)
        f._use_op_strata = True
        f._op_strata = OpStrataScheduler(rng=RandPool(seed=1))
        assert "op_strata" in operator_strategy_pool(f)

    def test_off_means_absent(self):
        f = SimpleNamespace(mc=None)
        install_scheduler_surface(f)
        assert "op_strata" not in operator_strategy_pool(f)


class TestSeedPickerHook:
    def _picker(self, guard=Guard.PRESENT):
        led = EdgeLedger(4)
        data = {}
        for k, fam in (("a", 1), ("b", 2), ("c", 3), ("d", 4)):
            led.observe(k, frozenset({_e(0, 0), _e(fam, 0)}))
            data[k] = k.encode()
        f = SimpleNamespace(
            corpus=list(data.values()),
            seed_meta={v: {} for v in data.values()},
            _strata_bytes=data,
            _seed_strata=StrataSeedScheduler(RandPool(seed=2), led, guard),
            _edge_ledger=led,
            _strata_live_len=-1,
        )
        p = SeedPicker.__new__(SeedPicker)
        p.f = f
        return p

    def test_returns_corpus_member(self):
        p = self._picker()
        assert p._pick_strata_seed() in p.f.corpus

    def test_evicted_seed_is_forgotten(self):
        p = self._picker()
        p.f.seed_meta.pop(b"b")
        p.f.corpus.remove(b"b")
        for _ in range(20):
            assert p._pick_strata_seed() in (b"a", b"c", b"d")
        assert p.f._edge_ledger.n_seeds == 3
        assert "b" not in p.f._strata_bytes

    def test_abstains_without_guard(self):
        assert self._picker(Guard.ABSENT)._pick_strata_seed() is None

    def test_absent_arm_declines(self):
        p = self._picker()
        p.f._seed_strata = None
        assert p._pick_strata_seed() is None


class TestCli:
    def test_flags_reach_fuzzer_and_hail_mary(self):
        from fuzzer_tool.cli.commands import _HAIL_MARY_FLAGS

        assert "strata" in _HAIL_MARY_FLAGS
        assert "op_strata" in _HAIL_MARY_FLAGS


class TestPersistence:
    def test_state_roundtrip(self, tmp_path):
        f = _fuzzer(tmp_path, strata=True, op_strata=True)
        f._strata_observe(b"a", {_e(1, 0)})
        f._strata_observe(b"b", {_e(2, 0)})
        f._strata_observe(b"c", {_e(3, 0)})
        f._save_strata()
        g = _fuzzer(tmp_path / "g", strata=True, op_strata=True, resume=True)
        g._state_store = f._state_store
        g._load_strata()
        assert g._edge_ledger.frontier() == f._edge_ledger.frontier()
        assert g._seed_strata.ledger is g._edge_ledger


class TestReportAndStats:
    def _f(self):
        led = EdgeLedger(4)
        for k, fam in (("a", 1), ("b", 2), ("c", 3)):
            led.observe(k, frozenset({_e(0, 0), _e(fam, 0)}))
        return SimpleNamespace(
            _edge_ledger=led,
            _seed_strata=StrataSeedScheduler(RandPool(seed=1), led, Guard.PRESENT),
            _op_strata=OpStrataScheduler(rng=RandPool(seed=1)),
        )

    def test_report_lines(self):
        from fuzzer_tool.services.report import _strata_lines

        lines = _strata_lines(self._f())
        assert any("Strata ledger" in ln and "frontier=3" in ln for ln in lines)
        assert any("Strata seed" in ln for ln in lines)
        assert any("Strata op" in ln for ln in lines)

    def test_stats_str(self):
        from fuzzer_tool.services.stats import _strata_str

        assert _strata_str(self._f()).startswith(" | strata: front=3")

    def test_off_is_empty(self):
        from unittest.mock import MagicMock

        from fuzzer_tool.services.report import _strata_lines
        from fuzzer_tool.services.stats import _strata_str

        assert _strata_lines(MagicMock()) == []
        assert _strata_str(MagicMock()) == ""
