"""Wiring of the OS / network scheduler ports: Fuzzer, SeedPicker, ballot, CLI."""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_p2c import OpP2CScheduler
from fuzzer_tool.core.schedulers.op_stride import OpStrideScheduler
from fuzzer_tool.core.schedulers.seed_aimd import SeedAIMDScheduler
from fuzzer_tool.core.schedulers.seed_bfq import SeedBFQScheduler
from fuzzer_tool.core.schedulers.seed_codel import SeedCoDelScheduler
from fuzzer_tool.core.schedulers.seed_eevdf import SeedEEVDFScheduler
from fuzzer_tool.core.schedulers.seed_mlfq import SeedMLFQScheduler
from fuzzer_tool.core.schedulers.seed_p2c import SeedP2CScheduler
from fuzzer_tool.core.schedulers.seed_sfq import SFQ_BUCKETS, SeedSFQScheduler, bucket_of
from fuzzer_tool.core.schedulers.seed_stride import SeedStrideScheduler
from fuzzer_tool.services.operators import operator_strategy_pool
from tests.support.operator_env import install_scheduler_surface
from tests.support.scripted_rng import ScriptedRng

SEED_A = b"\x01" * 4
SEED_B = b"\x02" * 4
SEED_C = b"\x03" * 4

# Elo name -> (Fuzzer kwarg, Fuzzer attribute, class)
SEED_ARMS = {
    "mlfq": ("seed_mlfq_scheduler", "_seed_mlfq", SeedMLFQScheduler),
    "stride": ("seed_stride_scheduler", "_seed_stride", SeedStrideScheduler),
    "eevdf": ("seed_eevdf_scheduler", "_seed_eevdf", SeedEEVDFScheduler),
    "bfq": ("seed_bfq_scheduler", "_seed_bfq", SeedBFQScheduler),
    "sfq": ("seed_sfq_scheduler", "_seed_sfq", SeedSFQScheduler),
    "codel": ("seed_codel_scheduler", "_seed_codel", SeedCoDelScheduler),
    "aimd": ("seed_aimd_scheduler", "_seed_aimd", SeedAIMDScheduler),
    "p2c": ("seed_p2c_scheduler", "_seed_p2c", SeedP2CScheduler),
}
OP_ARMS = {
    "op_stride": OpStrideScheduler,
    "op_p2c": OpP2CScheduler,
}


def _build(cls):
    try:
        return cls(rng=RandPool(seed=3))
    except TypeError:
        return cls()


def _profile():
    return SimpleNamespace(format_signature=None, boundary_markers=[], magic_bytes=[])


def _fuzzer(name=None, corpus=(SEED_A, SEED_B, SEED_C), meta=None, favored=()):
    f = SimpleNamespace(
        corpus=list(corpus),
        seed_meta=meta if meta is not None else {},
        _use_elo=True,
        _elo=SimpleNamespace(select_strategy=lambda keys, **_: f"seed_{name}"),
        _seed_strategy=None,
        _seed_strategy_pool=[],
        _seed_strategies_used=set(),
        _stall_recovery_active=False,
        _rng=RandPool(seed=4),
        ga=None,
        qea=None,
        markov_generate=False,
        markov_trained=False,
        _use_bayesian=False,
        _use_boltzmann=False,
        _use_ecofuzz=False,
        _profile=_profile(),
        _kruskal_count=None,
        _favored=set(favored),
        mean_exec_time=lambda: 0.001,
    )
    for arm, (_kw, attr, cls) in SEED_ARMS.items():
        setattr(f, attr, _build(cls) if arm == name else None)
    f._seed_key = lambda data: data
    return f


def _picker(f):
    from fuzzer_tool.services.seed_picker import SeedPicker

    sp = SeedPicker.__new__(SeedPicker)
    sp.f = f
    sp._rng = f._rng
    return sp


def _real_fuzzer(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir(parents=True)
    crashes.mkdir(parents=True)
    return Fuzzer(
        target="targets/test_target",
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        **kw,
    )


# --- SeedPicker -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SEED_ARMS))
class TestSeedPicker:
    def test_eligible_and_dispatched_under_elo(self, name):
        f = _fuzzer(name)
        picked = _picker(f)._pick_seed_elo()

        assert name in f._seed_strategy_pool
        assert f._seed_strategy == name
        assert picked in f.corpus

    def test_not_eligible_when_disabled_or_empty(self, name):
        for f in (_fuzzer(None), _fuzzer(name, corpus=())):
            f._use_boltzmann = True
            f._elo = SimpleNamespace(select_strategy=lambda keys, **_: "seed_unhandled")
            _picker(f)._pick_seed_elo()

            assert name not in f._seed_strategy_pool

    def test_declines_when_disabled_or_empty(self, name):
        assert _picker(_fuzzer(None))._pick_os_seed(name) is None
        assert _picker(_fuzzer(name, corpus=()))._pick_os_seed(name) is None

    def test_no_elo_fallback_runs_the_arm(self, name, monkeypatch):
        """Enabled without --elo, the arm is reached before drr / round robin."""
        f = _fuzzer(name)
        f._use_elo = False
        calls = []
        attr = SEED_ARMS[name][1]
        real = getattr(f, attr)
        select = real.select_seed

        def spy(ids, *signals):
            calls.append(ids)
            return select(ids, *signals)

        monkeypatch.setattr(real, "select_seed", spy)
        f._seed_drr = SimpleNamespace(select_seed=lambda *a: pytest.fail("drr won"))
        sp = _picker(f)
        monkeypatch.setattr(sp, "_update_temperature", lambda: None)

        assert sp.pick_seed() in f.corpus
        assert calls


class TestSeedSignals:
    def test_stride_favored_seed_weighs_double(self):
        f = _fuzzer("stride", corpus=(SEED_A, SEED_B), favored={SEED_A})
        sp = _picker(f)
        picks = [sp._pick_os_seed("stride") for _ in range(300)]

        assert picks.count(SEED_A) == pytest.approx(200, abs=1)

    def test_eevdf_slow_seed_gets_equal_time(self):
        """Cost comes from the ledger (total_time / cost_samples) relative to the mean."""
        meta = {
            SEED_A: {"total_time": 0.001, "cost_samples": 1},
            SEED_B: {"total_time": 0.004, "cost_samples": 1},
        }
        f = _fuzzer("eevdf", corpus=(SEED_A, SEED_B), meta=meta)
        sp = _picker(f)
        picks = [sp._pick_os_seed("eevdf") for _ in range(500)]

        assert picks.count(SEED_A) > 3 * picks.count(SEED_B)

    def test_sfq_groups_siblings_by_parent_key(self):
        """Siblings of one parent share a flow; the lone seed gets half the picks."""
        kids = [bytes([0x10 + i]) * 4 for i in range(6)]
        meta = {k: {"parent_key": "P"} for k in kids}
        f = _fuzzer("sfq", corpus=(*kids, SEED_A), meta=meta)
        salt = 11
        assert bucket_of(salt, "P", SFQ_BUCKETS) != bucket_of(salt, SEED_A, SFQ_BUCKETS)
        f._seed_sfq = SeedSFQScheduler(rng=ScriptedRng(randints=[salt]))
        sp = _picker(f)
        picks = [sp._pick_os_seed("sfq") for _ in range(100)]

        assert picks.count(SEED_A) == 50


# --- Fuzzer -----------------------------------------------------------------


class TestFuzzer:
    @pytest.mark.parametrize("name", sorted(SEED_ARMS))
    def test_seed_strategy_registered(self, name):
        from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES

        assert name in _SEED_STRATEGY_NAMES

    @pytest.mark.parametrize("name", sorted(OP_ARMS))
    def test_op_arm_elo_only_like_op_strata(self, name):
        """op_* arms stay off the pre-registered list and the no-elo precedence."""
        from fuzzer_tool.services.fuzzer import _OPERATOR_STRATEGY_NAMES
        from fuzzer_tool.services.operators import _FALLBACK_PRECEDENCE

        assert name not in _OPERATOR_STRATEGY_NAMES
        assert name not in _FALLBACK_PRECEDENCE

    def test_kwargs_default_off_and_appended(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        params = inspect.signature(Fuzzer.__init__).parameters
        names = list(params)
        kwargs = [kw for kw, _a, _c in SEED_ARMS.values()] + list(OP_ARMS)

        for kw in kwargs:
            assert params[kw].default is False
            assert names.index(kw) > names.index("seed_drr_scheduler")

    def test_off_by_default(self, tmp_path):
        f = _real_fuzzer(tmp_path)

        assert all(getattr(f, attr) is None for _kw, attr, _c in SEED_ARMS.values())
        assert f._op_stride is None and f._op_p2c is None
        assert f._seed_os_arms == ()

    def test_flags_build_the_arms(self, tmp_path):
        kwargs = {kw: True for kw, _a, _c in SEED_ARMS.values()}
        f = _real_fuzzer(tmp_path, op_stride=True, op_p2c=True, **kwargs)

        for _kw, attr, cls in SEED_ARMS.values():
            assert isinstance(getattr(f, attr), cls)
        assert isinstance(f._op_stride, OpStrideScheduler)
        assert isinstance(f._op_p2c, OpP2CScheduler)
        assert len(f._seed_os_arms) == len(SEED_ARMS)

    def test_seed_outcome_reaches_every_arm(self, tmp_path):
        """Falsification: one recorded corpus outcome lands in each enabled arm's ledger."""
        kwargs = {kw: True for kw, _a, _c in SEED_ARMS.values()}
        f = _real_fuzzer(tmp_path, **kwargs)
        f.corpus.append(SEED_A)
        f.seed_meta[SEED_A] = {}
        f._record_seed_os_arms(SEED_A, success=True, weight=1.0)

        key = f._seed_key(SEED_A)
        for arm in f._seed_os_arms:
            assert arm.bandit_stats() == {key: (1.0, 0.0)}

    def test_regression_non_corpus_parent_not_recorded(self, tmp_path):
        """Adversarial (PR #44 review): synthetic parents (Markov) never enter the ledgers."""
        kwargs = {kw: True for kw, _a, _c in SEED_ARMS.values()}
        f = _real_fuzzer(tmp_path, **kwargs)
        f._record_seed_os_arms(b"not-in-corpus", success=True, weight=1.0)

        for arm in f._seed_os_arms:
            assert arm.bandit_stats() == {}

    def test_regression_seed_meta_without_corpus_not_recorded(self, tmp_path):
        """Falsification (PR #46 review): standalone QEA fills seed_meta, not the corpus."""
        kwargs = {kw: True for kw, _a, _c in SEED_ARMS.values()}
        f = _real_fuzzer(tmp_path, **kwargs)
        f.seed_meta[SEED_B] = {}
        assert SEED_B not in f.corpus

        f._record_seed_os_arms(SEED_B, success=True, weight=1.0)

        for arm in f._seed_os_arms:
            assert arm.bandit_stats() == {}

    def test_adversarial_parent_admitted_later_is_recorded(self, tmp_path):
        """A cached 'not in corpus' verdict must not outlive the parent's admission."""
        kwargs = {kw: True for kw, _a, _c in SEED_ARMS.values()}
        f = _real_fuzzer(tmp_path, **kwargs)
        f._record_seed_os_arms(SEED_C, success=True, weight=1.0)

        f.corpus.append(SEED_C)
        f._record_seed_os_arms(SEED_C, success=True, weight=1.0)

        key = f._seed_key(SEED_C)
        for arm in f._seed_os_arms:
            assert arm.bandit_stats() == {key: (1.0, 0.0)}

    def test_regression_parent_replaced_in_place_not_recorded(self, tmp_path):
        """Falsification (PR #47 review): same-length in-place replacement retires the parent."""
        kwargs = {kw: True for kw, _a, _c in SEED_ARMS.values()}
        f = _real_fuzzer(tmp_path, **kwargs)
        f.corpus.append(SEED_A)
        f._record_seed_os_arms(SEED_A, success=True, weight=1.0)

        f.corpus[f.corpus.index(SEED_A)] = SEED_B
        f._record_seed_os_arms(SEED_A, success=True, weight=1.0)

        key = f._seed_key(SEED_A)
        for arm in f._seed_os_arms:
            assert arm.bandit_stats()[key] == (1.0, 0.0)

    def test_adversarial_parent_kept_across_corpus_rebuild(self, tmp_path):
        """A rebuilt corpus list that still holds the parent keeps recording it."""
        kwargs = {kw: True for kw, _a, _c in SEED_ARMS.values()}
        f = _real_fuzzer(tmp_path, **kwargs)
        f.corpus.append(SEED_A)
        f._record_seed_os_arms(SEED_A, success=True, weight=1.0)

        f.corpus = [SEED_C, SEED_A]
        f._record_seed_os_arms(SEED_A, success=True, weight=1.0)

        key = f._seed_key(SEED_A)
        for arm in f._seed_os_arms:
            assert arm.bandit_stats()[key] == (2.0, 0.0)

    def test_op_arms_get_priors_registration_and_records(self):
        """The op arms sit in _register_arms and in fuzz_one's shared record loop."""
        from fuzzer_tool.services import fuzzer as fz

        src = inspect.getsource(fz.Fuzzer)
        for attr in ("self._op_stride", "self._op_p2c"):
            assert f"_register_arms({attr})" in src
        from fuzzer_tool.services.fuzz_round import FuzzRound

        record_loop = inspect.getsource(FuzzRound)
        tuple_src = record_loop[record_loop.index("for scheduler in (") :]
        tuple_src = tuple_src[: tuple_src.index("):")]
        assert "f._op_stride" in tuple_src
        assert "f._op_p2c" in tuple_src


# --- Operator ballot --------------------------------------------------------


@pytest.mark.parametrize("name", sorted(OP_ARMS))
class TestOpBallot:
    def test_on_ballot_when_enabled(self, name):
        f = SimpleNamespace(mc=None)
        install_scheduler_surface(f)
        setattr(f, f"_use_{name}", True)
        setattr(f, f"_{name}", _build(OP_ARMS[name]))

        assert name in operator_strategy_pool(f)

    def test_absent_when_off(self, name):
        f = SimpleNamespace(mc=None)
        install_scheduler_surface(f)

        assert name not in operator_strategy_pool(f)

    def test_elo_dispatch_uses_the_arm(self, name):
        from fuzzer_tool.services.operators import OperatorEngine

        src = inspect.getsource(OperatorEngine)
        assert f'strategy == "{name}" and f._{name}' in src
        assert f"op = f._{name}.select_op(ops)" in src


# --- CLI --------------------------------------------------------------------


def test_cli_flags_reach_fuzzer_and_hail_mary():
    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
    ]
    dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))

    for kw in [kw for kw, _a, _c in SEED_ARMS.values()] + list(OP_ARMS):
        assert all(kw in {k.arg for k in c.keywords} for c in calls)
        assert kw in dests
        assert kw in commands._HAIL_MARY_FLAGS
