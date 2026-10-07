"""ConsolidatedV1Scheduler: the properties it is made of, one at a time.

Convergence and decay recovery live in test_scheduler_convergence (RELIABLE
and RECOVERS). These pin the mechanisms: category sharing reaches unsampled
arms, the cap forgets without moving the mean, format priors are a one-time
nudge, rewards are a single bounded observation, and the RandPool seed
reproduces a campaign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.core.blup import MIN_STRENGTH
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES, category_of
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import ConsolidatedV1Scheduler, ConsolidatedV2Scheduler
from fuzzer_tool.core.schedulers.op_consolidated_v1 import _BLUP_MAX_STRENGTH, PriorMode


def _two_categories():
    """Three registered operators from category A and one from category B."""
    cats = sorted(c for c, ops in OPERATOR_CATEGORIES.items() if len(ops) >= 3)
    a_ops = sorted(op for op in OPERATOR_CATEGORIES[cats[0]] if category_of(op) == cats[0])[:3]
    b_op = sorted(op for op in OPERATOR_CATEGORIES[cats[1]] if category_of(op) == cats[1])[0]
    assert len(a_ops) == 3
    return a_ops, b_op


def test_unsampled_arm_inherits_its_categorys_rate():
    """The point of the shrinkage prior: evidence on two operators of a
    category moves the third, never pulled, above an arm of a dead one."""
    (a1, a2, a_unseen), b = _two_categories()
    s = ConsolidatedV1Scheduler(rng=RandPool(1))
    for op in (a1, a2, a_unseen, b):
        s.init_arm(op)
    for _ in range(100):
        s.record(a1, True)
        s.record(a2, True)
        s.record(b, False)
    assert s.posterior_mean(a_unseen) > 0.6
    assert s.posterior_mean(b) < 0.1
    picks = [s.select_op([a_unseen, b]) for _ in range(200)]
    assert picks.count(a_unseen) > 180


def test_without_prior_strength_categories_are_ignored():
    (a1, a2, a_unseen), _ = _two_categories()
    s = ConsolidatedV1Scheduler(prior_strength=0.0, rng=RandPool(1))
    for _ in range(100):
        s.record(a1, True)
        s.record(a2, True)
    assert s.posterior_mean(a_unseen) == pytest.approx(0.5)


def test_cap_bounds_evidence_and_keeps_the_mean():
    s = ConsolidatedV1Scheduler(max_pseudocount=50.0, rng=RandPool(1))
    op = "bit_flip"
    for i in range(1000):
        s.record(op, i % 4 == 0)
    aid = s._index[op]
    assert s._alpha[aid] + s._beta[aid] == pytest.approx(50.0)
    assert s._alpha[aid] / 50.0 == pytest.approx(0.25, abs=0.05)


def test_cap_lets_a_dead_arm_be_left():
    """After the cap, 200 failures move the mean most of the way down --
    an uncapped posterior with 5000 prior successes would barely move."""
    s = ConsolidatedV1Scheduler(rng=RandPool(1))
    op = "bit_flip"
    for _ in range(5000):
        s.record(op, True)
    for _ in range(200):
        s.record(op, False)
    assert s.posterior_mean(op) < 0.5


def test_format_prior_is_a_one_time_nudge():
    s = ConsolidatedV1Scheduler(rng=RandPool(1))
    s.init_arm("bit_flip", 2.0, 1.0)
    aid = s._index["bit_flip"]
    assert (s._alpha[aid], s._beta[aid]) == (1.0, 0.0)
    s.init_arm("bit_flip", 2.0, 1.0)  # re-registration does not stack
    assert (s._alpha[aid], s._beta[aid]) == (1.0, 0.0)
    s.init_arm("byte_flip", 0.5, 0.5)  # weaker than uniform: floored at zero
    assert s._alpha[s._index["byte_flip"]] == 0.0


@pytest.mark.parametrize(
    ("success", "weight", "alpha"), [(True, 15.0, 1.0), (True, 0.25, 0.25), (False, 1.0, 0.0)]
)
def test_one_record_is_one_bounded_observation(success, weight, alpha):
    s = ConsolidatedV1Scheduler(rng=RandPool(1))
    s.record("bit_flip", success, weight=weight)
    aid = s._index["bit_flip"]
    assert s._alpha[aid] == pytest.approx(alpha)
    assert s._alpha[aid] + s._beta[aid] == pytest.approx(1.0)


def test_seeded_rng_reproduces_the_campaign():
    ops = ["bit_flip", "byte_flip", "arith_inc", "havoc"]

    def campaign(seed):
        s = ConsolidatedV1Scheduler(rng=RandPool(seed))
        out = []
        for i in range(300):
            op = s.select_op(ops)
            s.record(op, (i * 7 + len(op)) % 5 == 0)
            out.append(op)
        return out

    assert campaign(3) == campaign(3)
    assert campaign(3) != campaign(4)


def test_degenerate_candidate_lists():
    s = ConsolidatedV1Scheduler(rng=RandPool(1))
    assert s.select_op([]) == ""
    assert s.select_op(["havoc"]) == "havoc"


def test_unknown_operator_is_registered_lazily():
    s = ConsolidatedV1Scheduler(rng=RandPool(1))
    assert s.select_op(["not_a_real_op", "bit_flip"]) in ("not_a_real_op", "bit_flip")
    s.record("also_unknown", True)
    assert "also_unknown" in s._index


def test_bandit_stats_shape():
    s = ConsolidatedV1Scheduler(rng=RandPool(1))
    s.record("bit_flip", True)
    stats = s.bandit_stats()
    assert stats["consolidated_v1_pulls"] == 1
    assert stats["consolidated_v1_top"][0][0] == "bit_flip"


@pytest.mark.parametrize("kwargs", [{"prior_strength": -1.0}, {"max_pseudocount": 0.0}])
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ConsolidatedV1Scheduler(**kwargs)


_TARGET = Path(__file__).resolve().parent.parent / "targets" / "test_target"


@pytest.mark.skipif(not _TARGET.exists(), reason="targets/test_target not built")
def test_fuzzer_wiring_selects_and_learns(tmp_path):
    """--consolidated builds it, it selects ahead of the bandit without Elo,
    it learns from its rounds, and the bandit still learns from them too
    (the shared fan-out is kept on purpose)."""
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
        consolidated_v1=True,
        mc_bandit=True,
    )
    assert isinstance(f._consolidated_v1, ConsolidatedV1Scheduler)
    assert f._track_op_effect

    def bandit_mass():
        return sum(f.mc.arm_alpha.values()) + sum(f.mc.arm_beta.values())

    before = bandit_mass()
    selectors = set()
    for i in range(40):
        f.fuzz_one(bytes([65 + i % 26]) * 16)
        selectors.add(f._op_selector)
    assert "consolidated_v1" in selectors
    assert "bandit" not in selectors
    assert f._consolidated_v1.bandit_stats()["consolidated_v1_pulls"] > 0
    assert bandit_mass() > before, "the bandit stopped learning from rounds it did not select"


# -- pre-v2 compatibility (Copilot review on daedalus/fuzzer#53) -------------


def test_regression_alias_keeps_legacy_stats_keys():
    """ConsolidatedScheduler behaves as v1 but reports its pre-v2 keys."""
    from fuzzer_tool.core.schedulers import ConsolidatedScheduler

    s = ConsolidatedScheduler(rng=RandPool(1))
    assert isinstance(s, ConsolidatedV1Scheduler)
    s.record("bit_flip", True)
    stats = s.bandit_stats()
    assert stats["consolidated_pulls"] == 1
    assert stats["consolidated_top"][0][0] == "bit_flip"
    assert not any(k.startswith("consolidated_v1") for k in stats)


def test_regression_alias_is_exported():
    import fuzzer_tool.core.schedulers as S

    assert "ConsolidatedScheduler" in S.__all__


def test_regression_fuzzer_signature_keeps_positional_slots():
    """`consolidated` keeps its pre-v2 slot (just before `moss`); the
    versioned flags, then `clock`, are appended, so positional callers are not shifted."""
    import inspect

    from fuzzer_tool.services.fuzzer import Fuzzer

    params = list(inspect.signature(Fuzzer.__init__).parameters)
    assert params.index("moss") == params.index("consolidated") + 1
    assert params.index("consolidated_v2") == params.index("consolidated_v1") + 1
    assert params.index("consolidated_v1") > params.index("op_p2c")
    assert params[-1] == "clock"


@pytest.mark.skipif(not _TARGET.exists(), reason="targets/test_target not built")
def test_regression_legacy_consolidated_kwarg_builds_v1(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir()
    crashes.mkdir()
    f = Fuzzer(
        target=str(_TARGET), corpus_dir=str(corpus), crashes_dir=str(crashes), consolidated=True
    )
    assert isinstance(f._consolidated_v1, ConsolidatedV1Scheduler)
    assert f._use_consolidated_v1


# -- BLUP prior strength (core/blup.py) ---------------------------------------


def _strength(s, op):
    """Total pseudocount of *op*'s category prior."""
    pa, pb = s._prior(np.array([s._arm_id(op)]))
    return float(pa[0] + pb[0])


def _blup(**kw):
    return ConsolidatedV1Scheduler(prior_mode=PriorMode.BLUP, rng=RandPool(1), **kw)


def test_blup_falsify_identical_ops_pool_fully():
    """Falsification: two ops with exactly the same rate give the unseen op
    of their category the full ceiling of pooling, not the fixed 4."""
    (a1, a2, a_unseen), _ = _two_categories()
    s = _blup(prior_strength=4.0)
    for _ in range(150):
        s.record(a1, True, 0.2)
        s.record(a2, True, 0.2)
    assert _strength(s, a_unseen) == pytest.approx(_BLUP_MAX_STRENGTH)
    assert s.posterior_mean(a_unseen) == pytest.approx(0.2, abs=0.01)


def test_blup_adversarial_outlier_keeps_its_own_rate():
    """Adversarial: a strong op beside a weak one makes the category's ops
    unlike each other, so pooling drops to the floor and the strong op is
    not dragged toward the category mean."""
    (a1, a2, a_unseen), _ = _two_categories()
    s = _blup()
    for _ in range(150):
        s.record(a1, True, 0.02)
        s.record(a2, True, 0.5)
    assert _strength(s, a_unseen) == pytest.approx(MIN_STRENGTH)
    assert s.posterior_mean(a2) == pytest.approx(0.5, abs=0.01)


def test_blup_undefined_fit_falls_back_to_fixed_strength():
    """One sampled op in a category: no dispersion to measure."""
    (a1, _, a_unseen), _ = _two_categories()
    s = _blup(prior_strength=4.0)
    for _ in range(100):
        s.record(a1, True, 0.3)
    assert _strength(s, a_unseen) == pytest.approx(4.0)


def test_fixed_mode_keeps_the_constant_strength():
    (a1, a2, a_unseen), _ = _two_categories()
    s = ConsolidatedV1Scheduler(prior_mode=PriorMode.FIXED, prior_strength=4.0, rng=RandPool(1))
    for _ in range(150):
        s.record(a1, True, 0.2)
        s.record(a2, True, 0.2)
    assert _strength(s, a_unseen) == pytest.approx(4.0)


def test_v2_forwards_prior_mode():
    s = ConsolidatedV2Scheduler(prior_mode=PriorMode.BLUP, rng=RandPool(1))
    assert s.prior_mode is PriorMode.BLUP


def test_cli_consolidated_prior_default_and_value(monkeypatch):
    from tests.test_dirichlet_wiring import _parse

    assert _parse(monkeypatch).consolidated_prior == PriorMode.FIXED.value
    assert _parse(monkeypatch, "--consolidated-prior", "blup").consolidated_prior == "blup"


def test_adversarial_cli_rejects_unknown_prior(monkeypatch):
    from tests.test_dirichlet_wiring import _parse

    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--consolidated-prior", "reml")


def test_cmd_fuzz_forwards_consolidated_prior():
    from tests.test_dirichlet_wiring import _fuzzer_call_kwargs

    assert all("consolidated_prior" in k for k in _fuzzer_call_kwargs())


@pytest.mark.skipif(not _TARGET.exists(), reason="targets/test_target not built")
def test_fuzzer_builds_blup_schedulers(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir()
    crashes.mkdir()
    f = Fuzzer(
        target=str(_TARGET),
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        consolidated_v1=True,
        consolidated_v2=True,
        consolidated_prior=PriorMode.BLUP,
    )
    assert f._consolidated_v1.prior_mode is PriorMode.BLUP
    assert f._consolidated_v2.prior_mode is PriorMode.BLUP
