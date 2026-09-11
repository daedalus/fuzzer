"""ConsolidatedScheduler: the properties it is made of, one at a time.

Convergence and decay recovery live in test_scheduler_convergence (RELIABLE
and RECOVERS). These pin the mechanisms: category sharing reaches unsampled
arms, the cap forgets without moving the mean, format priors are a one-time
nudge, rewards are a single bounded observation, and the RandPool seed
reproduces a campaign.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES, category_of
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import ConsolidatedScheduler


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
    s = ConsolidatedScheduler(rng=RandPool(1))
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
    s = ConsolidatedScheduler(prior_strength=0.0, rng=RandPool(1))
    for _ in range(100):
        s.record(a1, True)
        s.record(a2, True)
    assert s.posterior_mean(a_unseen) == pytest.approx(0.5)


def test_cap_bounds_evidence_and_keeps_the_mean():
    s = ConsolidatedScheduler(max_pseudocount=50.0, rng=RandPool(1))
    op = "bit_flip"
    for i in range(1000):
        s.record(op, i % 4 == 0)
    aid = s._index[op]
    assert s._alpha[aid] + s._beta[aid] == pytest.approx(50.0)
    assert s._alpha[aid] / 50.0 == pytest.approx(0.25, abs=0.05)


def test_cap_lets_a_dead_arm_be_left():
    """After the cap, 200 failures move the mean most of the way down --
    an uncapped posterior with 5000 prior successes would barely move."""
    s = ConsolidatedScheduler(rng=RandPool(1))
    op = "bit_flip"
    for _ in range(5000):
        s.record(op, True)
    for _ in range(200):
        s.record(op, False)
    assert s.posterior_mean(op) < 0.5


def test_format_prior_is_a_one_time_nudge():
    s = ConsolidatedScheduler(rng=RandPool(1))
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
    s = ConsolidatedScheduler(rng=RandPool(1))
    s.record("bit_flip", success, weight=weight)
    aid = s._index["bit_flip"]
    assert s._alpha[aid] == pytest.approx(alpha)
    assert s._alpha[aid] + s._beta[aid] == pytest.approx(1.0)


def test_seeded_rng_reproduces_the_campaign():
    ops = ["bit_flip", "byte_flip", "arith_inc", "havoc"]

    def campaign(seed):
        s = ConsolidatedScheduler(rng=RandPool(seed))
        out = []
        for i in range(300):
            op = s.select_op(ops)
            s.record(op, (i * 7 + len(op)) % 5 == 0)
            out.append(op)
        return out

    assert campaign(3) == campaign(3)
    assert campaign(3) != campaign(4)


def test_degenerate_candidate_lists():
    s = ConsolidatedScheduler(rng=RandPool(1))
    assert s.select_op([]) == ""
    assert s.select_op(["havoc"]) == "havoc"


def test_unknown_operator_is_registered_lazily():
    s = ConsolidatedScheduler(rng=RandPool(1))
    assert s.select_op(["not_a_real_op", "bit_flip"]) in ("not_a_real_op", "bit_flip")
    s.record("also_unknown", True)
    assert "also_unknown" in s._index


def test_bandit_stats_shape():
    s = ConsolidatedScheduler(rng=RandPool(1))
    s.record("bit_flip", True)
    stats = s.bandit_stats()
    assert stats["consolidated_pulls"] == 1
    assert stats["consolidated_top"][0][0] == "bit_flip"


@pytest.mark.parametrize("kwargs", [{"prior_strength": -1.0}, {"max_pseudocount": 0.0}])
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ConsolidatedScheduler(**kwargs)


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
        consolidated=True,
        mc_bandit=True,
    )
    assert isinstance(f._consolidated, ConsolidatedScheduler)
    assert f._track_op_effect

    def bandit_mass():
        return sum(f.mc.arm_alpha.values()) + sum(f.mc.arm_beta.values())

    before = bandit_mass()
    selectors = set()
    for i in range(40):
        f.fuzz_one(bytes([65 + i % 26]) * 16)
        selectors.add(f._op_selector)
    assert "consolidated" in selectors
    assert "bandit" not in selectors
    assert f._consolidated.bandit_stats()["consolidated_pulls"] > 0
    assert bandit_mass() > before, "the bandit stopped learning from rounds it did not select"
