"""MOSSScheduler: the index, the fair-share cutoff, and where MOSS pays off.

Convergence on the 12-arm environments lives in test_scheduler_convergence
(RELIABLE). These pin the mechanisms one at a time: the index matches the
paper's formula, an arm at or above its fair share t/K gets no exploration
bonus, unpulled arms open first, ties go to the less-pulled arm, a record is
one bounded pull, the discount ages evidence, and the seeded RandPool
reproduces a campaign.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import DUCBScheduler, MOSSScheduler
from tests.support.bandit_env import run
from tests.support.scripted_rng import ScriptedRng


def _feed(s, name, pulls, successes):
    for i in range(pulls):
        s.record(name, i < successes)


def _reference_index(mean, n, t, k, alpha, exploration):
    """The MOSS-anytime index written out, one arm at a time."""
    log_plus = max(math.log(t / (k * n)), 0.0)
    return mean + exploration * math.sqrt((1.0 + alpha) / 2.0 * log_plus / n)


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


def test_scores_match_the_formula():
    """Vectorized scores against the scalar formula, on fractional rewards."""
    rng = random.Random(5)
    ops = [f"op{i}" for i in range(9)]
    s = MOSSScheduler(alpha=1.3, exploration=0.7, rng=RandPool(1))
    sums = dict.fromkeys(ops, 0.0)
    pulls = dict.fromkeys(ops, 0)
    for _ in range(400):
        op = rng.choice(ops)
        w = rng.random()
        s.record(op, True, weight=w)
        sums[op] += w
        pulls[op] += 1

    t = sum(pulls.values())
    want = [_reference_index(sums[o] / pulls[o], pulls[o], t, len(ops), 1.3, 0.7) for o in ops]
    assert np.allclose(s._scores(ops), want, rtol=1e-12, atol=0.0)


def test_arm_at_fair_share_gets_no_bonus():
    """The MOSS cutoff. Arm a has 60 of 100 pulls (fair share 50), so its
    index is its mean, 0.10; b sits below fair share and keeps a bonus.
    A log(t/n) width (UCB1) would give a a bonus too -- this pins that it
    does not."""
    s = MOSSScheduler(rng=RandPool(1))
    _feed(s, "a", 60, 6)
    _feed(s, "b", 40, 2)
    scores = s._scores(["a", "b"])
    assert scores[0] == pytest.approx(0.10, abs=1e-12)
    assert scores[1] > 0.05


def test_greedy_above_fair_share_where_ucb1_explores():
    """The consequence: a 60/200 (above fair share), b 3/20. At the same
    constant UCB1 opens b, because a's width still counts; MOSS stays on a."""
    s = MOSSScheduler(exploration=0.5, rng=RandPool(1))
    _feed(s, "a", 200, 60)
    _feed(s, "b", 20, 3)
    ops = ["a", "b"]
    t = 220

    def ucb1(mean, n):
        return mean + 0.5 * math.sqrt(math.log(t) / n)

    assert ucb1(0.15, 20) > ucb1(0.30, 200)
    assert s.select_op(ops) == "a"
    assert s._scores(ops)[0] == pytest.approx(0.30, abs=1e-12)


# ---------------------------------------------------------------------------
# Selection mechanics
# ---------------------------------------------------------------------------


def test_unpulled_arms_open_first():
    s = MOSSScheduler(rng=ScriptedRng(choice_idxs=[1]))
    _feed(s, "a", 50, 50)
    s.init_arm("b")
    s.init_arm("c")
    assert s.select_op(["a", "b", "c"]) == "c"


def test_runtime_registered_operator_is_opened():
    """An op never passed to init_arm (REGISTRY.register_mutator at
    runtime) is registered on sight and, having no evidence, opened."""
    s = MOSSScheduler(rng=RandPool(1))
    _feed(s, "a", 50, 50)
    assert s.select_op(["a", "late_op"]) == "late_op"


def test_ties_go_to_the_less_pulled_arm():
    """All arms above fair share with mean 0 tie at exactly 0.0; falling
    back to list order would favour whatever the caller listed first."""
    s = MOSSScheduler(rng=RandPool(1))
    _feed(s, "a", 5, 0)
    _feed(s, "b", 4, 0)
    _feed(s, "c", 5, 0)
    # t=14, K=3, fair share 4.67: b is below it and has the only bonus.
    assert s.select_op(["a", "b", "c"]) == "b"
    s.record("b", False)
    # t=15, all at 5 = fair share: exact three-way tie, rng decides.
    s._rng = ScriptedRng(choice_idxs=[2])
    assert s.select_op(["a", "b", "c"]) == "c"


def test_tie_between_unequal_counts_prefers_fewer_pulls():
    """a 4/40 and b 3/30 score exactly 0.1 (no bonus: both at or above
    fair share 30); c 0/20 keeps a bonus of ~0.071. b wins on fewer pulls
    without consulting the rng -- the scripted pool raises on any draw."""
    s = MOSSScheduler(rng=ScriptedRng())
    _feed(s, "a", 40, 4)
    _feed(s, "b", 30, 3)
    _feed(s, "c", 20, 0)
    scores = s._scores(["a", "b", "c"])
    assert scores[0] == scores[1] == 0.1
    assert 0.0 < scores[2] < 0.1
    assert s.select_op(["a", "b", "c"]) == "b"


def test_degenerate_candidate_lists():
    s = MOSSScheduler(rng=RandPool(1))
    assert s.select_op([]) == ""
    assert s.select_op(["havoc"]) == "havoc"


def test_duplicate_candidates_do_not_break_selection():
    s = MOSSScheduler(rng=RandPool(1))
    _feed(s, "a", 10, 5)
    _feed(s, "b", 10, 1)
    assert s.select_op(["a", "a", "b"]) in ("a", "b")


# ---------------------------------------------------------------------------
# Rewards and state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("success", "weight", "mean"),
    [
        (True, 15.0, 1.0),
        (True, 0.25, 0.25),
        (False, 1.0, 0.0),
        (True, -3.0, 0.0),
        (True, math.nan, 0.0),
        (True, math.inf, 1.0),
    ],
)
def test_one_record_is_one_bounded_pull(success, weight, mean):
    s = MOSSScheduler(rng=RandPool(1))
    s.record("a", success, weight=weight)
    stats = s.bandit_stats()
    assert stats["moss_pulls"] == 1
    assert stats["moss_effective_n"] == pytest.approx(1.0)
    s.record("b", False)
    assert s._scores(["a", "b"])[0] == pytest.approx(mean)


def test_discount_ages_evidence():
    s = MOSSScheduler(gamma=0.5, rng=RandPool(1))
    s.record("a", True)
    s.record("b", False)
    # a's pull has been discounted once, b's not at all.
    assert s.bandit_stats()["moss_effective_n"] == pytest.approx(1.5)


def test_discount_survives_underflow():
    """The global factor is folded back in before it underflows; the
    counts stay finite and positive."""
    s = MOSSScheduler(gamma=0.9, rng=RandPool(1))
    for i in range(5000):
        s.record("a" if i % 2 else "b", i % 3 == 0)
    n = s.bandit_stats()["moss_effective_n"]
    assert math.isfinite(n)
    assert n == pytest.approx(1.0 / (1.0 - 0.9), rel=1e-6)
    assert s.select_op(["a", "b"]) in ("a", "b")


def test_undiscounted_counts_are_exact():
    s = MOSSScheduler(rng=RandPool(1))
    for _ in range(1000):
        s.record("a", False)
    assert s.bandit_stats()["moss_effective_n"] == 1000.0


def test_seeded_rng_reproduces_the_campaign():
    ops = ["bit_flip", "byte_flip", "arith_inc", "havoc"]

    def campaign(seed):
        s = MOSSScheduler(rng=RandPool(seed))
        out = []
        for i in range(300):
            op = s.select_op(ops)
            s.record(op, (i * 7 + len(op)) % 5 == 0)
            out.append(op)
        return out

    assert campaign(3) == campaign(3)
    assert campaign(3) != campaign(4)


def test_bandit_stats_shape():
    s = MOSSScheduler(rng=RandPool(1))
    s.init_arm("a")
    s.record("b", True)
    stats = s.bandit_stats()
    assert stats["moss_pulls"] == 1
    assert stats["moss_arms"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [{"gamma": 0.0}, {"gamma": 1.5}, {"alpha": -0.1}, {"exploration": 0.0}, {"exploration": -1.0}],
)
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        MOSSScheduler(**kwargs)


def test_does_not_take_priors():
    assert MOSSScheduler.supports_priors is False


# ---------------------------------------------------------------------------
# Where MOSS pays off: many arms at fuzzing rates
# ---------------------------------------------------------------------------


class _ManyLowRateArms:
    """100 arms: one at 0.08, four at 0.03, the rest at 0.002-0.01.

    The regime of the real operator set (~200 arms, low yields) scaled
    down to run in about a second. UCB1-style widths spend the campaign
    opening every arm; MOSS stops paying an arm's bonus at its fair share.
    """

    def __init__(self):
        self.arms = [f"op{i:03d}" for i in range(100)]
        self.probs = {a: (0.01 if i % 3 == 0 else 0.002) for i, a in enumerate(self.arms)}
        for a in self.arms[10:14]:
            self.probs[a] = 0.03
        self.best = self.arms[57]
        self.probs[self.best] = 0.08

    def p(self, arm, t):  # noqa: ARG002 - stationary
        return self.probs[arm]

    def p_max(self, t):  # noqa: ARG002 - stationary
        return self.probs[self.best]


def test_many_low_rate_arms_beats_discounted_ucb():
    """Measured over seeds 1-5 and 92 at 20k pulls: MOSS 1150-1243
    successes with 0.82-0.90 of the tail on the best arm, DUCB 168-196
    (uniform selection expects ~129). DUCB's 10k-pull memory leaves ~100
    discounted pulls per arm, too few for its width to separate 0.08 from
    0.01."""
    env = _ManyLowRateArms()
    moss = run(MOSSScheduler(rng=RandPool(92)), env, seed=92, rounds=20_000)
    ducb = run(DUCBScheduler(rng=RandPool(92)), env, seed=92, rounds=20_000)
    assert moss.successes > 4 * ducb.successes
    assert moss.tail_share(env.best) > 0.7


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

_TARGET = Path(__file__).resolve().parent.parent / "targets" / "test_target"


@pytest.mark.skipif(not _TARGET.exists(), reason="targets/test_target not built")
def test_fuzzer_wiring_selects_and_learns(tmp_path):
    """--moss builds it with the fuzzer's pool, it selects ahead of the
    bandit without Elo, and it learns from its rounds."""
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
        moss=True,
        moss_gamma=0.999,
        mc_bandit=True,
    )
    assert isinstance(f._moss, MOSSScheduler)
    assert f._moss.gamma == 0.999
    assert f._moss._rng is f._rng
    assert f._track_op_effect

    selectors = set()
    for i in range(40):
        f.fuzz_one(bytes([65 + i % 26]) * 16)
        selectors.add(f._op_selector)
    assert "moss" in selectors
    assert "bandit" not in selectors
    assert f._moss.bandit_stats()["moss_pulls"] > 0
