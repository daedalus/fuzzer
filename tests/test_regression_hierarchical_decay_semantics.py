"""op_hierarchical decay and fractional rewards vs op_monte_carlo's math.

Two Thompson schedulers are armed in the same campaigns, so they must agree
about what forgotten evidence means. The 2026-11-24 review found they did
not, and one of the two is wrong by the other's documented derivation:

* Decay: op_monte_carlo decays each posterior toward its prior — forgotten
  evidence reads "unknown" (Beta(1,1)), not Beta(eps, eps). op_hierarchical
  multiplied both parameters by the decay factor, which preserves the
  posterior mean while evaporating the concentration: an arm that always
  succeeded kept mean 1.0 forever while its evidence vanished.
* Fractional reward: a success worth w is w of a hit and 1 - w of a miss
  (alpha/(alpha+beta) converges to E[reward]). op_hierarchical added only w
  to alpha, which converges to p*E[w] / (p*E[w] + 1 - p) instead.
"""

import pytest

from fuzzer_tool.core.schedulers.op_hierarchical import HierarchicalBanditScheduler

BASE_OPS = ["bit_flip", "byte_flip", "arith8"]

#: Uncategorized: recording it leaves bit_flip's op and category posteriors alone.
OTHER = "zz_other"


def _armed(**kwargs):
    sched = HierarchicalBanditScheduler(**kwargs)
    for op in BASE_OPS:
        sched.init_arm(op)
    return sched


class TestDecayTowardPrior:
    def test_decay_moves_toward_the_prior_not_zero(self):
        """The always-succeeding arm (101, 1) reads unknown after decay.

        Proportional decay left it at (101d, d): same 101/102 mean,
        vanishing evidence. Prior-anchored decay moves both parameters
        toward 1.0.
        """
        sched = _armed()
        sched.op_alpha["bit_flip"] = 101.0
        sched.op_beta["bit_flip"] = 1.0
        sched._total_pulls = sched.decay_interval - 1

        sched.record(OTHER, success=False)

        d = sched.arm_decay
        assert sched.op_alpha["bit_flip"] == pytest.approx(1.0 + 100.0 * d)
        assert sched.op_beta["bit_flip"] == pytest.approx(1.0)

    def test_repeated_decay_converges_to_fresh_prior(self):
        """Full forgetting lands on Beta(1,1), never near Beta(0, 0)."""
        sched = _armed(arm_decay=0.5, decay_interval=1)
        sched.op_alpha["bit_flip"] = 1000.0
        sched.op_beta["byte_flip"] = 1000.0

        for _ in range(80):  # 1000 * 0.5**80 is far below any epsilon
            sched.record(OTHER, success=False)

        assert sched.op_alpha["bit_flip"] == pytest.approx(1.0, abs=1e-6)
        assert sched.op_beta["byte_flip"] == pytest.approx(1.0, abs=1e-6)

    def test_category_level_decays_toward_prior_too(self):
        sched = _armed(arm_decay=0.5, decay_interval=1)
        cat = sched._cat_for("bit_flip")
        sched.cat_alpha[cat] = 500.0
        sched.cat_beta[cat] = 1.0

        sched.record(OTHER, success=False)

        assert sched.cat_alpha[cat] == pytest.approx(1.0 + 499.0 * 0.5)
        assert sched.cat_beta[cat] == pytest.approx(1.0)


class TestDecayClock:
    """Decay ticks on the record clock, once per ``decay_interval`` records.

    It ran in select_op() against a counter only record() advances: several
    selects per exec on a tick decayed several times, and records that
    stepped over a tick between selects skipped it.
    """

    def test_regression_select_does_not_decay(self):
        sched = _armed(arm_decay=0.5, decay_interval=1)
        sched.op_alpha["bit_flip"] = 101.0
        sched._total_pulls = 1

        for _ in range(5):
            sched.select_op(BASE_OPS)

        assert sched.op_alpha["bit_flip"] == 101.0

    def test_regression_decay_fires_once_per_interval(self):
        """7 records at interval 3 tick twice: 1 + 8 * 0.5**2."""
        sched = _armed(arm_decay=0.5, decay_interval=3)
        sched.op_alpha["bit_flip"] = 9.0

        for _ in range(7):
            sched.record(OTHER, success=False)

        assert sched.op_alpha["bit_flip"] == pytest.approx(1.0 + 8.0 * 0.5**2)

    def test_interleaved_selects_do_not_change_the_count(self):
        """Adversarial: selects between records on a tick add no decay."""
        sched = _armed(arm_decay=0.5, decay_interval=3)
        sched.op_alpha["bit_flip"] = 9.0

        for _ in range(7):
            sched.record(OTHER, success=False)
            for _ in range(10):
                sched.select_op(BASE_OPS)

        assert sched.op_alpha["bit_flip"] == pytest.approx(1.0 + 8.0 * 0.5**2)


class TestFractionalReward:
    def test_success_adds_complement_to_beta(self):
        """A success worth 0.25 is 0.25 hit + 0.75 miss, both levels."""
        sched = _armed()
        sched.record("bit_flip", success=True, weight=0.25)
        cat = sched._cat_for("bit_flip")

        assert sched.op_alpha["bit_flip"] == pytest.approx(1.25)
        assert sched.op_beta["bit_flip"] == pytest.approx(1.75)
        assert sched.cat_alpha[cat] == pytest.approx(1.25)
        assert sched.cat_beta[cat] == pytest.approx(1.75)

    def test_full_weight_success_leaves_beta_unchanged(self):
        """weight=1.0 is a plain hit: the complement is 0, as before."""
        sched = _armed()
        sched.record("bit_flip", success=True, weight=1.0)

        assert sched.op_alpha["bit_flip"] == pytest.approx(2.0)
        assert sched.op_beta["bit_flip"] == pytest.approx(1.0)

    def test_posterior_mean_converges_to_expected_reward(self):
        """alpha/(alpha+beta) tracks E[reward] for fractional successes.

        The one-sided update converged to p*E[w] / (p*E[w] + 1 - p): an
        arm "succeeding" at w=0.1 read ~0.091 under it, and 1.0 for
        p=1/w=0.1 scenarios; the fractional update reads 0.1.
        """
        sched = _armed()
        for _ in range(2000):
            sched.record("bit_flip", success=True, weight=0.1)

        a, b = sched.op_alpha["bit_flip"], sched.op_beta["bit_flip"]
        assert a / (a + b) == pytest.approx(0.1, abs=0.01)
