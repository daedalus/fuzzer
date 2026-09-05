"""Regression: the CEM refit interval was a one-way ratchet to base//4.

``_adapt_interval`` compared ``last_js_divergence`` -- a Jensen-Shannon
divergence in nats, bounded by ln 2 -- against ``ks_significance_threshold``, a
critical value for the Kolmogorov-Smirnov statistic, which is a sup-norm
distance between CDFs. Different quantities on different scales. It then derived
that value's sample size from ``sum(arm_alpha) + sum(arm_beta)``, the bandit's
pull counts, which have nothing to do with the elite-set byte histograms being
compared. Two unit errors stacked.

Measured over 8 refits with 60 elites: JS pinned at ~0.63 (91% of the ln 2
ceiling) against thresholds of 0.026-0.072, so the "stable" branch was
unreachable after the first refit -- where JS is 0 only because there is no
previous snapshot -- and the interval went 2000 -> 1000 -> 500 -> 250 and stayed
at base//4 for the rest of the campaign.

The saturated JS was not evidence that the distribution was moving. JS between
two *empirical* distributions does not approach zero when the underlying
distribution is unchanged: with a 256-value support sampled a few dozen times,
two draws share few cells. Measured on two independent draws from one fixed
distribution: JS 0.46 at 2 elites, 0.37 at 6, 0.33 at 20, 0.29 at 100. So "how
small is small" depends on the sample size and cannot be read off any table.

The reference is now a permutation null -- pool the two samples, re-split at the
observed sizes -- which is exact under exchangeability at any sparsity. A
parametric bootstrap from the pooled *proportions* was tried first and is biased
low here: the pooled support is roughly twice either sample's, so replicates
overlap more than two real draws do and the reference landed at 0.43 against a
true null near 0.63.

Measured calibration over 200 trials, 60 elites, 8-symbol alphabet:

    unchanged distribution:  double 93-94%, hold 4-5%, halve 2-3%
    genuinely shifted:       halve 100%

Before the fix the unchanged case halved 100% of the time.
"""

import math
import random

import pytest

from fuzzer_tool.core.schedulers.monte_carlo import MonteCarloScheduler

STABLE_ALPHABET = b"ABCDEFGH"
SHIFTED_ALPHABET = b"MNOPQRST"


def _elite(rng, n, alphabet, length=64):
    return [
        (rng.random(), bytes(rng.choice(alphabet) for _ in range(length))) for _ in range(n)
    ]


def _pull_arms(mc, pulls=2000.0):
    """Give the bandit a realistic pull history.

    The old threshold was ks_significance_threshold(sum(arm_alpha) +
    sum(arm_beta)), so with no pulls it evaluated to 1.36 and everything looked
    stable. The ratchet only appears once the bandit has run, which it always
    has in a real campaign -- so a test that reproduces the defect has to supply
    the pulls. Without them these tests pass against the broken code.
    """
    for arm in ("havoc", "splice", "bitflip", "arith"):
        mc.arm_alpha[arm] = pulls / 4
        mc.arm_beta[arm] = pulls / 4


def _two_refits(rng, shifted, base=1000, n=60):
    mc = MonteCarloScheduler(refit_interval=base)
    mc.elite_set = _elite(rng, n, STABLE_ALPHABET)
    _pull_arms(mc)
    mc.maybe_refit()
    mc.elite_set = _elite(rng, n, SHIFTED_ALPHABET if shifted else STABLE_ALPHABET)
    _pull_arms(mc)
    before = mc.refit_interval
    mc.maybe_refit()
    return mc, before


class TestRatchet:
    def test_unchanged_distribution_does_not_halve_forever(self):
        """The measured defect: resampling one distribution ratcheted to base//4."""
        rng = random.Random(5)
        mc = MonteCarloScheduler(refit_interval=1000)
        for _ in range(8):
            mc.elite_set = _elite(rng, 60, STABLE_ALPHABET)
            _pull_arms(mc)
            mc.maybe_refit()
        assert mc.refit_interval > 1000, (
            f"interval fell to {mc.refit_interval} on an unchanged distribution"
        )

    def test_shifting_distribution_still_halves(self):
        """The fix must not simply stop the interval from moving."""
        rng = random.Random(11)
        mc = MonteCarloScheduler(refit_interval=1000)
        for alphabet in (b"AB", b"CD", b"EF", b"GH", b"IJ"):
            mc.elite_set = _elite(rng, 60, alphabet)
            _pull_arms(mc)
            mc.maybe_refit()
        assert mc.refit_interval == 250  # base // 4

    def test_stable_branch_is_reachable(self):
        """It was unreachable after the first refit, where JS is 0 only because
        there is no previous snapshot to compare against."""
        rng = random.Random(5)
        mc, before = _two_refits(rng, shifted=False)
        assert mc.refit_interval > before


class TestCalibration:
    @pytest.mark.parametrize(
        "shifted,expected_action,min_rate",
        [(False, "double", 0.80), (True, "halve", 0.95)],
    )
    def test_decision_rates(self, shifted, expected_action, min_rate):
        rng = random.Random(11)
        actions = {"double": 0, "hold": 0, "halve": 0}
        trials = 120
        for _ in range(trials):
            mc, before = _two_refits(rng, shifted=shifted)
            if mc.refit_interval > before:
                actions["double"] += 1
            elif mc.refit_interval < before:
                actions["halve"] += 1
            else:
                actions["hold"] += 1
        rate = actions[expected_action] / trials
        assert rate >= min_rate, f"{expected_action} rate {rate:.2f} < {min_rate}: {actions}"

    def test_unchanged_rarely_halves(self):
        """The false-positive branch. Nominally ~1%, allow slack for 120 trials
        and the reduced replicate count under the time budget."""
        rng = random.Random(23)
        halved = 0
        trials = 120
        for _ in range(trials):
            mc, before = _two_refits(rng, shifted=False)
            if mc.refit_interval < before:
                halved += 1
        assert halved / trials < 0.15


class TestNullReference:
    def test_null_is_estimated_not_tabulated(self):
        """A KS critical value depends only on n; the reference here must move
        with the distributions themselves."""
        rng = random.Random(3)
        mc_narrow = MonteCarloScheduler(refit_interval=100)
        for _ in range(2):
            mc_narrow.elite_set = _elite(rng, 60, b"AB")
            mc_narrow.maybe_refit()

        mc_wide = MonteCarloScheduler(refit_interval=100)
        for _ in range(2):
            mc_wide.elite_set = _elite(rng, 60, bytes(range(256)))
            mc_wide.maybe_refit()

        # A 256-value support at the same n leaves far more room for
        # sampling noise than a 2-value one.
        assert mc_wide.last_js_null_p95 > mc_narrow.last_js_null_p95

    def test_null_quantiles_are_ordered_and_bounded(self):
        rng = random.Random(7)
        mc = MonteCarloScheduler(refit_interval=100)
        for _ in range(2):
            mc.elite_set = _elite(rng, 60, STABLE_ALPHABET)
            mc.maybe_refit()
        assert 0.0 < mc.last_js_null_p95 <= mc.last_js_null_p99 <= math.log(2) + 1e-12

    def test_bandit_pull_counts_do_not_move_the_reference(self):
        rng = random.Random(13)
        values = []
        for pulls in (2.0, 2000.0, 200000.0):
            mc = MonteCarloScheduler(refit_interval=100)
            trial_rng = random.Random(13)
            for _ in range(2):
                mc.elite_set = _elite(trial_rng, 60, STABLE_ALPHABET)
                mc.arm_alpha["a"] = pulls
                mc.arm_beta["a"] = pulls
                mc.maybe_refit()
            values.append(mc.last_js_null_p95)
        spread = max(values) - min(values)
        assert spread < 0.05, f"reference still tracks pull counts: {values}"
        del rng


class TestNullBackends:
    """The pure-Python path must agree with the numpy one."""

    @staticmethod
    def _pooled(rng, n=30, positions=6):
        pooled = []
        for _ in range(positions):
            counts = {}
            for _ in range(2 * n):
                v = rng.randrange(16)
                counts[v] = counts.get(v, 0) + 1
            pooled.append((counts, n, n))
        return pooled

    def test_backends_agree_in_distribution(self):
        rng = random.Random(31)
        pooled = self._pooled(rng)
        np_samples = MonteCarloScheduler._null_js_samples_numpy(pooled, 60)
        py_samples = MonteCarloScheduler._null_js_samples_python(pooled, 60)
        assert np_samples and py_samples
        np_mean = sum(np_samples) / len(np_samples)
        py_mean = sum(py_samples) / len(py_samples)
        assert abs(np_mean - py_mean) < 0.02, (np_mean, py_mean)

    def test_permutation_preserves_pooled_totals(self):
        """A re-split must move observations, never invent or drop them."""
        rng = random.Random(37)
        pooled = self._pooled(rng, n=20, positions=3)
        samples = MonteCarloScheduler._null_js_samples_numpy(pooled, 10)
        assert len(samples) == 10
        assert all(0.0 <= s <= math.log(2) + 1e-12 for s in samples)

    def test_budget_stops_early_but_not_below_the_floor(self):
        rng = random.Random(41)
        pooled = self._pooled(rng, n=40, positions=40)
        samples = MonteCarloScheduler._null_js_samples_numpy(
            pooled, 200, deadline=0.0, min_replicates=8
        )
        assert len(samples) == 8
