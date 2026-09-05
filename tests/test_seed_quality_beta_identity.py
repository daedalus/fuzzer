"""Beta with a floored parameter has a closed form; the seed picker uses it.

    Beta(a, 1) == U**(1/a)          # max of a uniforms, for integer a
    Beta(1, b) == 1 - (1-U)**(1/b)  # min of b uniforms, for integer b

Per-seed discovery is rare, so the alpha side of a seed posterior sits at
the prior for most of a campaign: 83% of arms degenerate at 500 seeds, 99%
at 8000. That makes this the one Thompson site where the identity beats the
general sampler, and it draws from `random` either way so --seed still
determines the campaign.

The statistical test below runs betavariate against *itself* as a control.
Without it a broken oracle looks like a passing test.
"""

import random

import numpy as np
import pytest
from scipy import stats

from fuzzer_tool.core.seed_quality import BayesianSeedQuality, _beta_sample


@pytest.mark.parametrize(
    ("a", "b", "u"),
    [(1.0, 1.0, 0.25), (1.0, 4.0, 0.5), (1.0, 200.0, 0.9), (3.0, 1.0, 0.5), (50.0, 1.0, 0.1)],
)
def test_closed_form_is_used_for_a_floored_parameter(monkeypatch, a, b, u):
    """Exact value for a pinned uniform draw, and no gamma sampler at all."""
    monkeypatch.setattr(random, "random", lambda: u)
    monkeypatch.setattr(
        random, "betavariate", lambda *_: pytest.fail("betavariate on a degenerate posterior")
    )
    expected = u ** (1.0 / a) if b == 1.0 else 1.0 - (1.0 - u) ** (1.0 / b)

    assert _beta_sample(a, b) == pytest.approx(expected)


def test_general_case_still_uses_betavariate(monkeypatch):
    """Falsification: the guard must not swallow non-degenerate posteriors."""
    seen = []
    monkeypatch.setattr(random, "betavariate", lambda a, b: seen.append((a, b)) or 0.5)

    assert _beta_sample(7.0, 3.0) == 0.5
    assert seen == [(7.0, 3.0)]


@pytest.mark.parametrize("a", [1.0, 2.5, 40.0])
@pytest.mark.parametrize("b", [1.0, 2.5, 40.0])
def test_sample_stays_in_the_unit_interval(a, b):
    random.seed(9)
    for _ in range(200):
        assert 0.0 <= _beta_sample(a, b) <= 1.0


@pytest.mark.parametrize(("a", "b"), [(1.0, 2.0), (1.0, 37.0), (2.0, 1.0), (12.0, 1.0)])
def test_matches_betavariate_in_distribution(a, b):
    """Two-sample KS, with betavariate against itself as the control."""
    n = 20_000
    random.seed(1234)
    ref = [random.betavariate(a, b) for _ in range(n)]
    control = [random.betavariate(a, b) for _ in range(n)]
    identity = [_beta_sample(a, b) for _ in range(n)]

    _, p_control = stats.ks_2samp(ref, control)
    _, p_identity = stats.ks_2samp(ref, identity)

    assert p_control > 0.001, "control failed — the oracle itself is broken"
    assert p_identity > 0.001


def test_degenerate_moments_match_the_analytic_values():
    """Adversarial: a wrong exponent passes KS badly but fails the mean."""
    random.seed(77)
    a, b = 1.0, 9.0
    draws = np.array([_beta_sample(a, b) for _ in range(60_000)])

    assert draws.mean() == pytest.approx(a / (a + b), abs=0.005)
    assert draws.var() == pytest.approx(a * b / ((a + b) ** 2 * (a + b + 1)), abs=0.002)


class TestSelectIndex:
    """select_seed() threw away the position, forcing a rehash to recover it."""

    def _quality(self, ids):
        q = BayesianSeedQuality()
        for sid in ids:
            q.init_seed(sid)
        return q

    def test_index_and_id_agree(self):
        ids = [f"seed{i}" for i in range(12)]
        q = self._quality(ids)
        random.seed(3)
        idx = q.select_index(ids)
        random.seed(3)

        assert ids[idx] == q.select_seed(ids)

    def test_index_in_range(self):
        ids = [f"seed{i}" for i in range(40)]
        q = self._quality(ids)
        random.seed(8)
        for _ in range(100):
            assert 0 <= q.select_index(ids) < len(ids)

    def test_single_candidate(self):
        q = self._quality(["only"])

        assert q.select_index(["only"]) == 0

    def test_duplicate_ids_resolve_to_a_real_position(self):
        """Adversarial: two corpus entries hashing alike must still index."""
        ids = ["dup", "dup", "other"]
        q = self._quality(ids)
        random.seed(2)
        for _ in range(50):
            assert q.select_index(ids) in (0, 1, 2)

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            self._quality([]).select_index([])

    def test_winner_is_the_highest_draw(self, monkeypatch):
        """Pin the policy, not a sample: highest posterior draw wins."""
        ids = ["a", "b", "c"]
        q = self._quality(ids)
        draws = iter([0.1, 0.9, 0.4])
        monkeypatch.setattr(q, "posterior_sample", lambda _sid: next(draws))

        assert q.select_index(ids) == 1
