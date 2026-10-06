"""BinRates: per-seed Beta-style rates over offset bins.

Covers core/schedulers/_bin_rates.py, shared by the ``changed`` and
``rare_mask`` position arms.
"""

import numpy as np

from fuzzer_tool.core.schedulers._bin_rates import MAX_BINS, MAX_SEEDS, BinRates


class ScriptedRng:
    """Scripted random()/randint(); randint asserts and logs its bounds."""

    def __init__(self, randoms=(), ints=()):
        self._randoms = list(randoms)
        self._ints = list(ints)
        self.bounds = []

    def random(self):
        return self._randoms.pop(0)

    def randint(self, a, b):
        self.bounds.append((a, b))
        v = self._ints.pop(0) if self._ints else a
        assert a <= v <= b
        return v


SEED = bytes(100)


def _expected_bin(weights, r):
    """Independent oracle: first bin whose running total exceeds r * total."""
    total = sum(weights)
    run = 0.0
    for i, w in enumerate(weights):
        run += w
        if r * total < run:
            return i
    return len(weights) - 1


class TestCredit:
    def test_unseen_seed_declines(self):
        assert BinRates(ScriptedRng(), 1.0, 1.0).propose(SEED, len(SEED)) is None

    def test_credit_counts_per_bin(self):
        br = BinRates(ScriptedRng(), 1.0, 1.0)
        br.credit(SEED, [3, 3, 50], 0.5)
        n, s = br.counts(SEED)
        assert n[3] == 2 and s[3] == 1.0
        assert n[50] == 1 and s[50] == 0.5

    def test_adversarial_negative_and_past_end_offsets(self):
        br = BinRates(ScriptedRng(), 1.0, 1.0)
        br.credit(SEED, [-1, 10_000], 1.0)
        n, s = br.counts(SEED)
        assert n[-1] == 1 and n.sum() == 1  # past-end clamps to the last bin

    def test_reset_forgets_a_seed(self):
        br = BinRates(ScriptedRng(), 1.0, 1.0)
        br.credit(SEED, [3], 1.0)
        br.reset(SEED)
        assert br.counts(SEED) is None

    def test_seeds_bounded(self):
        br = BinRates(ScriptedRng(), 1.0, 1.0)
        for i in range(MAX_SEEDS + 3):
            br.credit(SEED + bytes([i % 256, i // 256]), [0], 1.0)
        assert br.seed_count() == MAX_SEEDS

    def test_large_seed_is_binned(self):
        data = bytes(MAX_BINS * 3)
        br = BinRates(ScriptedRng(), 1.0, 1.0)
        br.credit(data, [len(data) - 1], 1.0)
        n, _ = br.counts(data)
        assert len(n) == MAX_BINS and n[-1] == 1

    def test_width_is_bytes_per_bin_and_none_when_unseen(self):
        br = BinRates(ScriptedRng(), 1.0, 1.0)
        data = bytes(MAX_BINS * 3)
        assert br.width(data) is None
        br.credit(data, [0], 1.0)
        assert br.width(data) == 3
        br.credit(SEED, [0], 1.0)
        assert br.width(SEED) == 1


class TestPropose:
    def test_picks_the_oracle_bin(self):
        br = BinRates(ScriptedRng(), 1.0, 1.0)
        br.credit(SEED, [10, 10, 10], 1.0)  # bin 10 -> rate 4/5
        br.credit(SEED, [20, 20, 20], 0.0)  # bin 20 -> rate 1/5
        n, s = br.counts(SEED)
        weights = [(s[i] + 1) / (n[i] + 2) for i in range(len(SEED))]
        for r in (0.0, 0.1, 0.105, 0.5, 0.999):
            br._rng = ScriptedRng(randoms=[r])
            assert br.propose(SEED, len(SEED)) == _expected_bin(weights, r)

    def test_falsification_high_rate_bin_wins_more_mass(self):
        br = BinRates(ScriptedRng(), 1.0, 1.0)
        br.credit(SEED, [7] * 50, 1.0)
        br.credit(SEED, [8] * 50, 0.0)
        w = br.weights(SEED, len(SEED))
        assert w[7] > w[0] > w[8]

    def test_adversarial_shrunk_buffer(self):
        br = BinRates(ScriptedRng(randoms=[0.999]), 1.0, 1.0)
        br.credit(SEED, [90], 1.0)
        assert br.propose(SEED, 5) == 4

    def test_wide_bin_draws_inside_the_buffer(self):
        data = bytes(MAX_BINS * 4)  # width 4
        rng = ScriptedRng(randoms=[0.999], ints=[1])
        br = BinRates(rng, 1.0, 1.0)
        br.credit(data, [0], 1.0)
        assert br.propose(data, 10) == 9  # last live bin [8, 10) -> 8 + 1
        assert rng.bounds == [(0, 1)]

    def test_weights_vectorised_matches_loop(self):
        br = BinRates(ScriptedRng(), 2.0, 5.0)
        br.credit(SEED, list(range(0, 100, 3)), 0.25)
        n, s = br.counts(SEED)
        loop = [(s[i] + 2.0) / (n[i] + 7.0) for i in range(len(SEED))]
        assert np.allclose(br.weights(SEED, len(SEED)), loop)
