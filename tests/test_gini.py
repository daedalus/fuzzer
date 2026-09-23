"""Tests for core/gini.py -- Gini coefficient of a count distribution."""

import pytest

from fuzzer_tool.core.gini import gini


class TestGini:
    def test_empty_returns_none(self):
        assert gini([]) is None

    def test_single_item_is_zero(self):
        assert gini([42.0]) == 0.0

    def test_all_zero_is_zero(self):
        assert gini([0, 0, 0, 0]) == 0.0

    def test_perfectly_even_is_zero(self):
        g = gini([5.0, 5.0, 5.0, 5.0])
        assert g == pytest.approx(0.0, abs=1e-12)

    def test_maximal_inequality_approaches_one_as_n_grows(self):
        # One item holds everything, n-1 hold nothing: G = 1 - 1/n.
        g10 = gini([100.0] + [0.0] * 9)
        assert g10 == pytest.approx(1.0 - 1.0 / 10, abs=1e-9)
        g100 = gini([100.0] + [0.0] * 99)
        assert g100 == pytest.approx(1.0 - 1.0 / 100, abs=1e-9)
        # Larger population under the same concentration -> closer to 1.
        assert g100 > g10

    def test_known_value_two_items(self):
        # n=2, values [1, 3]: mean 2, mean absolute difference 2,
        # G = MAD / (2 * mean) = 2 / 4 = 0.25.
        assert gini([1.0, 3.0]) == pytest.approx(0.25, abs=1e-12)

    def test_order_independent(self):
        assert gini([1.0, 5.0, 2.0, 8.0]) == gini([8.0, 2.0, 5.0, 1.0])

    def test_negative_value_rejected(self):
        with pytest.raises(ValueError):
            gini([1.0, -2.0, 3.0])

    def test_accepts_generator(self):
        # Callers (stats.py) pass dict_values / generator expressions, not
        # just lists -- must not require a sized/re-iterable input.
        assert gini(v for v in [2.0, 2.0, 2.0]) == pytest.approx(0.0, abs=1e-12)

    def test_bounded_zero_to_one(self):
        import random

        rng = random.Random(0)
        for _ in range(20):
            n = rng.randint(1, 30)
            values = [rng.uniform(0, 100) for _ in range(n)]
            g = gini(values)
            assert 0.0 <= g <= 1.0
