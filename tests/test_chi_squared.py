"""Tests for chi_squared.py — χ² tests, p-value, ContingencyTable."""

from fuzzer_tool.core.chi_squared import (
    ContingencyTable,
    chi_squared_critical_value,
    chi_squared_goodness_of_fit,
    chi_squared_homogeneity,
    chi_squared_independence,
    chi_squared_pvalue,
    cramers_v,
)


class TestPValue:
    def test_zero_chi2(self):
        assert chi_squared_pvalue(0.0, 5) == 1.0

    def test_large_chi2(self):
        p = chi_squared_pvalue(1e6, 5)
        assert p < 1e-10

    def test_zero_dof(self):
        assert chi_squared_pvalue(10.0, 0) == 1.0

    def test_dof_1_known_value(self):
        # At dof=1, χ²=3.841 → p ≈ 0.05 (the 95th percentile)
        p = chi_squared_pvalue(3.841, 1)
        assert 0.04 < p < 0.06

    def test_dof_2_known_value(self):
        # At dof=2, χ²=5.991 → p ≈ 0.05
        p = chi_squared_pvalue(5.991, 2)
        assert 0.04 < p < 0.06

    def test_dof_10_known_value(self):
        # At dof=10, χ²=18.307 → p ≈ 0.05
        p = chi_squared_pvalue(18.307, 10)
        assert 0.04 < p < 0.06


class TestCriticalValue:
    def test_alpha_05_dof_1(self):
        cv = chi_squared_critical_value(1, 0.05)
        assert 3.8 < cv < 3.9  # ≈ 3.841

    def test_alpha_05_dof_5(self):
        cv = chi_squared_critical_value(5, 0.05)
        assert 11.0 < cv < 11.2  # ≈ 11.070

    def test_alpha_01_dof_1(self):
        cv = chi_squared_critical_value(1, 0.01)
        assert 6.6 < cv < 6.7  # ≈ 6.635

    def test_alpha_099(self):
        # p near 1 → very small critical value
        cv = chi_squared_critical_value(3, 0.99)
        assert cv < 1.0

    def test_inverse_roundtrip(self):
        """chi_squared_critical_value inverse of chi_squared_pvalue."""
        for dof in [1, 2, 5, 10, 20]:
            cv = chi_squared_critical_value(dof, 0.05)
            p = chi_squared_pvalue(cv, dof)
            assert abs(p - 0.05) < 0.001, f"dof={dof}: cv={cv}, p={p}"


class TestGoodnessOfFit:
    def test_uniform(self):
        # Fair die: all faces observed equally
        observed = [100, 100, 100, 100, 100, 100]
        chi2, p, dof = chi_squared_goodness_of_fit(observed)
        assert chi2 == 0.0
        assert p == 1.0
        assert dof == 5

    def test_biased(self):
        # Biased die: significant deviation from uniform
        observed = [50, 200, 50, 50, 50, 50]
        chi2, p, dof = chi_squared_goodness_of_fit(observed)
        assert chi2 > 0
        assert p < 0.05
        assert dof == 5

    def test_custom_expected(self):
        observed = [50, 30, 20]
        expected = [40, 40, 20]
        chi2, p, dof = chi_squared_goodness_of_fit(observed, expected)
        assert chi2 > 0
        assert dof == 2

    def test_empty_observed(self):
        import pytest

        with pytest.raises(ValueError, match="must not be empty"):
            chi_squared_goodness_of_fit([])

    def test_negative_count(self):
        import pytest

        with pytest.raises(ValueError, match="negative"):
            chi_squared_goodness_of_fit([10, -1, 5])

    def test_expected_length_mismatch(self):
        import pytest

        with pytest.raises(ValueError, match="same length"):
            chi_squared_goodness_of_fit([10, 20, 30], [15, 15])

    def test_with_n_params(self):
        observed = [60, 40]
        # n_params=1 reduces dof from 1 to 1 (floor at 1)
        chi2, p, dof = chi_squared_goodness_of_fit(observed, n_params=1)
        assert dof >= 1


class TestIndependence:
    def test_independent(self):
        # Rows and columns independent: roughly proportional
        table = [
            [30, 20],
            [15, 10],
        ]
        chi2, p, dof = chi_squared_independence(table)
        assert p > 0.05  # independent
        assert dof == 1

    def test_dependent(self):
        # Clear association
        table = [
            [90, 10],
            [10, 90],
        ]
        chi2, p, dof = chi_squared_independence(table)
        assert p < 0.001  # strongly dependent
        assert dof == 1

    def test_empty_table(self):
        import pytest

        with pytest.raises(ValueError, match="at least one row"):
            chi_squared_independence([])

    def test_small_expected(self):
        # 2x2 table with small counts (should still compute)
        table = [[2, 0], [0, 2]]
        chi2, p, dof = chi_squared_independence(table)
        assert chi2 > 0
        assert dof == 1


class TestHomogeneity:
    def test_same_distribution(self):
        # Two rows, same proportions
        table = [
            [50, 50],
            [50, 50],
        ]
        chi2, p, dof = chi_squared_homogeneity(table)
        assert p > 0.05

    def test_different_distributions(self):
        table = [
            [90, 10],
            [10, 90],
        ]
        chi2, p, dof = chi_squared_homogeneity(table)
        assert p < 0.001

    def test_larger_table(self):
        # 3x3 table
        table = [
            [30, 20, 10],
            [10, 20, 30],
            [20, 20, 20],
        ]
        chi2, p, dof = chi_squared_homogeneity(table)
        assert dof == 4
        assert 0.0 <= p <= 1.0


class TestCramersV:
    def test_perfect_association(self):
        # 2x2: 90/10 split vs 10/90 → large V
        table = [[90, 10], [10, 90]]
        chi2, _, _ = chi_squared_independence(table)
        v = cramers_v(chi2, 200, 2, 2)
        assert v > 0.5

    def test_no_association(self):
        table = [[50, 50], [50, 50]]
        chi2, _, _ = chi_squared_independence(table)
        v = cramers_v(chi2, 200, 2, 2)
        assert v < 0.1

    def test_zero_chi2(self):
        assert cramers_v(0.0, 100, 2, 2) == 0.0


class TestContingencyTable:
    def test_builder_basic(self):
        ct = ContingencyTable()
        ct.add("A", "yes", 40)
        ct.add("A", "no", 10)
        ct.add("B", "yes", 20)
        ct.add("B", "no", 30)
        chi2, p, dof = ct.chi_squared()
        assert dof == 1
        assert p < 0.05  # A and B differ

    def test_marginals(self):
        ct = ContingencyTable()
        ct.add("A", "yes", 40)
        ct.add("A", "no", 10)
        ct.add("B", "yes", 20)
        ct.add("B", "no", 30)
        assert ct.grand_total == 100.0
        assert abs(ct.row_marginals["A"] - 50.0) < 1e-9
        assert abs(ct.row_marginals["B"] - 50.0) < 1e-9
        assert abs(ct.col_marginals["yes"] - 60.0) < 1e-9
        assert abs(ct.col_marginals["no"] - 40.0) < 1e-9

    def test_observed_matrix(self):
        ct = ContingencyTable()
        ct.add("A", "x", 10)
        ct.add("A", "y", 20)
        ct.add("B", "x", 30)
        ct.add("B", "y", 40)
        obs = ct.observed
        assert len(obs) == 2
        assert len(obs[0]) == 2

    def test_residuals(self):
        ct = ContingencyTable()
        ct.add("A", "yes", 40)
        ct.add("A", "no", 10)
        ct.add("B", "yes", 30)
        ct.add("B", "no", 20)
        res = ct.residuals
        assert len(res) == 2
        assert len(res[0]) == 2

    def test_cramers_v_on_table(self):
        ct = ContingencyTable()
        ct.add("A", "yes", 90)
        ct.add("A", "no", 10)
        ct.add("B", "yes", 10)
        ct.add("B", "no", 90)
        v = ct.cramers_v()
        assert v > 0.5

    def test_frozen_after_chi_squared(self):
        import pytest

        ct = ContingencyTable()
        ct.add("A", "x", 10)
        ct.add("B", "y", 20)
        ct.chi_squared()
        with pytest.raises(RuntimeError, match="frozen"):
            ct.add("C", "z", 5)


class TestRegression:
    """Regression tests against textbook known values."""

    def test_medical_example(self):
        """Known 2x2 from epidemiology textbooks."""
        # Treatment vs Control × Recovered/Not
        table = [[20, 10], [8, 22]]
        chi2, p, dof = chi_squared_independence(table)
        # Expected χ² ≈ 8.24, p ≈ 0.004
        assert dof == 1
        assert 7.0 < chi2 < 10.0
        assert p < 0.01

    def test_mendel_pea_example(self):
        """Mendel's classic 9:3:3:1 ratio test."""
        # Observed: 315 round+yellow, 108 round+green, 101 wrinkled+yellow, 32 wrinkled+green
        # Expected: 312.75, 104.25, 104.25, 34.75 (9:3:3:1 of 556)
        observed = [315, 108, 101, 32]
        expected = [312.75, 104.25, 104.25, 34.75]
        chi2, p, dof = chi_squared_goodness_of_fit(observed, expected)
        # Known: χ² ≈ 0.47, p ≈ 0.93
        assert dof == 3
        assert 0.3 < chi2 < 0.6
        assert p > 0.5
