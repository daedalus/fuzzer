"""Tests for Holm-Bonferroni / Benjamini-Hochberg multiple-testing correction."""

from types import SimpleNamespace

from fuzzer_tool.core.multiple_testing import (
    benjamini_hochberg,
    collect_and_correct,
    collect_current_pvalues,
    holm_bonferroni,
)


def _by_name(corrections):
    return {c.name: c for c in corrections}


# ── shared behavior: None handling, empty input ─────────────────────────


def test_holm_empty_input():
    assert holm_bonferroni({}) == []


def test_bh_empty_input():
    assert benjamini_hochberg({}) == []


def test_holm_all_none_returns_empty():
    assert holm_bonferroni({"a": None, "b": None}) == []


def test_bh_all_none_returns_empty():
    assert benjamini_hochberg({"a": None, "b": None}) == []


def test_holm_skips_none_entries_but_keeps_others():
    result = holm_bonferroni({"a": 0.001, "b": None, "c": 0.5})
    names = {c.name for c in result}
    assert names == {"a", "c"}


def test_bh_skips_none_entries_but_keeps_others():
    result = benjamini_hochberg({"a": 0.001, "b": None, "c": 0.5})
    names = {c.name for c in result}
    assert names == {"a", "c"}


# ── Holm-Bonferroni: known worked example ───────────────────────────────
# Classic textbook example (m=4): p = 0.01, 0.02, 0.03, 0.20 at alpha=0.05.
# Thresholds: 0.05/4=0.0125, 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05.
# 0.01<=0.0125 reject; 0.02<=0.0167? no -> stop. So only the smallest rejects.


def test_holm_bonferroni_worked_example():
    pvals = {"t1": 0.01, "t2": 0.02, "t3": 0.03, "t4": 0.20}
    result = _by_name(holm_bonferroni(pvals, alpha=0.05))
    assert result["t1"].rejected is True
    assert result["t2"].rejected is False
    assert result["t3"].rejected is False
    assert result["t4"].rejected is False


def test_holm_bonferroni_all_reject_when_all_tiny():
    pvals = {"t1": 1e-9, "t2": 1e-8, "t3": 1e-7}
    result = holm_bonferroni(pvals, alpha=0.05)
    assert all(c.rejected for c in result)


def test_holm_bonferroni_none_reject_when_all_large():
    pvals = {"t1": 0.9, "t2": 0.8, "t3": 0.99}
    result = holm_bonferroni(pvals, alpha=0.05)
    assert not any(c.rejected for c in result)


def test_holm_single_test_matches_plain_alpha():
    # With m=1, Holm's threshold degenerates to alpha itself.
    just_under = _by_name(holm_bonferroni({"a": 0.049}, alpha=0.05))
    just_over = _by_name(holm_bonferroni({"a": 0.051}, alpha=0.05))
    assert just_under["a"].rejected is True
    assert just_over["a"].rejected is False


def test_holm_more_conservative_than_uncorrected_alpha():
    # A p-value that would pass an uncorrected alpha=0.05 test can fail
    # once batched with others -- that is the entire point of the
    # correction. p=0.03 alone would reject at 0.05; batched with three
    # smaller p-values it must not, automatically, always reject.
    pvals = {"a": 0.03, "b": 0.001, "c": 0.002, "d": 0.004}
    result = holm_bonferroni(pvals, alpha=0.05)
    # a is the largest p-value in the batch; Holm's last threshold is
    # alpha/1 = 0.05 so it *can* still reject if every smaller one also
    # cleared its own threshold -- assert on the actual monotone contract
    # instead of a specific outcome for "a".
    ordered = sorted(result, key=lambda c: c.p_value)
    seen_non_reject = False
    for c in ordered:
        if seen_non_reject:
            assert not c.rejected, "rejections must be a prefix in sorted order"
        if not c.rejected:
            seen_non_reject = True


# ── Benjamini-Hochberg: known worked example ────────────────────────────
# m=4, p = 0.01, 0.02, 0.03, 0.20, alpha=0.05.
# BH thresholds: (1/4)*0.05=0.0125, (2/4)*0.05=0.025, (3/4)*0.05=0.0375,
# (4/4)*0.05=0.05. Compare sorted p to threshold: 0.01<=0.0125 ok,
# 0.02<=0.025 ok, 0.03<=0.0375 ok, 0.20<=0.05 fails. Largest k passing is 3
# (step-up finds the largest index where the inequality holds at all,
# scanning from the top) -- reject the smallest three.


def test_benjamini_hochberg_worked_example():
    pvals = {"t1": 0.01, "t2": 0.02, "t3": 0.03, "t4": 0.20}
    result = _by_name(benjamini_hochberg(pvals, alpha=0.05))
    assert result["t1"].rejected is True
    assert result["t2"].rejected is True
    assert result["t3"].rejected is True
    assert result["t4"].rejected is False


def test_bh_rejects_at_least_as_much_as_holm():
    # BH controls FDR (weaker guarantee) and is uniformly less conservative
    # than Holm's FWER control on the same batch -- the rejection set from
    # BH must be a superset of Holm's on identical input.
    pvals = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.20, "e": 0.5}
    holm = _by_name(holm_bonferroni(pvals, alpha=0.05))
    bh = _by_name(benjamini_hochberg(pvals, alpha=0.05))
    for name in pvals:
        if holm[name].rejected:
            assert bh[name].rejected, f"{name} rejected by Holm but not BH"


def test_bh_adjusted_values_are_monotone_nondecreasing_in_sorted_order():
    pvals = {"a": 0.5, "b": 0.001, "c": 0.3, "d": 0.01, "e": 0.02}
    result = sorted(benjamini_hochberg(pvals, alpha=0.05), key=lambda c: c.p_value)
    adjusted = [c.adjusted for c in result]
    assert all(a is not None for a in adjusted)
    assert all(adjusted[i] <= adjusted[i + 1] for i in range(len(adjusted) - 1))


def test_bh_adjusted_values_bounded_by_one():
    pvals = {"a": 0.9, "b": 0.95, "c": 0.99}
    result = benjamini_hochberg(pvals, alpha=0.05)
    assert all(c.adjusted <= 1.0 for c in result)


def test_bh_adjusted_at_least_raw_pvalue_scaled():
    # q(i) is built from (m/j)*p(j) terms, so for the single-test case
    # (m=1) the adjusted value must equal the raw p-value exactly.
    result = benjamini_hochberg({"only": 0.037}, alpha=0.05)
    assert len(result) == 1
    assert abs(result[0].adjusted - 0.037) < 1e-12


def test_holm_adjusted_is_none():
    # Holm's Correction objects don't carry a q-value-style adjusted figure.
    result = holm_bonferroni({"a": 0.01, "b": 0.5})
    assert all(c.adjusted is None for c in result)


# ── collect_current_pvalues: reading off a duck-typed fuzzer ───────────


def _fake_fuzzer(structure_fn=None, discovery_uniformity=None, garch=None):
    return SimpleNamespace(
        _structure_fn=structure_fn,
        _discovery_uniformity=discovery_uniformity,
        _garch=garch,
    )


def test_collect_pvalues_empty_when_nothing_wired():
    f = SimpleNamespace()  # no detector attributes at all
    assert collect_current_pvalues(f) == {}


def test_collect_pvalues_skips_structure_fn_when_not_enough_data():
    structure_fn = SimpleNamespace(dispersion_pvalue=lambda: None)
    f = _fake_fuzzer(structure_fn=structure_fn)
    assert collect_current_pvalues(f) == {}


def test_collect_pvalues_folds_structure_fn_two_sided():
    # Upper-tail p=0.01 (strongly overdispersed) folds to 0.02 two-sided.
    structure_fn = SimpleNamespace(dispersion_pvalue=lambda: 0.01)
    f = _fake_fuzzer(structure_fn=structure_fn)
    result = collect_current_pvalues(f)
    assert abs(result["structure_function_dispersion"] - 0.02) < 1e-12


def test_collect_pvalues_folds_structure_fn_lower_tail_too():
    # Upper-tail p=0.995 (strongly underdispersed) folds the same way:
    # 2*min(0.995, 0.005) = 0.01.
    structure_fn = SimpleNamespace(dispersion_pvalue=lambda: 0.995)
    f = _fake_fuzzer(structure_fn=structure_fn)
    result = collect_current_pvalues(f)
    assert abs(result["structure_function_dispersion"] - 0.01) < 1e-12


def test_collect_pvalues_includes_discovery_uniformity_verdict_p():
    discovery_uniformity = SimpleNamespace(verdict=lambda: {"p": 0.42, "n": 100})
    f = _fake_fuzzer(discovery_uniformity=discovery_uniformity)
    result = collect_current_pvalues(f)
    assert result["discovery_uniformity_dispersion"] == 0.42


def test_collect_pvalues_skips_discovery_uniformity_below_min_n():
    discovery_uniformity = SimpleNamespace(verdict=lambda: {"p": 1.0, "n": 1})
    f = _fake_fuzzer(discovery_uniformity=discovery_uniformity)
    assert collect_current_pvalues(f) == {}


def test_collect_pvalues_includes_garch_ljung_box():
    garch = SimpleNamespace(ljung_box=lambda: (12.3, 0.07))
    f = _fake_fuzzer(garch=garch)
    result = collect_current_pvalues(f)
    assert result["garch_ljung_box"] == 0.07


def test_collect_pvalues_combines_all_three():
    structure_fn = SimpleNamespace(dispersion_pvalue=lambda: 0.02)
    discovery_uniformity = SimpleNamespace(verdict=lambda: {"p": 0.5, "n": 200})
    garch = SimpleNamespace(ljung_box=lambda: (1.0, 0.9))
    f = _fake_fuzzer(structure_fn, discovery_uniformity, garch)
    result = collect_current_pvalues(f)
    assert set(result) == {
        "structure_function_dispersion",
        "discovery_uniformity_dispersion",
        "garch_ljung_box",
    }


def test_collect_and_correct_end_to_end_all_significant():
    # All three tests strongly significant on the same series -- BH should
    # reject all three even after correction.
    structure_fn = SimpleNamespace(dispersion_pvalue=lambda: 0.001)
    discovery_uniformity = SimpleNamespace(verdict=lambda: {"p": 0.002, "n": 200})
    garch = SimpleNamespace(ljung_box=lambda: (30.0, 0.003))
    f = _fake_fuzzer(structure_fn, discovery_uniformity, garch)
    result = collect_and_correct(f, alpha=0.05)
    assert len(result) == 3
    assert all(c.rejected for c in result)


def test_collect_and_correct_end_to_end_nothing_significant():
    structure_fn = SimpleNamespace(dispersion_pvalue=lambda: 0.5)
    discovery_uniformity = SimpleNamespace(verdict=lambda: {"p": 0.6, "n": 200})
    garch = SimpleNamespace(ljung_box=lambda: (0.1, 0.9))
    f = _fake_fuzzer(structure_fn, discovery_uniformity, garch)
    result = collect_and_correct(f, alpha=0.05)
    assert not any(c.rejected for c in result)


def test_collect_and_correct_borderline_single_test_no_longer_significant():
    # A single p=0.03 among three related tests: uncorrected it would pass
    # alpha=0.05. Batched with two very non-significant partners it still
    # must clear BH's own (less strict than Holm's) bar -- this checks BH
    # is actually being applied rather than p-values passed through raw.
    structure_fn = SimpleNamespace(dispersion_pvalue=lambda: 0.985)  # folds to 0.03
    discovery_uniformity = SimpleNamespace(verdict=lambda: {"p": 0.9, "n": 200})
    garch = SimpleNamespace(ljung_box=lambda: (0.1, 0.95))
    f = _fake_fuzzer(structure_fn, discovery_uniformity, garch)
    result = collect_and_correct(f, alpha=0.05)
    by_name = {c.name: c for c in result}
    # BH threshold for the smallest of 3 at alpha=0.05 is (1/3)*0.05=0.0167;
    # 0.03 > 0.0167, so it must not reject.
    assert by_name["structure_function_dispersion"].rejected is False
