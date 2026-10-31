"""Near-duplicate operator families (ShapleyAttribution.operator_families) and report wiring."""

import types

import pytest

from fuzzer_tool.core.shapley import ShapleyAttribution
from fuzzer_tool.services.report import _operator_redundancy


def _attribution(edges: dict[str, set[int]]) -> ShapleyAttribution:
    sa = ShapleyAttribution()
    for op, es in edges.items():
        sa.record({op}, len(es), es)
    return sa


DUPES = {
    "a": set(range(10)),
    "b": set(range(10)),
    "c": set(range(10)),
    "d": set(range(100, 110)),
    "e": set(range(100, 110)),
    "f": {200},
}


def test_groups_duplicates_drops_singletons():
    fams = _attribution(DUPES).operator_families(threshold=0.9)
    assert sorted(map(sorted, fams)) == [["a", "b", "c"], ["d", "e"]]


def test_falsify_unreachable_threshold_gives_none():
    """Falsification: Jaccard never exceeds 1, so no pair qualifies."""
    assert _attribution(DUPES).operator_families(threshold=1.01) == []


def test_partial_overlap_below_threshold_not_grouped():
    # J(a, b) = 5 / 15.
    sa = _attribution({"a": set(range(10)), "b": set(range(5, 15))})
    assert sa.operator_families(threshold=0.9) == []


def test_adversarial_transitive_chain_is_one_family():
    """a~b and b~c clear 0.9, a~c does not: components are transitive by design."""
    base = set(range(100))
    sa = _attribution(
        {"a": base, "b": base | {1000, 1001, 1002, 1003, 1004}, "c": base | set(range(1000, 1012))}
    )
    kernel = sa.operator_kernel()
    assert kernel["a"]["b"] >= 0.9 and kernel["b"]["c"] >= 0.9 > kernel["a"]["c"]

    assert sorted(map(sorted, sa.operator_families(threshold=0.9))) == [["a", "b", "c"]]


@pytest.mark.parametrize("edges", [{}, {"a": {1}}])
def test_adversarial_too_few_operators(edges):
    assert _attribution(edges).operator_families() == []


def test_layout_follows_fiedler_coordinate():
    """Order inside and between families is the spectral embedding's dim 1."""
    sa = _attribution({**DUPES, "g": set(range(5, 15)), "h": set(range(5, 15))})
    fiedler = {op: v[1] for op, v in sa.spectral_embedding(k=2).items()}

    fams = sa.operator_families(threshold=0.9)
    for fam in fams:
        assert fam == sorted(fam, key=lambda op: (fiedler[op], op))
    means = [sum(fiedler[op] for op in fam) / len(fam) for fam in fams]
    assert means == sorted(means)


def test_redundant_operators_unchanged_by_refactor():
    sa = _attribution(DUPES)
    pairs = sa.redundant_operators(threshold=0.9)
    assert {(a, b) for a, b, _ in pairs} == {("a", "b"), ("a", "c"), ("b", "c"), ("d", "e")}


def _fuzzer(sa):
    return types.SimpleNamespace(_use_shapley=sa is not None, _shapley=sa)


def test_report_lists_families():
    out = _operator_redundancy(_fuzzer(_attribution(DUPES)))
    assert "Operator Redundancy" in out
    assert "a, b, c" in out
    assert "d, e" in out
    assert " f" not in out.replace("families", "")


@pytest.mark.parametrize("sa", [None, ShapleyAttribution()])
def test_report_silent_without_data(sa):
    assert _operator_redundancy(_fuzzer(sa)) == ""


def test_report_wired_into_generate_report():
    import inspect

    from fuzzer_tool.services import report

    assert "_operator_redundancy(fuzzer)" in inspect.getsource(report.generate_report)


def test_report_caps_members_per_family():
    """A tiny target makes most operators duplicates: one row must stay readable."""
    from fuzzer_tool.services.report import _FAMILY_MEMBERS_MAX

    n = _FAMILY_MEMBERS_MAX + 5
    sa = _attribution({f"op{i:02d}": set(range(10)) for i in range(n)})
    rows = [
        line for line in _operator_redundancy(_fuzzer(sa)).splitlines() if line.startswith("    ")
    ]

    assert len(rows) == 1
    assert rows[0].count(",") == _FAMILY_MEMBERS_MAX - 1
    assert rows[0].lstrip().startswith(f"[{n}]")
    assert rows[0].endswith(f"+{n - _FAMILY_MEMBERS_MAX} more")
