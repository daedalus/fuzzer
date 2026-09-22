"""``detect_chained_clusters`` must catch single-linkage chaining without
touching ``cluster_crashes``'s own output.

Single-linkage only ever checks the one link that triggers a merge, so a
cluster can end up holding two members that are nowhere near each other,
bridged by some intermediate member. This file builds that scenario by
hand (A--B and B--C both clear the clustering threshold, but A--C does
not clear the looser diagnostic threshold) and checks:

* ``cluster_crashes`` really does chain A, B, C into one cluster (sanity
  check on the fixture, not the feature under test).
* ``detect_chained_clusters`` flags that cluster and reports its true
  worst pairwise similarity.
* A cluster whose members are all mutually close is left unflagged.
* Singleton clusters are never flagged.
* ``max_diagnostic_size`` skips oversized clusters rather than scanning
  them, and is independent of whether they *would* have been flagged.
* ``configure_crash_cluster``'s ``core_threshold`` is what a ``None``
  argument to ``detect_chained_clusters`` falls back to.
"""

import pytest

from fuzzer_tool.core.crash_metadata import (
    cluster_crashes,
    configure_crash_cluster,
    detect_chained_clusters,
)

# A--B differ in the last 6 of 20 chars (sim 0.70, clears the 0.7 cluster
# threshold exactly). B--C differ in a *different* 6 chars (also sim 0.70).
# A--C, compared directly, differ in 12 of 20 (sim 0.40) -- below both the
# 0.7 cluster threshold and the 0.5 diagnostic core threshold. So A and C
# only ever end up together by chaining through B.
_A = "A" * 20
_B = "A" * 14 + "B" * 6
_C = "C" * 6 + "A" * 8 + "B" * 6


def test_fixture_actually_chains():
    """Sanity check on the fixture itself, not the diagnostic."""
    clusters = cluster_crashes([_A, _B, _C], None, 0.7)
    assert len(clusters) == 1, f"expected A/B/C chained into one cluster, got {clusters}"


def test_chained_cluster_is_flagged_with_true_minimum():
    clusters = cluster_crashes([_A, _B, _C], None, 0.7)
    flagged = detect_chained_clusters(clusters, [_A, _B, _C], None)
    assert 0 in flagged
    assert flagged[0] == pytest.approx(0.4, abs=1e-9)


def test_cluster_crashes_output_is_unmodified_by_the_diagnostic():
    """Calling detect_chained_clusters must not mutate or resize clusters."""
    clusters = cluster_crashes([_A, _B, _C], None, 0.7)
    before = [list(c) for c in clusters]
    detect_chained_clusters(clusters, [_A, _B, _C], None)
    assert clusters == before


def test_tight_cluster_is_not_flagged():
    """All-mutually-close members must not be flagged, even at 3+ members."""
    base = "X" * 20
    s2 = base[:19] + "Y"  # 1 char from base
    s3 = base[:18] + "YY"  # 2 chars from base, 1 from s2
    sigs = [base, s2, s3]
    clusters = cluster_crashes(sigs, None, 0.7)
    assert len(clusters) == 1  # sanity: fixture does cluster
    flagged = detect_chained_clusters(clusters, sigs, None)
    assert flagged == {}


def test_singletons_never_flagged():
    sigs = [_A, "completely different signature entirely"]
    clusters = cluster_crashes(sigs, None, 0.7)
    assert all(len(c) == 1 for c in clusters)  # sanity: no merge happens
    assert detect_chained_clusters(clusters, sigs, None) == {}


def test_degenerate_inputs():
    assert detect_chained_clusters([], [], None) == {}
    assert detect_chained_clusters([[0]], ["only one"], None) == {}


def test_max_diagnostic_size_skips_oversized_clusters():
    # Same chain as above, plus a fourth member identical to C so the
    # cluster has 4 members but the same worst pair (A, C or A, D).
    sigs = [_A, _B, _C, _C]
    clusters = cluster_crashes(sigs, None, 0.7)
    assert len(clusters) == 1 and len(clusters[0]) == 4  # sanity

    # Large enough cap: flagged as before.
    flagged_uncapped = detect_chained_clusters(clusters, sigs, None, max_diagnostic_size=10)
    assert 0 in flagged_uncapped

    # Cap below the cluster's size: skipped, not "clean".
    flagged_capped = detect_chained_clusters(clusters, sigs, None, max_diagnostic_size=3)
    assert flagged_capped == {}


def test_core_threshold_override_changes_the_call():
    clusters = cluster_crashes([_A, _B, _C], None, 0.7)
    # Strict enough that even the tight cluster below would fail if reused
    # here; loose enough that our 0.4 worst pair still trips it.
    assert 0 in detect_chained_clusters(clusters, [_A, _B, _C], None, core_threshold=0.45)
    # Looser than the actual worst pair (0.4): no longer flagged.
    assert detect_chained_clusters(clusters, [_A, _B, _C], None, core_threshold=0.3) == {}


def test_configure_crash_cluster_sets_the_default_core_threshold():
    clusters = cluster_crashes([_A, _B, _C], None, 0.7)
    try:
        configure_crash_cluster(threshold=0.7, core_threshold=0.3)
        # 0.3 is looser than the fixture's 0.4 worst pair: default now clean.
        assert detect_chained_clusters(clusters, [_A, _B, _C], None) == {}
    finally:
        configure_crash_cluster(threshold=0.7, core_threshold=0.5)
