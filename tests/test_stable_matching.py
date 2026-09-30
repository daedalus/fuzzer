"""Many-to-one deferred acceptance (Gale-Shapley), ``core/stable_matching.py``."""

from __future__ import annotations

import itertools

import pytest

from fuzzer_tool.core.stable_matching import UNMATCHED, deferred_acceptance


def _blocking_pairs(prefs, scores, quotas, match):
    """Brute force, independent of the algorithm: (p, r) pairs that both prefer each other."""
    held = {r: [p for p, m in enumerate(match) if m == r] for r in range(len(quotas))}
    out = []
    for p, row in enumerate(prefs):
        for r in row:
            if r == match[p]:
                break  # p reached its own partner: every later r is worse for p
            has_room = len(held[r]) < quotas[r]
            beats = any(scores[r][p] > scores[r][q] for q in held[r])
            if has_room or beats:
                out.append((p, r))
    return out


def test_textbook_one_to_one():
    """Falsification: the classic 3x3 instance has one proposer-optimal answer."""
    prefs = [[0, 1, 2], [1, 0, 2], [0, 1, 2]]
    # reviewer 0 prefers p2 > p0 > p1; reviewer 1 prefers p0 > p2 > p1
    scores = [[2.0, 1.0, 3.0], [3.0, 1.0, 2.0], [1.0, 2.0, 3.0]]

    match = deferred_acceptance(prefs, scores, [1, 1, 1])

    # By hand: p2 bumps p0 at r0, p0 bumps p1 at r1, p1 lands at r2.
    assert match == [1, 2, 0]
    assert not _blocking_pairs(prefs, scores, [1, 1, 1], match)


def test_quota_admits_many():
    """Falsification: one reviewer with quota 3 takes all three proposers."""
    match = deferred_acceptance([[0], [0], [0]], [[1.0, 2.0, 3.0]], [3])

    assert match == [0, 0, 0]


def test_quota_evicts_the_worst_held():
    """Quota 1: the reviewer keeps its top-scored proposer, the rest fall to their next choice."""
    prefs = [[0, 1], [0, 1], [0, 1]]
    scores = [[1.0, 5.0, 3.0], [0.0, 0.0, 0.0]]

    match = deferred_acceptance(prefs, scores, [1, 2])

    assert match[1] == 0  # highest score at reviewer 0
    assert match[0] == 1 and match[2] == 1


def test_stable_on_every_small_instance():
    """Exhaustive over 3 proposers x 2 reviewers: no blocking pair, quotas respected."""
    orders = list(itertools.permutations(range(2)))
    score_rows = list(itertools.permutations([1.0, 2.0, 3.0]))
    for prefs in itertools.product(orders, repeat=3):
        for s0, s1 in itertools.product(score_rows, repeat=2):
            for quotas in ([1, 1], [2, 1], [1, 2], [3, 0]):
                match = deferred_acceptance(list(prefs), [list(s0), list(s1)], quotas)
                assert not _blocking_pairs(prefs, [s0, s1], quotas, match)
                for r, q in enumerate(quotas):
                    assert match.count(r) <= q


def test_short_capacity_leaves_unmatched():
    """Adversarial: total quota < proposers -> the lowest-scored proposer is left out."""
    match = deferred_acceptance([[0], [0]], [[2.0, 1.0]], [1])

    assert match == [0, UNMATCHED]


def test_zero_quota_and_empty_prefs():
    """Adversarial: a zero-quota reviewer never admits; an empty pref list stays unmatched."""
    match = deferred_acceptance([[0, 1], []], [[9.0, 9.0], [1.0, 1.0]], [0, 1])

    assert match == [1, UNMATCHED]


def test_no_proposers():
    assert deferred_acceptance([], [[]], [1]) == []


def test_bad_reviewer_index_raises():
    """Adversarial: a preference naming a reviewer that does not exist is a caller bug."""
    with pytest.raises(IndexError):
        deferred_acceptance([[5]], [[1.0]], [1])


def test_equal_scores_are_deterministic():
    """Adversarial: ties at a reviewer resolve the same way on every call (lower index kept)."""
    prefs = [[0], [0], [0]]
    first = deferred_acceptance(prefs, [[1.0, 1.0, 1.0]], [1])

    assert first == deferred_acceptance(prefs, [[1.0, 1.0, 1.0]], [1])
    assert first == [0, UNMATCHED, UNMATCHED]
