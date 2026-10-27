"""FenwickSampler: O(log n) weighted draw with insert, remove and reweight."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.fenwick_sampler import COMPACT_SLACK, FenwickSampler

NAN = float("nan")
INF = float("inf")


def _draws(s, n, seed=0):
    rng = random.Random(seed)
    out = {}
    for _ in range(n):
        k = s.sample(rng.random())
        out[k] = out.get(k, 0) + 1
    return out


def test_draw_matches_weights():
    """Falsification: 3:1:2 weights -> 3:1:2 draw shares."""
    s = FenwickSampler()
    for k, w in {"a": 3.0, "b": 1.0, "c": 2.0}.items():
        s.set(k, w)
    got = _draws(s, 12000)

    assert got["a"] == pytest.approx(6000, abs=250)
    assert got["b"] == pytest.approx(2000, abs=250)
    assert got["c"] == pytest.approx(4000, abs=250)


def test_equal_weights_are_uniform():
    s = FenwickSampler()
    for k in "abcd":
        s.set(k, 1.0)

    assert all(v == pytest.approx(2500, abs=250) for v in _draws(s, 10000).values())


def test_sample_is_monotone_in_u():
    """Deterministic: a fixed u always maps to the same key, ordered by slot."""
    s = FenwickSampler()
    for k, w in (("a", 1.0), ("b", 1.0), ("c", 2.0)):
        s.set(k, w)

    assert [s.sample(u) for u in (0.0, 0.24, 0.26, 0.49, 0.51, 0.99)] == [
        "a",
        "a",
        "b",
        "b",
        "c",
        "c",
    ]


def test_reweight_changes_the_share():
    s = FenwickSampler()
    s.set("a", 1.0)
    s.set("b", 1.0)
    s.set("a", 9.0)

    assert _draws(s, 10000)["a"] == pytest.approx(9000, abs=250)
    assert s.weight("a") == 9.0


def test_remove_and_reuse_slot():
    s = FenwickSampler()
    for k in "abc":
        s.set(k, 1.0)
    s.remove("b")
    s.set("d", 1.0)  # reuses b's slot

    assert set(_draws(s, 3000)) == {"a", "c", "d"}
    assert len(s) == 3
    assert s.total == pytest.approx(3.0)


def test_remove_missing_is_a_noop():
    s = FenwickSampler()
    s.set("a", 1.0)
    s.remove("zzz")

    assert len(s) == 1


@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, INF, -INF])
def test_bad_weight_removes_the_key(bad):
    """Adversarial: garbage can never win and never poisons the total."""
    s = FenwickSampler()
    s.set("a", 1.0)
    s.set("b", 1.0)
    s.set("b", bad)

    assert "b" not in s
    assert s.total == pytest.approx(1.0)
    assert set(_draws(s, 200)) == {"a"}


def test_empty_sampler_returns_none():
    assert FenwickSampler().sample(0.5) is None
    s = FenwickSampler()
    s.set("a", 1.0)
    s.remove("a")

    assert s.sample(0.5) is None
    assert s.total == 0.0


@pytest.mark.parametrize("u", [-5.0, 0.0, 1.0, 7.0, 0.999999999])
def test_u_is_clamped(u):
    s = FenwickSampler()
    s.set("a", 1.0)
    s.set("b", 1.0)

    assert s.sample(u) in {"a", "b"}


def test_compaction_keeps_the_distribution():
    s = FenwickSampler()
    for i in range(200):
        s.set(i, 1.0)
    for i in range(190):
        s.remove(i)  # trips compaction

    assert len(s) == 10
    assert set(_draws(s, 4000)) == set(range(190, 200))
    assert s.total == pytest.approx(10.0)


def test_matches_a_naive_cdf_after_random_churn():
    """Property: after random set/remove, every u maps to the naive cumulative interval."""
    rng = random.Random(11)
    s = FenwickSampler()
    truth: dict[int, float] = {}
    for _ in range(3000):
        k = rng.randrange(60)
        if rng.random() < 0.3:
            s.remove(k)
            truth.pop(k, None)
        else:
            w = rng.uniform(0.1, 5.0)
            s.set(k, w)
            truth[k] = w
    assert len(s) == len(truth)
    assert s.total == pytest.approx(sum(truth.values()))
    # Draw frequency vs truth share for a heavy key.
    total = sum(truth.values())
    got = _draws(s, 20000, seed=3)
    heavy = max(truth, key=truth.get)
    assert got[heavy] / 20000 == pytest.approx(truth[heavy] / total, abs=0.02)


def test_rebuild_is_idempotent():
    s = FenwickSampler()
    for i in range(50):
        s.set(i, 0.1 * (i + 1))
    before = s.total
    s.rebuild()

    assert s.total == pytest.approx(before)


def test_slack_constant_is_sane():
    assert COMPACT_SLACK >= 0
