"""``core.misra_gries.MisraGries``: bounded heavy-hitter counts."""

from __future__ import annotations

from collections import Counter

from fuzzer_tool.core.misra_gries import MisraGries, reduce_counts

K = 4


def _reference(stream: list[str], k: int) -> dict[str, int]:
    """Textbook Misra–Gries: full table + newcomer decrements all."""
    counts: dict[str, int] = {}
    for s in stream:
        if s in counts:
            counts[s] += 1
        elif len(counts) < k:
            counts[s] = 1
        else:
            counts = {key: c - 1 for key, c in counts.items() if c > 1}
    return counts


STREAM = list("aabcadeafgahaa")


def test_matches_reference():
    # Control: the oracle agrees with a second run of itself.
    assert _reference(STREAM, K) == _reference(list(STREAM), K)

    mg = MisraGries(K)
    for s in STREAM:
        mg.add(s)

    assert dict(mg.items()) == _reference(STREAM, K)


def test_add_returns_count_or_zero_when_dropped():
    mg = MisraGries(1)
    assert mg.add("a") == 1
    assert mg.add("a") == 2
    assert mg.add("b") == 0
    assert mg.get("a") == 1
    assert "b" not in mg


def test_exact_below_capacity():
    """Fewer distinct keys than K: counts equal an exact Counter."""
    stream = list("abcabca")
    mg = MisraGries(len(set(stream)))
    for s in stream:
        mg.add(s)

    assert dict(mg.items()) == dict(Counter(stream))


def test_falsify_heavy_hitter_survives_flood():
    """A key above n/(K+1) of the stream must survive any junk order."""
    stream = []
    for i in range(50):
        stream += ["hot", f"junk{i}"]
    mg = MisraGries(K)
    for s in stream:
        mg.add(s)

    assert "hot" in mg
    assert mg.most_common(1)[0][0] == "hot"


def test_adversarial_all_distinct_stays_bounded():
    mg = MisraGries(K)
    for i in range(1000):
        mg.add(i)
        assert len(mg) <= K


def test_most_common_orders_by_count():
    mg = MisraGries(K)
    for s in "abbccc":
        mg.add(s)

    assert [k for k, _ in mg.most_common(2)] == ["c", "b"]


def _reduce_reference(counts: dict, k: int) -> dict:
    """Batch (mergeable) Misra-Gries by hand: subtract the (k+1)-th largest count."""
    if len(counts) <= k:
        return dict(counts)
    c = sorted(counts.values(), reverse=True)[k]
    return {key: v - c for key, v in counts.items() if v > c}


def test_reduce_matches_reference_and_returns_evicted():
    counts = {"a": 5, "b": 1, "c": 3, "d": 1, "e": 2}
    want = _reduce_reference(counts, 2)
    assert want == _reduce_reference(dict(counts), 2)  # control
    evicted = reduce_counts(counts, 2)
    assert counts == want == {"a": 3, "c": 1}
    assert sorted(evicted) == ["b", "d", "e"]
    assert list(counts) == ["a", "c"]  # insertion order kept


def test_reduce_under_capacity_is_noop():
    counts = Counter({"a": 1, "b": 1})
    assert reduce_counts(counts, 2) == []
    assert counts == {"a": 1, "b": 1}


def test_reduce_batched_stream_keeps_error_bound():
    """Fed in batches, any key above n/(k+1) survives and stored <= true <= stored + n/(k+1)."""
    k, counts, true = 5, Counter(), Counter()
    for i in range(300):
        batch = [i % 3, i % 4 + 10, 100 + i, 200 + i]
        counts.update(batch)
        true.update(batch)
        reduce_counts(counts, k)
        assert len(counts) <= k
    n = sum(true.values())
    for key, c in true.items():
        assert c - n / (k + 1) <= counts.get(key, 0) <= c
        if c > n / (k + 1):
            assert key in counts


def test_reduce_adversarial_ties_and_huge_keys():
    """All tied: everything goes. Keys need only hash (ids beyond int64)."""
    big = 1 << 80
    counts = {big + i: 1 for i in range(4)}
    assert sorted(reduce_counts(counts, 3)) == [big + i for i in range(4)]
    assert counts == {}
