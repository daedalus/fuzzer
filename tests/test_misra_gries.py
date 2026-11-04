"""``core.misra_gries.MisraGries``: bounded heavy-hitter counts."""

from __future__ import annotations

from collections import Counter

from fuzzer_tool.core.misra_gries import MisraGries

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
