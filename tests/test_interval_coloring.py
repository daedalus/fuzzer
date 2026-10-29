"""core/interval_coloring and the ``redqueen_batch`` op built on it."""

from itertools import product

import pytest

from fuzzer_tool.core.interval_coloring import color_classes, color_intervals
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

NUM_RANDOM_CASES = 200
MAX_INTERVALS = 12
MAX_POS = 20
BRUTE_FORCE_MAX = 6
MANY = 20000


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _depth(spans: list[tuple[int, int]]) -> int:
    """Max number of spans covering one byte (the max clique), by scan."""
    if not spans:
        return 0
    return max(sum(s <= x < e for s, e in spans) for x in range(max(e for _, e in spans)))


def _min_colors(spans: list[tuple[int, int]]) -> int:
    """Chromatic number by brute force over all colorings."""
    n = len(spans)
    for k in range(1, n + 1):
        for col in product(range(k), repeat=n):
            if all(
                col[i] != col[j] or not _overlap(spans[i], spans[j])
                for i in range(n)
                for j in range(i + 1, n)
            ):
                return k
    return 0


def _proper(spans, colors) -> bool:
    n = len(spans)
    return all(
        colors[i] != colors[j] or not _overlap(spans[i], spans[j])
        for i in range(n)
        for j in range(i + 1, n)
    )


def _random_spans(rng: RandPool, n: int) -> list[tuple[int, int]]:
    out = []
    for _ in range(n):
        s = rng.randint(0, MAX_POS)
        out.append((s, s + rng.randint(1, 6)))
    return out


def test_oracle_control_depth_equals_brute_force_chromatic():
    # Interval graphs are perfect: the two oracles must agree on their own.
    rng = RandPool(seed=7)
    for _ in range(40):
        spans = _random_spans(rng, rng.randint(0, BRUTE_FORCE_MAX))
        assert _depth(spans) == _min_colors(spans)


def test_disjoint_share_one_color():
    assert color_intervals([0, 4, 8], [4, 8, 12]) == [0, 0, 0]


def test_nested_and_crossing_need_depth_colors():
    # [0,10) holds [2,4) and [3,6): depth 3 at byte 3.
    colors = color_intervals([0, 2, 3], [10, 4, 6])
    assert sorted(colors) == [0, 1, 2]


def test_color_reuse_after_end():
    # [0,2) ends before [2,5) starts: half-open, so they share a color.
    assert color_intervals([0, 1, 2], [2, 3, 5]) == [0, 1, 0]


def test_optimal_and_proper_on_random():
    rng = RandPool(seed=1234)
    for _ in range(NUM_RANDOM_CASES):
        spans = _random_spans(rng, rng.randint(0, MAX_INTERVALS))
        colors = color_intervals([s for s, _ in spans], [e for _, e in spans])
        assert _proper(spans, colors)
        assert (max(colors) + 1 if colors else 0) == _depth(spans)


def test_classes_partition_indices():
    classes = color_classes([0, 1, 5], [3, 4, 6])
    assert sorted(i for c in classes for i in c) == [0, 1, 2]
    assert classes == [[0, 2], [1]]


# --- adversarial ---------------------------------------------------------


def test_empty():
    assert color_intervals([], []) == []
    assert color_classes([], []) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        color_intervals([0, 1], [2])


def test_empty_interval_raises():
    with pytest.raises(ValueError):
        color_intervals([3], [3])


def test_identical_spans_all_distinct():
    colors = color_intervals([5] * 50, [9] * 50)
    assert sorted(colors) == list(range(50))


def test_many_disjoint_stay_one_color():
    starts = list(range(0, 2 * MANY, 2))
    assert set(color_intervals(starts, [s + 2 for s in starts])) == {0}


def test_unsorted_input_order_preserved():
    # Output is per input index, not per sorted position.
    assert color_intervals([8, 0], [9, 1]) == [0, 0]
    assert color_intervals([2, 0], [5, 3]) == [1, 0]


# --- redqueen_batch op ---------------------------------------------------


def test_batch_applies_one_non_overlapping_class():
    from fuzzer_tool.services.operators import _rq_batch_classes, _rq_replay_batch

    buf = bytearray(b"AAAABBBBCCCC")
    # Matches 0 and 2 are disjoint; 1 straddles both.
    matches = [(0, b"AAAA", b"wxyz"), (2, b"AABB", b"QQQQ"), (8, b"CCCC", b"1234")]
    _rq_replay_batch(buf, matches, _rq_batch_classes(matches), ScriptedRng(choice_idxs=[0]))
    assert bytes(buf) == b"wxyzBBBB1234"


def test_batch_second_class():
    from fuzzer_tool.services.operators import _rq_batch_classes, _rq_replay_batch

    buf = bytearray(b"AAAABBBBCCCC")
    matches = [(0, b"AAAA", b"wxyz"), (2, b"AABB", b"QQQQ"), (8, b"CCCC", b"1234")]
    _rq_replay_batch(buf, matches, _rq_batch_classes(matches), ScriptedRng(choice_idxs=[1]))
    assert bytes(buf) == b"AAQQQQBBCCCC"


def test_batch_span_covers_longer_replacement():
    from fuzzer_tool.services.operators import _rq_batch_classes, _rq_replay_batch

    # op_b (4 bytes) at 0 writes past op_a (2 bytes) into match 1: overlap.
    buf = bytearray(b"ABCDEF")
    matches = [(0, b"AB", b"wxyz"), (2, b"CD", b"..")]
    _rq_replay_batch(buf, matches, _rq_batch_classes(matches), ScriptedRng(choice_idxs=[0]))
    assert bytes(buf) == b"wxyzEF"


def test_batch_skips_stale_match():
    from fuzzer_tool.services.operators import _rq_batch_classes, _rq_replay_batch

    buf = bytearray(b"ZZZZ")
    stale = [(0, b"AAAA", b"wxyz")]
    _rq_replay_batch(buf, stale, _rq_batch_classes(stale), ScriptedRng(choice_idxs=[0]))
    assert bytes(buf) == b"ZZZZ"


def test_batch_registered_adaptive_and_gated():
    from tests.test_regression_operator_registry import _MockFuzzer

    assert REGISTRY.category_of("redqueen_batch") == "adaptive"
    fuzzer = _MockFuzzer()
    assert "redqueen_batch" not in REGISTRY.available(fuzzer, b"seed")
    fuzzer.seed_meta = {b"seed": {"redqueen_offsets": [1]}}
    assert "redqueen_batch" not in REGISTRY.available(fuzzer, b"seed")
    fuzzer.seed_meta = {b"seed": {"redqueen_matches": [(0, b"s", b"t")]}}
    assert "redqueen_batch" in REGISTRY.available(fuzzer, b"seed")


def test_batch_dispatch_mutates():
    from fuzzer_tool.services.operators import OperatorEngine
    from tests.test_regression_operator_registry import _MockFuzzer

    fuzzer = _MockFuzzer()
    fuzzer._rng = RandPool(seed=1)
    fuzzer.max_len = 4096
    fuzzer.seed_meta = {b"seed": {"redqueen_matches": [(0, b"se", b"XY")]}}
    dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
    buf = bytearray(b"seed")
    dispatch["redqueen_batch"](buf, 0, b"seed")
    assert bytes(buf) == b"XYed"


def test_batch_classes_cached_and_revalidated():
    from fuzzer_tool.services.operators import OperatorEngine
    from tests.test_regression_operator_registry import _MockFuzzer

    engine = OperatorEngine(_MockFuzzer())
    matches = [(0, b"AA", b"BB"), (1, b"AA", b"CC")]
    first = engine._rq_classes(matches)
    assert first == [[0], [1]]
    assert engine._rq_classes(matches) is first

    # Redqueen appends matches: a longer list must be recolored.
    matches.append((4, b"DD", b"EE"))
    assert engine._rq_classes(matches) == [[0, 2], [1]]
