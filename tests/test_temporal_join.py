"""Regression tests for ``core.temporal_join``."""

from __future__ import annotations

from fuzzer_tool.core.temporal_join import join_streams


def _mk(events: list[tuple[float, str]]) -> list[tuple[float, str]]:
    return events


def test_empty_streams_return_empty() -> None:
    assert join_streams([], 0.5) == []
    assert join_streams([[]], 0.5) == []
    assert join_streams([[], []], 0.5) == []


def test_single_stream_returns_empty() -> None:
    assert join_streams([_mk([(0.0, "a"), (1.0, "b")])], 0.5) == []


def test_exact_match_accepted() -> None:
    streams = [
        _mk([(0.0, "a1"), (2.0, "a2")]),
        _mk([(0.3, "b1"), (2.3, "b2")]),
    ]
    assert join_streams(streams, 0.5) == [("a1", "b1"), ("a2", "b2")]


def test_within_gap_accepted() -> None:
    streams = [
        _mk([(0.0, "a1")]),
        _mk([(0.4, "b1")]),
        _mk([(0.5, "c1")]),
    ]
    assert join_streams(streams, 0.5) == [("a1", "b1", "c1")]


def test_outside_gap_skips() -> None:
    streams = [
        _mk([(0.0, "a1"), (1.0, "a2")]),
        _mk([(2.0, "b1")]),
    ]
    assert join_streams(streams, 0.5) == []


def test_laggard_advance_only() -> None:
    streams = [
        _mk([(0.0, "a1"), (1.0, "a2")]),
        _mk([(0.0, "b1"), (3.0, "b2")]),
    ]
    # b at t=3 is too far from a at t=1 for a match, so only one pair.
    assert join_streams(streams, 0.5) == [("a1", "b1")]


def test_three_stream_all_match() -> None:
    streams = [
        _mk([(0.0, "a1"), (10.0, "a2")]),
        _mk([(0.2, "b1"), (10.2, "b2")]),
        _mk([(0.4, "c1"), (10.4, "c2")]),
    ]
    assert join_streams(streams, 0.5) == [("a1", "b1", "c1"), ("a2", "b2", "c2")]


def test_three_stream_mismatch_then_match() -> None:
    streams = [
        _mk([(0.0, "a1"), (5.0, "a2"), (10.0, "a3")]),
        _mk([(0.0, "b1"), (5.5, "b2"), (10.5, "b3")]),
        _mk([(10.0, "c1"), (11.0, "c2")]),
    ]
    # First two streams start too far behind the third; they all align
    # only once the laggards reach t=10.
    assert join_streams(streams, 1.5) == [("a3", "b3", "c1")]


def test_caller_applies_value_fn_after_join() -> None:
    streams = [
        _mk([(0.0, 10), (2.0, 20)]),
        _mk([(0.0, 1), (2.0, 2)]),
    ]
    joined = join_streams(streams, 0.5)
    assert joined == [(10, 1), (20, 2)]
    # Caller-side collapse is intentionally separate from the join.
    collapsed = [sum(vals) for vals in joined]
    assert collapsed == [11, 22]


def test_non_monotonic_input_undefined() -> None:
    # The docstring says non-monotonic input is undefined behavior;
    # this test simply asserts we do not crash on a short non-monotonic
    # stream rather than pinning exact output.
    streams = [
        _mk([(1.0, "a"), (0.0, "b")]),
        _mk([(0.5, "c"), (0.6, "d")]),
    ]
    join_streams(streams, 0.5)


def test_large_synthetic_streams() -> None:
    n = 5_000
    streams = [
        [(float(i), f"a{i}") for i in range(n)],
        [(float(i) + 0.1, f"b{i}") for i in range(n)],
    ]
    out = join_streams(streams, 0.5)
    assert len(out) == n
    assert out[0] == ("a0", "b0")
    assert out[-1] == (f"a{n - 1}", f"b{n - 1}")
