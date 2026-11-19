"""Regression: delocalised ops (`mutation_offset=None`) crashed `backtest()`.

`_predict_effect` treated `offset is None` as "covered" and then compared
`h.offset <= None` -> TypeError, aborting the whole fuzz run.
"""

from __future__ import annotations

from fuzzer_tool.core.analyzers.analyzer_format_learner import (
    FieldHypothesis,
    FormatCluster,
    TimelineEntry,
)

_CONFIDENT = 0.9


def _entry(op: str, offset: int | None) -> TimelineEntry:
    return TimelineEntry(
        input_hash="0" * 16,
        mutation_op=op,
        mutation_offset=offset,
        mutation_width=1,
        coverage_before=10,
        coverage_after=11,
        new_edges={1},
        lost_edges=set(),
    )


def _cluster(sensitive_op: str) -> FormatCluster:
    # One confident field at [4, 8) sensitive to *sensitive_op*.
    c = FormatCluster(signature="sig")
    c._add_hypothesis(
        FieldHypothesis(
            offset=4,
            width=4,
            field_type="data",
            confidence=_CONFIDENT,
            sensitive_ops={sensitive_op: 1},
        )
    )
    return c


def test_regression_none_offset_backtest_no_crash():
    c = _cluster("havoc")
    c.timeline.append(_entry("splice", None))

    # No span to match, no op match -> no prediction, backtest passes.
    assert c._predict_effect(c.timeline[0]) is None
    assert c.backtest() == (True, None)


def test_falsify_none_offset_still_uses_op_sensitivity():
    # Falsification: skipping the span scan must not skip the op scan.
    c = _cluster("splice")
    assert c._predict_effect(_entry("splice", None)) is True


def test_adversarial_none_offset_via_learner_record():
    # Adversarial: mixed None/int offsets through the public record path,
    # enough to trigger the periodic backtest.
    from fuzzer_tool.core.analyzers import analyzer_format_learner as afl

    c = _cluster("havoc")
    for i in range(afl.BACKTEST_INTERVAL):
        offset = None if i % 2 else 5
        c.record(_entry("splice", offset), b"\x00" * 16, afl.BACKTEST_INTERVAL)

    assert c.backtest_passes + c.backtest_fails >= 1
