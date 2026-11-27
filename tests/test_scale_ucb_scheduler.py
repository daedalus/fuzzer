"""ScaleUCBScheduler: pick a mutation scale band, then an operator in it."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_scale_ucb import (
    BAND_OF_CATEGORY,
    OTHER_BAND,
    ScaleUCBScheduler,
)


def _ops(category: str, n: int) -> list[str]:
    return sorted(OPERATOR_CATEGORIES[category])[:n]


def test_every_category_has_a_band() -> None:
    """Falsification: no registered category falls through to OTHER_BAND."""
    assert set(OPERATOR_CATEGORIES) <= set(BAND_OF_CATEGORY)


@pytest.mark.parametrize(
    ("category", "band"), [("bit", "bit"), ("byte", "byte"), ("block", "chunk"), ("dict", "field")]
)
def test_band_of_registered_op(category, band) -> None:
    """Ops inherit the band of their registry category."""
    s = ScaleUCBScheduler(rng=RandPool(1))
    op = _ops(category, 1)[0]
    assert s.band_of(op) == band


def test_unknown_op_is_still_selectable() -> None:
    """Adversarial: an unregistered name lands in OTHER_BAND, not dropped."""
    s = ScaleUCBScheduler(rng=RandPool(1))
    assert s.band_of("not_an_op") == OTHER_BAND
    assert s.select_op(["not_an_op"]) == "not_an_op"
    s.record("not_an_op", True, 1.0)
    assert s.select_op(["not_an_op", "also_not"]) in ("not_an_op", "also_not")


def test_empty_and_single() -> None:
    s = ScaleUCBScheduler(rng=RandPool(1))
    assert s.select_op([]) == ""
    op = _ops("bit", 1)[0]
    assert s.select_op([op]) == op


def test_record_credits_band_and_op() -> None:
    """One record reaches both levels."""
    s = ScaleUCBScheduler(rng=RandPool(1))
    op = _ops("block", 1)[0]
    s.record(op, True, 0.5)
    stats = s.bandit_stats()
    assert stats["scale_ucb_band_chunk"] == 1
    assert stats["scale_ucb_band_bit"] == 0
    assert stats["scale_ucb_pulls"] == 1


def test_concentrates_on_productive_band() -> None:
    """Only chunk ops pay: tail selections sit in the chunk band."""
    rng = RandPool(7)
    s = ScaleUCBScheduler(rng=RandPool(1))
    ops = _ops("bit", 10) + _ops("byte", 10) + _ops("block", 10) + _ops("dict", 10)
    payoff = {op: (0.3 if s.band_of(op) == "chunk" else 0.02) for op in ops}

    tail_chunk = 0
    rounds, tail = 6000, 1000
    for t in range(rounds):
        op = s.select_op(ops)
        assert op in ops
        hit = rng.random() < payoff[op]
        s.record(op, hit, 1.0)
        if t >= rounds - tail and s.band_of(op) == "chunk":
            tail_chunk += 1

    assert tail_chunk / tail >= 0.8


def test_never_selects_absent_band() -> None:
    """A band whose ops are not candidates is never chosen, however good."""
    s = ScaleUCBScheduler(rng=RandPool(1))
    good = _ops("block", 1)[0]
    for _ in range(200):
        s.record(good, True, 1.0)
    cands = _ops("bit", 3)
    for _ in range(50):
        assert s.select_op(cands) in cands
