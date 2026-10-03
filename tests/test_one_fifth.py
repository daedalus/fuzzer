"""--one-fifth: Rechenberg's success rule over the per-round mutation count."""

from __future__ import annotations

import ast
import inspect
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.core.one_fifth import (
    DEFAULT_TARGET,
    SCALE_MAX,
    SCALE_MIN,
    OneFifthRule,
    Outcome,
)
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer

# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def _feed(rule: OneFifthRule, period: int, cycles: int) -> None:
    """One HIT then period-1 MISSes, repeated: hit rate exactly 1/period."""
    for _ in range(cycles):
        rule.record(Outcome.HIT)
        for _ in range(period - 1):
            rule.record(Outcome.MISS)


@pytest.mark.parametrize(("target", "period"), [(0.2, 5), (0.1, 10), (0.02, 50)])
def test_rate_at_target_holds_scale(target, period):
    """Hit rate == target: each full cycle's log-steps sum to zero."""
    rule = OneFifthRule(target=target)
    _feed(rule, period, cycles=40)
    assert math.isclose(rule.scale(), 1.0, rel_tol=1e-9)


def test_rate_above_target_grows_below_shrinks():
    """Falsification: the direction must follow the rate, not just move."""
    above = OneFifthRule(target=0.2)
    _feed(above, period=2, cycles=5)
    below = OneFifthRule(target=0.2)
    _feed(below, period=20, cycles=5)
    assert above.scale() > 1.0 > below.scale()


def test_saturates_without_windup():
    """Adversarial: a long dry spell pins the floor; the next hit moves off it
    at once instead of first paying back an unbounded integral."""
    rule = OneFifthRule(target=0.2)
    for _ in range(100_000):
        rule.record(Outcome.MISS)
    assert math.isclose(rule.scale(), SCALE_MIN)
    rule.record(Outcome.HIT)
    assert rule.scale() > SCALE_MIN * (1 + 1e-6)

    for _ in range(100_000):
        rule.record(Outcome.HIT)
    assert math.isclose(rule.scale(), SCALE_MAX)


@pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 1.5, math.nan, math.inf])
def test_rejects_degenerate_target(target):
    """target 0 or 1 makes one direction impossible; NaN poisons the scale."""
    with pytest.raises(ValueError):
        OneFifthRule(target=target)


@pytest.mark.parametrize("damping", [0.0, -1.0, math.nan])
def test_rejects_bad_damping(damping):
    with pytest.raises(ValueError):
        OneFifthRule(damping=damping)


def test_stats_counts_rounds_and_hits():
    rule = OneFifthRule()
    _feed(rule, period=4, cycles=3)
    stats = rule.stats()
    assert stats["one_fifth_rounds"] == 12
    assert stats["one_fifth_hits"] == 3
    assert stats["one_fifth_target"] == DEFAULT_TARGET
    assert stats["one_fifth_scale"] == rule.scale()


# ---------------------------------------------------------------------------
# The engine: -M x perf score x rule scale, stall floor last
# ---------------------------------------------------------------------------


class _FixedScale:
    def __init__(self, s: float) -> None:
        self.s = s

    def scale(self) -> float:
        return self.s


def _count_ops(scale: float, mutations: int = 8, stall: bool = False) -> int:
    f = make_minimal_fuzzer(seed=3)
    f.mutations_per_input = mutations
    f._last_perf_score = 100.0
    f._stall_recovery_active = stall
    f._one_fifth = _FixedScale(scale)
    engine = OperatorEngine(f)
    applied = []

    def op(buf, _idx, _data):
        applied.append(1)
        buf[0] ^= 1

    engine.build_ops = lambda data: ["counted"]
    f._op_dispatch = {"counted": op}
    engine.mutate(b"A" * 64)
    return len(applied)


@pytest.mark.parametrize("scale", [0.5, 2.0, 3.0])
def test_scale_multiplies_round_mutations(scale):
    assert _count_ops(scale) == round(8 * scale)


def test_scale_never_drops_below_one_mutation():
    """Adversarial: a tiny scale must still mutate, or the round re-runs its parent."""
    assert _count_ops(1e-9) == 1


def test_stall_floor_still_applies():
    assert _count_ops(0.125, stall=True) == 16


def test_rule_off_leaves_count_alone():
    f = make_minimal_fuzzer(seed=3)
    assert f._one_fifth is None


# ---------------------------------------------------------------------------
# The round: every round is one observation
# ---------------------------------------------------------------------------


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory(prefix="one_fifth_") as tmp:
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            from fuzzer_tool.services.fuzzer import Fuzzer

            f = Fuzzer(
                target="/bin/true",
                corpus_dir=str(Path(tmp) / "corpus"),
                crashes_dir=str(Path(tmp) / "crashes"),
                max_len=256,
                timeout=1,
                mutations_per_input=2,
                one_fifth=True,
            )
        yield f


def test_fuzzer_builds_rule(fuzzer):
    assert isinstance(fuzzer._one_fifth, OneFifthRule)


def test_boring_rounds_record_misses(fuzzer):
    """Falsification: every round lands, and a boring one is a MISS."""
    n = 5
    with (
        patch.object(fuzzer, "_run_target", return_value=(0, "")),
        patch.object(fuzzer, "_is_crash", return_value=False),
        patch.object(fuzzer, "_is_interesting", return_value=False),
    ):
        for i in range(n):
            with patch.object(fuzzer, "_dedup_mutate", return_value=bytes([i + 1]) * 8):
                fuzzer.fuzz_one(fuzzer.corpus[0])
    stats = fuzzer._one_fifth.stats()
    assert stats["one_fifth_rounds"] == n
    assert stats["one_fifth_hits"] == 0


@pytest.mark.parametrize(("covered", "hits"), [(True, 1), (False, 0)])
def test_new_coverage_is_the_hit(fuzzer, covered, hits):
    rnd = FuzzRound(fuzzer, b"SEED")
    rnd._has_new_coverage = covered
    rnd._record_one_fifth()
    assert fuzzer._one_fifth.stats()["one_fifth_hits"] == hits


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def test_cmd_fuzz_forwards_one_fifth():
    from fuzzer_tool.cli import commands

    fn = next(
        n
        for n in ast.walk(ast.parse(inspect.getsource(commands)))
        if isinstance(n, ast.FunctionDef) and n.name == "cmd_fuzz"
    )
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Fuzzer"
    ]
    assert {"one_fifth", "one_fifth_target"} <= {k.arg for k in calls[0].keywords}


# ---------------------------------------------------------------------------
# The stats line
# ---------------------------------------------------------------------------


def test_stats_field_reports_scale_and_rate():
    from types import SimpleNamespace

    from fuzzer_tool.services.stats import _one_fifth_str

    rule = OneFifthRule()
    _feed(rule, period=4, cycles=2)
    line = _one_fifth_str(SimpleNamespace(_one_fifth=rule))
    assert line == f" | 1/5: x{rule.scale():.2f} hit={2 / 8:.1%}"


def test_stats_field_absent_when_off():
    """Adversarial: report consumers pass MagicMock fuzzers; no rule, no field."""
    from unittest.mock import MagicMock

    from fuzzer_tool.services.stats import _one_fifth_str

    assert _one_fifth_str(MagicMock()) == ""
