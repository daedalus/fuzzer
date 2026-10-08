"""--colorize ran up to 512 executions on every round's fresh mutant.

``_redqueen_scan`` colorized ``self._mutated``: never cached (each mutant is
new), and run between ``_execute`` and ``_scan_coverage``, so the colorize runs
overwrote the mutant's coverage map. Under --hail-mary on fuzzgoat, 3k execs
found 16 edges against 255 without it. Colorization now runs once per seed,
before ``_execute``, after the seed has had as many rounds as it costs; the
scan and Weizz collection only read cached taints.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from fuzzer_tool.services.fuzz_round import FuzzRound

SEED = b"seed"
MUTANT = b"xxABCDxxABCD"
OPERAND = b"ABCD"
BUDGET = 8


def _scan_round(taints) -> tuple[MagicMock, list]:
    f = MagicMock()
    f._cached_taints.return_value = taints
    f._cmplog.pending_new_pairs.return_value = [(OPERAND, b"WXYZ")]
    r = FuzzRound(f, SEED)
    r._meta = {}
    r._mutated = MUTANT
    matches: list = []
    r._redqueen_scan(matches, set())
    return f, matches


def test_regression_colorize_per_round():
    """The scan never colorizes (never executes) the mutant."""
    f, _ = _scan_round(None)
    f._colorize_seed.assert_not_called()
    f._cached_taints.assert_called_once_with(MUTANT)


def test_cached_taints_still_filter():
    """Falsification: a cached taint over the first occurrence drops it."""
    first = MUTANT.index(OPERAND)
    taint = SimpleNamespace(start=first, end=first + len(OPERAND) - 1)
    _, matches = _scan_round([taint])
    assert [m[0] for m in matches] == [MUTANT.index(OPERAND, first + 1)]


def _colorize_round(fuzz_count, *, meta=True, enabled=True) -> MagicMock:
    f = MagicMock()
    f.colorize = enabled
    f._colorize_budget.return_value = BUDGET
    r = FuzzRound(f, SEED)
    r._meta = {"fuzz_count": fuzz_count} if meta else None
    r._colorize()
    return f


def test_seed_colorized_once_budget_is_earned():
    """At fuzz_count == budget, colorization took at most half the seed's runs."""
    _colorize_round(BUDGET)._colorize_seed.assert_called_once_with(SEED)


def test_fresh_seed_not_colorized():
    """Falsification: one round short of the budget, nothing runs."""
    _colorize_round(BUDGET - 1)._colorize_seed.assert_not_called()


def test_seed_without_meta_not_colorized():
    """Adversarial: no fuzz count (generated seed) never pays for colorization."""
    _colorize_round(BUDGET, meta=False)._colorize_seed.assert_not_called()


def test_disabled_not_colorized():
    _colorize_round(BUDGET, enabled=False)._colorize_seed.assert_not_called()


def test_weizz_tags_never_colorize():
    """Weizz collection on admission reads cached taints; it never executes."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    f = MagicMock()
    f.colorize = True
    f.seed_meta = {}
    f._cached_taints.return_value = None
    Fuzzer._maybe_collect_weizz_tags(f, MUTANT)
    f._colorize_seed.assert_not_called()
    f._cached_taints.assert_called_once_with(MUTANT)


def test_run_orders_colorize_before_execute():
    """Adversarial: run() itself, not just the stage, must keep the order."""
    calls: list[str] = []
    stubs = {s: (lambda self, s=s: calls.append(s)) for s in _RUN_STAGES}
    stubs["_admits"] = lambda self: False
    recorder = type("RecordingRound", (FuzzRound,), stubs)
    r = recorder(MagicMock(), SEED)
    r._is_crash = False

    r.run()

    order = [c for c in calls if c in ("_colorize", "_execute", "_scan_coverage")]
    assert order == ["_colorize", "_execute", "_scan_coverage"]


# Every stage FuzzRound.run() calls, stubbed so the order test runs no real stage.
_RUN_STAGES = (
    "_begin",
    "_search_fixpoint",
    "_generalize",
    "_colorize",
    "_execute",
    "_mine_cmplog",
    "_periodic",
    "_count_ops",
    "_classify",
    "_ltl_observe",
    "_scan_coverage",
    "_observe",
    "_gate_explore",
    "_credit_seed",
    "_feed_models",
    "_record_edges",
    "_learn_format",
    "_track_edges",
    "_judge",
    "_credit_ops",
    "_push_recurrence",
    "_on_boring",
)
