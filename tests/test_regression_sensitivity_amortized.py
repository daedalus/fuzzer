"""--sensitivity probed every admitted seed at once: 19% of --hail-mary's execs.

Each admission spent ``sample_rate * len`` probe runs before the seed had
been fuzzed once. The analysis now runs on the parent, before ``_execute``,
once the seed has had as many rounds as the probes cost (like ``_colorize``).
"""

from unittest.mock import MagicMock

from fuzzer_tool.core.analyzers.analyzer_sensitivity import ByteSensitivityTracker
from fuzzer_tool.services.fuzz_round import FuzzRound

SEED = b"s" * 40
EDGES = {1, 2, 3}


def _round(fuzz_count: int, *, edges=EDGES) -> MagicMock:
    f = MagicMock()
    f._use_sensitivity = True
    f._sensitivity.cost.return_value = 4
    f._sensitivity.analyzed.return_value = False
    f._edge_tracker.seed_edges = {"k": edges} if edges else {}
    f._seed_key.return_value = "k"
    r = FuzzRound(f, SEED)
    r._meta = {"fuzz_count": fuzz_count}
    r._sensitize()
    return f


def test_regression_sensitivity_amortized():
    """A seed short of its probe cost is not analyzed."""
    _round(3)._sensitivity.analyze_seed.assert_not_called()


def test_seed_that_earned_cost_is_analyzed():
    f = _round(4)
    seed, edges, _exec = f._sensitivity.analyze_seed.call_args.args
    assert (seed, edges) == (SEED, EDGES)


def test_seed_without_recorded_edges_skipped():
    """Adversarial: no parent edges to compare against, no probes."""
    _round(9, edges=None)._sensitivity.analyze_seed.assert_not_called()


def test_admission_no_longer_probes():
    """Falsification: the admission path has no sensitivity hook left."""
    assert not hasattr(FuzzRound, "_analyze_sensitivity")


def test_cost_matches_probe_count():
    """cost() is exactly the number of runs analyze_seed spends."""
    t = ByteSensitivityTracker(sample_rate=0.1)
    runs = []
    t.analyze_seed(SEED, EDGES, lambda d: runs.append(d) or EDGES)
    assert t.cost(SEED) == len(runs) == 4
    assert t.cost(b"") == 0
    assert t.cost(b"x") == 1


def test_analyzed_seed_skips_lookup():
    """Once scored, later rounds do not hash the seed for its edges."""
    f = MagicMock()
    f._use_sensitivity = True
    f._sensitivity.cost.return_value = 1
    f._sensitivity.analyzed.return_value = True
    r = FuzzRound(f, SEED)
    r._meta = {"fuzz_count": 9}
    r._sensitize()
    f._seed_key.assert_not_called()
