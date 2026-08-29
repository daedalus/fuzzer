"""The two cost-ledger consumers, exercised through the real code paths.

``docs/handover/handover_persistence_mechanics_2026-08-29.md`` §1a and §1b.
Both consumers read ``effective_fuzz_count``, which converts the per-seed cost
ledger back into executions at the corpus mean rate.  That form is a no-op
exactly where per-execution cost does not vary, so each class below asserts
both halves: unchanged under uniform cost, changed under varying cost.

These call ``StatsReporter._print_summary_seeds`` and
``SeedPicker._pick_boltzmann_seed`` rather than recomputing their formulas.
The existing Boltzmann tests in ``tests/test_seed_picker.py`` re-derive the
weight expression inline and would pass against either energy term, so they
guard the arithmetic and not the wiring.
"""

from __future__ import annotations

import io
import random
from contextlib import redirect_stdout
from types import SimpleNamespace

from fuzzer_tool.services.seed_picker import SeedPicker
from fuzzer_tool.services.stats import STALE_SEED_EXEC_EQUIVALENTS, StatsReporter


def _meta(fuzz_count, exec_time, edges=0, mean=0.002):
    """Seed metadata for a seed fuzzed `fuzz_count` times at `exec_time` each."""
    return {
        "fuzz_count": fuzz_count,
        "cost_samples": fuzz_count,
        "total_time": fuzz_count * exec_time,
        "coverage_edges": edges,
        "lineage_depth": 0,
    }


class _StubEdgeTracker:
    def classify_seeds(self):
        return {}

    def find_redundant_seeds(self):
        return []


def _stale_line(seed_meta, mean_exec):
    f = SimpleNamespace(
        seed_meta=seed_meta,
        mean_exec_time=lambda: mean_exec,
        _edge_tracker=_StubEdgeTracker(),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        StatsReporter._print_summary_seeds(StatsReporter.__new__(StatsReporter), f)
    for line in buf.getvalue().splitlines():
        if "Stale seeds:" in line:
            return line
    raise AssertionError("no stale-seeds line printed")


class TestStaleSeedsIsCostBased:
    def test_matches_the_count_criterion_under_uniform_cost(self):
        """Every seed costs the mean, so the cost criterion is the old one."""
        mean = 0.002
        metas = {
            b"a": _meta(10, mean),
            b"b": _meta(49, mean),
            b"c": _meta(50, mean),
            b"d": _meta(300, mean),
        }
        assert "2/4" in _stale_line(metas, mean)

    def test_expensive_barren_seed_is_stale_below_fifty_picks(self):
        mean = 0.002
        # 10 picks at 20x the mean = 200 average-cost executions of budget.
        metas = {b"pricey": _meta(10, 0.040)}
        assert f"1/1 ({STALE_SEED_EXEC_EQUIVALENTS}+" in _stale_line(metas, mean)

    def test_cheap_barren_seed_is_not_stale_at_fifty_picks(self):
        """The case the count criterion got wrong in the other direction."""
        mean = 0.002
        metas = {b"cheap": _meta(50, 0.00002)}
        assert "0/1" in _stale_line(metas, mean)

    def test_productive_seed_is_never_stale_however_expensive(self):
        mean = 0.002
        metas = {b"good": _meta(300, 0.040, edges=7)}
        assert "0/1" in _stale_line(metas, mean)

    def test_unmeasured_seed_falls_back_to_its_count(self):
        """A seed restored from a pre-ledger state keeps its old reading."""
        mean = 0.002
        metas = {
            b"legacy": {
                "fuzz_count": 300,
                "cost_samples": 0,
                "total_time": 0.0,
                "coverage_edges": 0,
                "lineage_depth": 0,
            }
        }
        assert "1/1" in _stale_line(metas, mean)


class _ScriptedRandom:
    """Yields one scripted float per random() call; choice() is a tripwire."""

    def __init__(self, value):
        self._value = value
        self.choice_calls = 0

    def random(self):
        return self._value

    def choice(self, seq):
        self.choice_calls += 1
        return seq[0]


def _pick(seed_meta, mean_exec, r, temperature=1.0):
    corpus = list(seed_meta)
    rng = _ScriptedRandom(r)
    f = SimpleNamespace(
        corpus=corpus,
        seed_meta=seed_meta,
        _rand_pool=rng,
        _temperature=temperature,
        mean_exec_time=lambda: mean_exec,
    )
    sp = SeedPicker.__new__(SeedPicker)
    sp.f = f
    picked = sp._pick_boltzmann_seed()
    assert rng.choice_calls == 0, "fell through to the uniform fallback"
    return picked


class TestBoltzmannEnergyIsCostBased:
    def test_selection_unchanged_under_uniform_cost(self):
        """Equal cost per execution: the weights are the fuzz_count weights."""
        mean = 0.002
        metas = {b"rare": _meta(1, mean), b"common": _meta(100, mean)}
        # w(rare) = 1/2, w(common) = 1/101 -> rare owns 98% of the mass.
        assert _pick(metas, mean, 0.5) == b"rare"
        assert _pick(metas, mean, 0.99) == b"common"

    def test_equal_counts_split_by_cost(self):
        """The discriminant: same fuzz_count, so the old energy tied them."""
        mean = 0.002
        metas = {
            b"cheap": _meta(20, 0.0002),  # 2 average-cost executions
            b"pricey": _meta(20, 0.020),  # 200 average-cost executions
        }
        # New weights: cheap 1/3, pricey 1/201 -> cheap holds 98.5% of the mass,
        # so r=0.9 lands inside it. Under E = log(fuzz_count + 1) the two were
        # exactly equal at 1/21 each and r=0.9 landed on the second seed.
        assert _pick(metas, mean, 0.9) == b"cheap"

    def test_expensive_seed_still_reachable(self):
        mean = 0.002
        metas = {b"cheap": _meta(20, 0.0002), b"pricey": _meta(20, 0.020)}
        assert _pick(metas, mean, 0.999) == b"pricey"

    def test_unmeasured_seed_falls_back_to_its_count(self):
        mean = 0.002
        metas = {
            b"legacy": {"fuzz_count": 100, "cost_samples": 0, "total_time": 0.0},
            b"fresh": _meta(1, mean),
        }
        # Falling back keeps legacy at w = 1/101 against fresh at 1/2.
        assert _pick(metas, mean, 0.5) == b"fresh"

    def test_no_mean_yet_falls_back_to_counts(self):
        """Before anything is timed there is no rate to convert with."""
        metas = {b"rare": _meta(1, 0.0), b"common": _meta(100, 0.0)}
        assert _pick(metas, 0.0, 0.5) == b"rare"

    def test_empty_corpus_does_not_reach_the_weight_loop(self):
        f = SimpleNamespace(
            corpus=[],
            seed_meta={},
            _rand_pool=random,
            _temperature=1.0,
            mean_exec_time=lambda: 0.002,
        )
        sp = SeedPicker.__new__(SeedPicker)
        sp.f = f
        sp._format_aware_seed = lambda: b"fallback"
        assert sp._pick_boltzmann_seed() == b"fallback"
