"""Regression tests for Fuzzer._dedup_mutate's repeat filter.

Three wasted-exec paths, all deterministic (scripted ``mutate``):

  * a mutant byte-identical to its parent was executed (no re-roll);
  * a parent in the --cuckoo-seed-filter returned itself unmutated, so the
    round executed the parent verbatim;
  * re-roll counters were never reported in the run summary.
"""

import time
from types import SimpleNamespace
from unittest.mock import patch

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.cuckoo import CuckooFilter
from fuzzer_tool.services.fuzzer import EXEC_DEDUP_RETRIES, Fuzzer
from fuzzer_tool.services.stats import StatsReporter

PARENT = b"parent-seed"
NOVEL = b"novel-mutant"
PRUNED = b"pruned-seed"


class _Stub:
    """Only what _dedup_mutate touches; ``mutate`` pops a scripted list."""

    def __init__(self, mutants, dedup_execs=True, pruned=()):
        self._mutants = list(mutants)
        self._dedup_execs = dedup_execs
        self._exec_bloom = BloomFilter(capacity=1000, error_rate=1e-3)
        self._dedup_hits = 0
        self._dedup_gaveup = 0
        self.mutate_calls = 0
        self.cuckoo_seed_filter = None
        self._cuckoo_recovered: set[str] = set()
        if pruned:
            self.cuckoo_seed_filter = CuckooFilter(capacity=1000)
            for seed in pruned:
                self.cuckoo_seed_filter.add(hash_data(seed))

    def _seed_key(self, data):
        return hash_data(bytes(data))

    def mutate(self, data):
        self.mutate_calls += 1
        return self._mutants.pop(0)

    _dedup_mutate = Fuzzer._dedup_mutate
    _is_repeat = Fuzzer._is_repeat


def test_regression_parent_identical_rerolled():
    f = _Stub([PARENT, PARENT, NOVEL])

    assert f._dedup_mutate(PARENT) == NOVEL
    assert f.mutate_calls == 3
    assert f._dedup_hits == 2
    assert f._dedup_gaveup == 0


def test_regression_parent_identity_is_direct():
    """The parent is compared, not inserted: the bloom never sees it."""
    f = _Stub([bytearray(PARENT), NOVEL])

    assert f._dedup_mutate(PARENT) == NOVEL
    assert f._exec_bloom.update_bytes(PARENT) is False


def test_regression_pruned_parent_not_executed():
    """A pruned parent is mutated, never returned verbatim for execution."""
    f = _Stub([NOVEL], pruned=[PARENT])

    assert f._dedup_mutate(PARENT) == NOVEL
    assert f.mutate_calls == 1
    assert f._dedup_hits == 0


def test_regression_pruned_mutant_rerolled():
    """A mutant re-creating a pruned seed is a repeat (no re-discovery)."""
    f = _Stub([PRUNED, NOVEL], pruned=[PRUNED])

    assert f._dedup_mutate(PARENT) == NOVEL
    assert f._dedup_hits == 1


def test_regression_recovered_mutant_passes():
    f = _Stub([PRUNED], pruned=[PRUNED])
    f._cuckoo_recovered.add(hash_data(PRUNED))

    assert f._dedup_mutate(PARENT) == PRUNED
    assert f._dedup_hits == 0


def test_regression_pruned_check_ignores_dedup_flag():
    """--cuckoo-seed-filter is its own opt-in; --no-dedup-execs leaves it on."""
    f = _Stub([PRUNED, NOVEL], dedup_execs=False, pruned=[PRUNED])

    assert f._dedup_mutate(PARENT) == NOVEL
    assert f._dedup_hits == 1


def test_no_dedup_execs_runs_parent_identical():
    """--no-dedup-execs: execute whatever mutate() produced, parent included."""
    f = _Stub([PARENT], dedup_execs=False)

    assert f._dedup_mutate(PARENT) == PARENT
    assert f.mutate_calls == 1
    assert f._dedup_hits == 0


# Falsification: a novel first draw must cost exactly one mutate().
def test_novel_first_try_zero_rerolls():
    f = _Stub([NOVEL], pruned=[PRUNED])

    assert f._dedup_mutate(PARENT) == NOVEL
    assert f.mutate_calls == 1
    assert (f._dedup_hits, f._dedup_gaveup) == (0, 0)


# Adversarial: mutate() is a fixpoint -> bounded, executed anyway, counted.
def test_always_parent_bounded():
    f = _Stub([PARENT] * (EXEC_DEDUP_RETRIES + 5))

    assert f._dedup_mutate(PARENT) == PARENT
    assert f.mutate_calls == EXEC_DEDUP_RETRIES + 1
    assert f._dedup_hits == EXEC_DEDUP_RETRIES
    assert f._dedup_gaveup == 1


def test_empty_input_rerolled_then_bounded():
    f = _Stub([b"", b"x"])
    assert f._dedup_mutate(b"") == b"x"
    assert f._dedup_hits == 1

    g = _Stub([b""] * (EXEC_DEDUP_RETRIES + 1))
    assert g._dedup_mutate(b"") == b""
    assert g.mutate_calls == EXEC_DEDUP_RETRIES + 1
    assert g._dedup_gaveup == 1


def _summary(**attrs):
    base = {
        "start_time": time.time() - 10.0,
        "exec_count": 500,
        "_peak_eps": 1.0,
        "crash_count": 0,
        "crash_sigs": {},
        "corpus": [],
        "_total_corpus_attempts": 0,
        "_duplicate_reject_count": 0,
        "_pruned_count": 0,
        "_corpus_flux": None,
        "_stall_recovery_count": 0,
        "_use_poisson_disk_admission": False,
    }
    base.update(attrs)
    reporter = StatsReporter(SimpleNamespace(**base))
    with (
        patch.object(StatsReporter, "_print_temperature_control"),
        patch.object(StatsReporter, "_print_summary_coverage"),
        patch.object(StatsReporter, "_print_summary_seeds"),
        patch.object(StatsReporter, "_print_summary_rarity"),
        patch.object(StatsReporter, "_print_summary_gravity"),
        patch("builtins.print") as mock_print,
    ):
        reporter.print_run_summary()
    return [c.args[0] for c in mock_print.call_args_list if c.args]


def test_regression_dedup_counters_reported():
    lines = _summary(_dedup_hits=7, _dedup_gaveup=2)

    line = next(x for x in lines if "Exec dedup:" in x)
    assert "7 re-rolls" in line
    assert "2 gave up" in line


def test_dedup_counters_silent_when_zero():
    lines = _summary(_dedup_hits=0, _dedup_gaveup=0)

    assert not any("Exec dedup:" in x for x in lines)
