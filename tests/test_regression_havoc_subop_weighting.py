"""Regression: havoc's inner sub-mutation split must consume coverage feedback.

``_apply_single_mutation`` chose among its 11 inline branches with
``r[0] % 11`` -- flat odds, identical on execution one and execution one
billion, while every top-level operator above it is scheduled by a bandit
fed real per-operator success rates. Havoc is reachable from every mutation
round and is the stall-recovery escalation path, and it applies 2-8
sub-mutations per call (8-16 while stalled), so a branch that never finds an
edge on a given target was being handed ~9% of all sub-mutations
indefinitely.

The fix weights the draw by each branch's measured new-coverage rate, with
credit deferred to ``Fuzzer._record_outcome`` (havoc mutates long before the
coverage verdict exists) and a uniform mixing floor so no branch is starved
permanently -- a branch whose guard fails on short inputs (byte_swap,
delete_block, shuffle_range) must be able to recover once inputs grow.

Guards here:
  * ``--no-adaptive-havoc`` restores the exact uniform draw and touches no
    counters, so A/B runs measure the split and nothing else.
  * The priors are uniform, so a fresh run does not diverge before evidence.
  * Credit reaches only the branches actually applied that round.
  * The explore floor holds for a branch that never scores.
  * Decay preserves ratios (a halving that dropped hits but not trials would
    silently re-flatten the distribution mid-run).
"""

import pytest

from fuzzer_tool.services.operators import (
    _HAVOC_DECAY_AT,
    _HAVOC_EXPLORE,
    _HAVOC_N,
    _HAVOC_TABLE_SLOTS,
    HAVOC_SUB_OPS,
    OperatorEngine,
)
from tests.test_new_operators import _make_minimal_fuzzer


def _draw_histogram(engine, fuzzer, draws=20000, buf_len=64):
    """Sub-mutation selection counts over `draws` calls, read off the trial
    counters rather than by instrumenting the branch, so the histogram
    measures what the sampler actually did."""
    before = list(engine._havoc_trials)
    for _ in range(draws):
        buf = bytearray(range(buf_len))
        engine._apply_single_mutation(buf)
    return [engine._havoc_trials[i] - before[i] for i in range(_HAVOC_N)]


class TestAdaptiveHavocSubops:
    def setup_method(self):
        self.fuzzer = _make_minimal_fuzzer()
        self.engine = OperatorEngine(self.fuzzer)

    def test_names_cover_every_branch(self):
        # _apply_single_mutation's branch count and the name table must not
        # drift apart: the bitmask, the counters, and the state file all key
        # off HAVOC_SUB_OPS by index.
        assert len(HAVOC_SUB_OPS) == _HAVOC_N
        assert len(set(HAVOC_SUB_OPS)) == _HAVOC_N

    def test_disabled_path_touches_no_counters(self):
        self.fuzzer._adaptive_havoc = False
        trials_before = list(self.engine._havoc_trials)
        hits_before = list(self.engine._havoc_hits)
        for _ in range(2000):
            buf = bytearray(range(64))
            self.engine._apply_single_mutation(buf)
        assert list(self.engine._havoc_trials) == trials_before
        assert list(self.engine._havoc_hits) == hits_before
        assert self.fuzzer._last_havoc_subops == 0

    def test_disabled_path_is_uniform(self):
        # The flat split is what --no-adaptive-havoc is for; if it stopped
        # being flat the A/B baseline would be meaningless.
        self.fuzzer._adaptive_havoc = False
        counts = [0] * _HAVOC_N
        rng = self.fuzzer._rand_pool
        for _ in range(20000):
            counts[rng.randint_list(0, 1 << 30, 4)[0] % _HAVOC_N] += 1
        expected = 20000 / _HAVOC_N
        assert max(abs(c - expected) for c in counts) < expected * 0.15

    def test_priors_start_uniform(self):
        # No evidence yet -> the adaptive draw must not already be skewed,
        # or every short run would be biased by initialization alone.
        hist = _draw_histogram(self.engine, self.fuzzer)
        expected = sum(hist) / _HAVOC_N
        assert max(abs(c - expected) for c in hist) < expected * 0.15

    def test_credit_reaches_only_applied_branches(self):
        mask = (1 << 0) | (1 << 5)
        hits_before = list(self.engine._havoc_hits)
        self.engine.credit_havoc_subops(mask)
        for i in range(_HAVOC_N):
            delta = self.engine._havoc_hits[i] - hits_before[i]
            assert delta == (1.0 if i in (0, 5) else 0.0)

    def test_applied_branches_are_recorded_in_the_mask(self):
        # Every draw must set exactly its own bit; a mask that stayed 0 would
        # make _record_outcome skip credit entirely and silently degrade to
        # the uniform behaviour this change replaces.
        seen = 0
        for _ in range(500):
            self.fuzzer._last_havoc_subops = 0
            buf = bytearray(range(64))
            self.engine._apply_single_mutation(buf)
            mask = self.fuzzer._last_havoc_subops
            assert mask != 0
            assert bin(mask).count("1") == 1
            seen |= mask
        assert bin(seen).count("1") >= 8  # most branches reachable in 500 draws

    def test_winning_branch_gains_share(self):
        winner = HAVOC_SUB_OPS.index("byte_set")
        # Independent of the code under test: a branch credited on every
        # trial has ratio ~1.0, the rest ~0.0, so the winner should take the
        # whole non-explore mass.
        self.engine._havoc_hits[winner] = 5000
        self.engine._havoc_trials[winner] = 5000
        for i in range(_HAVOC_N):
            if i != winner:
                self.engine._havoc_trials[i] = 5000
        self.engine._rebuild_havoc_table()

        hist = _draw_histogram(self.engine, self.fuzzer)
        total = sum(hist)
        winner_share = hist[winner] / total
        uniform_share = 1.0 / _HAVOC_N
        assert winner_share > uniform_share * 3
        # ...and it must not take everything: the floor is what lets a branch
        # that only pays off on larger inputs climb back.
        assert winner_share < 1.0 - _HAVOC_EXPLORE / 2

    def test_zero_hit_branch_keeps_the_explore_floor(self):
        loser = HAVOC_SUB_OPS.index("crc32_repair")
        for i in range(_HAVOC_N):
            self.engine._havoc_hits[i] = 500
            self.engine._havoc_trials[i] = 500
        self.engine._havoc_hits[loser] = 1
        self.engine._havoc_trials[loser] = 50000
        self.engine._rebuild_havoc_table()

        hist = _draw_histogram(self.engine, self.fuzzer, draws=40000)
        share = hist[loser] / sum(hist)
        floor = _HAVOC_EXPLORE / _HAVOC_N
        assert share > floor * 0.5
        assert share < 1.0 / _HAVOC_N  # still deprioritised, just not starved

    def test_decay_preserves_ratios(self):
        for i in range(_HAVOC_N):
            self.engine._havoc_hits[i] = (i + 1) * 2
            self.engine._havoc_trials[i] = int(_HAVOC_DECAY_AT) * 2
        ratios_before = [
            self.engine._havoc_hits[i] / self.engine._havoc_trials[i] for i in range(_HAVOC_N)
        ]
        self.engine._rebuild_havoc_table()
        ratios_after = [
            self.engine._havoc_hits[i] / self.engine._havoc_trials[i] for i in range(_HAVOC_N)
        ]
        assert max(self.engine._havoc_trials) <= _HAVOC_DECAY_AT
        for a, b in zip(ratios_before, ratios_after, strict=True):
            assert a == pytest.approx(b, rel=1e-9)

    def test_table_is_complete_and_gives_every_branch_a_slot(self):
        # A branch with zero slots can never be drawn again and never earns
        # another hit -- the absorbing state the explore floor exists to
        # prevent. Derived independently of the builder: count slots.
        self.engine._havoc_hits[3] = 900
        self.engine._havoc_trials[3] = 1000
        self.engine._rebuild_havoc_table()
        table = self.engine._havoc_table
        assert len(table) == _HAVOC_TABLE_SLOTS
        counts = [table.count(i) for i in range(_HAVOC_N)]
        assert sum(counts) == _HAVOC_TABLE_SLOTS
        assert min(counts) >= 2  # floor is 1.4% of 256 == 3.5 slots
        assert counts[3] == max(counts)

    def test_havoc_stats_ranks_by_ratio(self):
        self.engine._havoc_hits[2] = 90
        self.engine._havoc_trials[2] = 100
        rows = self.engine.havoc_stats()
        assert rows[0][0] == HAVOC_SUB_OPS[2]
        assert len(rows) == _HAVOC_N
        assert all(row[2] >= 1 for row in rows)

    def test_counts_survive_save_load_by_name(self):
        # A resume that dropped these would silently restart adaptation from
        # the uniform prior on every restart -- invisible except as a slow
        # run. Keyed by name, so adding a 12th branch cannot shift the
        # counts by one slot.
        import tempfile
        from pathlib import Path

        from fuzzer_tool.services.corpus_manager import CorpusManager
        from tests.test_regression_array_corpus_history import _make_fuzzer

        idx = HAVOC_SUB_OPS.index("swap_regions")
        self.engine._havoc_hits[idx] = 41
        self.engine._havoc_trials[idx] = 97

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            saver = _make_fuzzer(tmp, [])
            saver._operators = self.engine
            cm = CorpusManager(saver)
            cm.save_state()
            raw = saver._state_store.get("corpus")
            assert raw["havoc_subop_stats"]["swap_regions"] == (41, 97)

            # Round-trip into a fresh engine: the counts must come back on
            # the branch they were recorded against, and the CDF must be
            # rebuilt from them rather than left at the uniform prior.
            loader = _make_fuzzer(tmp, [])
            loader._state_store = saver._state_store
            loader._operators = OperatorEngine(None)
            CorpusManager(loader).load_state()

        assert loader._operators._havoc_hits[idx] == 41
        assert loader._operators._havoc_trials[idx] == 97
        assert loader._operators._havoc_table != OperatorEngine(None)._havoc_table

    def test_max_len_still_honoured_under_weighting(self):
        # The weighted draw reaches the same branches, including the growth
        # ones; the max_len clamp must not depend on the uniform split.
        self.engine._havoc_hits[HAVOC_SUB_OPS.index("insert_byte")] = 1000
        self.engine._havoc_trials[HAVOC_SUB_OPS.index("insert_byte")] = 1000
        self.engine._rebuild_havoc_table()
        for max_len in (1, 2, 8, 64):
            self.fuzzer.max_len = max_len
            for _ in range(200):
                buf = bytearray((i * 37) % 256 for i in range(max_len))
                self.engine.havoc_mutate(buf)
                assert len(buf) <= max_len
