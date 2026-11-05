"""Tests for tools/sim_funnel_probe.py.

Linear probe placement is checked against ``adapters/shm.displacements`` (the
shim mirror), never against the simulator's own bookkeeping.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import displacements, home_slot
from fuzzer_tool.core.rand_pool import RandPool

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import sim_funnel_probe as sim  # noqa: E402

SIZE = 8192
WINDOW = 64
DELTA = 1 / 8
SEED = 7


def _ids_homed_at(slot: int, size: int, count: int) -> list[int]:
    """First *count* ids (1, 2, ...) whose home slot is *slot*: deterministic."""
    out = []
    eid = 1
    while len(out) < count:
        if home_slot(eid, size) == slot:
            out.append(eid)
        eid += 1
    return out


def _fill(table, load: float) -> list[int]:
    ids = sim.unique_ids(RandPool(seed=SEED), int(load * table.size))
    for eid in ids:
        table.insert(eid)
    return ids


class TestLinear:
    def test_hit_probes_match_shm_displacement(self):
        table = sim.LinearTable(SIZE, WINDOW)
        ids = [e for e in _fill(table, 0.85) if table.lookup(e).found]
        slots = np.array([table.slot_of(e) for e in ids], dtype=np.uint32)
        expect = displacements(slots, np.array(ids, dtype=np.uint32), SIZE) + 1
        got = np.array([table.lookup(e).probes for e in ids])
        assert np.array_equal(got, expect)

    def test_full_window_drops(self):
        size, window = 16, 4
        table = sim.LinearTable(size, window)
        ids = _ids_homed_at(0, size, window + 1)
        placed = [table.insert(e) for e in ids]
        assert placed[:window] == [sim.Insert.PLACED] * window
        assert placed[window] is sim.Insert.DROPPED

    def test_probe_crossing_line_touches_two_lines(self):
        size = 16
        last = next(s for s in range(size - 1) if sim.line_of(s) != sim.line_of(s + 1))
        table = sim.LinearTable(size, WINDOW)
        first, second = _ids_homed_at(last, size, 2)
        table.insert(first)
        table.insert(second)
        hit = table.lookup(second)
        assert (hit.probes, hit.lines) == (2, 2)


class TestFunnel:
    def test_levels_partition_table(self):
        table = sim.FunnelTable(SIZE, DELTA)
        assert table.layout_size() == SIZE

    def test_search_cost_equals_insert_cost(self):
        # Greedy, no reordering: a key's search probes equal its insert probes.
        table = sim.FunnelTable(SIZE, DELTA)
        ids = sim.unique_ids(RandPool(seed=SEED), int(0.85 * SIZE))
        costs = {e: table.insert_probes(e) for e in ids}
        for eid, cost in costs.items():
            if cost is None:
                continue
            assert table.lookup(eid).probes == cost

    @pytest.mark.parametrize("delta", [0.25, 0.1, 0.0])
    def test_rejects_bad_delta(self, delta):
        with pytest.raises(ValueError):
            sim.FunnelTable(SIZE, delta)


class TestAdversarial:
    @pytest.mark.parametrize("kind", list(sim.Kind))
    def test_overfull_table_terminates(self, kind):
        size = 256
        table = sim.make_table(kind, size, WINDOW, DELTA)
        ids = sim.unique_ids(RandPool(seed=SEED), 2 * size)
        outcomes = [table.insert(e) for e in ids]
        assert outcomes.count(sim.Insert.PLACED) <= size
        assert sim.Insert.DROPPED in outcomes

        absent = sim.unique_ids(RandPool(seed=SEED + 1), 64, exclude=set(ids))
        assert not any(table.lookup(e).found for e in absent)

    def test_unique_ids_excludes_and_is_distinct(self):
        taken = set(range(1, 1000))
        ids = sim.unique_ids(RandPool(seed=SEED), 500, exclude=taken)
        assert len(set(ids)) == 500
        assert not taken & set(ids)
        assert 0 not in ids


class TestControl:
    @pytest.mark.parametrize("kind", list(sim.Kind))
    def test_same_seed_same_stats(self, kind):
        # A/A: identical configs must report identical stats, else the harness leaks state.
        args = (kind, SIZE // 8, WINDOW, DELTA, 0.8, 200)
        a = sim.run_trial(*args, RandPool(seed=SEED))
        b = sim.run_trial(*args, RandPool(seed=SEED))
        assert a == b

    @pytest.mark.parametrize("kind", list(sim.Kind))
    def test_every_placed_key_is_found(self, kind):
        table = sim.make_table(kind, SIZE, WINDOW, DELTA)
        ids = sim.unique_ids(RandPool(seed=SEED), int(0.9 * SIZE))
        placed = [e for e in ids if table.insert(e) is sim.Insert.PLACED]
        assert all(table.lookup(e).found for e in placed)


class TestElastic:
    def test_arrays_halve_and_cover_table(self):
        table = sim.ElasticTable(SIZE, DELTA)
        sizes = table.array_sizes()
        assert sum(sizes) == SIZE
        assert len(sizes) == (SIZE - 1).bit_length()
        assert all(abs(2 * b - a) <= 2 for a, b in zip(sizes, sizes[1:], strict=False))

    def test_phi_injective_bounded_monotone(self):
        grid = [(i, j) for i in range(1, 20) for j in range(1, 200)]
        phis = [sim.phi(i, j) for i, j in grid]
        assert len(set(phis)) == len(grid)
        # Lemma 1: phi(i, j) <= O(i j^2); bit lengths give the constant 16.
        assert all(p < 16 * i * j * j for p, (i, j) in zip(phis, grid, strict=True))
        assert all(sim.phi(5, j) < sim.phi(5, j + 1) for j in range(1, 199))

    def test_hit_probes_equal_phi_rank(self):
        # Oracle: rank of the key's (i, j) among scanned pairs, counted from phi alone.
        table = sim.ElasticTable(SIZE // 4, DELTA)
        ids = _fill(table, 0.8)
        max_j = table.max_j()
        pairs = [(i, j) for i, top in enumerate(max_j, 1) for j in range(1, top + 1)]
        for eid in ids[:: len(ids) // 50]:
            i, j = table.placement(eid)
            rank = sum(sim.phi(a, b) <= sim.phi(i, j) for a, b in pairs)
            hit = table.lookup(eid)
            assert (hit.found, hit.probes) == (True, rank)

    def test_miss_scans_every_used_probe(self):
        # Non-greedy: an empty slot proves nothing, so a miss scans all pairs in use.
        table = sim.ElasticTable(SIZE // 4, DELTA)
        ids = _fill(table, 0.8)
        absent = sim.unique_ids(RandPool(seed=SEED + 1), 20, exclude=set(ids))
        assert {table.lookup(e).probes for e in absent} == {sum(table.max_j())}

    def test_finished_batches_leave_delta_half_free(self):
        # Paper §2 batch invariant: every A_j before the current batch is 1 - delta/2 full.
        table = sim.ElasticTable(SIZE, DELTA)
        _fill(table, 1 - DELTA)
        sizes, fill = table.array_sizes(), table.fill()
        for j in range(table.batch - 1):
            assert fill[j] >= sizes[j] - int(DELTA * sizes[j] / 2)

    @pytest.mark.parametrize("delta", [0.3, 1.5, 0.0])
    def test_rejects_bad_delta(self, delta):
        with pytest.raises(ValueError):
            sim.ElasticTable(SIZE, delta)


class TestHybrid:
    def test_buckets_are_line_aligned(self):
        table = sim.HybridTable(SIZE, DELTA)
        width = sim.LINE_ENTRIES
        for unit in range(sum(table.array_sizes())):
            base = table.bucket_base(unit)
            assert sim.line_of(base) == sim.line_of(base + width - 1)
            assert base + width <= SIZE

    def test_lone_key_costs_one_line(self):
        # One key: batch 0 puts it in A_1's first bucket; a miss sees that
        # bucket's empty slots, rules A_1 out, and no other array is in use.
        table = sim.HybridTable(SIZE, DELTA)
        table.insert(1)
        assert table.lookup(1) == sim.Lookup(True, 1, 1)
        assert table.lookup(2) == sim.Lookup(False, 1, 1)

    def test_miss_stops_before_full_scan(self):
        # Falsification of the early exit: elastic-style misses scan sum(max_j).
        table = sim.HybridTable(SIZE, DELTA)
        ids = _fill(table, 0.6)
        absent = sim.unique_ids(RandPool(seed=SEED + 1), 50, exclude=set(ids))
        probes = [table.lookup(e).probes for e in absent]
        assert max(probes) <= sum(table.max_j())
        assert sum(probes) < len(probes) * sum(table.max_j())

    def test_same_seed_same_stats(self):
        _args = (sim.Kind.HYBRID, SIZE // 8, WINDOW, DELTA, 0.8, 200)
        assert sim.run_trial(*_args, RandPool(seed=SEED)) == sim.run_trial(
            *_args, RandPool(seed=SEED)
        )

    @pytest.mark.parametrize("delta", [0.3, 1.5, 0.0])
    def test_rejects_bad_delta(self, delta):
        with pytest.raises(ValueError):
            sim.HybridTable(SIZE, delta)


def _tracker(n_edges: int, map_size: int):
    from fuzzer_tool.core.edge_tracker import EdgeTracker

    t = EdgeTracker(map_size=map_size)
    for i in range(n_edges):
        t._global_edge_hits[i] = 1
    return t


class TestGrowth:
    @pytest.mark.parametrize("n", [50, 99, 100, 5000, 5734, 5735, 7000, 60_000])
    @pytest.mark.parametrize("size", [8192, 16384])
    @pytest.mark.parametrize("dropped", [0, 1, 3000])
    def test_policy_matches_edge_tracker_at_shipped_threshold(self, n, size, dropped):
        # Oracle: the fuzzer's own recommended_map_size, which hardcodes 0.7.
        want = _tracker(n, size).recommended_map_size(dropped_edges=dropped)
        got = sim.recommend_size(n, size, dropped, sim.SHIPPED_THRESHOLD, sim.MAP_SIZE_MAX)
        assert got == want

    def test_same_seed_same_stats(self):
        # A/A control.
        args = (sim.Kind.LINEAR, 3000, 0.7, 256, 50)
        a = sim.run_growth(*args, RandPool(seed=SEED))
        b = sim.run_growth(*args, RandPool(seed=SEED))
        assert a == b

    def test_low_threshold_resizes_before_any_drop(self):
        # Growth needs HEADROOM x edges > size: 6000 edges cross 0.6 at 5120 (load 0.625).
        stats = sim.run_growth(sim.Kind.LINEAR, 6000, 0.6, 256, 20, RandPool(seed=SEED))
        assert stats.resizes >= 1
        assert stats.drop_events == 0

    def test_disabled_threshold_without_drops_never_resizes(self):
        # Falsification: a load trigger above 1 can only fire through drops.
        stats = sim.run_growth(sim.Kind.LINEAR, 500, 2.0, 100, 20, RandPool(seed=SEED))
        assert (stats.resizes, stats.drop_events, stats.final_size) == (0, 0, sim.MAP_SIZE_DEFAULT)

    def test_capped_map_keeps_dropping_and_terminates(self):
        stats = sim.run_growth(
            sim.Kind.LINEAR, 12_000, 0.7, 512, 20, RandPool(seed=SEED), cap=sim.MAP_SIZE_DEFAULT
        )
        assert stats.resizes == 0
        assert stats.final_size == sim.MAP_SIZE_DEFAULT
        assert stats.drop_events >= 12_000 - sim.MAP_SIZE_DEFAULT


class TestHot:
    def test_zero_exponent_is_plain_mean(self):
        # Control: uniform weights must reproduce the unweighted mean.
        lines = np.array([1, 2, 2, 5, 1])
        assert sim.weighted_lines(lines, 0.0) == pytest.approx(lines.mean())

    def test_two_edge_weights(self):
        # w = [1, 1/2]: (1*1 + 3*0.5) / 1.5
        assert sim.weighted_lines(np.array([1, 3]), 1.0) == pytest.approx(2.5 / 1.5)

    def test_steep_exponent_is_hottest_edge(self):
        assert sim.weighted_lines(np.array([1, 9, 9, 9]), 60.0) == pytest.approx(1.0)

    def test_empty_and_negative(self):
        assert sim.weighted_lines(np.array([], dtype=np.int64), 1.0) == 0.0
        with pytest.raises(ValueError):
            sim.weighted_lines(np.array([1]), -0.5)

    @pytest.mark.parametrize("kind", [sim.Kind.LINEAR, sim.Kind.HYBRID])
    def test_uniform_matches_run_trial(self, kind):
        # Oracle: s = 0 must equal run_trial's hit_lines on the same draws.
        args = (kind, SIZE // 4, WINDOW, DELTA, 0.8)
        (hot,) = sim.run_hot(*args, (0.0,), RandPool(seed=SEED))
        ref = sim.run_trial(*args, 10, RandPool(seed=SEED))
        assert hot == pytest.approx(ref.hit_lines)

    def test_same_seed_same_result(self):
        args = (sim.Kind.HYBRID, SIZE // 4, WINDOW, DELTA, 0.8, (0.0, 1.0))
        assert sim.run_hot(*args, RandPool(seed=SEED)) == sim.run_hot(*args, RandPool(seed=SEED))


class TestRealWeights:
    def test_weighted_mean_arithmetic(self):
        assert sim.weighted_mean(np.array([1, 3]), np.array([1.0, 0.5])) == pytest.approx(2.5 / 1.5)

    def test_zero_weight_total_raises(self):
        with pytest.raises(ValueError):
            sim.weighted_mean(np.array([1, 2]), np.zeros(2))

    @pytest.mark.parametrize("kind", [sim.Kind.LINEAR, sim.Kind.HYBRID])
    def test_unit_weights_match_hot_uniform(self, kind):
        # Oracle: equal real weights must equal hot mode at s = 0 on the same draws.
        n, load = 1600, 0.8
        got = sim.run_real(kind, np.ones(n), load, RandPool(seed=SEED))
        (ref,) = sim.run_hot(kind, int(n / load), WINDOW, DELTA, load, (0.0,), RandPool(seed=SEED))
        assert got == pytest.approx(ref)
