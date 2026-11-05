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
