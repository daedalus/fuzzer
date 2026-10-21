"""PositionPCAScheduler (core/schedulers/pos_pca.py) and its engine capture."""

import numpy as np
import pytest

from fuzzer_tool.core.byte_pca import ByteGroups, sketch_row
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_pca import (
    EPSILON,
    MAX_SEEDS,
    PositionPCAScheduler,
)

SEED = bytes(range(64))
NO_ESCAPE = 0.99


class ScriptedRng:
    """Scripted random(): escape draw, then group CDF point, then byte CDF point."""

    def __init__(self, randoms=()):
        self._randoms = list(randoms)

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_ESCAPE


def _groups(offsets=(3, 5, 9, 20, 22), comp=(0, 0, 0, 1, 1), weight=None):
    weight = weight if weight is not None else [1.0] * len(offsets)
    return ByteGroups(
        offsets=np.array(offsets, dtype=np.uint32),
        comp=np.array(comp, dtype=np.int8),
        weight=np.array(weight, dtype=np.float32),
        count=4,
    )


def _sched(rng=None, groups=None, ready=True):
    box = {"groups": groups if groups is not None else _groups()}
    arm = PositionPCAScheduler(
        rng or ScriptedRng(),
        groups_of=lambda d: box["groups"],
        ready=lambda: ready,
        key_of=lambda d: bytes(d[:4]).hex(),
    )
    return arm, box


def _pick(arm, u_group, u_byte, buf_len=64, data=SEED):
    arm._rng = ScriptedRng([NO_ESCAPE, u_group, u_byte])
    return arm.propose(data, buf_len)


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_sched()[0], PositionScheduler)

    def test_name_and_gate(self):
        arm, _ = _sched(ready=False)
        assert arm.name == "pca" and not arm.active()
        assert _sched(ready=True)[0].active()


class TestPropose:
    def test_offsets_come_from_the_chosen_group(self):
        arm, _ = _sched()
        assert _pick(arm, 0.0, 0.0) in (3, 5, 9)  # group 0
        assert _pick(arm, 0.99, 0.0) in (20, 22)  # group 1

    def test_within_group_choice_follows_the_weights(self):
        arm, _ = _sched(groups=_groups(weight=[1, 1, 8, 1, 1]))
        picks = [_pick(arm, 0.0, u) for u in (0.05, 0.3, 0.5, 0.9)]
        assert picks.count(9) == 3  # 8/10 of group 0's mass sits on byte 9

    def test_declines_without_groups(self):
        arm, box = _sched()
        box["groups"] = None
        assert arm.propose(SEED, 64) is None

    def test_declines_on_empty_buffer_or_data(self):
        arm, _ = _sched()
        assert arm.propose(SEED, 0) is None
        assert arm.propose(b"", 8) is None

    def test_declines_when_no_member_is_inside_the_buffer(self):
        arm, _ = _sched()
        assert arm.propose(SEED, 2) is None

    @pytest.mark.parametrize("buf_len", [4, 10, 21])
    def test_never_proposes_past_the_live_buffer(self, buf_len):
        arm, _ = _sched(rng=RandPool(seed=3))
        for _ in range(200):
            off = arm.propose(SEED, buf_len)
            assert off is None or 0 <= off < buf_len

    def test_epsilon_escape_declines(self):
        arm, _ = _sched()
        arm._rng = ScriptedRng([EPSILON / 2])
        assert arm.propose(SEED, 64) is None


class TestLearning:
    def test_record_credits_the_group_an_offset_fell_in(self):
        arm, _ = _sched()
        base = arm.group_rate(SEED, 0)
        arm.record(SEED, [3, 5], Outcome.GAIN)  # both in group 0: one credit
        assert arm.group_rate(SEED, 0) > base
        assert arm.group_rate(SEED, 1) == base

    def test_a_miss_lowers_the_rate(self):
        arm, _ = _sched()
        base = arm.group_rate(SEED, 1)
        arm.record(SEED, [20], Outcome.MISS)
        assert arm.group_rate(SEED, 1) < base

    def test_offsets_outside_every_group_credit_nothing(self):
        arm, _ = _sched()
        arm.record(SEED, [0, 1, 60], Outcome.GAIN)
        assert arm.seeds_tracked() == 0

    def test_ignores_negative_and_empty_offsets(self):
        arm, _ = _sched()
        arm.record(SEED, [], Outcome.GAIN)
        arm.record(SEED, [-5], Outcome.GAIN)
        assert arm.seeds_tracked() == 0

    def test_weight_is_ignored(self):
        a, _ = _sched()
        b, _ = _sched()
        a.record(SEED, [3], Outcome.GAIN, weight=0.01)
        b.record(SEED, [3], Outcome.GAIN, weight=1000.0)
        assert a.group_rate(SEED, 0) == b.group_rate(SEED, 0)

    def test_gain_heavy_group_is_drawn_more(self):
        arm, _ = _sched(rng=RandPool(seed=5))
        for _ in range(40):
            arm.record(SEED, [3], Outcome.GAIN)
            arm.record(SEED, [20], Outcome.MISS)
        hits = [arm.propose(SEED, 64) for _ in range(600)]
        in0 = sum(1 for h in hits if h in (3, 5, 9))
        in1 = sum(1 for h in hits if h in (20, 22))
        assert in0 > 3 * in1

    def test_seed_table_is_lru_bounded(self):
        arm, _ = _sched()
        for i in range(MAX_SEEDS + 5):
            arm.record(i.to_bytes(4, "big") + SEED, [3], Outcome.GAIN)
        assert arm.seeds_tracked() == MAX_SEEDS


class TestEngineCapture:
    """OperatorEngine sketches LIVE byteflips and groups them at the drain."""

    @staticmethod
    def _fuzzer(tmp_path, seed):
        from fuzzer_tool.services.fuzzer import Fuzzer

        corpus = tmp_path / "corpus"
        (corpus / "seeds").mkdir(parents=True)
        (tmp_path / "crashes").mkdir()
        (corpus / "seeds" / "seed1").write_bytes(seed)
        f = Fuzzer(
            "/bin/true",
            corpus_dir=str(corpus),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=4096,
            deterministic=True,
        )
        key = f._seed_key(seed)
        f._favored = {key}
        f._edge_tracker.seed_edges[key] = {1, 2, 3}
        return f

    @staticmethod
    def _trace(byte):
        """Three fields (byte // 8 % 3), each with its own edges, on a shared base."""
        field = (byte // 8) % 3
        own = np.arange(10_000 * (field + 1), 10_000 * (field + 1) + 30)
        return np.concatenate([np.arange(1, 400), own])

    def _drain(self, f, data, live):
        eng = f._operators
        while eng.maybe_deterministic_mutation(data) is not None:
            pending = eng._det_pending
            if pending is None:
                continue
            idx = pending[1]
            changed = idx in live
            ids = self._trace(idx) if changed and eng.wants_det_edges(pending[0]) else None
            eng.note_deterministic_result(changed, 7 if changed else 0, 1, ids)

    def test_capture_is_off_by_default(self, tmp_path):
        data = bytes((0x41 + i) & 0xFF for i in range(48))
        f = self._fuzzer(tmp_path, data)
        eng = f._operators
        eng.maybe_deterministic_mutation(data)
        assert not eng.wants_det_edges(f._seed_key(data))
        self._drain(f, data, set(range(48)))
        assert eng.effector_groups(data) is None and not eng.has_effector_groups()

    def test_groups_survive_the_drain(self, tmp_path):
        data = bytes((0x41 + i) & 0xFF for i in range(48))
        f = self._fuzzer(tmp_path, data)
        eng = f._operators
        eng.enable_pca_capture()
        self._drain(f, data, set(range(48)))
        got = eng.effector_groups(data)
        assert got is not None and eng.has_effector_groups()
        by_group = {}
        for off, comp in zip(got.offsets, got.comp, strict=True):
            by_group.setdefault(int(comp), set()).add((int(off) // 8) % 3)
        assert all(len(v) == 1 for v in by_group.values())

    def test_inert_flips_are_not_rows(self, tmp_path):
        data = bytes((0x41 + i) & 0xFF for i in range(16))
        f = self._fuzzer(tmp_path, data)
        eng = f._operators
        eng.enable_pca_capture()
        self._drain(f, data, set())
        assert eng.effector_groups(data) is None

    def test_row_count_is_capped(self, tmp_path):
        from fuzzer_tool.core.byte_pca import MAX_ROWS

        data = bytes((0x41 + i) & 0xFF for i in range(MAX_ROWS + 40))
        f = self._fuzzer(tmp_path, data)
        eng = f._operators
        eng.enable_pca_capture()
        key = f._seed_key(data)
        eng.maybe_deterministic_mutation(data)
        eff = eng._det_eff[key]
        for i in range(MAX_ROWS + 40):
            eff.rows.append((i, sketch_row(np.arange(5)))) if eng.wants_det_edges(key) else None
        assert len(eff.rows) == MAX_ROWS

    def test_concurrent_capture_is_bounded(self, tmp_path):
        from fuzzer_tool.services.operators import MAX_PCA_CAPTURES

        data = bytes(range(8))
        f = self._fuzzer(tmp_path, data)
        eng = f._operators
        eng.enable_pca_capture()
        from fuzzer_tool.services.operators import DeterministicEffectorMap

        for i in range(MAX_PCA_CAPTURES + 3):
            eng._det_eff[f"s{i}"] = DeterministicEffectorMap(8, capture=eng._may_capture())
        assert sum(m.rows is not None for m in eng._det_eff.values()) == MAX_PCA_CAPTURES

    def test_group_retention_is_bounded(self, tmp_path):
        from fuzzer_tool.services.operators import MAX_EFF_SEEDS

        f = self._fuzzer(tmp_path, bytes(8))
        eng = f._operators
        rows = [(i, sketch_row(np.arange(1, 5 + 30 * (i % 3)))) for i in range(12)]
        for i in range(MAX_EFF_SEEDS + 3):
            eng._keep_groups(f"k{i}", rows)
        assert len(eng._det_groups) <= MAX_EFF_SEEDS

    def test_unsorted_rows_are_sorted_before_grouping(self, tmp_path):
        f = self._fuzzer(tmp_path, bytes(8))
        eng = f._operators
        rng = np.random.default_rng(2)
        rows = [
            (i, sketch_row(np.concatenate([np.arange(1, 50), rng.integers(1000, 9000, 20)])))
            for i in range(30)
        ]
        rows.reverse()
        eng._keep_groups("k", rows)
        got = eng._det_groups.get("k")
        assert got is None or list(got.offsets) == sorted(got.offsets)
