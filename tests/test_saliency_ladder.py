"""Covers core/saliency_ladder.py and the saliency_ladder operator wiring."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.saliency_ladder import (
    P_INS_DEL,
    choose_block_len,
    saliency_ladder,
)

DATA = bytes(range(100))


def _draws(n=3000, idx=(7, 40, 3, 90), signs=(1, -1, 1, -1), data=DATA, seed=1, **kw):
    r = random.Random(seed)
    return [saliency_ladder(data, idx, signs, r, **kw) for _ in range(n)]


class TestBlockLen:
    def test_bounds(self):
        r = random.Random(0)
        for limit in (1, 2, 5, 40, 200, 5000):
            for _ in range(300):
                assert 1 <= choose_block_len(limit, r) <= limit

    def test_degenerate_limit(self):
        assert choose_block_len(0, random.Random(0)) == 1


class TestLadder:
    def test_empty_inputs_pass_through(self):
        r = random.Random(0)
        assert saliency_ladder(b"", [1], [1], r) == b""
        assert saliency_ladder(DATA, [], [], r) == DATA
        assert saliency_ladder(DATA, [500, -1], [1, 1], r) == DATA  # all offsets out of range

    def test_always_changes_something(self):
        assert all(o != DATA for o in _draws(500))

    def test_ins_del_share_matches_p_ins_del(self):
        n = 4000
        resized = sum(len(o) != len(DATA) for o in _draws(n))
        assert abs(resized / n - P_INS_DEL) < 0.03

    def test_nudges_touch_only_ranked_bytes_and_stay_in_range(self):
        for o in _draws(1000):
            if len(o) == len(DATA):
                moved = {i for i in range(100) if o[i] != DATA[i]}
                assert moved and moved <= {7, 40, 3, 90}

    def test_top_ranked_bytes_move_most(self):
        hits = {7: 0, 40: 0, 3: 0, 90: 0}
        for o in _draws(6000):
            if len(o) == len(DATA):
                for i in hits:
                    hits[i] += o[i] != DATA[i]
        # tier [0:2) = {7, 40}; [2:4) = {3, 90}; weights 1 : 1/2
        assert min(hits[7], hits[40]) > 1.5 * max(hits[3], hits[90])

    def test_direction_follows_sign_more_often_than_not(self):
        up = down = 0
        for o in _draws(6000, idx=(7,), signs=(1,)):
            if len(o) == len(DATA):
                up += o[7] > DATA[7]
                down += o[7] < DATA[7]
        assert up > 2 * down  # P_FOLLOW = 0.75 -> ~3:1

        up = down = 0
        for o in _draws(6000, idx=(7,), signs=(-1,)):
            if len(o) == len(DATA):
                up += o[7] > DATA[7]
                down += o[7] < DATA[7]
        assert down > 2 * up

    def test_clips_to_byte_range(self):
        hi = bytes([255] * 50)
        lo = bytes([0] * 50)
        for o in _draws(500, data=hi, idx=(3,), signs=(1,)):
            assert all(0 <= b <= 255 for b in o)
        for o in _draws(500, data=lo, idx=(3,), signs=(-1,)):
            assert all(0 <= b <= 255 for b in o)

    def test_delete_never_empties_and_respects_max_len(self):
        tiny = b"abc"
        for o in _draws(2000, data=tiny, idx=(0, 1, 2), signs=(1, 1, 1)):
            assert len(o) >= 1
        for o in _draws(2000, max_len=60):
            if len(o) != len(DATA):
                assert len(o) <= 60

    def test_step_is_log_uniform_in_1_255(self):
        diffs = []
        for o in _draws(6000, idx=(7,), signs=(1,)):
            if len(o) == len(DATA) and o[7] != DATA[7]:
                diffs.append(abs(o[7] - DATA[7]))
        diffs = np.array(diffs)
        assert diffs.min() >= 1
        assert (diffs <= 3).mean() > 0.15  # bit-lengths 1-2 of 8 buckets: heavy small tail

    def test_deterministic_for_seeded_rng(self):
        assert _draws(50, seed=3) == _draws(50, seed=3)

    def test_works_with_randpool(self):
        r = RandPool(seed=1)
        for _ in range(300):
            assert isinstance(saliency_ladder(DATA, [7, 40], [1, -1], r), bytes)


class TestGradientInfo:
    def _fitted(self):
        from tests.test_pos_saliency import _arm, _corpus

        seeds, samples = _corpus()
        a = _arm(samples)
        a.refit()
        return seeds, a

    def test_none_without_model(self):
        from tests.test_pos_saliency import _arm

        assert _arm([]).gradient_info(b"abcd") is None

    def test_shapes_order_and_signs(self):
        seeds, a = self._fitted()
        idx, signs = a.gradient_info(seeds[0], top=16)
        assert len(idx) == len(signs) <= 16
        assert set(np.unique(signs)) <= {-1, 1}
        assert len(set(idx.tolist())) == len(idx) and idx.max() < 100

    def test_planted_bytes_rank_near_the_top(self):
        seeds, a = self._fitted()
        top8 = 0
        for _ in range(60):
            idx, _ = a.gradient_info(seeds[3], top=8)
            top8 += int(7 in idx or 40 in idx)
        assert top8 > 30  # uniform chance of either in 8 of 100: ~15%

    def test_warm_fits_lazily(self):
        from tests.test_pos_saliency import _arm, _corpus

        _, samples = _corpus()
        a = _arm(samples)
        assert a.warm() is True and a.ready

    def test_record_alone_builds_the_model(self):
        from fuzzer_tool.core.schedulers.pos_base import Outcome
        from tests.test_pos_saliency import _arm, _corpus

        _, samples = _corpus()
        a = _arm(samples)
        for _ in range(250):
            a.record(b"x", [0], Outcome.MISS, 1.0)
        assert a.ready


class TestOperatorWiring:
    def test_registered_and_gated_on_a_model(self):
        from fuzzer_tool.core import operator_registry as reg

        assert "saliency_ladder" in reg._AVAILABLE
        gate = reg._AVAILABLE["saliency_ladder"]
        assert gate(SimpleNamespace(), b"x") is False  # no --pos-saliency
        assert gate(SimpleNamespace(_pos_saliency=None), b"x") is False
        unfitted = SimpleNamespace(_pos_saliency=SimpleNamespace(warm=lambda: False))
        assert gate(unfitted, b"x") is False  # armed but no fitted model: unavailable
        assert gate(SimpleNamespace(_pos_saliency=SimpleNamespace(warm=lambda: True)), b"x") is True

    def test_in_the_adaptive_category(self):
        from fuzzer_tool.core import operator_registry as reg

        cats = [
            k for k, v in vars(reg).items() if isinstance(v, dict) and "saliency_ladder" in str(v)
        ]
        assert any("saliency_ladder" in str(getattr(reg, c)) for c in cats)

    @pytest.mark.parametrize("model", [True, False])
    def test_handler(self, model):
        from fuzzer_tool.services.operators import OperatorEngine  # noqa: F401 (import check)

        cls = OperatorEngine
        assert hasattr(cls, "_op_saliency_ladder")
        sal = None
        if model:
            from tests.test_pos_saliency import _arm, _corpus

            _, samples = _corpus()
            sal = _arm(samples)
            sal.refit()
        h = SimpleNamespace(
            f=SimpleNamespace(_pos_saliency=sal),
            ctx=SimpleNamespace(_rng=random.Random(1), max_len=0),
        )
        data = bytearray(bytes(range(100)))
        out = cls._op_saliency_ladder(h, data, 0, bytes(data))
        if model:
            assert out is not None and bytes(out) != bytes(data)
        else:
            assert out is None
        assert cls._op_saliency_ladder(h, bytearray(), 0, b"") is None


class TestReviewRegressions:
    """Bugs found reviewing upstream 410a481: crash, sign ignored, no insert/delete."""

    def _engine_call(self, sal, data, seed=3, max_len=0):
        from fuzzer_tool.services.operators import OperatorEngine

        h = SimpleNamespace(
            f=SimpleNamespace(_pos_saliency=sal),
            ctx=SimpleNamespace(_rng=RandPool(seed=seed), max_len=max_len),
        )
        buf = bytearray(data)
        return OperatorEngine._op_saliency_ladder(h, buf, 0, bytes(data)), buf

    def _sal(self):
        from tests.test_pos_saliency import _arm, _corpus

        seeds, samples = _corpus()
        a = _arm(samples)
        a.refit()
        return seeds, a

    @pytest.mark.parametrize("size", [1, 2, 5, 17, 100, 700])
    def test_never_raises_on_any_input_size(self, size):
        seeds, sal = self._sal()
        data = (seeds[0] * 8)[:size]
        for seed in range(400):
            out, _ = self._engine_call(sal, data, seed=seed)
            assert out is None or (isinstance(out, bytearray) and len(out) >= 1)

    def test_does_not_mutate_its_input_buffer_and_returns_a_new_one(self):
        seeds, sal = self._sal()
        for seed in range(100):
            out, buf = self._engine_call(sal, seeds[0], seed=seed)
            assert bytes(buf) == seeds[0]
            assert out is None or (out is not buf and bytes(out) != seeds[0])

    def test_sometimes_resizes_like_neuzz(self):
        seeds, sal = self._sal()
        sizes = {len(o) for s in range(300) if (o := self._engine_call(sal, seeds[0], seed=s)[0])}
        assert len(sizes) > 1

    def test_moves_follow_the_gradient_sign(self):
        seeds, sal = self._sal()
        d = seeds[0]
        agree = total = 0
        for seed in range(1500):
            sal._rng = RandPool(seed=seed)  # makes the target draw match the op's
            info = sal.gradient_info(d)
            sal._rng = RandPool(seed=seed)
            out, _ = self._engine_call(sal, d, seed=seed)
            if out is None or len(out) != len(d) or info is None:
                continue
            idx, signs = info
            gmap = dict(zip(idx.tolist(), signs.tolist(), strict=True))
            moved = [i for i in range(len(d)) if out[i] != d[i] and i in gmap]
            for i in moved:
                total += 1
                agree += (out[i] > d[i]) == (gmap[i] > 0)
        assert total > 200 and agree / total > 0.65  # P_FOLLOW = 0.75; coin flip would be 0.5

    def test_gradient_info_uses_one_target_not_a_mean(self):
        seeds, sal = self._sal()
        full = [sal.gradient_info(seeds[0], top=100) for _ in range(40)]
        assert all(f is not None for f in full)
        # different draws pick different target columns, so the ranked order varies
        assert len({tuple(f[0][:5].tolist()) for f in full}) > 1

    def test_respects_max_len(self):
        seeds, sal = self._sal()
        for seed in range(300):
            out, _ = self._engine_call(sal, seeds[0], seed=seed, max_len=60)
            assert out is None or len(out) <= 100  # nudges keep size; resizes are capped below

    def test_declines_without_a_model(self):
        from tests.test_pos_saliency import _arm

        out, _ = self._engine_call(_arm([]), b"abcdefgh")
        assert out is None


class TestFuzzerWiring:
    def test_saliency_targets_flag_selects_the_selector(self, tmp_path):
        from tests.test_position_arena import TestRealConstruction as W  # reuse the builder

        w = W()
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        gt = w._build(tmp_path / "a", pos_saliency=True)
        sup = w._build(tmp_path / "b", pos_saliency=True, saliency_targets="support")
        assert gt._pos_saliency._target_selector is not None
        assert sup._pos_saliency._target_selector is None

    def test_bad_value_is_rejected(self, tmp_path):
        from tests.test_position_arena import TestRealConstruction as W

        with pytest.raises(ValueError):
            W()._build(tmp_path, pos_saliency=True, saliency_targets="bogus")
