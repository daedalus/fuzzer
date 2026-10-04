"""Covers core/schedulers/pos_saliency.py."""

from collections import Counter

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_saliency import (
    MIN_SEEDS,
    PositionSaliencyScheduler,
    _adjusted_counts,
    gt_rarity_selector,
)


def _corpus(n=200, length=100, seed=0):
    r = np.random.default_rng(seed)
    seeds = [bytes(r.integers(0, 256, length, dtype=np.uint8)) for _ in range(n)]

    def edges(s):
        e = {0}
        if s[7] > 128:
            e.add(1)
        if s[40] < 60:
            e.add(2)
        if s[7] > 128 and s[40] < 60:
            e.add(3)
        return e

    return seeds, [(s, edges(s)) for s in seeds]


def _arm(samples, **kw):
    kw.setdefault("refit_interval", 10)
    return PositionSaliencyScheduler(RandPool(seed=1), lambda: samples, **kw)


class TestContract:
    def test_satisfies_protocol(self):
        assert isinstance(_arm([]), PositionScheduler)

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            PositionSaliencyScheduler(None, lambda: [])

    @pytest.mark.parametrize("kw", [{"input_cap": 0}, {"hidden": 0}, {"explore": 1.5}])
    def test_rejects_bad_params(self, kw):
        with pytest.raises(ValueError):
            _arm([], **kw)

    def test_record_is_safe_without_model(self):
        a = _arm([])
        a.record(b"abc", [0], Outcome.GAIN, 1.0)


class TestWarmup:
    def test_declines_while_corpus_is_empty_then_fits_on_first_data(self):
        seeds, samples = _corpus()
        live: list = []
        a = PositionSaliencyScheduler(RandPool(seed=1), lambda: live, refit_interval=10)
        assert a.propose(seeds[0], 100) is None and not a.ready
        live.extend(samples)
        for _ in range(250):  # retry cadence while no model exists
            a.propose(seeds[0], 100)
        assert a.ready

    def test_declines_below_min_seeds(self):
        seeds, samples = _corpus(n=MIN_SEEDS - 1)
        a = _arm(samples)
        assert a.refit() is False
        assert "fewer than" in a.stats()["skip_reason"]
        assert a.propose(seeds[0], 100) is None

    def test_no_informative_edges(self):
        seeds = [bytes([i]) * 4 for i in range(MIN_SEEDS + 2)]
        a = _arm([(s, {1}) for s in seeds])  # one edge hit by every seed: constant column
        assert a.refit() is False
        assert a.stats()["skip_reason"] == "no informative edges"

    def test_empty_input_or_buffer_declines(self):
        _, samples = _corpus()
        a = _arm(samples)
        a.refit()
        assert a.propose(b"", 10) is None
        assert a.propose(b"abc", 0) is None


class TestFit:
    def test_refit_builds_model_and_columns(self):
        _, samples = _corpus()
        a = _arm(samples)
        assert a.refit() is True
        st = a.stats()
        assert st["ready"] and st["fits"] == 1 and st["targets"] == 3  # edge 0 is constant
        assert st["width"] == 100 and st["fit_seconds"] > 0

    def test_unchanged_sample_is_not_refit(self):
        _, samples = _corpus()
        a = _arm(samples)
        a.refit()
        assert a.refit() is False and a.stats()["fits"] == 1
        assert a.refit(force=True) is True and a.stats()["fits"] == 2

    def test_identical_columns_merge(self):
        # edges 5 and 6 are hit by the same seeds: one class, one output column
        seeds = [bytes([i, i]) for i in range(20)]
        samples = [(s, {5, 6} if s[0] < 10 else {7}) for s in seeds]
        a = _arm(samples)
        a.refit()
        assert a.stats()["targets"] == 2

    def test_target_columns_are_capped(self):
        r = np.random.default_rng(0)
        seeds = [bytes(r.integers(0, 256, 16, dtype=np.uint8)) for _ in range(40)]
        samples = [(s, {int(e) for e in r.integers(0, 500, 60)}) for s in seeds]
        a = _arm(samples)
        a.refit()
        assert a.stats()["targets"] <= 128

    def test_refit_failure_is_swallowed(self):
        def boom():
            raise RuntimeError("x")

        a = PositionSaliencyScheduler(RandPool(seed=1), boom, refit_interval=1)
        for _ in range(300):
            assert a.propose(b"abcdefgh", 8) is None
        assert a.stats()["skip_reason"] == "refit raised"


class TestPropose:
    def _ready(self, **kw):
        seeds, samples = _corpus()
        a = _arm(samples, **kw)
        a.refit()
        return seeds, a

    def test_offsets_in_range(self):
        seeds, a = self._ready()
        for _ in range(500):
            o = a.propose(seeds[0], 100)
            assert o is not None and 0 <= o < 100

    def test_respects_shrunk_buffer(self):
        seeds, a = self._ready()
        for _ in range(300):
            assert 0 <= a.propose(seeds[0], 30) < 30

    def test_bytes_beyond_the_net_still_get_proposals(self):
        seeds, a = self._ready()
        hits = Counter(a.propose(seeds[0], 400) >= 100 for _ in range(1500))
        assert 0.6 < hits[True] / 1500 < 0.9  # ~300/400 by construction

    def test_concentrates_on_planted_bytes(self):
        seeds, a = self._ready()
        c = Counter(a.propose(seeds[5], 100) for _ in range(4000))
        planted = (c[7] + c[40]) / 4000
        assert planted > 2 * (2 / 100)  # clearly above uniform; explore floor keeps all bytes live

    def test_explore_floor_keeps_every_byte_reachable(self):
        seeds, a = self._ready(explore=1.0)
        c = Counter(a.propose(seeds[5], 100) for _ in range(6000))
        assert len(c) > 95

    def test_seeded_runs_are_reproducible(self):
        _, samples = _corpus()
        runs = []
        for _ in range(2):
            a = _arm(samples)
            a.refit()
            runs.append([a.propose(samples[0][0], 100) for _ in range(50)])
        assert runs[0] == runs[1]

    def test_cap_limits_the_net_width(self):
        seeds, samples = _corpus(length=300)
        a = _arm(samples, input_cap=64)
        a.refit()
        assert a.stats()["width"] == 64
        assert a.saliency(seeds[0]).shape == (64,)


class TestCadence:
    def test_first_fit_happens_via_propose(self):
        seeds, samples = _corpus()
        a = _arm(samples, refit_interval=5)
        for _ in range(400):
            a.propose(seeds[0], 100)
        assert a.ready and a.stats()["fits"] == 1  # later ticks see an unchanged sample


class TestTargetSelector:
    @staticmethod
    def _spectrum(by_edge):
        sp = {}
        for rows in by_edge.values():
            sp[len(rows)] = sp.get(len(rows), 0) + 1
        return sp

    def test_gt_weights_fall_with_support(self):
        # supports 1,1,2,2,3,5,8,13 over 20 seeds: a plausible heavy-tailed spectrum
        sup = [1, 1, 1, 2, 2, 3, 5, 8, 13]
        by_edge = {100 + i: list(range(c)) for i, c in enumerate(sup)}
        ids = np.array(sorted(by_edge), dtype=np.int64)
        w = gt_rarity_selector(ids, by_edge, 20)
        assert len(w) == len(sup) and (w > 0).all()
        order = np.argsort(sup, kind="stable")
        assert (np.diff(w[order]) <= 1e-12).all()  # never inverts the rarity order

    def test_gt_uses_real_edge_ids(self):
        # ids far from 0..n: a selector indexing by position instead of id sees zero support
        by_edge = {1000: [0], 1001: [0, 1], 2000: list(range(10))}
        w = gt_rarity_selector(np.array([1000, 1001, 2000], dtype=np.int64), by_edge, 10)
        assert (w > 0).all() and w[0] > w[2]

    def test_gt_singletons_stand_out_more_than_one_over_support(self):
        sup = [1] * 6 + [2] * 3 + [3, 4, 6, 9]
        by_edge = {i: list(range(c)) for i, c in enumerate(sup)}
        ids = np.array(sorted(by_edge), dtype=np.int64)
        w = gt_rarity_selector(ids, by_edge, 20)
        s = np.array(sup, dtype=np.float64)
        # relative singleton:doubleton weight is sharper than the raw-count 2:1
        assert (w[0] / w[6]) > (1 / s[0]) / (1 / s[6])

    def test_adjusted_counts_fall_back_when_slope_is_flat(self):
        sp = {1: 3, 2: 3, 3: 3}  # flat spectrum: slope ~ 0 > -1, no adjustment
        c = _adjusted_counts(np.array([1, 2, 3]), sp)
        assert np.array_equal(c, [1.0, 2.0, 3.0])

    def test_gt_empty_and_unseen(self):
        assert len(gt_rarity_selector(np.array([], dtype=np.int64), {}, 20)) == 0
        w = gt_rarity_selector(np.array([5], dtype=np.int64), {}, 20)
        assert w.tolist() == [0.0]  # an edge no seed hit has no weight

    def test_selector_receives_representative_edge_ids(self):
        seen = {}

        def spy(edge_ids, by_edge, n):
            seen["ids"], seen["keys"] = edge_ids.tolist(), set(by_edge)
            return np.ones(len(edge_ids))

        seeds = [bytes([i]) * 8 for i in range(20)]
        samples = [(s, {1000, 1001 + s[0] % 3, 2000 + s[0]}) for s in seeds]
        a = PositionSaliencyScheduler(RandPool(seed=1), lambda: samples, target_selector=spy)
        assert a.refit()
        assert seen["ids"] and set(seen["ids"]) <= seen["keys"]  # real ids, not seed indices

    def test_selector_weights_replace_one_over_support(self):
        seeds = [bytes([i]) * 8 for i in range(20)]
        samples = [(s, {5} if s[0] < 4 else {6}) for s in seeds]  # supports 4 and 16

        def heavy_common(ids, by_edge, n):
            return np.array([1.0 if len(by_edge[int(e)]) == 16 else 1e-6 for e in ids])

        a = PositionSaliencyScheduler(
            RandPool(seed=1), lambda: samples, target_selector=heavy_common
        )
        a.refit()
        cum = a._cum_support
        assert cum[0] < 1e-3 and cum[-1] > 0.99  # the 16-seed column carries ~all the mass

    def test_default_weight_is_exactly_one_over_support(self):
        seeds = [bytes([i]) * 8 for i in range(20)]
        samples = [(s, {5} if s[0] < 4 else {6}) for s in seeds]
        a = PositionSaliencyScheduler(RandPool(seed=1), lambda: samples)
        a.refit()
        assert np.allclose(np.diff(np.concatenate(([0.0], a._cum_support))), [1 / 4, 1 / 16])

    @pytest.mark.parametrize(
        "bad", [lambda *a: np.ones(1), lambda *a: -np.ones(len(a[0])), lambda *a: 1 / 0]
    )
    def test_bad_selector_falls_back_without_stopping_the_fit(self, bad):
        seeds = [bytes([i]) * 8 for i in range(20)]
        samples = [(s, {5} if s[0] < 4 else {6, 7}) for s in seeds]
        a = PositionSaliencyScheduler(RandPool(seed=1), lambda: samples, target_selector=bad)
        assert a.refit() and a.ready
        assert np.allclose(np.diff(np.concatenate(([0.0], a._cum_support))), [1 / 4, 1 / 16])

    def test_failed_fit_leaves_model_and_weights_in_step(self, monkeypatch):
        _, samples = _corpus()
        a = _arm(samples)
        a.refit()
        cum, fits = a._cum_support.copy(), a.stats()["fits"]
        monkeypatch.setattr(
            "fuzzer_tool.core.schedulers.pos_saliency.TinyMLP.fit",
            lambda *x, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError):
            a.refit(force=True)
        assert np.array_equal(a._cum_support, cum) and a.stats()["fits"] == fits


class TestSharedCadence:
    """Refit policy is core.edge_matrix.RefitCadence: executions, not calls."""

    def _driven(self, samples, clock, **kw):
        return PositionSaliencyScheduler(
            RandPool(seed=1), lambda: samples, exec_count_fn=lambda: clock[0], **kw
        )

    def test_uses_refit_cadence_with_substrate_defaults(self):
        from fuzzer_tool.core.edge_matrix import DEFAULT_REFIT_INTERVAL, RefitCadence
        from fuzzer_tool.core.schedulers.pos_saliency import REFIT_INTERVAL

        a = _arm([])
        assert isinstance(a._cadence, RefitCadence)
        assert REFIT_INTERVAL == DEFAULT_REFIT_INTERVAL == 2000

    def test_driven_mode_never_fits_lazily_from_propose_record_or_warm(self):
        _, samples = _corpus()
        a = self._driven(samples, [0])
        for _ in range(300):
            assert a.propose(samples[0][0], 100) is None
            a.record(b"x", [0], Outcome.MISS, 1.0)
            assert a.warm() is False
        assert not a.ready  # only maybe_refit (the discovery hook) builds it

    def test_interval_is_counted_in_executions(self):
        _, samples = _corpus()
        clock = [0]
        a = self._driven(samples, clock, refit_interval=1000)
        assert a.maybe_refit() is True
        samples.append((b"\x07" * 100, {0, 1, 99}))  # corpus grew
        clock[0] = 999
        assert a.maybe_refit() is False  # interval not elapsed
        clock[0] = 1000
        assert a.maybe_refit() is True and a.stats()["fits"] == 2

    def test_cheap_precheck_does_not_walk_the_corpus(self):
        calls = []
        _, samples = _corpus()

        def spy():
            calls.append(1)
            return samples

        clock = [0]
        a = PositionSaliencyScheduler(RandPool(seed=1), spy, exec_count_fn=lambda: clock[0])
        a.maybe_refit()
        n = len(calls)
        for t in range(1, 50):
            clock[0] = t
            a.maybe_refit()
        assert len(calls) == n

    def test_too_few_seeds_does_not_stamp_the_clock(self):
        _, full = _corpus()
        live = full[: MIN_SEEDS - 1]
        clock = [0]
        a = self._driven(live, clock, refit_interval=5000)
        assert a.maybe_refit() is False
        assert "fewer than" in a.stats()["skip_reason"]
        live.extend(full[MIN_SEEDS - 1 : MIN_SEEDS + 5])  # first real discovery batch
        clock[0] = 1  # far inside the interval: a stamped clock would bar this
        assert a.maybe_refit() is True

    def test_unchanged_corpus_is_not_refit_even_when_due(self):
        _, samples = _corpus()
        clock = [0]
        a = self._driven(samples, clock, refit_interval=10)
        a.maybe_refit()
        clock[0] = 10_000
        assert a.maybe_refit() is False
        assert a.stats()["skip_reason"] == "sample unchanged" and a.stats()["fits"] == 1

    def test_failed_attempt_is_not_stamped(self):
        # all seeds hit the same single edge: no informative column, no fit
        seeds = [bytes([i]) * 8 for i in range(12)]
        live = [(s, {1}) for s in seeds]
        clock = [0]
        a = self._driven(live, clock, refit_interval=5000)
        assert a.maybe_refit() is False
        assert a.stats()["skip_reason"] == "no informative edges"
        live[:] = [(s, {1, 2 + (s[0] % 2)}) for s in seeds]
        clock[0] = 1
        assert a.maybe_refit() is True  # retried at once: nothing was stamped

    def test_maybe_refit_swallows_errors_refit_does_not(self):
        def boom():
            raise RuntimeError("x")

        a = PositionSaliencyScheduler(RandPool(seed=1), boom, exec_count_fn=lambda: 0)
        assert a.maybe_refit() is False and a.stats()["skip_reason"] == "refit raised"
        with pytest.raises(RuntimeError):
            a.refit()

    def test_standalone_retries_are_spaced_out(self):
        calls = []

        def spy():
            calls.append(1)
            return []

        a = PositionSaliencyScheduler(RandPool(seed=1), spy)
        for _ in range(450):
            a.propose(b"abcdefgh", 8)
        assert len(calls) <= 4  # not one corpus walk per proposal


class TestTrustGate:
    """Same preflight as the matrix arms: unstable edge ids -> abstain."""

    def _gated(self, trusted):
        _, samples = _corpus()
        flag = [trusted]
        a = PositionSaliencyScheduler(
            RandPool(seed=1), lambda: samples, trust_fn=lambda: flag[0], refit_interval=10
        )
        return samples, flag, a

    def test_untrusted_refuses_to_fit(self):
        _, _, a = self._gated(False)
        assert a.refit() is False
        assert a.stats()["skip_reason"] == "untrusted coverage ids" and not a.ready

    def test_model_fitted_while_trusted_abstains_once_distrusted(self):
        samples, flag, a = self._gated(True)
        assert a.refit()
        assert a.propose(samples[0][0], 100) is not None
        assert a.warm() is True and a.gradient_info(samples[0][0]) is not None
        flag[0] = False  # the stability probe found moving ids
        assert a.propose(samples[0][0], 100) is None
        assert a.warm() is False and a.gradient_info(samples[0][0]) is None
        flag[0] = True
        assert a.propose(samples[0][0], 100) is not None  # recovers; model was kept

    def test_broken_trust_fn_fails_open(self):
        _, samples = _corpus()

        def boom():
            raise RuntimeError("x")

        a = PositionSaliencyScheduler(RandPool(seed=1), lambda: samples, trust_fn=boom)
        assert a.refit() and a.warm()

    def test_force_overrides_the_gate_for_tools(self):
        _, _, a = self._gated(False)
        assert a.refit(force=True) is True
