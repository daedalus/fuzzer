"""Tests for the Kruskal-count seed strategy (core/schedulers/kruskal_count.py).

Every coupled/non-coupled fixture is a hand-built jump table over 8 bytes,
with walkers starting at 0, 2, 4, 6. Byte ``b`` at position ``p`` jumps to
``(p + max(1, b)) % 8``, so a fixture is designed by picking the target of
each position and writing ``(target - p) % 8`` (``8`` for a self-loop).

No retry-until-hit loops (Hard Rule 39): draws are scripted or come from a
fixed ``RandPool`` seed.
"""

import math
import time
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.kruskal_count import (
    MAX_STEPS,
    STATE_VERSION,
    WALKER_COUNT,
    KruskalCountSeedStrategy,
    _score,
    batch_scores,
    boundaries,
    jump_position,
    trace,
    walker_starts,
)
from tests.support.scripted_rng import ScriptedRng

# Every position funnels to 0 in one step; 0 self-loops. All 6 pairs couple at step 1.
FUNNEL_1 = bytes([8, 7, 6, 5, 4, 3, 2, 1])
# Starts step to 1/3/5/7 (distinct), then all to 0; 0 <-> 1 afterwards. Couple at step 2.
FUNNEL_2 = bytes([1, 7, 1, 5, 1, 3, 1, 1])
# Every position funnels to 5; 5 self-loops.
FUNNEL_TO_5 = bytes([5, 4, 3, 2, 1, 8, 7, 6])
# Rotation by 2: a permutation, walkers never meet.
ROTATION = bytes([2] * 8)
# Pair (2, 3) meets on 7 at step 1; pair (0, 1) meets on 5 at step 2.
TWO_CYCLES = bytes([1, 4, 1, 2, 3, 8, 1, 8])
# 2 -> 0 (self-loop), 4 and 6 circle 4->5->6->7->4 two apart. Only pair (0, 2) couples.
PARTIAL = bytes([8, 1, 6, 1, 1, 1, 1, 5])


def _profile(signature=None, markers=(), magic=()):
    return SimpleNamespace(
        format_signature=signature,
        boundary_markers=list(markers),
        magic_bytes=list(magic),
    )


NO_PROFILE = _profile()


def _strategy(rng=None, profile=NO_PROFILE):
    return KruskalCountSeedStrategy(rng or RandPool(seed=1), profile)


class TestJump:
    def test_byte_fallback_is_exact_and_wraps(self):
        seed = b"\x03\x00\x05\x01"
        assert jump_position(seed, 0, None) == 3
        assert jump_position(seed, 2, None) == 3  # (2 + 5) % 4
        assert jump_position(seed, 3, None) == 0  # wraps

    def test_zero_byte_jumps_one_and_cannot_self_loop(self):
        seed = bytes(4)
        for p in range(4):
            assert jump_position(seed, p, None) == (p + 1) % 4

    def test_format_boundary_wins_over_byte_fallback(self):
        seed = b"a,b,c"
        bounds = boundaries(seed, _profile("csv", [b","]))
        assert bounds == [1, 3]
        assert jump_position(seed, 0, bounds) == 1
        assert jump_position(seed, 1, bounds) == 3  # strictly after

    def test_past_last_boundary_falls_back_to_bytes(self):
        seed = b"a,b,c"
        bounds = boundaries(seed, _profile("csv", [b","]))
        assert jump_position(seed, 3, bounds) == (3 + ord(",")) % 5

    def test_multibyte_marker_lands_on_its_start(self):
        seed = b"xxGET yyGET "
        assert boundaries(seed, _profile("http", [b"GET "])) == [2, 8]

    def test_empty_markers_are_skipped(self):
        assert boundaries(b"abc", _profile("csv", [b"", b""])) is None

    def test_markers_ignored_without_signature(self):
        assert boundaries(b"a,b", _profile(None, [b","])) is None

    def test_no_marker_occurrence_falls_back(self):
        assert boundaries(b"abc", _profile("csv", [b","])) is None


class TestStarts:
    def test_evenly_spaced_when_long_enough(self):
        assert walker_starts(8) == [0, 2, 4, 6]

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 7, 4096])
    def test_starts_are_distinct_and_bounded(self, n):
        starts = walker_starts(n)
        assert len(starts) == min(n, WALKER_COUNT)
        assert len(set(starts)) == len(starts)
        assert all(0 <= s < n for s in starts)


class TestScore:
    def test_empty_and_one_byte_seeds_score_zero(self):
        s = _strategy()
        assert s.score_seed(b"") == 0.0
        assert s.score_seed(b"\x00") == 0.0

    def test_known_coupled_trace_records_step(self):
        t = trace(FUNNEL_1, None)
        assert set(t.couple_steps.values()) == {1}
        assert len(t.couple_steps) == 6
        assert trace(FUNNEL_2, None).couple_steps[(0, 1)] == 2

    def test_non_coupled_trace_scores_zero(self):
        assert trace(ROTATION, None).couple_steps == {}
        assert _strategy().score_seed(ROTATION) == 0.0

    def test_exact_scores(self):
        s = _strategy()
        assert s.score_seed(FUNNEL_1) == pytest.approx(1 - 1 / MAX_STEPS)
        assert s.score_seed(PARTIAL) == pytest.approx((1 / 6) * (1 - 1 / MAX_STEPS))

    def test_faster_coupling_scores_higher_same_pair_count(self):
        s = _strategy()
        assert len(trace(FUNNEL_1, None).couple_steps) == len(trace(FUNNEL_2, None).couple_steps)
        assert s.score_seed(FUNNEL_1) > s.score_seed(FUNNEL_2)

    def test_adversarial_constant_seeds_are_bounded(self):
        """All-zero rotates, 0xFF over 255 bytes self-loops: neither couples."""
        s = _strategy()
        t0 = time.perf_counter()
        assert s.score_seed(bytes(4096)) == 0.0
        assert s.score_seed(b"\xff" * 255) == 0.0
        assert time.perf_counter() - t0 < 1.0

    def test_scores_are_in_unit_interval(self):
        s = _strategy()
        rng = RandPool(seed=3)
        for _ in range(50):
            v = s.score_seed(rng.randbytes(rng.randint(0, 300)))
            assert 0.0 <= v < 1.0


class TestGenerate:
    def test_uses_converged_trajectory_and_donor(self):
        donor = bytes(range(0x10, 0x18))
        out = _strategy(ScriptedRng()).generate(FUNNEL_2, [FUNNEL_2, donor])
        # Trajectory after coupling is {0, 1}: only those bytes come from the donor.
        assert out == bytes([0x10, 0x11]) + FUNNEL_2[2:]

    def test_earliest_coupled_pair_owns_the_mask(self):
        t = trace(TWO_CYCLES, None)
        assert t.couple_steps == {(2, 3): 1, (0, 1): 2}
        out = _strategy(ScriptedRng()).generate(TWO_CYCLES, [b"\xaa" * 8])
        assert out == TWO_CYCLES[:7] + b"\xaa"

    def test_length_mismatch_uses_modulo(self):
        out = _strategy(ScriptedRng()).generate(FUNNEL_TO_5, [b"\xab\xcd\xef\x01"])
        assert out == FUNNEL_TO_5[:5] + b"\xcd" + FUNNEL_TO_5[6:]  # 5 % 4 == 1

    def test_magic_prefix_is_preserved(self):
        s = _strategy(ScriptedRng(), _profile(magic=[b"\x01"]))
        out = s.generate(FUNNEL_2, [b"\xaa" * 8])
        assert out == bytes([0x01, 0xAA]) + FUNNEL_2[2:]

    def test_no_coupling_returns_none(self):
        assert _strategy().generate(ROTATION, [b"\xaa" * 8]) is None

    def test_no_donor_returns_none(self):
        s = _strategy()
        assert s.generate(FUNNEL_1, []) is None
        assert s.generate(FUNNEL_1, [FUNNEL_1]) is None
        assert s.generate(FUNNEL_1, [b""]) is None

    def test_mask_fully_inside_prefix_returns_none(self):
        s = _strategy(profile=_profile(magic=[FUNNEL_1[:1]]))
        assert s.generate(FUNNEL_1, [b"\xaa" * 8]) is None

    def test_identical_recombination_changes_one_trajectory_byte(self):
        """Donor matches the anchor on the mask: one scripted XOR, no retry."""
        donor = bytes([8, 0, 0, 0, 0, 0, 0, 0])
        rng = ScriptedRng(randints=[0x55], choice_idxs=[0])
        out = _strategy(rng).generate(FUNNEL_1, [donor])
        assert out == bytes([8 ^ 0x55]) + FUNNEL_1[1:]

    def test_donor_weighted_by_score(self):
        """A zero-score donor carries only MIN_WEIGHT against a coupled one."""
        s = _strategy(RandPool(seed=11))
        picks = [s.generate(FUNNEL_TO_5, [ROTATION, FUNNEL_1]) for _ in range(20)]
        assert all(p[5] == FUNNEL_1[5] for p in picks)

    def test_generated_counter(self):
        s = _strategy(ScriptedRng())
        s.generate(FUNNEL_2, [b"\xaa" * 8])
        assert s.stats()["generated"] == 1


class TestSelfControl:
    def test_same_input_twice_is_identical(self):
        """Hard Rule 46: the reference against itself before any comparison."""
        rng = RandPool(seed=5)
        corpus = [rng.randbytes(rng.randint(2, 64)) for _ in range(30)] + [FUNNEL_1, FUNNEL_2]
        a, b = _strategy(RandPool(seed=7)), _strategy(RandPool(seed=7))
        assert a.scores(corpus) == b.scores(corpus)
        assert [a.generate(x, corpus) for x in corpus] == [b.generate(x, corpus) for x in corpus]


class TestBatch:
    def _corpus(self):
        rng = RandPool(seed=13)
        seeds = [rng.randbytes(rng.randint(WALKER_COUNT, 700)) for _ in range(300)]
        return seeds + [
            FUNNEL_1,
            FUNNEL_2,
            PARTIAL,
            ROTATION,
            TWO_CYCLES,
            bytes(4096),
            b"\xff" * 255,
        ]

    def _scalar(self, seeds):
        out = []
        for s in seeds:
            t = trace(s, None)
            out.append((_score(t, len(walker_starts(len(s)))), len(t.couple_steps)))
        return out

    def test_vectorized_matches_scalar_reference(self):
        seeds = self._corpus()
        # Control first (Hard Rule 46): each path against a second run of itself.
        assert self._scalar(seeds) == self._scalar(seeds)
        assert batch_scores(seeds) == batch_scores(seeds)
        assert batch_scores(seeds) == self._scalar(seeds)

    def test_strategy_counters_match_scalar_path(self):
        seeds = self._corpus()
        fast = _strategy()
        fast.scores(seeds)
        slow = _strategy()
        for s in seeds:
            slow.score_seed(s)
        assert fast.stats() == slow.stats()

    def test_format_hints_bypass_batch(self):
        prof = _profile("csv", [b","])
        s = _strategy(profile=prof)
        seed = b"a,b,c,d,e,f"
        assert s.scores([seed]) == [_strategy(profile=prof).score_seed(seed)]


class TestSelect:
    def test_empty_returns_none(self):
        assert _strategy().select([]) is None

    def test_scores_cache_matches_fresh(self):
        s = _strategy()
        corpus = [FUNNEL_1, PARTIAL, ROTATION]
        first = s.scores(corpus)
        assert s.scores(corpus) == first == [_strategy().score_seed(x) for x in corpus]

    def test_cache_drops_seeds_left_corpus(self):
        s = _strategy()
        s.scores([FUNNEL_1, PARTIAL])
        s.scores([PARTIAL])
        assert s.stats()["cached"] == 1

    def test_mask_cache_drops_seeds_left_corpus(self):
        s = _strategy(ScriptedRng())
        s.generate(FUNNEL_2, [b"\xaa" * 8])
        s.scores([FUNNEL_1])
        assert FUNNEL_2 not in s._masks

    def test_select_prefers_coupled(self):
        s = _strategy(RandPool(seed=2))
        assert all(s.select([ROTATION, FUNNEL_1]) == FUNNEL_1 for _ in range(20))


class TestState:
    def test_round_trip(self):
        s = _strategy()
        s.scores([FUNNEL_1, PARTIAL, ROTATION])
        s.generate(FUNNEL_2, [b"\xaa" * 8])
        data = s.to_dict()
        assert data["version"] == STATE_VERSION
        r = KruskalCountSeedStrategy.from_dict(data, RandPool(seed=1), NO_PROFILE)
        assert r.stats() | {"cached": 0} == s.stats() | {"cached": 0}

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "x",
            {},
            {"version": 99, "scored": 1},
            {"version": STATE_VERSION, "scored": "1"},
            {"version": STATE_VERSION, "scored": -1},
            {"version": STATE_VERSION, "scored": True},
            {"version": STATE_VERSION, "score_sum": math.nan},
            {"version": STATE_VERSION, "score_sum": [1.0]},
            {"version": STATE_VERSION, "generated": object()},
        ],
    )
    def test_malformed_state_is_ignored_whole(self, bad):
        r = KruskalCountSeedStrategy.from_dict(bad, RandPool(seed=1), NO_PROFILE)
        assert r.stats() == _strategy().stats()


class TestSeedPickerWiring:
    """Elo pool eligibility, handler dispatch, and the non-Elo fallback."""

    def _fuzzer(self, strategy=True, corpus=(FUNNEL_2, ROTATION)):
        f = SimpleNamespace(
            corpus=list(corpus),
            seed_meta={},
            _use_elo=True,
            _elo=SimpleNamespace(select_strategy=lambda keys, **_: "seed_kruskal_count"),
            _seed_strategy=None,
            _seed_strategy_pool=[],
            _seed_strategies_used=set(),
            _stall_recovery_active=False,
            _rng=RandPool(seed=4),
            ga=None,
            qea=None,
            markov_generate=False,
            markov_trained=False,
            _use_bayesian=False,
            _use_boltzmann=False,
            _use_ecofuzz=False,
            _profile=NO_PROFILE,
        )
        f._kruskal_count = KruskalCountSeedStrategy(f._rng, NO_PROFILE) if strategy else None
        return f

    def _picker(self, f):
        from fuzzer_tool.services.seed_picker import SeedPicker

        sp = SeedPicker.__new__(SeedPicker)
        sp.f = f
        sp._rng = f._rng
        return sp

    def test_eligible_and_dispatched_under_elo(self):
        f = self._fuzzer()
        picked = self._picker(f)._pick_seed_elo()
        assert "kruskal_count" in f._seed_strategy_pool
        assert f._seed_strategy == "kruskal_count"
        # Anchor FUNNEL_2 (ROTATION scores 0) recombined with the only donor.
        assert picked == bytes([ROTATION[0], ROTATION[1]]) + FUNNEL_2[2:]

    def test_not_eligible_when_disabled_or_empty(self):
        for f in (self._fuzzer(strategy=False), self._fuzzer(corpus=())):
            f._use_boltzmann = True  # two arms, so Elo is consulted
            f._elo = SimpleNamespace(select_strategy=lambda keys, **_: "seed_unhandled")
            self._picker(f)._pick_seed_elo()
            assert "kruskal_count" not in f._seed_strategy_pool

    def test_uncoupled_corpus_returns_anchor(self):
        f = self._fuzzer(corpus=(ROTATION,))
        assert self._picker(f)._pick_kruskal_count_seed() == ROTATION

    def test_empty_corpus_falls_back_to_format_seed(self, monkeypatch):
        f = self._fuzzer(corpus=())
        sp = self._picker(f)
        monkeypatch.setattr(sp, "_format_aware_seed", lambda: b"fmt")
        assert sp._pick_kruskal_count_seed() == b"fmt"

    def test_non_elo_fallback_dispatches_before_bayesian(self, monkeypatch):
        f = self._fuzzer()
        f._use_elo = False
        f._use_bayesian = True
        f._seed_quality = {"x": 1}
        sp = self._picker(f)
        monkeypatch.setattr(sp, "_update_temperature", lambda: None)
        monkeypatch.setattr(sp, "_pick_bayesian_seed", lambda: pytest.fail("bayesian won"))
        assert sp.pick_seed() == bytes([ROTATION[0], ROTATION[1]]) + FUNNEL_2[2:]


class TestFuzzerWiring:
    def test_registered_as_seed_strategy(self):
        from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES

        assert "kruskal_count" in _SEED_STRATEGY_NAMES

    def test_constructor_flag_is_appended_last(self):
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        params = list(inspect.signature(Fuzzer.__init__).parameters)
        assert params[-1] == "kruskal_count"
        assert inspect.signature(Fuzzer.__init__).parameters["kruskal_count"].default is False

    def test_cli_passes_flag_to_both_constructions(self):
        import ast
        import inspect

        from fuzzer_tool.cli import commands
        from fuzzer_tool.services import parallel

        def kws(fn, callee):
            tree = ast.parse(inspect.getsource(fn))
            calls = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == callee
            ]
            return [{k.arg for k in c.keywords} for c in calls]

        assert all("kruskal_count" in k for k in kws(commands.cmd_fuzz, "Fuzzer"))
        assert all("kruskal_count" in k for k in kws(commands.cmd_fuzz, "run_parallel"))
        assert all("kruskal_count" in k for k in kws(parallel._worker_main, "Fuzzer"))
        assert "kruskal_count" in commands._HAIL_MARY_FLAGS

    def test_parser_declares_flag(self):
        import ast
        import inspect

        from fuzzer_tool.cli import commands
        from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

        assert "kruskal_count" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
