"""--elo-stale-bench: stalest-first strategy bench and revive (soksolv port).

soksolv lane.go ringOver retires the engine with the most expansions since
its last improvement; lane_revive.go restarts a dead lane fresh. Here a
meta-scheduler strategy that played many rounds without a gain is benched
(out of select_strategy) and revived after a cool-down with a prior rating.
No RNG is drawn on any path tested here: one armed arm, or a cold arm,
returns without a Thompson draw.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker
from fuzzer_tool.core.elo_stale_bench import (
    CHECK_EVERY,
    COOLDOWN,
    STALE_FLOOR,
    STALE_MULT,
    StaleBench,
)
from fuzzer_tool.core.run_mode import Switch

A, B, C = "op_a", "op_b", "op_c"
SA, SB = "seed_a", "seed_b"

# Tiny knobs so tests stay short: floor 3, 2x median, cool-down 5, check every round.
KNOBS = {"floor": 3, "mult": 2, "cooldown": 5, "check_every": 1}


def _play(bench: StaleBench, name: str, score: float, n: int = 1) -> None:
    for _ in range(n):
        bench.record(name, score)


def _never_revive(name: str) -> None:
    raise AssertionError(f"unexpected revive of {name}")


class TestBench:
    def test_stalest_benched_above_floor_and_median(self):
        b = StaleBench(**KNOBS)
        _play(b, A, 0.0, 4)  # 4 > max(3, 2 * median 0)
        _play(b, B, 1.0)
        _play(b, C, 1.0)

        assert b.armed([A, B, C], _never_revive) == [B, C]
        assert b.benched() == {A}

    def test_below_floor_stays_armed(self):
        """Falsification: stale but not past the floor -> no bench."""
        b = StaleBench(**KNOBS)
        _play(b, A, 0.0, 3)  # == floor, not above

        assert b.armed([A, B], _never_revive) == [A, B]

    def test_below_median_multiple_stays_armed(self):
        """Falsification: everyone equally stale -> nobody is the outlier."""
        b = StaleBench(**KNOBS)
        for s in (A, B, C):
            _play(b, s, 0.0, 10)

        assert b.armed([A, B, C], _never_revive) == [A, B, C]

    def test_gain_resets_counter(self):
        b = StaleBench(**KNOBS)
        _play(b, A, 0.0, 10)
        _play(b, A, 0.5)

        assert b.armed([A, B], _never_revive) == [A, B]

    def test_one_bench_per_pass(self):
        """soksolv retires one engine per check, stalest first."""
        b = StaleBench(**KNOBS)
        _play(b, A, 0.0, 9)
        _play(b, B, 0.0, 8)
        rest = ["op_d", "op_e", "op_f"]
        for s in rest:
            _play(b, s, 1.0)
        pool = [A, B, *rest]

        assert b.armed(pool, _never_revive) == [B, *rest]
        _play(b, "op_d", 1.0)  # a pass needs a new round
        assert b.armed(pool, _never_revive) == rest

    def test_never_empties_pool(self):
        """Adversarial: two stale arms; the last armed one is never benched."""
        b = StaleBench(**{**KNOBS, "cooldown": 10_000})
        _play(b, A, 0.0, 50)
        _play(b, B, 0.0, 1)

        assert b.armed([A, B], _never_revive) == [B]
        _play(b, B, 0.0, 500)
        assert b.armed([A, B], _never_revive) == [B]

    def test_single_arm_pool_untouched(self):
        b = StaleBench(**KNOBS)
        _play(b, A, 0.0, 100)

        assert b.armed([A], _never_revive) == [A]

    def test_arenas_tick_and_check_separately(self):
        """Seed rounds neither age nor check the operator arena."""
        b = StaleBench(floor=3, mult=2, cooldown=5, check_every=4)
        _play(b, A, 0.0, 4)
        _play(b, B, 1.0)
        assert b.armed([A, B], _never_revive) == [B]  # first check is due at once
        _play(b, SA, 0.0, 50)

        assert b.armed([SA, SB], _never_revive) == [SB]
        assert b.benched() == {A, SA}


class TestRevive:
    def test_revived_after_cooldown_fresh(self):
        b = StaleBench(**KNOBS)
        _play(b, A, 0.0, 4)
        _play(b, B, 1.0)
        b.armed([A, B], _never_revive)
        revived: list[str] = []

        _play(b, B, 1.0, KNOBS["cooldown"] - 1)
        assert b.armed([A, B], revived.append) == [B]
        _play(b, B, 1.0)
        assert b.armed([A, B], revived.append) == [A, B]
        assert revived == [A]
        assert b.stale(A) == 0

    def test_absent_from_pool_not_revived(self):
        """Edge: only a strategy offered this pick can come back."""
        b = StaleBench(**KNOBS)
        _play(b, A, 0.0, 4)
        _play(b, B, 1.0)
        b.armed([A, B], _never_revive)
        _play(b, B, 1.0, 50)

        assert b.armed([B, C], _never_revive) == [B, C]
        assert b.benched() == {A}


class TestPersist:
    def test_round_trip(self):
        b = StaleBench(**KNOBS)
        _play(b, A, 0.0, 4)
        _play(b, B, 1.0)
        b.armed([A, B], _never_revive)

        c = StaleBench(**KNOBS)
        c.from_dict(b.to_dict())
        assert c.benched() == {A}
        assert c.to_dict() == b.to_dict()

    @pytest.mark.parametrize("bad", [None, 3, "x", {"stale": 5}, {"benched": {"op_a": "z"}}])
    def test_malformed_starts_fresh(self, bad):
        """Adversarial: a corrupt section never raises."""
        c = StaleBench(**KNOBS)
        c.from_dict(bad)

        assert c.benched() == set()


def test_constants_documented_shape():
    assert STALE_FLOOR > 0 and STALE_MULT > 1 and COOLDOWN > STALE_FLOOR
    assert 0 < CHECK_EVERY < STALE_FLOOR


# --- Tracker integration -------------------------------------------------------


class _FavourFirst:
    """Thompson draw that puts the first rated arm far ahead."""

    def normal_array(self, n):
        import numpy as np

        return np.array([100.0] + [-100.0] * (n - 1))


def _tracker() -> BayesianEloTracker:
    elo = BayesianEloTracker(min_matches=1)
    elo.enable_stale_bench(StaleBench(**KNOBS))
    return elo


class TestTracker:
    def test_benched_never_selected(self):
        elo = _tracker()
        for _ in range(4):
            elo.record_strategy_matches(A, [B], 0.0)
        elo.record_strategy_matches(B, [A], 1.0)

        assert elo.select_strategy([A, B]) == B

    def test_revive_resets_rating_to_prior(self):
        elo = _tracker()
        elo.record_strategy_matches(A, [B], 0.0)
        elo.record_strategy_matches(B, [A], 1.0)
        for _ in range(4):
            elo.record_strategy_matches(A, [B], 0.0)
        elo.select_strategy([A, B])
        assert elo._strategy_mu[A] != elo.initial_mu
        for _ in range(KNOBS["cooldown"]):
            elo.record_strategy_matches(B, [A], 1.0)

        # A comes back cold (match count 0), so the cold-arm rule picks it.
        assert elo.select_strategy([A, B]) == A
        assert elo._strategy_mu[A] == elo.initial_mu
        assert elo._strategy_sigma_sq[A] == elo.initial_sigma**2

    def test_off_is_plain_tracker(self):
        """Falsification: no bench -> the stale arm wins a draw that favours it."""
        elo = BayesianEloTracker(min_matches=1)
        elo.record_strategy_matches(A, [B], 0.0)
        elo.record_strategy_matches(B, [A], 1.0)
        for _ in range(50):
            elo.record_strategy_matches(A, [B], 0.0)
        elo._rng = _FavourFirst()

        assert "select_strategy" not in vars(elo)
        assert "stale_bench" not in elo.to_dict()
        assert elo.select_strategy([A, B]) == A

    def test_on_same_draw_skips_benched(self):
        """Control for the test above: identical history, bench on -> B."""
        elo = _tracker()
        elo.record_strategy_matches(A, [B], 0.0)
        elo.record_strategy_matches(B, [A], 1.0)
        for _ in range(50):
            elo.record_strategy_matches(A, [B], 0.0)
        elo._rng = _FavourFirst()

        assert elo.select_strategy([A, B]) == B

    def test_state_persists_with_elo_section(self):
        elo = _tracker()
        for _ in range(4):
            elo.record_strategy_matches(A, [B], 0.0)
        elo.record_strategy_matches(B, [A], 1.0)
        elo.select_strategy([A, B])
        data = elo.to_dict()

        # from_dict before enabling (the Fuzzer order) still restores it.
        fresh = BayesianEloTracker(min_matches=1)
        fresh.from_dict(data)
        bench = StaleBench(**KNOBS)
        fresh.enable_stale_bench(bench)
        assert bench.benched() == {A}

        # And after enabling.
        other = _tracker()
        other.from_dict(data)
        assert other._stale_bench.benched() == {A}


# --- Fuzzer + CLI wiring ----------------------------------------------------------


def _real_fuzzer(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir(parents=True, exist_ok=True)
    crashes.mkdir(parents=True, exist_ok=True)
    return Fuzzer(
        target="targets/test_target",
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        **kw,
    )


class TestFuzzerWiring:
    def test_on_with_elo(self, tmp_path):
        f = _real_fuzzer(tmp_path, elo=True, elo_stale_bench=Switch.ON)

        assert isinstance(f._elo._stale_bench, StaleBench)

    def test_off_by_default(self, tmp_path):
        f = _real_fuzzer(tmp_path, elo=True)

        assert f._elo._stale_bench is None

    def test_on_without_elo_is_inert(self, tmp_path):
        """Edge: the flag without --elo builds nothing and does not raise."""
        f = _real_fuzzer(tmp_path, elo_stale_bench=Switch.ON)

        assert not f._elo


def test_cli_flag_reaches_fuzzer_not_hail_mary():
    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
    ]
    dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))

    assert all("elo_stale_bench" in {k.arg for k in c.keywords} for c in calls)
    assert "elo_stale_bench" in dests
    assert "elo_stale_bench" not in commands._HAIL_MARY_FLAGS
