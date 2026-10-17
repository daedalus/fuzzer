"""FrontierSeedScheduler: pick seeds that execute rarely reached open frontier conditions."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.schedulers.seed_frontier import DEFAULT_P_FRONTIER, FrontierSeedScheduler
from tests.support.scripted_rng import ScriptedRng

A, B, C = ("f.c", 1, 1), ("f.c", 2, 1), ("f.c", 3, 1)


class _Oracle:
    """Scripted reach: key -> open conditions; cond -> reachers."""

    def __init__(self, reach, reachers):
        self._reach = reach
        self._reachers = reachers
        self.epoch = 0

    def rare_reach(self, keys):
        out = []
        for k in keys:
            reach = self._reach.get(k) or ()
            out.append(sum(1.0 / self._reachers[c] for c in reach))
        return out


def _sched(oracle, *randoms, p=1.0):
    return FrontierSeedScheduler(ScriptedRng(randoms=list(randoms)), oracle, p_frontier=p)


def test_weight_is_rare_reach():
    # s1: A (1 reacher) -> 1.0; s2: B (2 reachers) + C (4) -> 0.75; s3 unmeasured -> 0.
    o = _Oracle({"s1": {A}, "s2": {B, C}}, {A: 1, B: 2, C: 4})
    keys = ["s1", "s2", "s3"]
    assert _sched(o).weights(keys) == [1.0, 0.75, 0.0]


def test_roulette_follows_weights():
    o = _Oracle({"s1": {A}, "s2": {B, C}}, {A: 1, B: 2, C: 4})
    keys = ["s1", "s2", "s3"]
    total = 1.75
    assert _sched(o, 0.5 / total).select_seed(keys) == "s1"
    assert _sched(o, 1.2 / total).select_seed(keys) == "s2"


def test_mix_gate_hands_off_to_default_picker():
    # p=0.5: a gate draw >= p declines (default picker, any seed); < p runs the roulette.
    o = _Oracle({"s1": {A}}, {A: 1})
    keys = ["s0", "s1"]
    assert _sched(o, 0.5, p=0.5).select_seed(keys) == ""
    assert _sched(o, 0.49, 0.5, p=0.5).select_seed(keys) == "s1"
    assert DEFAULT_P_FRONTIER == 0.5


def test_adversarial_p_bounds():
    o = _Oracle({"s1": {A}}, {A: 1})
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            FrontierSeedScheduler(ScriptedRng(), o, p_frontier=bad)
    # p=0 never weights, never draws a roulette value.
    assert _sched(o, 0.0, p=0.0).select_seed(["s1"]) == ""


def test_falsification_unmeasured_corpus_declines():
    # No seed reaches an open condition: decline (empty key), never a uniform guess.
    o = _Oracle({}, {})
    assert _sched(o).select_seed(["s1", "s2"]) == ""
    assert _sched(o).select_seed([]) == ""


def test_adversarial_last_draw_never_overruns():
    o = _Oracle({"s1": {A}}, {A: 1})
    assert _sched(o, 0.999999999).select_seed(["s0", "s1"]) == "s1"


def test_weights_cached_until_oracle_changes():
    o = _Oracle({"s1": {A}}, {A: 1})
    s = _sched(o)
    assert s.weights(["s1"]) == [1.0]
    o._reachers[A] = 2
    assert s.weights(["s1"]) == [1.0]  # same epoch, same keys
    o.epoch += 1
    assert s.weights(["s1"]) == [0.5]


def test_requires_rng():
    with pytest.raises(ValueError):
        FrontierSeedScheduler(None, _Oracle({}, {}))


def test_registered_as_seed_arm():
    from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES
    from fuzzer_tool.services.seed_picker import _OS_SEED_ARMS

    assert "frontier" in _SEED_STRATEGY_NAMES
    assert _OS_SEED_ARMS["frontier"] == "_seed_frontier"


# ── wiring ──────────────────────────────────────────────────────────────────


def _make_fuzzer(**kw):
    import tempfile
    from unittest.mock import patch

    from fuzzer_tool.services.fuzzer import Fuzzer

    tmpdir = tempfile.mkdtemp(prefix="fuzz_frontier_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            **kw,
        )


def test_off_by_default():
    f = _make_fuzzer()
    assert f._seed_frontier is None and f._frontier_oracle is None


def test_frontier_bin_builds_arm_and_oracle(tmp_path):
    from fuzzer_tool.adapters.frontier_cov import FrontierOracle

    f = _make_fuzzer(frontier_bin=str(tmp_path / "x_cov"))
    try:
        assert isinstance(f._frontier_oracle, FrontierOracle)
        assert isinstance(f._seed_frontier, FrontierSeedScheduler)
        assert f._seed_frontier in f._seed_os_arms
    finally:
        f._frontier_oracle.close()


def test_admitted_seed_is_submitted(tmp_path):
    f = _make_fuzzer(frontier_bin=str(tmp_path / "x_cov"))
    seen = []
    f._frontier_oracle.close()
    f._frontier_oracle = type("O", (), {"submit": lambda self, k, d: seen.append((k, d))})()
    f._frontier_observe(b"seed")
    assert seen == [(f._seed_key(b"seed"), b"seed")]


def test_cli_flag_wired():
    import ast
    import inspect

    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
    ]
    assert calls and all("frontier_bin" in {k.arg for k in c.keywords} for c in calls)
    assert "frontier_bin" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
    assert "frontier_bin" not in commands._HAIL_MARY_FLAGS


def test_stats_field():
    import types

    from fuzzer_tool.services.stats import _frontier_str

    class _O:
        def stats(self):
            return {"seeds": 5, "pending": 1, "dropped": 0, "frontier": 7, "closed": 2}

    assert _frontier_str(types.SimpleNamespace(_frontier_oracle=None)) == ""
    assert _frontier_str(types.SimpleNamespace(_frontier_oracle=_O())) == (
        " | frontier: open=7 closed=2 seeds=5 queue=1"
    )


def test_frontier_p_reaches_arm(tmp_path):
    f = _make_fuzzer(frontier_bin=str(tmp_path / "x_cov"), frontier_p=0.3)
    try:
        assert f._seed_frontier._p == 0.3
    finally:
        f._frontier_oracle.close()
