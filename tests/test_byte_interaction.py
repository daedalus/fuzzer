"""Second-order byte interaction (core/analyzers/analyzer_interaction.py).

Targets here are fake ``exec_fn`` closures whose edges depend only on WHICH
bytes differ from the seed, never on the drawn values, so every assertion is
deterministic regardless of the RNG stream (Hard Rule 39).
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from fuzzer_tool.cli import commands
from fuzzer_tool.core.analyzer_registry import REGISTRY as ANALYZERS
from fuzzer_tool.core.analyzer_registry import (
    _activate_byte_interaction,
    _deactivate_byte_interaction,
)
from fuzzer_tool.core.analyzers.analyzer_interaction import (
    INTERACTION_MIN,
    ByteInteractionTracker,
    mixed_difference,
)
from fuzzer_tool.core.operator_registry import _AVAILABLE, REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

SEED = bytes(range(16))
BASE = {1, 2, 3}


def _target(rules):
    """Fake target: *rules* maps an edge to a predicate over the changed-offset set."""
    calls = []

    def exec_fn(candidate: bytes) -> set[int]:
        changed = {i for i, (a, b) in enumerate(zip(SEED, candidate, strict=False)) if a != b}
        calls.append(changed)
        return BASE | {e for e, pred in rules.items() if pred(changed)}

    return exec_fn, calls


def _scores(*hot):
    """First-order scores: 0.5 on *hot* offsets, 0 elsewhere."""
    return [0.5 if i in hot else 0.0 for i in range(len(SEED))]


def _tracker(top_k=8):
    return ByteInteractionTracker(top_k=top_k, rng=RandPool(seed=7))


# -- mixed_difference -------------------------------------------------------


def test_mixed_difference_zero_when_additive():
    # Each byte contributes its own edge: f(ij) = f(i) + f(j) - f(0).
    assert mixed_difference(BASE, BASE | {10}, BASE | {11}, BASE | {10, 11}) == 0.0


def test_mixed_difference_and_gate():
    # Edge 99 needs both bytes: one of 6 union edges carries |+1|.
    got = mixed_difference(BASE, BASE | {10}, BASE | {11}, BASE | {10, 11, 99})
    assert got == 1 / len(BASE | {10, 11, 99})


def test_mixed_difference_empty_sets():
    assert mixed_difference(set(), set(), set(), set()) == 0.0


# -- tracker ----------------------------------------------------------------


def test_falsify_independent_bytes_yield_no_pairs():
    """Falsification: purely additive bytes must not be reported as coupled."""
    exec_fn, _ = _target({10: lambda c: 3 in c, 11: lambda c: 7 in c, 12: lambda c: 9 in c})
    t = _tracker()

    assert t.analyze_seed(SEED, BASE, exec_fn, _scores(3, 7, 9)) == []
    assert t.analyzed(SEED)
    assert not t.has_data(SEED)


def test_adversarial_and_coupled_pair_found():
    """Adversarial: an edge gated on BOTH bytes is invisible to first order."""
    rules = {10: lambda c: 5 in c, 11: lambda c: 9 in c, 99: lambda c: {5, 9} <= c}
    exec_fn, _ = _target(rules)
    t = _tracker()

    assert t.analyze_seed(SEED, BASE, exec_fn, _scores(5, 9, 12)) == [(5, 9)]
    assert t.pairs(SEED) == [(5, 9)]
    assert t.has_data(SEED)


def test_adversarial_xor_cancellation_found():
    # Edge fires when exactly one byte changes; changing both cancels it.
    exec_fn, _ = _target({50: lambda c: len(c & {2, 4}) == 1})
    t = _tracker()

    assert t.analyze_seed(SEED, BASE, exec_fn, _scores(2, 4)) == [(2, 4)]


def test_pairs_ranked_by_strength():
    rules = {
        90: lambda c: {1, 2} <= c,
        91: lambda c: {6, 8} <= c,
        92: lambda c: {6, 8} <= c,
    }
    exec_fn, _ = _target(rules)
    t = _tracker()

    assert t.analyze_seed(SEED, BASE, exec_fn, _scores(1, 2, 6, 8)) == [(6, 8), (1, 2)]


def test_exec_count_bounded_by_cost():
    exec_fn, calls = _target({})
    t = _tracker(top_k=4)

    t.analyze_seed(SEED, BASE, exec_fn, _scores(*range(16)))

    assert len(calls) == t.cost() == 4 + 4 * 3 // 2
    # Singles first (one changed byte each), then pairs (two each).
    assert all(len(c) in (1, 2) for c in calls)


def test_top_k_picks_highest_first_order_offsets():
    exec_fn, calls = _target({})
    scores = [0.0] * 16
    scores[11], scores[3], scores[14] = 0.9, 0.8, 0.1
    t = _tracker(top_k=2)

    t.analyze_seed(SEED, BASE, exec_fn, scores)

    assert set().union(*calls) == {3, 11}


def test_fewer_than_two_hot_offsets_spends_nothing():
    exec_fn, calls = _target({})
    t = _tracker()

    assert t.analyze_seed(SEED, BASE, exec_fn, _scores(4)) == []
    assert calls == []
    assert t.analyzed(SEED)


def test_empty_seed_and_short_scores():
    exec_fn, calls = _target({})
    t = _tracker()

    assert t.analyze_seed(b"", BASE, exec_fn, []) == []
    # Scores longer than the seed: offsets past the end are ignored.
    assert t.analyze_seed(SEED[:2], BASE, exec_fn, _scores(0, 1, 9)) == []
    assert all(max(c) < 2 for c in calls)


def test_reanalysis_is_cached():
    exec_fn, calls = _target({99: lambda c: {5, 9} <= c})
    t = _tracker()

    t.analyze_seed(SEED, BASE, exec_fn, _scores(5, 9))
    spent = len(calls)
    t.analyze_seed(SEED, BASE, exec_fn, _scores(5, 9))

    assert len(calls) == spent


def test_memory_bounded_by_max_seeds():
    exec_fn, _ = _target({})
    t = ByteInteractionTracker(max_seeds=2, rng=RandPool(seed=1))

    for k in range(5):
        t.analyze_seed(bytes([k]) + SEED, BASE, exec_fn, [])

    assert t.tracked() == 2
    assert t.analyzed(bytes([4]) + SEED)
    assert not t.analyzed(bytes([0]) + SEED)


def test_threshold_filters_noise():
    # One flaky edge in a large trace stays below INTERACTION_MIN.
    big = set(range(1000, 1000 + int(2 / INTERACTION_MIN)))
    got = mixed_difference(big, big, big, big | {7})
    assert got < INTERACTION_MIN


# -- operator ---------------------------------------------------------------


def _engine(pairs, rng):
    tracker = SimpleNamespace(pairs=lambda _d: pairs)
    eng = SimpleNamespace(
        f=SimpleNamespace(_byte_interaction=tracker),
        ctx=SimpleNamespace(_rng=rng),
        declined=[],
    )
    eng._op_declined = lambda op, _buf: eng.declined.append(op)
    return eng


def test_op_coupled_pair_writes_both_bytes():
    buf = bytearray(SEED)
    eng = _engine([(2, 9), (4, 12)], ScriptedRng(choice_idxs=[1], randints=[0xAA, 0xBB]))

    OperatorEngine._op_coupled_pair(eng, buf, 0, SEED)

    expect = bytearray(SEED)
    expect[4], expect[12] = 0xAA, 0xBB
    assert buf == expect


def test_op_coupled_pair_skips_pairs_past_buffer():
    buf = bytearray(SEED[:10])
    # Only (2, 9) fits; choice index 0 must land on it, not on (4, 12).
    eng = _engine([(4, 12), (2, 9)], ScriptedRng(choice_idxs=[0], randints=[1, 2]))

    OperatorEngine._op_coupled_pair(eng, buf, 0, SEED)

    assert (buf[2], buf[9]) == (1, 2)


def test_op_coupled_pair_declines_without_pairs():
    buf = bytearray(SEED[:3])
    eng = _engine([(4, 12)], ScriptedRng())

    OperatorEngine._op_coupled_pair(eng, buf, 0, SEED)

    assert eng.declined == ["coupled_pair"]
    assert buf == bytearray(SEED[:3])


def test_op_coupled_pair_registered_and_gated():
    assert REGISTRY.category_of("coupled_pair") == "adaptive"
    gate = _AVAILABLE["coupled_pair"]

    on = SimpleNamespace(_byte_interaction=SimpleNamespace(has_data=lambda _d: True))
    off = SimpleNamespace(_byte_interaction=SimpleNamespace(has_data=lambda _d: False))
    assert gate(on, SEED)
    assert not gate(off, SEED)
    assert not gate(SimpleNamespace(), SEED)


# -- wiring -----------------------------------------------------------------


def _round(fuzz_count, *, scores=(0.5, 0.5), analyzed=False):
    f = MagicMock()
    f._use_byte_interaction = True
    f._byte_interaction.cost.return_value = 36
    f._byte_interaction.analyzed.return_value = analyzed
    f._sensitivity.scores.return_value = list(scores)
    f._edge_tracker.seed_edges = {"k": BASE}
    f._seed_key.return_value = "k"
    r = FuzzRound(f, SEED)
    r._meta = {"fuzz_count": fuzz_count}
    r._interact()
    return f, r


def test_round_interacts_once_cost_earned():
    f, r = _round(36)
    seed, edges, run, scores = f._byte_interaction.analyze_seed.call_args.args
    assert (seed, edges, run, scores) == (SEED, BASE, r._probe_run, [0.5, 0.5])


def test_round_skips_short_of_cost():
    f, _ = _round(35)
    f._byte_interaction.analyze_seed.assert_not_called()


def test_round_skips_without_first_order_scores():
    f, _ = _round(99, scores=())
    f._byte_interaction.analyze_seed.assert_not_called()


def test_round_skips_analyzed_seed():
    f, _ = _round(99, analyzed=True)
    f._seed_key.assert_not_called()


def test_round_runs_interact_before_execute():
    src = inspect.getsource(FuzzRound._steps)
    assert src.index("_sensitize()") < src.index("_interact()") < src.index("_execute()")


def test_analyzer_registry_wiring():
    on = SimpleNamespace(_use_byte_interaction=True, _rng=RandPool(seed=3))
    off = SimpleNamespace(_use_byte_interaction=False)

    _activate_byte_interaction(on)
    _deactivate_byte_interaction(off)
    assert isinstance(on._byte_interaction, ByteInteractionTracker)
    assert off._byte_interaction is None
    assert "byte_interaction" in ANALYZERS.names()


def test_cli_flag_reaches_fuzzer():
    assert inspect.signature(Fuzzer.__init__).parameters["byte_interaction"].default is False
    src = inspect.getsource(commands)
    assert '"--byte-interaction"' in src
    assert 'byte_interaction=getattr(args, "byte_interaction", False)' in src


# -- observability ----------------------------------------------------------


def test_coupled_counts_seeds_with_pairs():
    exec_fn, _ = _target({99: lambda c: {5, 9} <= c})
    t = _tracker()

    t.analyze_seed(SEED, BASE, exec_fn, _scores(5, 9))
    t.analyze_seed(b"\xff" + SEED[1:], BASE, exec_fn, _scores(4))

    assert (t.coupled(), t.tracked()) == (1, 2)


def test_stats_line_reports_pairs():
    from fuzzer_tool.services.stats import StatsReporter

    sens = SimpleNamespace(_analyzed={b"a", b"b"})
    inter = SimpleNamespace(coupled=lambda: 1, tracked=lambda: 2)
    f = SimpleNamespace(_sensitivity=sens, _byte_interaction=inter)

    line = StatsReporter._print_stats_sens_str(None, f)

    assert line == " | sens: 2 seeds | pairs: 1/2 seeds"


def test_stats_line_without_interaction_unchanged():
    from fuzzer_tool.services.stats import StatsReporter

    f = SimpleNamespace(_sensitivity=SimpleNamespace(_analyzed={b"a"}), _byte_interaction=None)

    assert StatsReporter._print_stats_sens_str(None, f) == " | sens: 1 seeds"
