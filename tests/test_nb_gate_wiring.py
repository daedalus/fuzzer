"""--exec-gate nb wiring: registry binding, re-roll loop, outcome feed, CLI."""

from __future__ import annotations

import ast
import inspect
import types

from fuzzer_tool.core.analyzer_registry import REGISTRY
from fuzzer_tool.core.nb_gate import GateMode, NBGate, OpIds, Outcome
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import EXEC_DEDUP_RETRIES, Fuzzer
from tests.support.scripted_rng import ScriptedRng


class _Store(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def _fuzzer(mode=GateMode.NB, store=None, rng=None):
    f = Fuzzer.__new__(Fuzzer)
    f._exec_gate_mode = mode
    f._exec_gate_lift = 0.5
    f._exec_gate_eps = 0.1
    f._state_store = store or _Store()
    f._rng = rng or ScriptedRng()
    spec = REGISTRY._specs["nb_gate"]
    if spec.available(f):
        spec.activate(f)
    else:
        spec.deactivate(f)
    return f


def _script_mutants(f, ops):
    """Raw dedup stub: mutant i ran op ops[i] at offset 0."""
    calls = []

    def raw(data):
        i = len(calls)
        calls.append(i)
        f._last_ops_used = [ops[i]] if ops[i] else []
        f._last_mutation_offset = 0
        return b"m%d" % i

    f._dedup_raw = raw
    return calls


def _train_dead(f, dead="dead", live="live"):
    gate = f._nb_gate
    gate.warmup = 0
    data = b"seed"
    for _ in range(50):
        f._last_ops_used, f._last_mutation_offset = [live], 0
        gate.record(f._nb_features(data), Outcome.NOVEL)
        f._last_ops_used = [dead]
        gate.record(f._nb_features(data), Outcome.BORING)


# --- Registry -----------------------------------------------------------------


def test_off_by_default_leaves_dedup_unbound():
    assert inspect.signature(Fuzzer.__init__).parameters["exec_gate"].default is GateMode.OFF
    f = _fuzzer(GateMode.OFF)
    assert f._nb_gate is None
    assert "_dedup_mutate" not in vars(f)  # class method, no per-exec wrapper


def test_nb_binds_gated_mutate_once():
    f = _fuzzer()
    assert isinstance(f._nb_gate, NBGate)
    assert f._dedup_mutate == f._gated_mutate
    assert (f._nb_gate.lift, f._nb_gate.eps) == (0.5, 0.1)


def test_restores_counts_and_op_slots():
    src = _fuzzer()
    _train_dead(src)
    state = {"gate": src._nb_gate.to_dict(), "ops": src._nb_ops.names()}
    f = _fuzzer(store=_Store(nb_gate=state))
    assert f._nb_ops.names() == ["live", "dead"]
    assert f._nb_gate.total == src._nb_gate.total


# --- Re-roll loop ---------------------------------------------------------------


def test_rerolls_dead_mutants_until_live():
    f = _fuzzer(rng=ScriptedRng(randoms=[0.9, 0.9]))  # no exploration
    _train_dead(f)
    calls = _script_mutants(f, ["dead", "dead", "live"])
    assert f._dedup_mutate(b"seed") == b"m2"
    assert len(calls) == 3 and f._nb_gate.skipped == 2
    assert f._nb_feats == f._nb_features(b"seed")


def test_gives_up_after_retry_budget():
    retries = EXEC_DEDUP_RETRIES
    f = _fuzzer(rng=ScriptedRng(randoms=[0.9] * retries))
    _train_dead(f)
    calls = _script_mutants(f, ["dead"] * (retries + 1))
    assert f._dedup_mutate(b"seed") == b"m%d" % retries
    assert len(calls) == retries + 1


def test_opless_mutant_always_runs():
    # Deterministic stage / TCP seed / advisor: no op drew it, never gated.
    f = _fuzzer()
    _train_dead(f)
    calls = _script_mutants(f, [None])
    assert f._dedup_mutate(b"seed") == b"m0" and len(calls) == 1
    assert f._nb_feats is None


# --- Outcome feed ---------------------------------------------------------------


def _round(f, novel):
    r = FuzzRound.__new__(FuzzRound)
    r._f = f
    r._has_new_coverage = novel
    return r


def test_round_records_outcome():
    f = _fuzzer()
    f._nb_feats = (0, 0, 3)
    _round(f, novel=True)._feed_nb_gate()
    _round(f, novel=False)._feed_nb_gate()
    assert f._nb_gate.total == 2
    assert f._nb_gate.posterior((0, 0, 3)) == f._nb_gate.posterior((0, 0, 3))


def test_round_feed_noop_off_and_opless():
    _round(types.SimpleNamespace(), novel=True)._feed_nb_gate()
    f = _fuzzer()
    f._nb_feats = None
    _round(f, novel=True)._feed_nb_gate()
    assert f._nb_gate.total == 0


def test_features_shape():
    f = _fuzzer()
    f._nb_ops = OpIds()
    f._last_ops_used, f._last_mutation_offset = ["a", "b"], None
    # Last op, delocalised position, log2 size class of the parent.
    assert f._nb_features(b"x" * 5) == (0, 8, 3)


# --- CLI ----------------------------------------------------------------------


def test_cli_dests_registered():
    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
    assert {"exec_gate", "exec_gate_lift", "exec_gate_eps"} <= dests


def test_rejects_bad_flag_values():
    import pytest

    f = Fuzzer.__new__(Fuzzer)
    f._exec_gate_mode, f._exec_gate_lift, f._exec_gate_eps = GateMode.NB, 0.25, 2.0
    f._state_store, f._rng = _Store(), ScriptedRng()
    with pytest.raises(ValueError):
        REGISTRY._specs["nb_gate"].activate(f)
