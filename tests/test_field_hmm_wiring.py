"""--field-hmm wiring: analyzer, hmm_segment op, gate, position proposer, CLI.

RNG-dependent paths run on ScriptedRng (Hard Rule 39); expected outputs are
rebuilt from the scripted draws.
"""

import ast
import inspect
from types import SimpleNamespace

import numpy as np

from fuzzer_tool.cli import commands
from fuzzer_tool.core.analyzer_registry import REGISTRY as ANALYZERS
from fuzzer_tool.core.analyzer_registry import _activate_field_hmm, _deactivate_field_hmm
from fuzzer_tool.core.analyzers.analyzer_field_hmm import (
    RETRAIN_EVERY,
    SPAN_CACHE_MAX,
    FieldHmmAnalyzer,
)
from fuzzer_tool.core.field_hmm import FieldKind
from fuzzer_tool.core.operator_registry import _AVAILABLE, REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.state_store import StateStore
from fuzzer_tool.core.weizz_tags import ByteTag, StructureMap, TagFlags, attach_tags_to_meta
from fuzzer_tool.services.operators import _DELOCALISED_OPS, OperatorEngine
from tests import test_commands_extended
from tests import test_regression_cli_fuzzer_kwargs as cli_kwargs
from tests.support.scripted_rng import ScriptedRng

TWO = b"\x00" * 40 + b"\xff" * 40
FLAT = b"\x41" * 50


def _trained(n_states: int = 2, rng=None) -> FieldHmmAnalyzer:
    a = FieldHmmAnalyzer(n_states=n_states, rng=rng or RandPool(seed=1))
    a.fit_corpus([TWO] * 4, {})
    return a


def _state_of(a: FieldHmmAnalyzer, data: bytes, offset: int) -> int:
    starts, ends, states = a.segments(data)
    i = int(np.searchsorted(ends, offset, side="right"))
    return int(states[i])


# -- analyzer ---------------------------------------------------------------


def test_segments_split_two_regions():
    a = _trained()
    starts, ends, _ = a.segments(TWO)
    assert (starts.tolist(), ends.tolist()) == ([0, 40], [40, 80])
    assert a.has_spans(TWO)


def test_falsify_single_span_seed_not_offered():
    a = _trained()
    assert not a.has_spans(b"\x00" * 30)
    assert not a.has_spans(b"")
    assert a.segments(b"") is None


def test_refit_on_retrain_cadence(monkeypatch):
    a = _trained()
    calls = []
    monkeypatch.setattr(a, "fit_corpus", lambda corpus, meta: calls.append(len(corpus)))

    for _ in range(RETRAIN_EVERY - 1):
        a.on_admit([TWO], {})
    assert calls == []

    a.on_admit([TWO], {})
    assert calls == [1]


def test_refit_clears_span_cache():
    a = _trained()
    a.segments(TWO)
    assert a.cached() == 1
    a.fit_corpus([TWO], {})
    assert a.cached() == 0


def test_adversarial_span_cache_bounded():
    a = _trained()
    for i in range(SPAN_CACHE_MAX + 5):
        a.segments(TWO + i.to_bytes(2, "big"))
    assert a.cached() <= SPAN_CACHE_MAX


def test_donor_returns_same_state_span():
    a = _trained()
    a.segments(TWO)
    ff = _state_of(a, TWO, 50)

    assert a.donor(ff, ScriptedRng(choice_idxs=[0])) == b"\xff" * 40


def test_donor_none_for_unseen_state():
    a = FieldHmmAnalyzer(n_states=4, rng=RandPool(seed=1))
    assert a.donor(3, ScriptedRng()) is None


def test_propose_picks_a_span_start():
    a = _trained(rng=ScriptedRng(choice_idxs=[1]))
    assert a.propose(TWO, len(TWO)) == 40


def test_propose_none_without_structure():
    a = _trained(rng=ScriptedRng())
    assert a.propose(FLAT, len(FLAT)) is None


def test_propose_skips_starts_past_buffer():
    a = _trained(rng=ScriptedRng(choice_idxs=[0]))
    # Only start 0 is inside a 30-byte buffer.
    assert a.propose(TWO, 30) == 0


def test_states_named_from_weizz_tags():
    tags = [ByteTag(cmp_id=1, flags=TagFlags.IS_MAGIC) for _ in range(30)]
    tags += [ByteTag() for _ in range(50)]
    meta = {TWO: attach_tags_to_meta({}, StructureMap(tags=tags, input_len=80))}

    a = FieldHmmAnalyzer(n_states=2, rng=RandPool(seed=1))
    a.fit_corpus([TWO] * 4, meta)

    assert a.kind(_state_of(a, TWO, 0)) is FieldKind.MAGIC
    assert a.kind(_state_of(a, TWO, 50)) is FieldKind.UNKNOWN


def test_state_round_trip_and_mismatch(tmp_path):
    a = _trained()
    store = StateStore(tmp_path)
    store.set("field_hmm", a.to_state())
    assert store.save()

    b = FieldHmmAnalyzer(n_states=2, rng=RandPool(seed=1))
    assert b.load_state(StateStore(tmp_path).get("field_hmm"))
    assert b.segments(TWO)[0].tolist() == a.segments(TWO)[0].tolist()

    # Different K: rejected, caller refits.
    c = FieldHmmAnalyzer(n_states=3, rng=RandPool(seed=1))
    assert not c.load_state(a.to_state())


# -- operator ---------------------------------------------------------------


def _engine(analyzer, rng):
    eng = SimpleNamespace(
        f=SimpleNamespace(_field_hmm=analyzer),
        ctx=SimpleNamespace(_rng=rng),
        declined=[],
    )
    eng._op_declined = lambda op, _buf: eng.declined.append(op)
    return eng


def test_op_havoc_rewrites_one_span():
    a = _trained()
    blob = bytes(range(1, 41))
    eng = _engine(a, ScriptedRng(randints=[1, 0], randbytes=[blob]))
    buf = bytearray(TWO)

    out = OperatorEngine._op_hmm_segment(eng, buf, 0, TWO)

    assert out is None
    assert buf == bytearray(TWO[:40] + blob)


def test_op_splice_inserts_same_state_donor():
    a = _trained()
    a.segments(TWO)
    eng = _engine(a, ScriptedRng(randints=[0, 1], choice_idxs=[0]))

    out = OperatorEngine._op_hmm_segment(eng, bytearray(TWO), 0, TWO)

    # Span 0's donor is itself: the result equals the parent.
    assert bytes(out) == TWO


def test_op_dup_and_delete():
    a = _trained()
    dup = OperatorEngine._op_hmm_segment(
        _engine(a, ScriptedRng(randints=[0, 2])), bytearray(TWO), 0, TWO
    )
    assert bytes(dup) == TWO[:40] * 2 + TWO[40:]

    gone = OperatorEngine._op_hmm_segment(
        _engine(a, ScriptedRng(randints=[1, 3])), bytearray(TWO), 0, TWO
    )
    assert bytes(gone) == TWO[:40]


def test_regression_op_declines_after_length_change():
    a = _trained()
    eng = _engine(a, ScriptedRng())
    buf = bytearray(b"X" + TWO)

    OperatorEngine._op_hmm_segment(eng, buf, 0, TWO)

    assert eng.declined == ["hmm_segment"]
    assert buf == bytearray(b"X" + TWO)


def test_op_declines_without_analyzer_or_structure():
    eng = _engine(None, ScriptedRng())
    OperatorEngine._op_hmm_segment(eng, bytearray(TWO), 0, TWO)

    flat = _engine(_trained(), ScriptedRng())
    OperatorEngine._op_hmm_segment(flat, bytearray(FLAT), 0, FLAT)

    assert eng.declined == flat.declined == ["hmm_segment"]


def test_op_splice_declines_without_donor():
    a = FieldHmmAnalyzer(n_states=2, rng=RandPool(seed=1))
    a.fit_corpus([TWO] * 4, {})
    a.segments(TWO)
    a._donors.clear()
    eng = _engine(a, ScriptedRng(randints=[0, 1]))

    OperatorEngine._op_hmm_segment(eng, bytearray(TWO), 0, TWO)

    assert eng.declined == ["hmm_segment"]


def test_op_registered_gated_and_delocalised():
    assert REGISTRY.category_of("hmm_segment") == "adaptive"
    assert "hmm_segment" in _DELOCALISED_OPS

    gate = _AVAILABLE["hmm_segment"]
    a = _trained()
    assert gate(SimpleNamespace(_field_hmm=a), TWO)
    assert not gate(SimpleNamespace(_field_hmm=a), FLAT)
    assert not gate(SimpleNamespace(_field_hmm=None), TWO)
    assert not gate(SimpleNamespace(), TWO)


# -- wiring -----------------------------------------------------------------


def _fuzzer_ns(tmp_path, **kw):
    return SimpleNamespace(
        _use_field_hmm=True,
        _field_hmm_states=2,
        _rng=RandPool(seed=2),
        _state_store=StateStore(tmp_path),
        corpus=[TWO] * 3,
        seed_meta={},
        **kw,
    )


def test_analyzer_registry_wiring(tmp_path):
    on = _fuzzer_ns(tmp_path)
    off = SimpleNamespace(_use_field_hmm=False)

    _activate_field_hmm(on)
    _deactivate_field_hmm(off)

    assert isinstance(on._field_hmm, FieldHmmAnalyzer)
    assert on._field_hmm.has_spans(TWO)
    assert off._field_hmm is None
    assert "field_hmm" in ANALYZERS.names()


def test_activate_restores_saved_model(tmp_path, monkeypatch):
    saved = _trained()
    f = _fuzzer_ns(tmp_path)
    f._state_store.set("field_hmm", saved.to_state())

    def _no_fit(*_a):
        raise AssertionError("saved model must load without a refit")

    monkeypatch.setattr(FieldHmmAnalyzer, "fit_corpus", _no_fit)

    _activate_field_hmm(f)

    assert f._field_hmm.segments(TWO)[0].tolist() == saved.segments(TWO)[0].tolist()


def test_after_admit_feeds_analyzer():
    from fuzzer_tool.services.corpus_manager import CorpusManager

    src = inspect.getsource(CorpusManager._after_admit)
    assert "_field_hmm" in src


def test_end_of_run_saves_model():
    from fuzzer_tool.services import fuzzer as fuzzer_mod

    assert '"field_hmm"' in inspect.getsource(fuzzer_mod.Fuzzer)


def test_regression_cli_flags_reach_fuzzer(monkeypatch, tmp_path):
    """Parser dests and Fuzzer() keywords must both be field_hmm / field_hmm_states."""
    tree = ast.parse(inspect.getsource(commands))
    dests = cli_kwargs._fuzz_parser_dests(tree)
    assert {"field_hmm", "field_hmm_states"} <= dests

    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    args.field_hmm = True
    args.field_hmm_states = 5
    captured = {}

    def fake_fuzzer(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(run=lambda max_execs=0: 0)

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake_fuzzer)
    assert commands.cmd_fuzz(args) == 0
    assert (captured.get("field_hmm"), captured.get("field_hmm_states")) == (True, 5)
