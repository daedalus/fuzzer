"""``--ltl`` wiring: round hooks, operator, crash signature, CLI loader."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters.filesystem import classify_crash
from fuzzer_tool.cli.commands import _load_ltl_arg
from fuzzer_tool.core.ltl import (
    LtlChannel,
    Record,
    parse_hoa,
    violation_signature,
)
from fuzzer_tool.core.operator_registry import _CATEGORIES, REGISTRY
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.operators import _DELOCALISED_OPS
from tests.support.scripted_rng import ScriptedRng
from tests.test_ltl import CHAIN, GF_E1, NEVER_E1, _pack


def _channel(tmp_path, hoa=NEVER_E1):
    return LtlChannel(parse_hoa(hoa), str(tmp_path / "ev"))


def _fuzzer(ch):
    return SimpleNamespace(ltl=ch)


def _round(f, mutated=b"abcd", is_crash=False, is_timeout=False):
    r = FuzzRound(f, b"parent")
    r._mutated = mutated
    r._stderr = ""
    r._is_crash = is_crash
    r._is_timeout = is_timeout
    return r


def _write(ch, *recs):
    with open(ch.events_path, "wb") as fh:
        fh.write(_pack(*recs))


# ── round hooks ──────────────────────────────────────────────────────


def test_round_clears_events_before_the_run(tmp_path):
    ch = _channel(tmp_path)
    _write(ch, (1, 0, 0))
    _round(_fuzzer(ch))._ltl_clear()
    with open(ch.events_path, "rb") as fh:
        assert fh.read() == b""


def test_round_flags_violation_as_crash_with_marker(tmp_path):
    ch = _channel(tmp_path)
    _write(ch, (2, 0, 0), (1, 1, 0))
    r = _round(_fuzzer(ch))
    r._ltl_observe()
    assert r._is_crash is True
    assert "LTL-VIOLATION: trap" in r._stderr


def test_round_leaves_a_real_crash_untouched(tmp_path):
    ch = _channel(tmp_path)
    _write(ch, (1, 0, 0))
    r = _round(_fuzzer(ch), is_crash=True)
    r._stderr = "ASAN report"
    r._ltl_observe()
    assert r._stderr == "ASAN report"


def test_round_novel_transition_marks_admission(tmp_path):
    ch = _channel(tmp_path)
    _write(ch, (2, 0, 0))
    r = _round(_fuzzer(ch))
    r._ltl_observe()
    assert r._ltl_new is True
    assert r._admits() is True
    r2 = _round(_fuzzer(ch))
    r2._ltl_observe()
    assert r2._ltl_new is False


def test_round_skips_timeouts(tmp_path):
    ch = _channel(tmp_path)
    _write(ch, (2, 0, 0), (1, 1, 0))
    r = _round(_fuzzer(ch), is_timeout=True)
    r._ltl_observe()
    assert r._is_crash is False
    assert r._ltl_new is False


def test_round_without_channel_is_inert():
    r = _round(SimpleNamespace())
    r._ltl_clear()
    r._ltl_observe()
    assert r._ltl_new is False
    assert r._is_crash is False


def test_round_with_missing_event_file_is_quiet(tmp_path):
    ch = _channel(tmp_path)
    r = _round(_fuzzer(ch))
    r._ltl_observe()
    assert r._is_crash is False


# ── crash signature ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("stderr", "sig"),
    [
        ("x\nLTL-VIOLATION: trap\n", "ltl:trap"),
        ("LTL-VIOLATION: lasso (candidate)\n", "ltl:lasso"),
        ("", None),
        ("LTL-VIOLATION: nonsense\n", None),
    ],
)
def test_violation_signature(stderr, sig):
    assert violation_signature(stderr) == sig


def test_classify_crash_uses_the_ltl_signature():
    v = classify_crash(b"x", 0, "LTL-VIOLATION: trap\n", set(), {})
    assert v.signature == "ltl:trap"
    assert v.novel is True


def test_ltl_kinds_get_separate_buckets():
    seen = {"ltl:trap": 1}
    v = classify_crash(b"y", 0, "LTL-VIOLATION: lasso (candidate)\n", set(), seen)
    assert v.novel is True


def test_classify_crash_without_marker_is_unchanged():
    v = classify_crash(b"x", -11, "", set(), {})
    assert v.signature == "signal:11"


# ── operator ─────────────────────────────────────────────────────────


class _Gate:
    def __init__(self, ltl):
        self.ltl = ltl


def test_op_registered_and_gated():
    assert "ltl_prefix" in _CATEGORIES["block"]
    assert "ltl_prefix" in REGISTRY.names()
    assert "ltl_prefix" not in REGISTRY.available(_Gate(None), b"abc")
    assert "ltl_prefix" in REGISTRY.available(_Gate(object()), b"abc")


def test_op_is_delocalised():
    assert "ltl_prefix" in _DELOCALISED_OPS


def test_op_replaces_buffer_with_prefix_plus_tail(tmp_path):
    from fuzzer_tool.services.operators import OperatorEngine

    ch = _channel(tmp_path, CHAIN)
    _write(ch, (1, 4, 0))
    ch.collect(b"abcdefgh")

    ctx = SimpleNamespace(
        ltl=ch, max_len=64, _rng=ScriptedRng(randoms=[0.0], randints=[2], randbytes=[b"ZZ"])
    )
    eng = OperatorEngine.__new__(OperatorEngine)
    eng._ctx_cache = ctx
    eng.f = SimpleNamespace()
    buf = bytearray(b"XXXXXXXXXXXXXXXX")
    out = eng._op_ltl_prefix(buf, 0, b"")
    assert out == bytearray(b"abcdZZ")


def test_op_declines_without_a_frontier(tmp_path):
    from fuzzer_tool.services.operators import OperatorEngine

    ch = _channel(tmp_path, GF_E1)
    ctx = SimpleNamespace(ltl=ch, max_len=64, _rng=ScriptedRng(randoms=[0.0]))
    eng = OperatorEngine.__new__(OperatorEngine)
    eng._ctx_cache = ctx
    eng.f = SimpleNamespace()
    assert eng._op_ltl_prefix(bytearray(b"abc"), 0, b"") is None


# ── splice ───────────────────────────────────────────────────────────


def test_splice_respects_max_len(tmp_path):
    ch = _channel(tmp_path, CHAIN)
    _write(ch, (1, 4, 0))
    ch.collect(b"abcdefgh")
    rng = ScriptedRng(randoms=[0.0], randints=[1], randbytes=[b"Q"])
    assert ch.splice(rng, max_len=5) == b"abcdQ"


def test_splice_none_when_prefix_fills_the_budget(tmp_path):
    ch = _channel(tmp_path, CHAIN)
    _write(ch, (1, 4, 0))
    ch.collect(b"abcdefgh")
    assert ch.splice(ScriptedRng(randoms=[0.0]), max_len=4) is None


# ── CLI loader ───────────────────────────────────────────────────────


def test_loader_builds_channel_and_exports_env(tmp_path, monkeypatch):
    monkeypatch.delenv("__LTL_EVENTS_OUT", raising=False)
    hoa = tmp_path / "p.hoa"
    hoa.write_text(NEVER_E1)
    ch = _load_ltl_arg(str(hoa))
    assert isinstance(ch, LtlChannel)
    assert os.environ["__LTL_EVENTS_OUT"] == ch.events_path
    assert os.path.exists(ch.events_path)
    os.unlink(ch.events_path)
    os.environ.pop("__LTL_EVENTS_OUT")


def test_loader_rejects_bad_automaton(tmp_path, capsys):
    hoa = tmp_path / "bad.hoa"
    hoa.write_text("not hoa")
    assert _load_ltl_arg(str(hoa)) is None
    assert "--ltl" in capsys.readouterr().out


def test_loader_rejects_missing_file(tmp_path, capsys):
    assert _load_ltl_arg(str(tmp_path / "nope.hoa")) is None
    assert "--ltl" in capsys.readouterr().out


def test_record_tuple_shape():
    assert Record(1, 2, 3) == (1, 2, 3)
