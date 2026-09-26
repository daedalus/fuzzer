"""--fsm wiring: genseed seeds, fsm_regen operator, fuzz CLI plumbing."""

import argparse
import re

from fuzzer_tool.cli.commands import cmd_genseed
from fuzzer_tool.core.format_fsm import parse_fsm
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

AB_C = "start A\nfinal D\nA -> B : 'a' | 'b'\nB -> B : 'a' | 'b'\nB -> D : 'c'\n"
AB_C_RE = re.compile(rb"[ab]+c")


def _genseed_args(**kw):
    defaults = dict(format="all", corpus=None, count=1, max_len=4096, seed=None, fsm=None)
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class _Ctx:
    """Fuzzer-shaped context for OperatorEngine handlers."""

    def __init__(self, fsm, rng, max_len=64):
        self.fsm = fsm
        self._rng = rng
        self.max_len = max_len


class TestRegistry:
    def test_registered_in_adaptive_band(self):
        assert "fsm_regen" in REGISTRY.names()
        assert "fsm_regen" in OPERATOR_CATEGORIES["adaptive"]

    def test_gated_on_fsm(self):
        off = type("F", (), {"fsm": None})()
        on = type("F", (), {"fsm": parse_fsm(AB_C)})()
        assert "fsm_regen" not in REGISTRY.available(off, b"ab")
        assert "fsm_regen" in REGISTRY.available(on, b"ab")

    def test_dispatch_has_handler(self):
        engine = OperatorEngine(_Ctx(None, None))
        assert callable(REGISTRY.dispatch(engine)["fsm_regen"])


class TestHandler:
    def test_regenerates_tail_exactly(self):
        # Pair idx 2 = (2, "B"); B: option 1 -> D, 'c' forced.
        ctx = _Ctx(parse_fsm(AB_C), ScriptedRng(randints=[2, 1]))
        out = OperatorEngine(ctx)._op_fsm_regen(bytearray(b"abXYZ"), 0, b"abXYZ")
        assert out == bytearray(b"abc")

    def test_no_fsm_is_noop(self):
        engine = OperatorEngine(_Ctx(None, None))
        assert engine._op_fsm_regen(bytearray(b"ab"), 0, b"ab") is None

    def test_output_capped_at_max_len(self):
        # The forced completion may overrun max_len; the handler clips it.
        fsm = parse_fsm('start A\nfinal B\nA -> B : "LONGWORD"\n')
        ctx = _Ctx(fsm, ScriptedRng(), max_len=4)
        out = OperatorEngine(ctx)._op_fsm_regen(bytearray(b""), 0, b"")
        assert out == bytearray(b"LONG")


class TestGenseed:
    def test_writes_accepted_seeds(self, tmp_path):
        spec = tmp_path / "ab.fsm"
        spec.write_text(AB_C)
        corpus = tmp_path / "corpus"
        rc = cmd_genseed(_genseed_args(fsm=str(spec), corpus=str(corpus), count=20, seed=3))
        assert rc == 0

        files = list((corpus / "seeds").rglob("id_*"))
        assert len(files) > 1
        for f in files:
            assert AB_C_RE.fullmatch(f.read_bytes())

    def test_bad_spec_fails(self, tmp_path):
        spec = tmp_path / "bad.fsm"
        spec.write_text("start A\nA -> B : 'a'\n")
        rc = cmd_genseed(_genseed_args(fsm=str(spec), corpus=str(tmp_path / "c")))
        assert rc == 1
        assert not (tmp_path / "c").exists()


class TestFuzzCli:
    def test_fuzz_parser_accepts_fsm(self, tmp_path, monkeypatch):
        from fuzzer_tool.cli import commands

        seen = {}
        monkeypatch.setattr(commands, "cmd_fuzz", lambda a: seen.setdefault("fsm", a.fsm) and 0)
        monkeypatch.setattr("sys.argv", ["fuzzer-tool", "fuzz", "t", "--fsm", "x.fsm"])
        commands.main()
        assert seen["fsm"] == "x.fsm"
