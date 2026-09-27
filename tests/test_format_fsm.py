"""Constraint-labelled FSM message formats (StateLifter, USENIX Sec '23 §2, §6.3).

Oracles are independent of the module: a ``re`` regex for the (a|b)+c
running example, arithmetic for integer constraints, ``int.from_bytes`` for
encodings.
"""

import re

import pytest

from fuzzer_tool.core.format_fsm import load_fsm, parse_fsm
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

# Paper Figure 1: (a|b)+c.
AB_C = """
start A
final D
A -> B : 'a' | 'b'
B -> B : 'a' | 'b'
B -> D : 'c'
"""
AB_C_RE = re.compile(rb"[ab]+c")


class TestGenerate:
    def test_every_message_matches_regex(self):
        fsm = parse_fsm(AB_C)
        rng = RandPool(seed=11)
        for _ in range(200):
            assert AB_C_RE.fullmatch(fsm.generate(rng, max_len=64))

    def test_scripted_walk_exact(self):
        # Forced choices draw nothing. A: byte idx 1 ('b'); B: option 0
        # (loop), byte idx 0 ('a'); B: option 1 (-> D), 'c'; D: stop.
        fsm = parse_fsm(AB_C)
        rng = ScriptedRng(randints=[0, 1], choice_idxs=[1, 0])
        assert fsm.generate(rng, max_len=64) == b"bac"

    def test_final_state_can_stop_or_continue(self):
        fsm = parse_fsm("start A\nfinal A\nA -> A : 'x'\n")
        # Options at A: [loop, stop]; loop twice, then stop.
        rng = ScriptedRng(randints=[0, 0, 1])
        assert fsm.generate(rng, max_len=64) == b"xx"

    def test_int_range_modulus(self):
        fsm = parse_fsm("start A\nfinal B\nA -> B : u8[0,255]%10=4\n")
        rng = RandPool(seed=3)
        for _ in range(100):
            (v,) = fsm.generate(rng, max_len=8)
            assert v % 10 == 4

    def test_int_range_scripted_value(self):
        fsm = parse_fsm("start A\nfinal B\nA -> B : u16le[11,65535]%7=2\n")
        first = 11 + (2 - 11) % 7
        rng = ScriptedRng(randints=[5])
        out = fsm.generate(rng, max_len=8)
        assert int.from_bytes(out, "little") == first + 7 * 5

    @pytest.mark.parametrize(
        ("kind", "order", "width"),
        [("u16be", "big", 2), ("u32le", "little", 4), ("u32be", "big", 4)],
    )
    def test_int_encodings(self, kind, order, width):
        fsm = parse_fsm(f"start A\nfinal B\nA -> B : {kind}[4660,4660]\n")
        out = fsm.generate(ScriptedRng(), max_len=8)
        assert len(out) == width
        assert int.from_bytes(out, order) == 4660

    def test_literal_and_range(self):
        fsm = parse_fsm("start A\nfinal C\nA -> B : \"GET \"\nB -> C : 'a'-'z'\n")
        rng = ScriptedRng(choice_idxs=[25])
        assert fsm.generate(rng, max_len=16) == b"GET z"


class TestMatch:
    def test_accepts_falsification(self):
        fsm = parse_fsm(AB_C)
        assert fsm.accepts(b"abbac")
        for bad in (b"", b"c", b"abca", b"abx", b"ab"):
            assert not fsm.accepts(bad), bad

    def test_prefix_states(self):
        fsm = parse_fsm(AB_C)
        assert fsm.prefix_states(b"abX") == [(0, "A"), (1, "B"), (2, "B")]

    def test_regenerate_keeps_valid_prefix(self):
        fsm = parse_fsm(AB_C)
        # Pair idx 2 = (2, "B"); B: option 1 (-> D), 'c' forced.
        rng = ScriptedRng(randints=[2, 1])
        assert fsm.regenerate(b"abXYZ", rng, max_len=64) == b"abc"


class TestAdversarial:
    def test_budget_forces_shortest_completion(self):
        # At max_len 1 the 'x' loop cannot fit with a completion after it:
        # the only fitting option is forced, so no draw happens.
        fsm = parse_fsm("start A\nfinal B\nA -> A : 'x'\nA -> B : 'y'\n")
        assert fsm.generate(ScriptedRng(), max_len=1) == b"y"

    def test_regression_generation_never_exceeds_max_len(self):
        # Clipping a finished message can split a label (C1/C5): every
        # message must fit the budget and stay accepted instead.
        fsm = parse_fsm('start A\nfinal C\nA -> A : "xyz"\nA -> B : "LONG"\nB -> C : u16le[0,9]\n')
        rng = RandPool(seed=4)
        for budget in range(6, 20):
            out = fsm.generate(rng, max_len=budget)
            assert len(out) <= budget
            assert fsm.accepts(out)

    def test_regression_infeasible_budget_returns_none(self):
        fsm = parse_fsm('start A\nfinal B\nA -> B : "LONGWORD"\n')
        assert fsm.generate(ScriptedRng(), max_len=7) is None
        assert fsm.regenerate(b"", ScriptedRng(), max_len=7) is None

    def test_regression_regenerate_respects_budget(self):
        # Prefix pairs whose cheapest completion overruns are not picked.
        fsm = parse_fsm(AB_C)
        rng = RandPool(seed=6)
        for _ in range(50):
            out = fsm.regenerate(b"abab", rng, max_len=4)
            assert len(out) <= 4 and AB_C_RE.fullmatch(out)

    def test_dead_states_never_entered(self):
        # C cannot reach a final state: generation must never take A -> C.
        fsm = parse_fsm("start A\nfinal B\nA -> C : 'z'\nA -> B : 'y'\nC -> C : 'z'\n")
        rng = RandPool(seed=5)
        for _ in range(50):
            assert fsm.generate(rng, max_len=16) == b"y"

    def test_garbage_regenerates_valid(self):
        fsm = parse_fsm(AB_C)
        rng = RandPool(seed=9)
        junk = RandPool(seed=1).randbytes(4096)
        for _ in range(50):
            assert AB_C_RE.fullmatch(fsm.regenerate(junk, rng, max_len=64))

    def test_regression_inline_comments(self):
        spec = (
            "start A  # entry\n"
            "final C\n"
            "A -> B : '#' | 'a'  # quoted hash is a byte\n"
            'B -> C : "x#y"  # hash inside a literal\n'
        )
        fsm = parse_fsm(spec)
        assert fsm.accepts(b"#x#y") and fsm.accepts(b"ax#y")
        assert not fsm.accepts(b"ax")

    def test_quoted_pipe_is_a_byte(self):
        fsm = parse_fsm("start A\nfinal B\nA -> B : '|' | 0x7e\n")
        assert fsm.accepts(b"|") and fsm.accepts(b"~")
        assert not fsm.accepts(b" ")

    @pytest.mark.parametrize(
        "spec",
        [
            "final B\nA -> B : 'a'\n",  # no start
            "start A\nA -> B : 'a'\n",  # no final
            "start A\nfinal B\nA -> C : 'a'\n",  # final unreachable
            "start A\nfinal B\nA -> B : ''\n",  # empty byte
            'start A\nfinal B\nA -> B : ""\n',  # empty literal
            "start A\nfinal B\nA -> B : u8[0,300]\n",  # overflows width
            "start A\nfinal B\nA -> B : u8[9,3]\n",  # lo > hi
            "start A\nfinal B\nA -> B : u8[1,3]%10=5\n",  # no congruent value
            "start A\nfinal B\nA -> B : 'z'-'a'\n",  # inverted range
            "start A\nfinal B\nA -> B : 'a' junk\n",  # trailing garbage
            "start A\nfinal B\nA => B : 'a'\n",  # bad arrow
            "start A\nfinal B\nA -> B : any |\n",  # dangling pipe after any
        ],
    )
    def test_rejects_bad_spec(self, spec):
        with pytest.raises(ValueError):
            parse_fsm(spec)

    def test_load_fsm_from_file(self, tmp_path):
        p = tmp_path / "ab.fsm"
        p.write_text(AB_C)
        assert load_fsm(p).accepts(b"ac")
