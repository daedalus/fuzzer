"""tools/tla_lean.py: TLA+ fragment -> Lean 4 model, checked by a compiled BFS."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import tla_extract  # noqa: E402
import tla_lean  # noqa: E402

from tests.support import tla_fixtures as fx  # noqa: E402

_LEAN = tla_lean.find_lean()
_JAR = tla_extract.find_jar()
needs_lean = pytest.mark.skipif(_LEAN is None, reason="Lean missing (tools/vendor_lean.sh)")
needs_both = pytest.mark.skipif(
    _LEAN is None or _JAR is None, reason="Lean or TLC missing (vendor_lean.sh / vendor_tla.sh)"
)

# x counts up by one per step; Bad is violated once x reaches 3.
_COUNTER = """---- MODULE Counter ----
EXTENDS Naturals
CONSTANT Max
VARIABLE w
Init == w = [x |-> 0]
Step(v, Ok(_)) == IF v.x < Max THEN Ok([v EXCEPT !.x = @ + 1]) ELSE FALSE
Next == Step(w, LAMBDA v2: w' = v2)
Bad == w.x < 3
====
"""


def _lean_check(text: str, consts: dict[str, int], invs: list[str], out: Path):
    src = tla_lean.translate([text], consts, invs)
    return tla_lean.check("Counter", src, out, _LEAN)


# ── Parser ──────────────────────────────────────────────────────────


def test_bullets_group_by_column():
    """A deeper bullet list stays inside its item; the next outer bullet starts a new one."""
    expr = tla_lean.parse_expr("\\/ \\E x \\in S :\n     /\\ a\n     /\\ b\n\\/ c")
    assert expr.kind == "or"
    first, second = expr.args
    assert first.kind == "exists" and second.kind == "ident"
    assert first.args[2].kind == "and"


def test_set_filter_vs_map():
    flt = tla_lean.parse_expr("{q \\in 1..M : q > 0}")
    mp = tla_lean.parse_expr("{q + 1 : q \\in S}")
    assert (flt.kind, mp.kind) == ("setfilter", "setmap")


def test_unsupported_construct_rejected():
    """Adversarial: a construct outside the fragment names its line, writes no Lean."""
    with pytest.raises(tla_lean.TranslateError, match="line 5"):
        tla_lean.translate(
            [
                "---- MODULE M ----\nVARIABLE w\nInit == w = 0\nNext ==\n  w' = CHOOSE x \\in S : x\n===="
            ],
            {},
            [],
        )


def test_two_actions_in_conjunction_rejected():
    """Adversarial: `w' = a /\\ w' = b` has no single successor; refuse, don't guess."""
    with pytest.raises(tla_lean.TranslateError, match="conjunction"):
        tla_lean.translate(
            ["---- MODULE M ----\nVARIABLE w\nInit == w = 0\nNext == w' = 1 /\\ w' = 2\n===="],
            {},
            [],
        )


# ── Falsification + control (Hard Rules 23, 46) ─────────────────────


@needs_lean
def test_counter_fails_only_past_bound(tmp_path):
    ok = _lean_check(_COUNTER, {"Max": 2}, ["Bad"], tmp_path / "ok")
    assert ok.completed and not ok.violations
    assert ok.states == 3  # x = 0, 1, 2

    bad = _lean_check(_COUNTER, {"Max": 5}, ["Bad"], tmp_path / "bad")
    assert bad.states == 6
    # Shortest trace: x = 0 -> 1 -> 2 -> 3.
    assert bad.violations["Bad"].count("State ") == 4


# Operators the thread models use: \\ (difference), :> / @@ (TLC), DOMAIN, ENABLED.
_THREAD_OPS = """---- MODULE Counter ----
VARIABLE w
Init == w = ("a" :> 1) @@ [a |-> 9, b |-> 2]
Next == w' = w
Merge == w = [a |-> 1, b |-> 2]
Dom == DOMAIN w = {"a", "b"}
Diff == {1, 2, 3} \\ {2} = {1, 3}
En == ENABLED (w.a = 1 /\\ w' = w)
NotEn == ~ENABLED (w.a = 2 /\\ w' = w)
Wrong == ENABLED (w.a = 2 /\\ w' = w)
====
"""


@needs_lean
def test_thread_model_operators(tmp_path):
    """@@ keeps the left value; ENABLED is true exactly when the action has a successor."""
    invs = ["Merge", "Dom", "Diff", "En", "NotEn", "Wrong"]
    res = _lean_check(_THREAD_OPS, {}, invs, tmp_path)
    assert res.completed and res.states == 1
    assert set(res.violations) == {"Wrong"}


def _verdicts(res) -> tuple[bool, frozenset[str], int]:
    return res.completed, frozenset(res.violations), res.states


@needs_both
@pytest.mark.parametrize(
    "cls",
    [fx.CleanRunner, fx.LeakyRunner, fx.StaleReplyRunner, fx.FreshReplyRunner, fx.LeakyJob,
     fx.CleanJob, fx.ShortReadRunner, fx.StaleKillRunner, fx.TimerFirstSampler,
     fx.HandlerFirstSampler, fx.InstallOnceRunner],
    ids=lambda c: c.__name__,
)  # fmt: skip
def test_lean_matches_tlc(cls, tmp_path):
    """Same verdicts and state count as TLC; each backend run twice agrees with itself first."""
    model = tla_extract.extract(cls)
    bounds = tla_extract.Bounds()
    tlc = [tla_extract.check(model, tmp_path / f"tlc{i}", _JAR, bounds) for i in (0, 1)]
    lean = [tla_extract.check_lean(model, tmp_path / f"lean{i}", _LEAN, bounds) for i in (0, 1)]

    assert _verdicts(tlc[0]) == _verdicts(tlc[1])
    assert _verdicts(lean[0]) == _verdicts(lean[1])
    assert _verdicts(lean[0]) == _verdicts(tlc[0])
    # Trace lengths too: both explore breadth-first, so each shortest trace has one length.
    assert tla_extract._disagree(tlc[0], tlc[1]) is None
    assert tla_extract._disagree(lean[0], lean[1]) is None
    assert tla_extract._disagree(tlc[0], lean[0]) is None


# TLC prints "State n: <action ...>" per step; an Init violation is one bare state.
_TLC_3 = "State 1: <Initial predicate>\nw = 0\n\nState 2: <Next>\nw = 1\n\nState 3: <Next>\nw = 2"
_LEAN_3 = "State 1: <step>\nw = 0\n\nState 2: <step>\nw = 1\n\nState 3: <step>\nw = 2"
_TLC_INIT = 'w = [setup |-> "OSError"]'
_LEAN_INIT = 'State 1: <step>\nw = [setup |-> "OSError"]'


def _res(**violations: str) -> tla_extract.CheckResult:
    return tla_extract.CheckResult(True, violations, 7)


def test_trace_len_formats():
    assert tla_extract._trace_len(_TLC_3) == tla_extract._trace_len(_LEAN_3) == 3
    assert tla_extract._trace_len(_TLC_INIT) == tla_extract._trace_len(_LEAN_INIT) == 1


def test_trace_length_mismatch_flagged():
    """Falsification: same verdict, different shortest-trace length is a disagreement."""
    assert tla_extract._disagree(_res(A=_TLC_3), _res(A=_LEAN_3)) is None
    diff = tla_extract._disagree(_res(A=_TLC_3), _res(A=_LEAN_INIT))
    assert diff is not None and "A" in diff and "3" in diff and "1" in diff


def test_trace_without_states_flagged():
    """Adversarial: a trace that cannot be measured (error text) is never silently equal."""
    diff = tla_extract._disagree(_res(A="TLC error (no verdict):\nboom"), _res(A=_LEAN_3))
    assert diff is not None


@needs_lean
def test_cli_lean_backend(tmp_path):
    rc = tla_extract.main(
        ["tests.support.tla_fixtures:LeakyRunner", "--check", "--backend", "lean",
         "--out", str(tmp_path)]
    )  # fmt: skip
    assert rc == 1
    assert (tmp_path / "lean" / "LeakyRunner.lean").is_file()
