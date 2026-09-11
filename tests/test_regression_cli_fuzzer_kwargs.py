"""Regression: every fuzz flag that names a Fuzzer parameter reaches Fuzzer().

cmd_fuzz builds the fuzzer in one of two calls: run_parallel(...) for -j>1
and Fuzzer(...) otherwise. Both are hand-written keyword lists, and the
single-process one -- the default mode -- had dropped kl_ducb,
kl_ducb_gamma, kl_swucb, kl_swucb_window and markov_blend while the parallel
one passed them. --kl-ducb, --kl-swucb and --markov-blend parsed, were
accepted, and built nothing; --elo all set args.kl_ducb/kl_swucb and they
were discarded the same way. test_regression_elo_all's flag list never
named the two KL schedulers, which is why it did not notice.

The structural test reads both sides from source: the fuzz parser's option
dests and the keywords of the Fuzzer(...) call in cmd_fuzz. A flag whose
dest is also a Fuzzer.__init__ parameter must be passed.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.services.fuzzer import Fuzzer
from tests import test_commands_extended


def _fuzz_parser_dests(tree: ast.AST) -> set[str]:
    dests = set()
    for n in ast.walk(tree):
        if not (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_argument"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "fuzz_parser"
        ):
            continue
        dest = next(
            (
                k.value.value
                for k in n.keywords
                if k.arg == "dest" and isinstance(k.value, ast.Constant)
            ),
            None,
        )
        if dest is None:
            dest = next(
                (
                    a.value[2:].replace("-", "_")
                    for a in n.args
                    if isinstance(a, ast.Constant) and str(a.value).startswith("--")
                ),
                None,
            )
        if dest:
            dests.add(dest)
    return dests


def _single_process_fuzzer_keywords(tree: ast.AST) -> set[str]:
    cmd = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cmd_fuzz")
    calls = [
        n
        for n in ast.walk(cmd)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Fuzzer"
    ]
    assert len(calls) == 1, "premise: cmd_fuzz has exactly one Fuzzer(...) call"
    return {k.arg for k in calls[0].keywords if k.arg}


def test_every_fuzzer_named_flag_is_passed():
    tree = ast.parse(inspect.getsource(commands))
    dests = _fuzz_parser_dests(tree)
    assert len(dests) > 100, f"premise: found only {len(dests)} fuzz flags"
    params = set(inspect.signature(Fuzzer.__init__).parameters) - {"self"}
    passed = _single_process_fuzzer_keywords(tree)
    dropped = sorted((dests & params) - passed)
    assert not dropped, f"fuzz flags naming a Fuzzer parameter but not passed to it: {dropped}"


@pytest.mark.parametrize(
    ("flag", "value"),
    [("kl_ducb", True), ("kl_swucb", True), ("markov_blend", True)],
)
def test_flag_reaches_the_single_process_fuzzer(monkeypatch, tmp_path, flag, value):
    # Module attribute, not a from-import: importing the Test class by name
    # would make pytest collect and rerun its tests from this file.
    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    setattr(args, flag, value)
    captured = {}

    def fake_fuzzer(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(run=lambda iterations: 0)

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake_fuzzer)
    assert commands.cmd_fuzz(args) == 0
    assert captured.get(flag) is value, f"--{flag.replace('_', '-')} did not reach Fuzzer()"
