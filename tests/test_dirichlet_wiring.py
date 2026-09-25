"""Wiring of the Dirichlet features: CLI → Fuzzer → Markov / CEM / operators."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.dirichlet import AlphaMode
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.test_regression_enabled_features_entropy_deviation import _make_fake_fuzzer

_TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")


@pytest.fixture
def build(tmp_path):
    def _build(**kwargs) -> Fuzzer:
        corpus = tmp_path / "corpus"
        crashes = tmp_path / "crashes"
        corpus.mkdir(exist_ok=True)
        crashes.mkdir(exist_ok=True)
        return Fuzzer(
            target=_TARGET, corpus_dir=str(corpus), crashes_dir=str(crashes), max_len=4096, **kwargs
        )

    return _build


def _parse(monkeypatch, *argv: str):
    seen = {}
    monkeypatch.setattr(commands, "cmd_fuzz", lambda args: seen.setdefault("a", args) and 0)
    monkeypatch.setattr("sys.argv", ["fuzzer-tool", "fuzz", "/bin/true", *argv])
    commands.main()
    return seen["a"]


def _fuzzer_call_kwargs() -> list[set[str]]:
    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    return [
        {k.arg for k in c.keywords}
        for c in ast.walk(tree)
        if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "Fuzzer"
    ]


class TestDirichletAlpha:
    def test_default_fixed(self, build):
        f = build(markov_order="0,1", mc_cem=True)
        assert all(c.alpha_mode is AlphaMode.FIXED for c in f.markov.chains.values())
        assert f.mc._cem_alpha() == 1.0

    def test_learned_reaches_markov_and_cem(self, build):
        f = build(markov_order="0,1", mc_cem=True, dirichlet_alpha=AlphaMode.LEARNED)
        assert all(c.alpha_mode is AlphaMode.LEARNED for c in f.markov.chains.values())
        assert f.mc._cem_dirichlet_concentration > 0

    def test_learned_single_chain(self, build):
        f = build(dirichlet_alpha=AlphaMode.LEARNED)
        assert f.markov.alpha_mode is AlphaMode.LEARNED

    def test_cli_passes_enum(self, monkeypatch):
        assert _parse(monkeypatch).dirichlet_alpha == AlphaMode.FIXED.value
        assert _parse(monkeypatch, "--dirichlet-alpha", "learned").dirichlet_alpha == "learned"

    def test_cmd_fuzz_forwards(self):
        calls = _fuzzer_call_kwargs()
        assert calls
        assert all("dirichlet_alpha" in k for k in calls)

    def test_adversarial_cli_rejects_unknown(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse(monkeypatch, "--dirichlet-alpha", "bogus")

    def test_hail_mary_learns(self, monkeypatch):
        assert _parse(monkeypatch, "--hail-mary").dirichlet_alpha == AlphaMode.LEARNED.value

    def test_banner(self, capsys):
        Fuzzer._print_enabled_features(_make_fake_fuzzer())
        assert "dirichlet-alpha" not in capsys.readouterr().out
        Fuzzer._print_enabled_features(_make_fake_fuzzer(_dirichlet_alpha=AlphaMode.LEARNED))
        assert "dirichlet-alpha=learned" in capsys.readouterr().out


class TestDictThompson:
    TOKENS = [b"GET", b"POST", b"HEAD"]

    def test_off_by_default(self, build):
        f = build(dictionary=list(self.TOKENS))
        assert f._dict_picker is None

    def test_mutate_draws_from_picker(self, build):
        """The round's scratch indices are exactly the picker's pending draw."""
        f = build(dictionary=list(self.TOKENS), dict_thompson=True)
        f.mutate(b"GET / HTTP/1.1\r\n\r\n")
        assert f._dict_scratch
        assert f._dict_scratch is f._dict_picker._pending
        assert f._dict_picker._drawn_from is f._operators.ctx.dictionary

    def test_cli_and_hail_mary(self, monkeypatch):
        assert _parse(monkeypatch).dict_thompson is False
        assert _parse(monkeypatch, "--dict-thompson").dict_thompson is True
        assert _parse(monkeypatch, "--hail-mary").dict_thompson is True
        assert "dict_thompson" in commands._HAIL_MARY_FLAGS

    def test_cmd_fuzz_forwards(self):
        assert all("dict_thompson" in k for k in _fuzzer_call_kwargs())

    def test_banner(self, capsys, build):
        Fuzzer._print_enabled_features(_make_fake_fuzzer())
        assert "dict-thompson" not in capsys.readouterr().out
        f = build(dict_thompson=True)
        Fuzzer._print_enabled_features(_make_fake_fuzzer(_dict_picker=f._dict_picker))
        assert "dict-thompson" in capsys.readouterr().out


requires_test_target = pytest.mark.skipif(
    not Path(_TARGET).exists(), reason="targets/test_target not built"
)


@requires_test_target
def test_fuzz_one_rewards_iff_new_coverage(build):
    """Every round settles the picker: reward on new coverage, clear otherwise.

    The coverage verdict is scripted (Hard Rule 39) so both branches run.
    """
    f = build(dictionary=[b"GET", b"POST", b"HEAD"], dict_thompson=True, use_coverage=True)
    picker = f._dict_picker
    script = [True, False, False, True, False, True]
    verdicts = iter(script)
    events: list[str] = []
    real_confirm, real_reward, real_clear = f._confirm_new_coverage, picker.reward, picker.clear

    def confirm(*a, **k):
        return next(verdicts), real_confirm(*a, **k)[1]

    def reward(n_used):
        events.append("reward")
        real_reward(n_used)

    def clear():
        events.append("clear")
        real_clear()

    f._confirm_new_coverage = confirm
    picker.reward = reward
    picker.clear = clear
    for i in range(len(script)):
        f.fuzz_one(b"GET /" + bytes([65 + i]) * 8)

    # reward() ends in clear(); drop that inner call to get one event per round
    settled = [e for i, e in enumerate(events) if not (e == "clear" and events[i - 1] == "reward")]
    assert settled == ["reward" if v else "clear" for v in script]
