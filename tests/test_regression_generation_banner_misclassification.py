"""Regression test: `_print_enabled_features` must not double-list
`mcts`/`alphabeta` under both "Seed selection" and "Generation", and
`bootstrap` (bootstrap-percolation corpus *minimization*, see
core/percolation.py::bootstrap_minimize_corpus) must be reported under
"Analysis", not "Generation" -- it removes redundant seeds, it does not
generate new inputs.

Context: `mcts` (MCTSSeedScheduler) and `alphabeta`
(AlphaBetaMCTSSeedScheduler, core/schedulers/seed_mcts.py) both pick which
*existing* corpus seed to fuzz next via search over the lineage tree; they
were correctly appended to "Seed selection" but were also being appended a
second time to "Generation", which is not what either arm does.
"""

from __future__ import annotations

from types import SimpleNamespace

from fuzzer_tool.services.fuzzer import Fuzzer


def _make_fake_fuzzer(**overrides):
    """Minimal stand-in exposing every attribute `_print_enabled_features`
    reads unconditionally, all falsy/default, so the method runs without
    AttributeErrors and prints nothing unless overridden.
    """
    base = dict(
        _adaptive_timeout=False,
        _calibrate=0,
        _corpus_boost=0,
        _diff_target=None,
        _exp4=False,
        _format_learner=None,
        _inprocess_runner=False,
        _successive_elim=False,
        _tracer=None,
        branch_cov=False,
        colorize=False,
        debug=False,
        enable_arm_mutator=False,
        enable_regex_bomb=False,
        enable_x86_mutator=False,
        ga=False,
        honggfuzz=False,
        hw_perf=False,
        markov_generate=False,
        markov_trained=False,
        mc_bandit=False,
        mc_cem=False,
        persistent=False,
        pt_cov=False,
        qea=False,
        weizz_tags=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _lines(out: str) -> dict[str, str]:
    return {
        line.split(":", 1)[0].strip(): line
        for line in out.splitlines()
        if ":" in line
    }


def test_mcts_appears_only_under_seed_selection(capsys):
    fake = _make_fake_fuzzer(_use_mcts=True)
    Fuzzer._print_enabled_features(fake)
    lines = _lines(capsys.readouterr().out)
    assert "mcts" in lines.get("Seed selection", "")
    assert "mcts" not in lines.get("Generation", "")


def test_alphabeta_appears_only_under_seed_selection(capsys):
    fake = _make_fake_fuzzer(_use_alphabeta=True)
    Fuzzer._print_enabled_features(fake)
    lines = _lines(capsys.readouterr().out)
    assert "alphabeta" in lines.get("Seed selection", "")
    assert "alphabeta" not in lines.get("Generation", "")


def test_bootstrap_appears_under_analysis_not_generation(capsys):
    fake = _make_fake_fuzzer(_use_bootstrap=True)
    Fuzzer._print_enabled_features(fake)
    lines = _lines(capsys.readouterr().out)
    assert "bootstrap" in lines.get("Analysis", "")
    assert "bootstrap" not in lines.get("Generation", "")


def test_wfc_and_corpus_boost_remain_under_generation(capsys):
    fake = _make_fake_fuzzer(_wfc_enabled=True, _corpus_boost=4)
    Fuzzer._print_enabled_features(fake)
    lines = _lines(capsys.readouterr().out)
    assert "wfc" in lines.get("Generation", "")
    assert "corpus-boost=4" in lines.get("Generation", "")
