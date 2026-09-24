"""P3-3 step 6: ``last_picked`` on seed_meta and the LST revisit override.

``lst_pick`` (core/job_scheduling.py) returns the job with the least slack
``due - now - p`` once any slack is negative. ``SeedPicker._pick_lst_seed``
feeds it the corpus with ``due = max(last_picked | added_at, start_time) + D``
so no seed waits longer than ``--lst-revisit D`` seconds between visits.
"""

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.job_scheduling import Job, lst_pick

SEED_A = b"\x01" * 4
SEED_B = b"\x02" * 4
SEED_C = b"\x03" * 4
START = 1000.0
REVISIT = 10.0
BUDGET = 8


# --------------------------------------------------------------------------
# lst_pick
# --------------------------------------------------------------------------


def test_lst_pick_none_when_no_job_is_late():
    jobs = [Job("a", 1.0, due_date=20.0), Job("b", 2.0, due_date=30.0)]
    assert lst_pick(jobs, now=10.0) is None


def test_lst_pick_returns_least_slack():
    # slack a = 12 - 10 - 1 = 1, b = 11 - 10 - 3 = -2, c = 9 - 10 - 0 = -1
    jobs = [Job("a", 1.0, due_date=12.0), Job("b", 3.0, due_date=11.0), Job("c", 0.0, due_date=9.0)]
    assert lst_pick(jobs, now=10.0) == "b"


def test_lst_pick_falsification_edd_would_disagree():
    """EDD picks the earliest due date; LST must not.

    c is due first but is short, b is due later but long: b has less slack.
    If lst_pick were EDD in disguise this returns "c".
    """
    jobs = [Job("c", 0.1, due_date=9.0), Job("b", 5.0, due_date=9.5)]
    assert min(jobs, key=lambda j: j.due_date).id == "c"
    assert lst_pick(jobs, now=10.0) == "b"


def test_lst_pick_zero_slack_is_not_late():
    assert lst_pick([Job("a", 2.0, due_date=12.0)], now=10.0) is None


def test_lst_pick_adversarial_empty_and_infinite_due():
    assert lst_pick([], now=0.0) is None
    assert lst_pick([Job("a", 1e9)], now=1e18) is None


def test_lst_pick_ties_keep_first():
    jobs = [Job("x", 1.0, due_date=5.0), Job("y", 1.0, due_date=5.0)]
    assert lst_pick(jobs, now=10.0) == "x"


def test_lst_pick_rejects_negative_processing_time():
    with pytest.raises(ValueError):
        lst_pick([Job("a", -1.0, due_date=5.0)], now=10.0)


# --------------------------------------------------------------------------
# SeedPicker wiring
# --------------------------------------------------------------------------


def _fuzzer(revisit=REVISIT, metas=None):
    metas = metas if metas is not None else {}
    corpus = list(metas) or [SEED_A, SEED_B, SEED_C]
    f = SimpleNamespace(
        corpus=corpus,
        seed_meta={s: metas.get(s, {"added_at": START}) for s in corpus},
        start_time=START,
        mutations_per_input=BUDGET,
        _lst_revisit=revisit,
        _seed_strategy=None,
    )
    f.mean_exec_time = lambda: 0.0
    return f


def _picker(f):
    from fuzzer_tool.services.seed_picker import SeedPicker

    sp = SeedPicker.__new__(SeedPicker)
    sp.f = f
    return sp


def test_picker_disabled_by_default():
    f = _fuzzer(revisit=0.0)
    assert _picker(f)._pick_lst_seed(now=START + 1e6) is None


def test_picker_idle_before_revisit_bound():
    f = _fuzzer()
    assert _picker(f)._pick_lst_seed(now=START + REVISIT - 1.0) is None


def test_picker_returns_longest_waiting_seed():
    metas = {
        SEED_A: {"added_at": START, "last_picked": START + 5.0},
        SEED_B: {"added_at": START, "last_picked": START + 1.0},
        SEED_C: {"added_at": START, "last_picked": START + 3.0},
    }
    f = _fuzzer(metas=metas)
    assert _picker(f)._pick_lst_seed(now=START + REVISIT + 4.0) == SEED_B
    assert f._seed_strategy == "lst"


def test_picker_slow_seed_wins_on_slack():
    """Same wait, different cost: the seed that takes longer to fuzz is
    closer to breaching the bound, so it goes first."""
    metas = {
        SEED_A: {"added_at": START, "total_time": 0.001, "cost_samples": 1},
        SEED_B: {"added_at": START, "total_time": 0.5, "cost_samples": 1},
    }
    f = _fuzzer(metas=metas)
    now = START + REVISIT - 0.5 * BUDGET + 0.1
    assert _picker(f)._pick_lst_seed(now=now) == SEED_B


def test_picker_admission_before_session_counts_from_start():
    """A resumed seed's added_at predates the session; clamp it to
    start_time so resume does not mark the whole corpus overdue."""
    metas = {SEED_A: {"added_at": START - 1e6}}
    f = _fuzzer(metas=metas)
    assert _picker(f)._pick_lst_seed(now=START + 1.0) is None


def test_picker_empty_corpus():
    f = _fuzzer()
    f.corpus = []
    assert _picker(f)._pick_lst_seed(now=START + 1e6) is None


def test_pick_seed_consults_lst_first(monkeypatch):
    from fuzzer_tool.services.seed_picker import SeedPicker

    metas = {SEED_A: {"added_at": START}}
    f = _fuzzer(metas=metas)
    f._stall_recovery_active = False
    f._rng = None
    sp = _picker(f)
    monkeypatch.setattr(SeedPicker, "_update_temperature", lambda self: 1.0)
    monkeypatch.setattr(SeedPicker, "_pick_seed_elo", lambda self: pytest.fail("LST must win"))
    monkeypatch.setattr("fuzzer_tool.services.seed_picker.time.time", lambda: START + 2 * REVISIT)
    assert sp.pick_seed() == SEED_A


def test_regression_fuzzer_pick_seed_stamps_last_picked(monkeypatch):
    from fuzzer_tool.services.fuzzer import Fuzzer

    meta = {"added_at": START}
    f = SimpleNamespace(
        seed_meta={SEED_A: meta},
        _seed_picker=SimpleNamespace(pick_seed=lambda: SEED_A),
    )
    monkeypatch.setattr("fuzzer_tool.services.fuzzer.time.time", lambda: START + 7.0)
    assert Fuzzer._pick_seed(f) == SEED_A
    assert meta["last_picked"] == START + 7.0


# --------------------------------------------------------------------------
# CLI / constructor wiring
# --------------------------------------------------------------------------


def test_cli_passes_lst_revisit_to_fuzzer():
    import ast
    import inspect

    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
    ]
    assert calls
    assert all("lst_revisit" in {k.arg for k in c.keywords} for c in calls)
    assert "lst_revisit" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))


def test_constructor_clamps_negative_revisit(tmp_path):
    from pathlib import Path

    from fuzzer_tool.services.fuzzer import Fuzzer

    target = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")
    (tmp_path / "corpus").mkdir()
    (tmp_path / "crashes").mkdir()
    kwargs = {"corpus_dir": str(tmp_path / "corpus"), "crashes_dir": str(tmp_path / "crashes")}
    assert Fuzzer(target=target, lst_revisit=-5.0, **kwargs)._lst_revisit == 0.0
    assert Fuzzer(target=target, lst_revisit=3.0, **kwargs)._lst_revisit == 3.0


# --------------------------------------------------------------------------
# least_slack kernel and the next-check gate
# --------------------------------------------------------------------------


def test_least_slack_matches_scalar_definition():
    import numpy as np

    from fuzzer_tool.core.job_scheduling import least_slack

    due = np.array([12.0, 11.0, 9.0])
    proc = np.array([1.0, 3.0, 0.0])
    idx, slack = least_slack(due, proc, now=10.0)
    expected = [d - 10.0 - p for d, p in zip(due, proc, strict=True)]
    assert idx == expected.index(min(expected))
    assert slack == min(expected)


def test_least_slack_empty():
    import numpy as np

    from fuzzer_tool.core.job_scheduling import least_slack

    idx, slack = least_slack(np.array([]), np.array([]), now=0.0)
    assert idx == -1
    assert slack == float("inf")


def test_gate_skips_scan_until_earliest_possible_lateness():
    """A None scan caches now + min_slack; nothing can go late before then."""
    metas = {SEED_A: {"added_at": START, "last_picked": START + 4.0}}
    f = _fuzzer(metas=metas)
    sp = _picker(f)
    assert sp._pick_lst_seed(now=START + 5.0) is None
    assert f._lst_next_check == START + 4.0 + REVISIT

    # Adversarial: poison meta so a scan would fire; the gate must hold it.
    metas[SEED_A]["last_picked"] = START - 1e6
    assert sp._pick_lst_seed(now=START + 4.0 + REVISIT - 0.5) is None
    assert sp._pick_lst_seed(now=START + 4.0 + REVISIT) == SEED_A
