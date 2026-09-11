"""Regression: ``fuzz -j N`` starts, and forwards options it does not name.

run_parallel and _worker_main had closed, hand-written signatures that
mirrored part of Fuzzer's. cmd_fuzz passed thirteen options they did not
list -- mc_cycle_detect, fpl, fpl_epsilon, invasion, garch, continuum,
poisson_disk_admission, poisson_disk_min_jaccard, sharpe_kelly_blend,
ducb_gamma, cucb_gamma, swucb_window -- unconditionally, with getattr
defaults, so every parallel campaign died with

    TypeError: run_parallel() got an unexpected keyword argument 'mc_cycle_detect'

before a worker started, whatever flags were given. The same failure was
fixed once before (C2 in test_regression_bugreport_critical, which keeps the
signatures closed on purpose) by adding the names it knew about; nothing
checked the next ones.

These tests check the whole chain from source instead of a list of names:
cmd_fuzz -> run_parallel's signature -> the worker_kwargs dict ->
_worker_main's signature -> the worker's Fuzzer(...) call.
"""

from __future__ import annotations

import ast
import inspect
import multiprocessing
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.services import parallel

_ROOT = Path(__file__).resolve().parent.parent
_TARGET = _ROOT / "targets" / "test_target"


def _run_parallel_keywords() -> set[str]:
    tree = ast.parse(inspect.getsource(commands))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "run_parallel"
    ]
    assert len(calls) == 1, "premise: cmd_fuzz has exactly one run_parallel(...) call"
    return {k.arg for k in calls[0].keywords if k.arg}


def test_run_parallel_accepts_everything_cmd_fuzz_passes():
    sig = inspect.signature(parallel.run_parallel)
    dummy = dict.fromkeys(_run_parallel_keywords(), None)
    sig.bind(**dummy)  # TypeError on the old closed signature


def _call_keywords(fn, callee: str) -> list[set[str]]:
    tree = ast.parse(inspect.getsource(fn))
    return [
        {k.arg for k in n.keywords if k.arg}
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == callee
    ]


def test_run_parallel_hands_every_worker_option_to_the_worker():
    worker_params = set(inspect.signature(parallel._worker_main).parameters)
    rp_params = set(inspect.signature(parallel.run_parallel).parameters)
    (dict_kw,) = [kw for kw in _call_keywords(parallel.run_parallel, "dict") if "target" in kw]
    dropped = sorted((rp_params & worker_params) - dict_kw)
    assert not dropped, f"run_parallel accepts but never gives the worker: {dropped}"


def test_worker_hands_every_fuzzer_option_to_fuzzer():
    from fuzzer_tool.services.fuzzer import Fuzzer

    fuzzer_params = set(inspect.signature(Fuzzer.__init__).parameters)
    worker_params = set(inspect.signature(parallel._worker_main).parameters)
    (call_kw,) = _call_keywords(parallel._worker_main, "Fuzzer")
    dropped = sorted((worker_params & fuzzer_params) - call_kw)
    assert not dropped, f"_worker_main accepts but never gives Fuzzer: {dropped}"


def test_worker_forwards_options_to_fuzzer(monkeypatch, tmp_path):
    captured = {}

    class _FakeFuzzer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.start_time = 1.0
            self.exec_count = self.crash_count = self.timeout_count = 0
            self.corpus = []
            self.shm_cov = None

        def _dump_stats(self):
            pass

        def _dump_coverage_report(self):
            pass

    monkeypatch.setattr("fuzzer_tool.services.fuzzer.Fuzzer", _FakeFuzzer)
    stop = multiprocessing.Event()
    stop.set()
    queue = SimpleNamespace(put=lambda item: None)
    named = {
        name: p.default
        for name, p in inspect.signature(parallel._worker_main).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    parallel._worker_main(
        worker_id=0,
        result_queue=queue,
        target="t",
        corpus_dir=str(tmp_path),
        crashes_dir=str(tmp_path),
        max_len=64,
        timeout=1,
        mutations_per_input=1,
        use_coverage=False,
        deep_coverage=False,
        max_bps=0,
        dictionary=[],
        file_mode=False,
        target_args=[],
        markov_order=0,
        markov_generate=False,
        mc_bandit=False,
        mc_cem=False,
        mc_elite_frac=0.1,
        mc_refit_interval=100,
        stats_file=None,
        stats_interval=0,
        coverage_report=None,
        iterations=0,
        sync_interval=10,
        stop_event=stop,
        **{k: v for k, v in named.items() if k not in ("rng_seed", "fpl", "mc_cycle_detect")},
        fpl=True,
        mc_cycle_detect=True,
    )
    assert captured.get("fpl") is True
    assert captured.get("mc_cycle_detect") is True


@pytest.mark.skipif(not _TARGET.exists(), reason="targets/test_target not built")
def test_fuzz_j2_runs_to_completion(tmp_path):
    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir()
    crashes.mkdir()
    (corpus / "seed").write_bytes(b"AAAA")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "fuzzer_tool",
            "fuzz",
            str(_TARGET),
            "--corpus",
            str(corpus),
            "--crashes",
            str(crashes),
            "-j",
            "2",
            "--iterations",
            "10",
        ],
        capture_output=True,
        text=True,
        timeout=12,
        cwd=_ROOT,
    )
    out = proc.stdout + proc.stderr
    assert "TypeError" not in out, out[-2000:]
    assert out.count("Done. execs=") == 2, out[-2000:]
