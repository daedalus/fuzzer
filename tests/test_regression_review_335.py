"""Regressions for review findings on PR #335."""

import os
import signal
import sys
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters import persistent_subprocess
from fuzzer_tool.adapters.filesystem import (
    hash_data,
    load_corpus,
    rehydrate_by_hash,
    save_to_corpus,
)
from fuzzer_tool.core.running_stats import RunningMoments
from fuzzer_tool.services.runner import TargetRunner
from tests.test_regression_persistent_runner import _CRASH, _OK, exe, runner  # noqa: F401

_PARENT = b"parentparent"
_DELTA_HASH = "00000000000000aa"
# Past json's C recursion limit on every supported interpreter.
_JSON_DEPTH = 200_000


class TestPersistentPid:
    """A reaped pid was kept: stop() could SIGKILL whoever reused it."""

    def test_regression_crash_forgets_the_pid(self, runner):  # noqa: F811
        assert runner.run_one(_CRASH)[0] == -signal.SIGABRT

        assert runner.pid is None
        assert not runner.alive

    def test_regression_vanished_child_is_restarted(self, runner):  # noqa: F811
        """Killed between inputs: the next exec must restart, not report -2 forever."""
        os.kill(runner.pid, signal.SIGKILL)
        os.waitpid(runner.pid, 0)
        run = TargetRunner(
            SimpleNamespace(_persistent_runner=runner, _perf_counters=None)
        )._run_other

        rcs = [run(_OK, None)[0] for _ in range(3)]

        assert rcs[1:] == [0, 0]

    def test_live_child_keeps_its_pid(self, runner):  # noqa: F811
        """Falsification: an ordinary iteration leaves the same child parked."""
        pid = runner.pid
        assert runner.run_one(_OK)[0] == 0

        assert runner.pid == pid


class TestDeepJsonDelta:
    """``json.loads`` raises ``RecursionError`` on deep nesting: not in the skip list."""

    def test_regression_nested_delta_is_skipped(self, tmp_path):
        save_to_corpus(_PARENT, tmp_path, set())
        deltas = tmp_path / "deltas"
        deltas.mkdir(exist_ok=True)
        (deltas / f"delta_{_DELTA_HASH}.json").write_bytes(b"[" * _JSON_DEPTH + b"]" * _JSON_DEPTH)

        loaded, _seen, _irr = load_corpus(tmp_path, add_default=False)

        assert loaded == [_PARENT]
        assert rehydrate_by_hash(_DELTA_HASH, tmp_path) is None
        assert rehydrate_by_hash(hash_data(_PARENT), tmp_path) == _PARENT


class TestLegacyMomentsBuffer:
    """A state saved before the buffer fix kept its samples through load and save."""

    def test_regression_unwindowed_load_drops_samples(self):
        legacy = {
            "n": 3,
            "mean": 2.0,
            "m2": 2.0,
            "m3": 0.0,
            "m4": 2.0,
            "window": None,
            "buf": [1.0, 2.0, 3.0],
        }
        rm = RunningMoments()
        rm.load(legacy)

        assert rm.save()["buf"] == []
        assert rm.mean == pytest.approx(2.0)

    def test_windowed_load_keeps_samples(self):
        """Falsification: the window still needs its samples back."""
        src = RunningMoments(window=4)
        for x in (1.0, 2.0, 3.0):
            src.update(x)
        rm = RunningMoments(window=4)
        rm.load(src.save())

        assert rm.save()["buf"] == [1.0, 2.0, 3.0]


class TestTminGrammarStdout:
    """Grammar-load progress was printed before stdout was redirected."""

    def test_regression_grammar_lines_stay_off_stdout(self, tmp_path, monkeypatch, capfdbinary):
        from fuzzer_tool.cli.commands import main

        grammar = tmp_path / "g.txt"
        grammar.write_text('root = "x"\n')
        minimized = b"\x00MIN\xff"
        monkeypatch.setattr("fuzzer_tool.services.tmin.tmin", lambda **_kw: minimized)
        argv = ["fuzzer-tool", "tmin", "/bin/true", str(tmp_path / "c.bin"), "-g", str(grammar)]
        monkeypatch.setattr(sys, "argv", argv)

        assert main() == 0
        out, err = capfdbinary.readouterr()
        assert out == minimized
        assert b"Grammar loaded" in err


class TestLoaderFailedInit:
    """A loader that answered INIT with something else stayed alive and unready."""

    def test_regression_non_ready_reply_stops_the_loader(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            persistent_subprocess,
            "_PERSISTENT_LOADER",
            "import sys\nsys.stdin.readline()\nprint('NOPE', flush=True)\nsys.stdin.read()\n",
        )
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        loader = persistent_subprocess.PersistentLoader("/bin/true", "f", timeout=1)

        assert loader.start() is False
        assert loader._proc is None
        assert list(tmp_path.iterdir()) == []
