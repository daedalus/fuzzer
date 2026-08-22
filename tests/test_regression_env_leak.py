"""Regression tests for bug report 2026-08-21, E1 and E2.

E2 is the expensive one and the reason these are worth pinning. Its symptom is
not an exception, it is a *quiet* wrong answer: an ASAN target whose subprocess
inherits a leaked cmplog LD_PRELOAD executes at full speed, reports healthy
throughput, and finds zero crashes, because the shim resolves the coverage
symbols the ASAN runtime wanted. The bug report reached it by bisecting a suite
after the same two tests passed in isolation and failed in a full run.

Three separate defects, each with its own falsification:

  * ``_clean_env({})`` returned the full parent environment. An explicitly
    empty dict is falsy, so ``env or os.environ`` silently substituted the
    caller's opposite intent.
  * ``setup_env_for_run()`` mutated ``os.environ`` with no way to undo it.
  * ``pytest`` had no timeout, so a native-code hang wedged the run forever.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.adapters.process import _clean_env
from fuzzer_tool.core.cmplog import CmplogCollector

TESTS_DIR = Path(__file__).parent


def _conftest():
    """Load ``tests/conftest.py`` by path, as test_seed_discipline.py does.

    pytest imports conftest under an internal name that ``import conftest``
    cannot reach under the default import mode, and the hook is what is under
    test here.
    """
    spec = importlib.util.spec_from_file_location("_conftest_uut", TESTS_DIR / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── E2a: _clean_env must not confuse "empty" with "unset" ──────────────────


class TestCleanEnvEmptyDict:
    def test_empty_dict_yields_empty_env(self, monkeypatch):
        """An empty dict means an empty environment, not the parent's."""
        monkeypatch.setenv("LD_PRELOAD", "/tmp/should_not_appear.so")
        monkeypatch.setenv("SOME_UNRELATED_KEY", "leaked")
        result = _clean_env({})
        assert result == {}

    def test_none_still_inherits_parent(self, monkeypatch):
        """Falsification: the fix must not break the documented None case."""
        monkeypatch.setenv("_CLEAN_ENV_PROBE", "present")
        # _clean_env caches the env=None result; clear it so this test sees
        # the value it just set rather than a snapshot from an earlier test.
        import fuzzer_tool.adapters.process as proc

        monkeypatch.setattr(proc, "_clean_env_cache", None)
        assert proc._clean_env(None).get("_CLEAN_ENV_PROBE") == "present"

    def test_explicit_dict_is_not_merged_with_parent(self, monkeypatch):
        """Adversarial: a caller's dict is authoritative, not a set of overrides."""
        monkeypatch.setenv("PARENT_ONLY", "1")
        result = _clean_env({"CHILD_ONLY": "1"})
        assert "PARENT_ONLY" not in result
        assert result["CHILD_ONLY"] == "1"

    def test_ksm_preload_still_stripped_from_explicit_dict(self):
        """The function's original job still works through the new path."""
        result = _clean_env({"LD_PRELOAD": "/lib/ksm_preload.so:/lib/keep.so"})
        assert result["LD_PRELOAD"] == "/lib/keep.so"


# ── E2b: cmplog env mutations must be reversible ───────────────────────────


@pytest.fixture
def collector(tmp_path):
    c = CmplogCollector(workdir=str(tmp_path))
    c._shim_path = str(tmp_path / "fuzz_cmplog_shim.so")
    return c


class TestCmplogEnvRestore:
    def test_setup_then_restore_is_identity(self, collector, monkeypatch):
        monkeypatch.delenv("LD_PRELOAD", raising=False)
        monkeypatch.delenv("_CMPLOG_OUT", raising=False)

        collector.setup_env_for_run()
        assert collector._shim_path in os.environ["LD_PRELOAD"]
        assert "_CMPLOG_OUT" in os.environ

        collector.restore_env()
        assert "LD_PRELOAD" not in os.environ, "absent key must be deleted, not emptied"
        assert "_CMPLOG_OUT" not in os.environ

    def test_preexisting_preload_is_preserved(self, collector, monkeypatch):
        """A preload we did not set must survive us untouched."""
        monkeypatch.setenv("LD_PRELOAD", "/lib/libasan.so.8")
        collector.setup_env_for_run()
        assert "/lib/libasan.so.8" in os.environ["LD_PRELOAD"]
        collector.restore_env()
        assert os.environ["LD_PRELOAD"] == "/lib/libasan.so.8"

    def test_repeated_setup_does_not_capture_our_own_mutation(self, collector, monkeypatch):
        """The bug this guards: run_target calls setup before *every* exec.

        Snapshotting on each call would capture the LD_PRELOAD we set on the
        previous call, and restore would then be a no-op that leaves the shim
        in place forever -- the original defect, reintroduced through the fix.
        """
        monkeypatch.delenv("LD_PRELOAD", raising=False)
        for _ in range(5):
            collector.setup_env_for_run()
        collector.restore_env()
        assert "LD_PRELOAD" not in os.environ

    def test_restore_without_setup_is_a_noop(self, collector, monkeypatch):
        monkeypatch.setenv("LD_PRELOAD", "/lib/untouched.so")
        collector.restore_env()
        assert os.environ["LD_PRELOAD"] == "/lib/untouched.so"

    def test_restore_is_idempotent(self, collector, monkeypatch):
        monkeypatch.delenv("LD_PRELOAD", raising=False)
        collector.setup_env_for_run()
        collector.restore_env()
        os.environ["LD_PRELOAD"] = "/lib/set_by_someone_else.so"
        collector.restore_env()
        assert os.environ["LD_PRELOAD"] == "/lib/set_by_someone_else.so"

    def test_stop_restores_env(self, collector, monkeypatch):
        monkeypatch.delenv("LD_PRELOAD", raising=False)
        collector.setup_env_for_run()
        collector.stop()
        assert "LD_PRELOAD" not in os.environ

    def test_subprocess_does_not_inherit_shim_after_restore(self, collector, monkeypatch):
        """End to end, at the layer that actually broke: a child process.

        The unit assertions above check os.environ in-process. This checks the
        thing that mattered -- that an exec'd child no longer sees the shim --
        because os.environ writes reach a child through putenv, and a fix that
        only satisfied the Python-level dict would not have helped ASAN.
        """
        monkeypatch.delenv("LD_PRELOAD", raising=False)
        collector.setup_env_for_run()
        collector.restore_env()
        out = subprocess.run(
            [sys.executable, "-c", "import os; print(os.environ.get('LD_PRELOAD', '<unset>'))"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert out.stdout.strip() == "<unset>"


# ── E1: the suite must be bounded ──────────────────────────────────────────


def test_regression_pytest_timeout_is_configured(request):
    """A bare `pytest` must not be able to hang forever.

    Skipped rather than failed when the plugin is absent: conftest applies the
    default defensively so a dev env without pytest-timeout still runs, and a
    test that fails on a missing optional dev dep teaches people to ignore it.
    """
    if not request.config.pluginmanager.hasplugin("timeout"):
        pytest.skip("pytest-timeout not installed (declared in the 'dev' extra)")
    assert request.config.getoption("timeout"), "no default per-test timeout configured"


def test_regression_default_timeout_method_spawns_no_thread(request):
    """The session default must not make the pytest process multi-threaded.

    The obvious fix for the hang -- ``--timeout-method=thread`` globally -- is
    the wrong one here, and this pins the reason. That method arms a
    threading.Timer per test, and this suite forks from the pytest process in
    at least three places. fork() from a multi-threaded process is the hazard
    behind docs/handover/test_shm_hang_2026-08-14.md, so buying a bound on
    native hangs by making every fork riskier is a net loss.

    Measured, on a test that forks and is otherwise silent: under the thread
    method CPython emits its multi-threaded-fork DeprecationWarning; under the
    signal method it does not.
    """
    if not request.config.pluginmanager.hasplugin("timeout"):
        pytest.skip("pytest-timeout not installed (declared in the 'dev' extra)")
    assert request.config.getoption("timeout_method") == "signal", (
        "session default must be thread-free; native-hang modules opt into "
        "the thread method individually via pytest_collection_modifyitems"
    )


def test_regression_native_hang_modules_opt_into_thread_method():
    """The Z3 modules must still get the method that can actually stop them.

    Falsification for the test above: if the signal default were applied
    uniformly, the case that motivated the timeout in the first place -- >9 min
    inside solver.add(), uninterruptible by SIGTERM -- would be unbounded
    again, and the suite would look fixed while being exactly as wedgeable.
    """
    conftest = _conftest()
    assert "test_structural_constraints" in conftest._NATIVE_HANG_MODULES

    marked: list[str] = []

    class _Item:
        def __init__(self, nodeid: str) -> None:
            self.nodeid = nodeid

        def add_marker(self, mark) -> None:
            marked.append((self.nodeid, mark.kwargs.get("method")))

    class _PM:
        def hasplugin(self, _name):
            return True

    class _Config:
        pluginmanager = _PM()

    items = [
        _Item("tests/test_structural_constraints.py::test_a"),
        _Item("tests/test_operator_smoke.py::test_b"),
    ]
    conftest.pytest_collection_modifyitems(_Config(), items)
    assert marked == [("tests/test_structural_constraints.py::test_a", "thread")], (
        "exactly the native-hang modules get the thread method, and nothing else"
    )
