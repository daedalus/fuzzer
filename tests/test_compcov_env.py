"""Tests for CmplogCollector's compcov_level -> $__AFL_COMPCOV_LEVEL wiring.

The shim-level behavior (partial matches actually landing in the edge
table) is covered by tests/test_compcov.py. This file covers only the
Python-side contract: the env var is set/restored with the same
reversible-mutation discipline as _CMPLOG_OUT (see
test_regression_env_leak.py, which this mirrors), and is left alone
entirely when compcov_level is 0.
"""

from __future__ import annotations

import os

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector


@pytest.fixture
def collector_off(tmp_path):
    c = CmplogCollector(workdir=str(tmp_path))
    c._shim_path = str(tmp_path / "fuzz_cmplog_shim.so")
    return c


@pytest.fixture
def collector_level2(tmp_path):
    c = CmplogCollector(workdir=str(tmp_path), compcov_level=2)
    c._shim_path = str(tmp_path / "fuzz_cmplog_shim.so")
    return c


class TestCompcovLevelClamping:
    def test_negative_clamps_to_zero(self, tmp_path):
        assert CmplogCollector(workdir=str(tmp_path), compcov_level=-3).compcov_level == 0

    def test_above_two_clamps_to_two(self, tmp_path):
        assert CmplogCollector(workdir=str(tmp_path), compcov_level=9).compcov_level == 2

    def test_valid_levels_pass_through(self, tmp_path):
        assert CmplogCollector(workdir=str(tmp_path), compcov_level=1).compcov_level == 1
        assert CmplogCollector(workdir=str(tmp_path), compcov_level=2).compcov_level == 2

    def test_default_is_off(self, tmp_path):
        assert CmplogCollector(workdir=str(tmp_path)).compcov_level == 0


class TestCompcovEnvSetupEnv:
    """setup_env() -- the dict-returning path used by subprocess exec."""

    def test_default_off_never_sets_the_env_var(self, collector_off):
        env = collector_off.setup_env({})
        assert "__AFL_COMPCOV_LEVEL" not in env

    def test_level_set_appears_in_returned_env(self, collector_level2):
        env = collector_level2.setup_env({})
        assert env["__AFL_COMPCOV_LEVEL"] == "2"

    def test_does_not_mutate_caller_dict_or_os_environ(self, collector_level2, monkeypatch):
        monkeypatch.delenv("__AFL_COMPCOV_LEVEL", raising=False)
        caller_env = {"UNRELATED": "1"}
        result = collector_level2.setup_env(caller_env)
        assert "__AFL_COMPCOV_LEVEL" not in caller_env, "setup_env must copy, not mutate in place"
        assert "__AFL_COMPCOV_LEVEL" not in os.environ
        assert result["__AFL_COMPCOV_LEVEL"] == "2"


class TestCompcovEnvSetupEnvForRun:
    """setup_env_for_run()/restore_env() -- the os.environ path used by
    inprocess/persistent execution, with the same reversible-mutation
    contract test_regression_env_leak.py pins for _CMPLOG_OUT."""

    def test_off_leaves_var_untouched(self, collector_off, monkeypatch):
        monkeypatch.delenv("__AFL_COMPCOV_LEVEL", raising=False)
        collector_off.setup_env_for_run()
        assert "__AFL_COMPCOV_LEVEL" not in os.environ
        collector_off.restore_env()
        assert "__AFL_COMPCOV_LEVEL" not in os.environ

    def test_level_set_then_restored(self, collector_level2, monkeypatch):
        monkeypatch.delenv("__AFL_COMPCOV_LEVEL", raising=False)
        collector_level2.setup_env_for_run()
        assert os.environ["__AFL_COMPCOV_LEVEL"] == "2"
        collector_level2.restore_env()
        assert "__AFL_COMPCOV_LEVEL" not in os.environ, "absent key must be deleted, not emptied"

    def test_preexisting_value_is_preserved_across_restore(self, collector_off, monkeypatch):
        """A value we did not set (compcov_level=0) must survive us untouched."""
        monkeypatch.setenv("__AFL_COMPCOV_LEVEL", "1")
        collector_off.setup_env_for_run()
        assert os.environ["__AFL_COMPCOV_LEVEL"] == "1"
        collector_off.restore_env()
        assert os.environ["__AFL_COMPCOV_LEVEL"] == "1"

    def test_repeated_setup_does_not_capture_our_own_mutation(self, collector_level2, monkeypatch):
        monkeypatch.delenv("__AFL_COMPCOV_LEVEL", raising=False)
        for _ in range(5):
            collector_level2.setup_env_for_run()
        collector_level2.restore_env()
        assert "__AFL_COMPCOV_LEVEL" not in os.environ
