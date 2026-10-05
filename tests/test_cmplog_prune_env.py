"""CmplogCollector prune_const -> $__AFL_CMP_PRUNE_CONST wiring.

Shim behaviour is covered by tests/test_cmplog_const_prune.py; this pins the
Python contract, mirroring tests/test_compcov_env.py: set and restored with
the same reversible discipline as _CMPLOG_OUT, untouched when off.
"""

from __future__ import annotations

import os

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector

_VAR = "__AFL_CMP_PRUNE_CONST"


def _collector(tmp_path, **kw) -> CmplogCollector:
    c = CmplogCollector(workdir=str(tmp_path), **kw)
    c._shim_path = str(tmp_path / "fuzz_cmplog_shim.so")
    return c


class TestPruneEnvSetupEnv:
    def test_default_off_never_sets_the_var(self, tmp_path):
        assert _VAR not in _collector(tmp_path).setup_env({})

    def test_on_appears_in_returned_env(self, tmp_path):
        assert _collector(tmp_path, prune_const=True).setup_env({})[_VAR] == "1"

    def test_does_not_mutate_caller_or_os_environ(self, tmp_path, monkeypatch):
        monkeypatch.delenv(_VAR, raising=False)
        caller = {"UNRELATED": "1"}
        out = _collector(tmp_path, prune_const=True).setup_env(caller)
        assert _VAR not in caller
        assert _VAR not in os.environ
        assert out[_VAR] == "1"


class TestPruneEnvForRun:
    def test_off_leaves_var_untouched(self, tmp_path, monkeypatch):
        monkeypatch.delenv(_VAR, raising=False)
        c = _collector(tmp_path)
        c.setup_env_for_run()
        assert _VAR not in os.environ
        c.restore_env()
        assert _VAR not in os.environ

    def test_on_then_restored_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv(_VAR, raising=False)
        c = _collector(tmp_path, prune_const=True)
        c.setup_env_for_run()
        assert os.environ[_VAR] == "1"
        c.restore_env()
        assert _VAR not in os.environ

    def test_preexisting_value_survives_restore(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_VAR, "0")
        c = _collector(tmp_path, prune_const=True)
        c.setup_env_for_run()
        c.restore_env()
        assert os.environ[_VAR] == "0"


@pytest.mark.parametrize("val", [0, "", None])
def test_falsy_values_are_off(tmp_path, val):
    assert _VAR not in _collector(tmp_path, prune_const=val).setup_env({})
