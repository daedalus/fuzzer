"""Regression: setup_env_for_run rewrote unchanged env vars on every exec.

It runs before every execution and assigned 4+ ``os.environ`` keys each
time, each a ``putenv`` (fuzzgoat: 621k putenv per 154k execs, ~36 us/exec
under cProfile). Values only change when the log path rotates, so writes
are now skipped when ``os.environ`` already holds the value.
"""

import os

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector

_KEYS = ("_CMPLOG_OUT", "_CMPLOG_COUNTS", "_CMPLOG_SITE_COUNTS")


@pytest.fixture
def collector(tmp_path, monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    c = CmplogCollector(workdir=str(tmp_path), site_counts=True)
    yield c
    c.stop()


def _count_putenv(monkeypatch) -> dict:
    calls = {"n": 0}
    real = os.putenv

    def counting(k, v):
        calls["n"] += 1
        real(k, v)

    monkeypatch.setattr(os, "putenv", counting)
    return calls


def test_regression_cmplog_env_rewrite_skipped(collector, monkeypatch):
    """Falsification: a second call with unchanged values writes nothing."""
    collector.setup_env_for_run()
    calls = _count_putenv(monkeypatch)
    collector.setup_env_for_run()
    assert calls["n"] == 0
    assert os.environ["_CMPLOG_OUT"] == collector.log_path


def test_external_change_is_repaired(collector, monkeypatch):
    """Adversarial: a key changed behind our back is written again."""
    collector.setup_env_for_run()
    os.environ["_CMPLOG_OUT"] = "/elsewhere"
    calls = _count_putenv(monkeypatch)
    collector.setup_env_for_run()
    assert calls["n"] == 1
    assert os.environ["_CMPLOG_OUT"] == collector.log_path


def test_restore_then_setup_rewrites(collector):
    """Adversarial: after restore_env removed the keys, setup puts them back."""
    collector.setup_env_for_run()
    collector.restore_env()
    assert "_CMPLOG_OUT" not in os.environ
    collector.setup_env_for_run()
    assert os.environ["_CMPLOG_OUT"] == collector.log_path
    assert os.environ["_CMPLOG_COUNTS"] == collector.counts_path
