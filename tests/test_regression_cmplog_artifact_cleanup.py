"""Regression: cmplog run artifacts must not accumulate across runs.

``CmplogCollector.stop()`` unlinks the ``.cmplog``/``.counts``/``.sites``
trio for its own run, and ``_cleanup_stale_cmplog_files()`` exists to sweep
what a killed run left behind. Neither was called from anywhere in the
package -- only from tests -- so every run minted a fresh uuid-named trio
under ``~/.cache/fuzzer_cmplog`` and nothing ever removed it.

The sweep is age-gated on purpose: parallel workers are separate processes
sharing that directory, and the filenames carry a uuid rather than a pid, so
an unconditional sweep at startup would delete the log a concurrent worker is
writing to.
"""

import os
import time

import pytest

from fuzzer_tool.core import cmplog as cmplog_mod
from fuzzer_tool.core.cmplog import (
    _CMPLOG_STALE_AGE_S,
    _cleanup_stale_cmplog_files,
)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "fuzzer_cmplog"
    d.mkdir()
    monkeypatch.setattr(cmplog_mod, "_CMPLOG_DIR", str(d))
    return d


def _write(dir_path, name: str, age_s: float = 0.0):
    p = dir_path / name
    p.write_text("x")
    if age_s:
        old = time.time() - age_s
        os.utime(p, (old, old))
    return p


class TestStaleSweep:
    def test_removes_all_three_artifact_kinds(self, cache_dir):
        old = _CMPLOG_STALE_AGE_S + 60
        for suffix in (".cmplog", ".counts", ".sites"):
            _write(cache_dir, f"fuzz_cmplog_deadbeef{suffix}", age_s=old)
        assert _cleanup_stale_cmplog_files() == 3
        assert list(cache_dir.iterdir()) == []

    def test_sites_files_were_previously_missed(self, cache_dir):
        """The sweep matched only .cmplog and .counts, so the per-site
        counter files -- the largest of the three -- were never swept."""
        _write(cache_dir, "fuzz_cmplog_a.cmplog.sites", age_s=_CMPLOG_STALE_AGE_S + 60)
        assert _cleanup_stale_cmplog_files() == 1

    def test_fresh_files_survive(self, cache_dir):
        """Adversarial: a concurrent worker's live log must not be deleted.
        Its mtime is recent because the collector truncates on every
        setup_env() and the shim writes on every execution."""
        live = _write(cache_dir, "fuzz_cmplog_live.cmplog", age_s=0.0)
        assert _cleanup_stale_cmplog_files() == 0
        assert live.exists()

    def test_age_boundary_is_respected(self, cache_dir):
        _write(cache_dir, "fuzz_cmplog_recent.cmplog", age_s=_CMPLOG_STALE_AGE_S / 2)
        _write(cache_dir, "fuzz_cmplog_ancient.cmplog", age_s=_CMPLOG_STALE_AGE_S * 2)
        assert _cleanup_stale_cmplog_files() == 1
        assert (cache_dir / "fuzz_cmplog_recent.cmplog").exists()

    def test_shim_objects_are_left_alone(self, cache_dir):
        """The compiled shim is cached for reuse and pruned by digest
        elsewhere; the artifact sweep must not touch it."""
        _write(cache_dir, "fuzz_cmplog_shim.abc123.so", age_s=_CMPLOG_STALE_AGE_S * 10)
        assert _cleanup_stale_cmplog_files() == 0

    def test_missing_directory_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cmplog_mod, "_CMPLOG_DIR", str(tmp_path / "absent"))
        assert _cleanup_stale_cmplog_files() == 0

    def test_custom_age_argument(self, cache_dir):
        _write(cache_dir, "fuzz_cmplog_x.cmplog", age_s=10)
        assert _cleanup_stale_cmplog_files(max_age_s=3600) == 0
        assert _cleanup_stale_cmplog_files(max_age_s=1) == 1


class TestStopIsWiredIntoShutdown:
    def test_run_shutdown_calls_stop(self):
        """The tail of Fuzzer.run() must release the collector. Asserted on
        the source rather than by driving a full campaign: reaching this
        line needs a built target, and the defect was purely that no call
        site existed."""
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        src = inspect.getsource(Fuzzer.run)
        assert "self._cmplog.stop()" in src

    def test_stop_removes_the_files_it_owns(self, cache_dir):
        from fuzzer_tool.core.cmplog import CmplogCollector

        c = CmplogCollector()
        c.log_path = str(_write(cache_dir, "fuzz_cmplog_own.cmplog"))
        c.counts_path = str(_write(cache_dir, "fuzz_cmplog_own.counts"))
        c.sites_path = str(_write(cache_dir, "fuzz_cmplog_own.cmplog.sites"))
        c.stop()
        assert list(cache_dir.iterdir()) == []

    def test_stop_is_idempotent(self, cache_dir):
        from fuzzer_tool.core.cmplog import CmplogCollector

        c = CmplogCollector()
        c.log_path = str(_write(cache_dir, "fuzz_cmplog_own.cmplog"))
        c.stop()
        c.stop()
        assert list(cache_dir.iterdir()) == []
