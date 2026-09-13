"""Tests for the --job-scheduler gate on Fuzzer (P3-3, steps 4-5 of 6).

Default (job_scheduler=False) must reproduce the pre-P3-3 maintenance-tick
gating exactly: independent i % 500 / 1000-exec-throttle checks, not
services.maintenance.MaintenanceQueue. Opting in with job_scheduler=True
switches maintenance-tick housekeeping to the queue.
"""

import tempfile
from unittest.mock import patch

from fuzzer_tool.services.fuzzer import Fuzzer


def _make_fuzzer(**kwargs):
    tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
    defaults = dict(
        target="/bin/true",
        corpus_dir=f"{tmpdir}/corpus",
        crashes_dir=f"{tmpdir}/crashes",
        max_len=256,
        timeout=1,
        mutations_per_input=2,
    )
    defaults.update(kwargs)
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(**defaults)


def test_default_is_legacy_mode():
    f = _make_fuzzer()
    assert f.job_scheduler is False


def test_job_scheduler_flag_is_stored():
    f = _make_fuzzer(job_scheduler=True)
    assert f.job_scheduler is True


def test_maintenance_queue_always_constructed_regardless_of_flag():
    # Constructed unconditionally (cheap, no I/O) -- only whether it is
    # *ticked* depends on the flag. Both modes get a usable queue object
    # so toggling the flag doesn't need to rebuild anything.
    legacy = _make_fuzzer(job_scheduler=False)
    scheduled = _make_fuzzer(job_scheduler=True)
    assert len(legacy._maintenance) == 4
    assert len(scheduled._maintenance) == 4


def test_legacy_memory_prune_tick_respects_threshold_off_switch():
    f = _make_fuzzer(job_scheduler=False, prune_corpus_max_memory=0)
    calls = []
    f._check_memory_and_prune = lambda: calls.append(1)
    f.exec_count = 5000
    f._legacy_memory_prune_tick()
    assert calls == []  # prune_corpus_max_memory <= 0 -> never runs


def test_legacy_memory_prune_tick_throttles_to_1000_execs():
    f = _make_fuzzer(job_scheduler=False, prune_corpus_max_memory=80)
    calls = []
    f._check_memory_and_prune = lambda: calls.append(f.exec_count)
    f.exec_count = 500
    f._legacy_memory_prune_tick()
    assert calls == []  # under the 1000-exec throttle
    f.exec_count = 1000
    f._legacy_memory_prune_tick()
    assert calls == [1000]


def test_maintenance_queue_memory_prune_job_active_only_when_configured():
    off = _make_fuzzer(job_scheduler=True, prune_corpus_max_memory=0)
    on = _make_fuzzer(job_scheduler=True, prune_corpus_max_memory=80)
    assert "memory_prune" not in off._maintenance.due_jobs(0)
    assert "memory_prune" in on._maintenance.due_jobs(0)


def test_maintenance_queue_tick_runs_memory_prune_without_internal_throttle():
    # Under --job-scheduler, _check_memory_and_prune itself has no internal
    # gate left (P3-3 step 4) -- the queue is the only gate. The queue
    # captured a bound method at construction time, so the job's action is
    # replaced directly rather than via an instance-attribute override
    # (which a plain self._check_memory_and_prune = ... would not reach).
    f = _make_fuzzer(job_scheduler=True, prune_corpus_max_memory=80)
    calls = []
    f._maintenance._jobs["memory_prune"].action = lambda: calls.append(f.exec_count)
    f._maintenance.tick(0)
    assert calls == [0]


def test_maintenance_queue_gc_job_registered_in_both_modes():
    legacy = _make_fuzzer(job_scheduler=False)
    scheduled = _make_fuzzer(job_scheduler=True)
    assert "gc" in legacy._maintenance.due_jobs(0)
    assert "gc" in scheduled._maintenance.due_jobs(0)


def test_maintenance_queue_replay_jobs_gated_by_replay_n_and_sanitizer_targets():
    f = _make_fuzzer(job_scheduler=True, replay_n=0, asan_target=None, ubsan_target=None)
    due = f._maintenance.due_jobs(0)
    assert "crash_replays" not in due
    assert "sanitizer_replays" not in due
