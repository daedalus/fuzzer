"""Regression: Fuzzer must hand os.environ back the way it found it.

Finding #10 (docs/bugreport_2026-08-21_merged.md, HIGH): __init__ and run()
write __AFL_DIST_SHM_ID, __AFL_SHM_ID, AFL_MAP_SIZE, an ASAN LD_PRELOAD
injection and UBSAN_OPTIONS straight into the process environment, because
those keys need to be visible to every subprocess.Popen()/os.exec* call made
during the run. Only the cmplog shim's own LD_PRELOAD edit was ever undone
(the "Hand os.environ back" block at the end of run() called
``self._cmplog.restore_env()`` and nothing else) -- everything else leaked
into whatever ran next in the same process: a second target in a
multi-target session, a caller embedding Fuzzer as a library, or the next
test in a pytest run.

These tests drive the real snapshot/restore functions rather than
reimplementing the diff logic, so a change to the restore semantics is
caught here rather than only in the helper it copies.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

from fuzzer_tool.services import fuzzer as fuzzer_mod


def _reset_module_snapshot(monkeypatch):
    """Isolate each test from the process-wide, set-once snapshot."""
    monkeypatch.setattr(fuzzer_mod, "_environ_snapshot", None)


def test_restore_removes_keys_added_after_the_snapshot(monkeypatch):
    _reset_module_snapshot(monkeypatch)
    monkeypatch.delenv("__AFL_DIST_SHM_ID", raising=False)
    fuzzer_mod._snapshot_environ_once()

    os.environ["__AFL_DIST_SHM_ID"] = "shm_deadbeef"
    fuzzer_mod._restore_environ()

    assert "__AFL_DIST_SHM_ID" not in os.environ


def test_restore_puts_back_a_key_that_was_only_modified(monkeypatch):
    _reset_module_snapshot(monkeypatch)
    monkeypatch.setenv("UBSAN_OPTIONS", "print_stacktrace=0")
    fuzzer_mod._snapshot_environ_once()

    os.environ["UBSAN_OPTIONS"] = "halt_on_error=1:abort_on_error=1:print_stacktrace=1"
    fuzzer_mod._restore_environ()

    assert os.environ["UBSAN_OPTIONS"] == "print_stacktrace=0"


def test_snapshot_is_only_taken_once_per_process(monkeypatch):
    """A second Fuzzer() built while the first is still mutating the
    environment must not re-baseline over those mutations and adopt a
    dirty environment as "original" -- that would make the leak permanent
    instead of merely unrestored until process exit."""
    _reset_module_snapshot(monkeypatch)
    monkeypatch.delenv("__AFL_SHM_ID", raising=False)
    fuzzer_mod._snapshot_environ_once()

    os.environ["__AFL_SHM_ID"] = "shm_from_first_instance"
    fuzzer_mod._snapshot_environ_once()  # simulates a second Fuzzer() ctor

    fuzzer_mod._restore_environ()
    assert "__AFL_SHM_ID" not in os.environ


def test_fuzzer_init_snapshots_before_any_mutation(monkeypatch):
    """The constructor must capture the baseline before touching the
    environment itself, or its own writes would be adopted as original."""
    _reset_module_snapshot(monkeypatch)
    monkeypatch.delenv("__AFL_SHM_ID", raising=False)

    from fuzzer_tool.services.fuzzer import Fuzzer

    tmpdir = tempfile.mkdtemp(prefix="environ_restore_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            cmplog=False,
        )

    assert fuzzer_mod._environ_snapshot is not None
    assert "__AFL_SHM_ID" not in fuzzer_mod._environ_snapshot

    os.environ["__AFL_SHM_ID"] = "leaked"
    fuzzer_mod._restore_environ()
    assert "__AFL_SHM_ID" not in os.environ
