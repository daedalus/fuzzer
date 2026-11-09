"""Rust/Go panics exit with an ordinary code and were read as clean."""

import types

from fuzzer_tool.services.runner import TargetRunner


def _runner() -> TargetRunner:
    r = TargetRunner.__new__(TargetRunner)
    r.f = types.SimpleNamespace(last_report=None, extra_crash_codes=set())
    return r


def test_regression_rust_panic_exit_101_is_crash():
    assert _runner().is_crash(101, "thread 'main' panicked at src/main.rs:3:5\n")


def test_regression_go_panic_exit_2_is_crash():
    err = "panic: runtime error: index out of range\n\ngoroutine 1 [running]:\n"
    assert _runner().is_crash(2, err)


def test_regression_panic_text_on_clean_exit_is_not_crash():
    # Falsification: a program that merely logs the phrase and exits 0.
    assert not _runner().is_crash(0, "note: handler panicked at startup, recovered\n")


def test_regression_unrelated_nonzero_exit_stays_clean():
    assert not _runner().is_crash(101, "usage: prog <file>\n")
