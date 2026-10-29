"""--crash-tmin-execs: shrink novel crashes at maintenance ticks.

Saved crashes kept the mutant's full size; `tmin` had to be run by hand.
CrashShrinker reduces each novel crash (core/reducer.py) under an exec and
time budget with its signature pinned, and writes ``<base>.min`` beside
the untouched ``.bin``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fuzzer_tool.services.crash_shrink import MAX_PENDING, CrashShrinker

_BUG = b"BUG"


def _target(calls: list[bytes]):
    """Crashes "S" when the input holds BUG, "T" (another bug) when only B."""

    def run(data: bytes) -> tuple[int, str]:
        calls.append(data)
        if _BUG in data:
            return -11, "S"
        if b"B" in data:
            return -11, "T"
        return 0, ""

    return run


def _sig(rc: int, stderr: str) -> str | None:
    return stderr or None


def _crash(tmp_path: Path, data: bytes) -> Path:
    p = tmp_path / "crash_1_abcd1234_sig_signal11.bin"
    p.write_bytes(data)
    return p


def _shrinker(calls, execs: int = 512, budget_ms: float = 1e6) -> CrashShrinker:
    return CrashShrinker(_target(calls), _sig, max_execs=execs, budget_ms=budget_ms)


def test_shrinks_to_trigger(tmp_path) -> None:
    calls: list[bytes] = []
    s = _shrinker(calls)
    crash = _crash(tmp_path, b"xxxxxxxx" + _BUG + b"yyyyyyyy")
    s.enqueue(crash, "S")
    s.tick()

    assert crash.with_suffix(".min").read_bytes() == _BUG
    assert crash.read_bytes() == b"xxxxxxxx" + _BUG + b"yyyyyyyy"  # original kept


def test_signature_pinned(tmp_path) -> None:
    # Falsification: "B" alone crashes too, as another bug; never accepted.
    calls: list[bytes] = []
    s = _shrinker(calls)
    crash = _crash(tmp_path, b"aaaa" + _BUG + b"bbbb")
    s.enqueue(crash, "S")
    s.tick()

    assert _BUG in crash.with_suffix(".min").read_bytes()


def test_exec_budget(tmp_path) -> None:
    calls: list[bytes] = []
    s = _shrinker(calls, execs=4)
    crash = _crash(tmp_path, bytes(range(200)) + _BUG)
    s.enqueue(crash, "S")
    s.tick()

    assert len(calls) <= 4
    out = crash.with_suffix(".min")
    assert not out.exists() or _BUG in out.read_bytes()  # only verified shrinks


def test_not_reproducing_writes_nothing(tmp_path) -> None:
    # Adversarial: a flaky crash that does not reproduce is left alone.
    calls: list[bytes] = []
    s = _shrinker(calls)
    crash = _crash(tmp_path, b"no trigger here")
    s.enqueue(crash, "S")
    s.tick()

    assert not crash.with_suffix(".min").exists()
    assert len(calls) == 1


def test_already_minimal_writes_nothing(tmp_path) -> None:
    # Edge: nothing smaller reproduces: no .min duplicate.
    s = _shrinker([])
    crash = _crash(tmp_path, _BUG)
    s.enqueue(crash, "S")
    s.tick()

    assert not crash.with_suffix(".min").exists()


def test_queue_bounded(tmp_path) -> None:
    s = _shrinker([])
    crash = _crash(tmp_path, _BUG)
    for _ in range(MAX_PENDING + 10):
        s.enqueue(crash, "S")

    assert s.pending == MAX_PENDING


def test_time_budget_defers(tmp_path) -> None:
    # Edge: an exhausted tick budget leaves the rest queued.
    s = _shrinker([], budget_ms=0)
    for i in range(3):
        (tmp_path / str(i)).mkdir()
        s.enqueue(_crash(tmp_path / str(i), _BUG + b"x"), "S")
    s.tick()

    assert s.pending == 2  # one crash per tick at least, then the budget stops it


def test_missing_file_skipped(tmp_path) -> None:
    # Adversarial: the crash file was removed before its turn.
    calls: list[bytes] = []
    s = _shrinker(calls)
    s.enqueue(tmp_path / "gone.bin", "S")
    s.tick()

    assert calls == [] and s.pending == 0


@pytest.fixture
def cm_fuzzer(tmp_path, monkeypatch):
    from fuzzer_tool.services.corpus_manager import CorpusManager
    from tests.test_regression_crash_triage_cost import _Fuzzer

    f = _Fuzzer(tmp_path)
    cm = CorpusManager(f)
    monkeypatch.setattr(cm, "_novel_crash_meta", lambda *a, **k: None)
    return f, cm


def test_novel_crash_enqueued(cm_fuzzer) -> None:
    f, cm = cm_fuzzer
    f._crash_shrinker = _shrinker([])
    asan = "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n    #0 0x4 in f\n"
    assert cm.save_crash(b"in", 1, asan)
    assert not cm.save_crash(b"in2", 1, asan)  # repeat: not queued

    assert f._crash_shrinker.pending == 1


def test_off_by_default(cm_fuzzer) -> None:
    # Falsification: no shrinker, saving works as before.
    f, cm = cm_fuzzer
    assert cm.save_crash(b"in", -11, "")
