"""Regression: direct_lite ptrace triage ran for every crash, repeats included.

The triage re-runs the input in a ptrace loader to recover the fault address.
A sanitizer signature does not use it and a repeat input is dropped unread,
so for those the subprocess bought nothing, once per crashing execution.
"""

from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.services import fuzz_round
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.fuzz_round import FuzzRound

ASAN = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n"
    "    #0 0x401111 in parse_chunk\n"
)
SEGV = -int(signal.SIGSEGV)


class _Triage:
    """Counts ptrace triage runs."""

    calls: list[bytes] = []

    def __init__(self, f) -> None:
        pass

    def _run_triage_ptrace(self, data: bytes):
        _Triage.calls.append(data)
        return SEGV, ""


@pytest.fixture
def fuzzer(monkeypatch):
    _Triage.calls = []
    monkeypatch.setattr(fuzz_round, "TargetRunner", _Triage)
    f = SimpleNamespace(
        target="t.so",
        _last_fault_addr=None,
        _triage_ok=True,
        _inprocess_runner=SimpleNamespace(direct_lite=True),
        crash_hashes=set(),
        crash_sigs={},
        crash_blocklist=set(),
        crash_allowlist=set(),
        crash_min_sizes={},
        save_smaller=False,
    )
    f._corpus_manager = CorpusManager(f)
    return f


def _triage(f, data: bytes, rc: int, stderr: str) -> None:
    rnd = SimpleNamespace(_f=f, _mutated=data, _returncode=rc, _stderr=stderr)
    FuzzRound._triage_crash(rnd)  # type: ignore[arg-type]


def test_regression_known_sanitizer_crash_not_triaged(fuzzer) -> None:
    fuzzer.crash_sigs[SanitizerReport.parse(ASAN).signature] = 1
    _triage(fuzzer, b"new-input", 1, ASAN)

    assert _Triage.calls == []


def test_regression_repeat_input_not_triaged(fuzzer) -> None:
    fuzzer.crash_hashes.add(hash_data(b"seen"))
    _triage(fuzzer, b"seen", SEGV, "")

    assert _Triage.calls == []


def test_novel_sanitizer_crash_triaged(fuzzer) -> None:
    # Falsification: a crash that will be written still gets its registers.
    _triage(fuzzer, b"new-input", 1, ASAN)

    assert _Triage.calls == [b"new-input"]


def test_known_signal_crash_still_triaged(fuzzer) -> None:
    # Edge: the fault address may split a known signal signature.
    fuzzer.crash_sigs["signal:11"] = 1
    _triage(fuzzer, b"new-input", SEGV, "")

    assert _Triage.calls == [b"new-input"]
