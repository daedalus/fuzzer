"""--novel-crash-reward: only a crash that would be saved pays the operators.

By default every crash, repeats included, is an operator success (``_judge``)
and an Elo crash win; once a shallow bug is found, operators that re-trigger
it keep winning. The flag rewards novel crashes only (A/B pending).
"""

from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.crash_symbols import CrashSymbol, SymFrame
from fuzzer_tool.core.run_mode import CrashReward
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.fuzz_round import FuzzRound

SEGV = -int(signal.SIGSEGV)
_QUIET = dict(
    _is_interesting=False,
    _has_new_coverage=False,
    _is_slow=False,
    _is_new_max=False,
    _is_cmp_progress=False,
    _is_new_valid_coverage=False,
)


def _judge(mode: CrashReward, is_crash: bool, novel: bool) -> SimpleNamespace:
    calls: list[bytes] = []

    def _novel(data, rc, stderr):
        calls.append(data)
        return novel

    f = SimpleNamespace(_crash_reward=mode, _corpus_manager=SimpleNamespace(crash_is_novel=_novel))
    rnd = SimpleNamespace(
        _f=f, _is_crash=is_crash, _mutated=b"m", _returncode=SEGV, _stderr="", **_QUIET
    )
    FuzzRound._judge(rnd)  # type: ignore[arg-type]
    rnd.novelty_calls = calls
    return rnd


def test_regression_repeat_crash_not_rewarded() -> None:
    rnd = _judge(CrashReward.NOVEL, is_crash=True, novel=False)

    assert not rnd._success
    assert not rnd._rewarded_crash


def test_novel_crash_rewarded() -> None:
    rnd = _judge(CrashReward.NOVEL, is_crash=True, novel=True)

    assert rnd._success and rnd._rewarded_crash


def test_default_rewards_every_crash() -> None:
    # Falsification: default behaviour unchanged, and no extra classify.
    rnd = _judge(CrashReward.ANY, is_crash=True, novel=False)

    assert rnd._success and rnd._rewarded_crash
    assert rnd.novelty_calls == []


def test_no_crash_no_novelty_check() -> None:
    # Edge: non-crashing rounds never pay for classification.
    rnd = _judge(CrashReward.NOVEL, is_crash=False, novel=True)

    assert not rnd._rewarded_crash
    assert rnd.novelty_calls == []


# ── The early novelty check must not steal the shim crash record ────────


class _Sink:
    """One shim record, drained once (as the real sink truncates)."""

    def __init__(self, sym: CrashSymbol) -> None:
        self._sym: CrashSymbol | None = sym

    def drain(self) -> CrashSymbol | None:
        sym, self._sym = self._sym, None
        return sym


@pytest.fixture
def fuzzer(tmp_path):
    from tests.test_regression_crash_triage_cost import _Fuzzer

    f = _Fuzzer(tmp_path)
    sym = CrashSymbol(signal=11, pc=0x401111)
    sym.frames = [SymFrame(pc=0x401111), SymFrame(pc=0x402222)]
    f._crash_sym_sink = _Sink(sym)
    return f


def test_regression_novelty_check_keeps_record_for_save(fuzzer, monkeypatch) -> None:
    cm = CorpusManager(fuzzer)
    monkeypatch.setattr(cm, "_novel_crash_meta", lambda *a, **k: None)

    assert cm.crash_is_novel(b"in", SEGV, "")
    cm.save_crash(b"in", SEGV, "")

    assert list(fuzzer.crash_sigs) == ["signal:11@0x111@0x222"]


def test_known_site_not_novel(fuzzer, monkeypatch) -> None:
    fuzzer.crash_sigs["signal:11@0x111@0x222"] = 1

    assert not CorpusManager(fuzzer).crash_is_novel(b"in", SEGV, "")
