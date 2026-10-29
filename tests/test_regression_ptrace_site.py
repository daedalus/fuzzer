"""Regression: ptrace signal signatures split one bug by its fault address.

``signal:11@<si_addr>`` names the data a wild pointer held, not the code:
one bad dereference fed different bytes filed one crash per address. The
faulting instruction (``rip``, page offset, ASLR-stable) now names the site.
"""

from __future__ import annotations

import json
import signal

from fuzzer_tool.adapters.filesystem import saved_crash_sigs
from fuzzer_tool.core.crash_symbols import CrashSymbol, SymFrame
from fuzzer_tool.services.corpus_manager import CorpusManager
from tests.test_regression_crash_triage_cost import _Fuzzer, counted  # noqa: F401

SEGV = -int(signal.SIGSEGV)


def _mgr(tmp_path, monkeypatch):
    f = _Fuzzer(tmp_path)
    cm = CorpusManager(f)
    monkeypatch.setattr(cm, "_novel_crash_meta", lambda *a, **k: None)
    return f, cm


def _crash(f, cm, data: bytes, rip: int, fault: int):
    f._last_regs = {"rip": rip, "rsp": 0, "rbp": 0}
    f._last_fault_addr = fault
    return cm.save_crash(data, SEGV, "")


def test_regression_fault_addr_not_identity(tmp_path, monkeypatch) -> None:
    f, cm = _mgr(tmp_path, monkeypatch)
    assert _crash(f, cm, b"a", 0x55D0004011D6, 0x4141414141)
    assert not _crash(f, cm, b"b", 0x55D0004011D6, 0x4242424242)

    assert list(f.crash_sigs) == ["signal:11@0x1d6"]


def test_other_rip_other_bug(tmp_path, monkeypatch) -> None:
    # Falsification: another faulting instruction is another site.
    f, cm = _mgr(tmp_path, monkeypatch)
    assert _crash(f, cm, b"a", 0x4011D6, 0)
    assert _crash(f, cm, b"b", 0x4022E8, 0)


def test_no_rip_keeps_fault_addr(tmp_path, monkeypatch) -> None:
    # Edge: GETREGS failed (rip 0 / no regs): the address is all there is.
    f, cm = _mgr(tmp_path, monkeypatch)
    assert _crash(f, cm, b"a", 0, 0xDEAD0000)
    f._last_regs = None
    f._last_fault_addr = 0xBEEF0000
    assert cm.save_crash(b"b", SEGV, "")

    assert list(f.crash_sigs) == ["signal:11@0xdead0000", "signal:11@0xbeef0000"]


def test_shim_pcs_win_over_rip(tmp_path, monkeypatch) -> None:
    # Edge: a backtrace beats a single instruction.
    f, cm = _mgr(tmp_path, monkeypatch)
    sym = CrashSymbol(signal=11, pc=0x401111)
    sym.frames = [SymFrame(pc=0x401111), SymFrame(pc=0x402222)]
    monkeypatch.setattr(cm, "_drain_sym", lambda: sym)

    assert _crash(f, cm, b"a", 0x4033E8, 0x10)
    assert list(f.crash_sigs) == ["signal:11@0x111@0x222"]


def test_sanitizer_report_ignores_rip(tmp_path, monkeypatch) -> None:
    # Adversarial: registers of a reported crash must not reshape its signature.
    f, cm = _mgr(tmp_path, monkeypatch)
    asan = "==1==ERROR: AddressSanitizer: SEGV on unknown address 0x0\n    #0 0x4 in f\n"
    f._last_regs = {"rip": 0x4011D6}
    assert cm.save_crash(b"a", SEGV, asan)

    assert list(f.crash_sigs) == ["AddressSanitizer:SEGV@f"]


def test_resync_uses_sidecar_rip(tmp_path, counted) -> None:  # noqa: F811
    f = _Fuzzer(tmp_path)
    cm = CorpusManager(f)
    _crash(f, cm, b"a", 0x4011D6, 0x4141)
    (side,) = f.crashes_dir.glob("*.json")

    assert json.loads(side.read_text())["registers"]["rip"] == 0x4011D6
    assert saved_crash_sigs(f.crashes_dir) == ["signal:11@0x1d6"]


def test_raised_signal_ignores_rip(tmp_path, monkeypatch) -> None:
    # Edge: no kernel fault address (abort, kill): rip is in libc, unused.
    f, cm = _mgr(tmp_path, monkeypatch)
    f._last_regs = {"rip": 0x7F00001234C0}
    f._last_fault_addr = None
    assert cm.save_crash(b"a", -int(signal.SIGABRT), "")

    assert list(f.crash_sigs) == ["signal:6"]
