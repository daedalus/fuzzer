"""Regression: the ROP gadget scan is wired into the crash sidecar.

Covers the full path from a GadgetReport through CrashMetadata rendering and
serialization, and the corpus-manager attach hook (scan once, cached).
"""

from __future__ import annotations

from types import SimpleNamespace

from fuzzer_tool.core import gadgets
from fuzzer_tool.core.crash_metadata import CrashMetadata

BASE = 0x400000
# pop rax;ret pop rdi;ret pop rsi;ret pop rdx;ret syscall;ret -> a feasible chain.
_FEASIBLE = b"\x58\xc3\x5f\xc3\x5e\xc3\x5a\xc3\x0f\x05\xc3"


def _feasible_report_dict():
    rep = gadgets.build_report(gadgets.scan(_FEASIBLE, BASE))
    assert rep.chain_feasible
    return rep.to_dict()


def test_sidecar_renders_gadget_section():
    meta = CrashMetadata()
    meta.sanitizer = "AddressSanitizer"
    meta.gadgets = _feasible_report_dict()

    text = meta.format_sidecar()

    assert "=== rop gadgets ===" in text
    assert "chain:" in text
    # The syscall-chain roles appear as a skeleton.
    assert "syscall" in text
    assert "rax" in text and "rdi" in text


def test_empty_gadgets_render_nothing():
    meta = CrashMetadata()
    meta.gadgets = {}
    assert "rop gadgets" not in meta.format_sidecar()

    # A scanned-but-empty inventory (no gadgets) also stays out of the sidecar.
    meta.gadgets = gadgets.build_report([]).to_dict()
    assert "rop gadgets" not in meta.format_sidecar()


def test_to_dict_carries_gadgets():
    meta = CrashMetadata()
    meta.gadgets = _feasible_report_dict()
    d = meta.to_dict()
    assert d["gadgets"]["chain_feasible"] is True
    assert d["gadgets"]["reg_control"]["rdi"].startswith("0x")


def test_corpus_manager_attach_caches_and_fills(tmp_path, monkeypatch):
    from fuzzer_tool.services import corpus_manager

    target = tmp_path / "t.bin"
    target.write_bytes(b"ignored; scan_target is stubbed")

    calls = {"n": 0}

    def fake_scan(path):
        calls["n"] += 1
        return gadgets.build_report(gadgets.scan(_FEASIBLE, BASE))

    monkeypatch.setattr(
        corpus_manager, "CorpusManager", corpus_manager.CorpusManager, raising=False
    )
    monkeypatch.setattr("fuzzer_tool.core.gadgets.scan_target", fake_scan)

    f = SimpleNamespace(target=str(target), scan_gadgets=True)
    mgr = corpus_manager.CorpusManager.__new__(corpus_manager.CorpusManager)
    mgr.f = f

    m1, m2 = CrashMetadata(), CrashMetadata()
    mgr._attach_gadgets(m1)
    mgr._attach_gadgets(m2)

    assert calls["n"] == 1  # scanned once, cached on f
    assert m1.gadgets["chain_feasible"] is True
    assert m2.gadgets == m1.gadgets


def test_attach_gadgets_survives_scan_failure(tmp_path, monkeypatch):
    from fuzzer_tool.services import corpus_manager

    def boom(_path):
        raise RuntimeError("corrupt binary")

    monkeypatch.setattr("fuzzer_tool.core.gadgets.scan_target", boom)

    f = SimpleNamespace(target="whatever", scan_gadgets=True)
    mgr = corpus_manager.CorpusManager.__new__(corpus_manager.CorpusManager)
    mgr.f = f

    meta = CrashMetadata()
    mgr._attach_gadgets(meta)  # must not raise
    assert meta.gadgets == {}
    assert f._gadget_report is None
