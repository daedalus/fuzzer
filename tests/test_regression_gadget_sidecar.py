"""Regression: the ROP gadget scan is wired into the crash sidecar.

Covers the full path from a GadgetReport through CrashMetadata rendering and
serialization, and the corpus-manager attach hook (scan once, cached).
"""

from __future__ import annotations

import struct
from types import SimpleNamespace

from fuzzer_tool.core import gadgets
from fuzzer_tool.core.crash_metadata import CrashMetadata

BASE = 0x400000
# pop rax;ret pop rdi;ret pop rsi;ret pop rdx;ret syscall;ret -> a feasible chain.
_FEASIBLE = b"\x58\xc3\x5f\xc3\x5e\xc3\x5a\xc3\x0f\x05\xc3"
# aarch64 equivalent: ldp x8, x0, [sp]; ret  svc #0; ret
# (words verified against capstone; see tests/test_gadgets.py).
_FEASIBLE_A64 = struct.pack("<IIII", 0xA94003E8, 0xD65F03C0, 0xD4000001, 0xD65F03C0)


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


def test_sidecar_renders_new_gadget_fields():
    # arch, per-section attribution, libc targets and the ret2libc skeleton
    # all render; the header line stays byte-identical to the old format.
    gs = gadgets.scan(_FEASIBLE_A64, BASE, arch="aarch64")
    rep = gadgets.build_report(
        gs, arch="aarch64", sections={".text": 9}, libc_targets={"system": 0x401020}
    )
    meta = CrashMetadata()
    meta.sanitizer = "AddressSanitizer"
    meta.gadgets = rep.to_dict()

    text = meta.format_sidecar()

    assert "=== rop gadgets ===" in text
    assert "arch: aarch64" in text
    assert "sections: .text=9" in text
    assert "libc targets: system@0x401020" in text
    # The ret2libc skeleton renders like the syscall one: role + address.
    assert "system" in text.split("chain:")[1]


def test_sidecar_x86_default_adds_no_new_lines():
    # FALSIFICATION: a plain x86 report without libc data must render no
    # arch/sections/libc lines -- old sidecars stay byte-identical.
    meta = CrashMetadata()
    meta.gadgets = gadgets.build_report(gadgets.scan(_FEASIBLE, BASE)).to_dict()
    text = meta.format_sidecar()
    for absent in ("arch:", "sections:", "libc targets:", "ret2libc"):
        assert absent not in text


def test_to_dict_carries_new_keys_additively():
    rep = gadgets.build_report(
        gadgets.scan(_FEASIBLE_A64, BASE, arch="aarch64"),
        arch="aarch64",
        sections={".text": 9},
        libc_targets={"system": 0x401020},
    )
    d = rep.to_dict()
    # Old keys still present and shaped as before.
    assert d["chain_feasible"] is True
    assert d["reg_control"]["x0"].startswith("0x")
    assert d["syscall_sites"][0].startswith("0x")
    # New keys are additive.
    assert d["arch"] == "aarch64"
    assert d["sections"] == {".text": 9}
    assert d["libc_targets"] == {"system": "0x401020"}
    assert d["chain_libc_feasible"] is True
    assert d["chain_libc"] == [["x0", f"0x{BASE:x}"], ["system", "0x401020"]]
