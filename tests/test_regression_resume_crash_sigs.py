"""Regression: resumed campaigns re-saved known crashes once.

``crash_sigs`` is restored from state.json as written by the old parser; a
signature format change (frames, stack rules) missed every known bug once,
and the on-disk check keyed by cluster id missed it too. Resume now re-derives
each saved crash's signature from its ``.json`` sidecar.
"""

from __future__ import annotations

import json
import signal

from fuzzer_tool.adapters.filesystem import saved_crash_sigs
from fuzzer_tool.services.corpus_manager import CorpusManager
from tests.test_regression_crash_triage_cost import _Fuzzer

SEGV = -int(signal.SIGSEGV)
_ASAN = (
    "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010\n"
    "    #0 0x401111 in json_parse_ex\n    #1 0x402222 in json_parse\n"
)
_OTHER = _ASAN.replace("heap-use-after-free", "heap-buffer-overflow")


def _old_campaign(tmp_path, monkeypatch):
    """A crashes dir saved under another signature format, and its state."""
    f = _Fuzzer(tmp_path)
    cm = CorpusManager(f)
    monkeypatch.setattr(cm, "_novel_crash_meta", lambda *a, **k: None)
    assert cm.save_crash(b"first", 1, _ASAN)

    # Old format: other signature string, other cluster id in the names.
    (sig,) = f.crash_sigs
    for p in f.crashes_dir.iterdir():
        parts = p.name.split("_")
        parts[2] = "0ld0c1d0"
        p.rename(p.with_name("_".join(parts)))
    f.crash_sigs = {sig + "@fuzz_shm_run@__afl_guard_run@libffi.so.8+0x7b15@libffi.so.8+0x43ee": 1}
    f.crash_hashes = set()
    return f, cm


def test_regression_resume_known_crash_not_resaved(tmp_path, monkeypatch) -> None:
    f, cm = _old_campaign(tmp_path, monkeypatch)
    before = sorted(p.name for p in f.crashes_dir.iterdir())

    cm.resync_crash_sigs()

    assert not cm.save_crash(b"second", 1, _ASAN)
    assert sorted(p.name for p in f.crashes_dir.iterdir()) == before


def test_resume_new_bug_still_saved(tmp_path, monkeypatch) -> None:
    # Falsification: resync must not swallow a different bug.
    f, cm = _old_campaign(tmp_path, monkeypatch)
    cm.resync_crash_sigs()

    assert cm.save_crash(b"other", 1, _OTHER)


def test_resync_keeps_counts(tmp_path, monkeypatch) -> None:
    # Edge: a signature already known keeps its count.
    f, cm = _old_campaign(tmp_path, monkeypatch)
    sigs = saved_crash_sigs(f.crashes_dir)
    f.crash_sigs[sigs[0]] = 7

    cm.resync_crash_sigs()

    assert f.crash_sigs[sigs[0]] == 7


def test_signal_crash_sig_from_shim_record(tmp_path) -> None:
    # Edge: no report; the sidecar's shim record names the site.
    frame = {"pc": "0x55d000401111", "module": "/t/t.so", "function": "", "file": ""}
    side = {
        "returncode": SEGV,
        "raw_stderr": "",
        "fault_addr": "",
        "shim_symbol": {"frames": [frame]},
    }
    (tmp_path / "crash_1_abcd1234_sig_signal11.json").write_text(json.dumps(side))

    assert saved_crash_sigs(tmp_path) == ["signal:11@0x111"]


def test_corrupt_sidecar_skipped(tmp_path) -> None:
    # Adversarial: truncated / wrong-typed sidecars must not abort a resume.
    (tmp_path / "crash_1_aaaa0000_x_y.json").write_text('{"returncode": ')
    (tmp_path / "crash_2_bbbb0000_x_y.json").write_text("[1, 2]")
    (tmp_path / "crash_3_cccc0000_x_y.json").write_text('{"returncode": "x", "raw_stderr": 5}')

    assert saved_crash_sigs(tmp_path) == []


def test_missing_dir(tmp_path) -> None:
    assert saved_crash_sigs(tmp_path / "absent") == []
