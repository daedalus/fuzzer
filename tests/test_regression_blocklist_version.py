"""Regression: --crash-blocklist files from an older stack-hash format.

The stack hash changed (ordered XXH64 over page offsets); old lists match
nothing and silently block nothing. Lists now carry ``# stack-hash vN``; an
untagged or other-version list warns. Saved crashes record their hash so a
list can be built from them.
"""

from __future__ import annotations

import json

from fuzzer_tool.cli.commands import _load_hash_list
from fuzzer_tool.core.sanitizer import STACK_HASH_TAG, STACK_HASH_VERSION, SanitizerReport
from fuzzer_tool.services.corpus_manager import CorpusManager
from tests.test_regression_crash_triage_cost import _Fuzzer

_ASAN = (
    "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010\n"
    "    #0 0x401111 in json_parse_ex\n    #1 0x402222 in json_parse\n"
)


def _write(tmp_path, text: str) -> str:
    p = tmp_path / "block.txt"
    p.write_text(text)
    return str(p)


def test_regression_untagged_list_warns(tmp_path, capsys) -> None:
    hashes = _load_hash_list(_write(tmp_path, "00112233445566778899\n"))

    assert hashes == {"00112233445566778899"}
    assert "stack-hash" in capsys.readouterr().out


def test_old_version_warns(tmp_path, capsys) -> None:
    old = f"# stack-hash v{STACK_HASH_VERSION - 1}\nabcd\n"
    _load_hash_list(_write(tmp_path, old))

    assert f"v{STACK_HASH_VERSION - 1}" in capsys.readouterr().out


def test_current_tag_silent(tmp_path, capsys) -> None:
    # Falsification: a current list loads without a warning.
    hashes = _load_hash_list(_write(tmp_path, f"{STACK_HASH_TAG}\nabcd\n# note\n"))

    assert hashes == {"abcd"}
    assert capsys.readouterr().out == ""


def test_tag_lookalike_not_a_tag(tmp_path, capsys) -> None:
    # Adversarial: "v2x" / "v22" are not the current version.
    _load_hash_list(_write(tmp_path, f"{STACK_HASH_TAG}x\nabcd\n"))
    assert "stack-hash" in capsys.readouterr().out

    _load_hash_list(_write(tmp_path, f"{STACK_HASH_TAG}2\nabcd\n"))
    assert "stack-hash" in capsys.readouterr().out


def test_empty_list_silent(tmp_path, capsys) -> None:
    # Edge: nothing to mismatch.
    assert _load_hash_list(_write(tmp_path, "")) == set()
    assert capsys.readouterr().out == ""


def test_sidecar_records_stack_hash(tmp_path, monkeypatch) -> None:
    f = _Fuzzer(tmp_path)
    cm = CorpusManager(f)
    monkeypatch.setattr(cm, "_novel_crash_meta", lambda *a, **k: None)
    assert cm.save_crash(b"in", 1, _ASAN)

    want = SanitizerReport.parse(_ASAN).stack_hash()
    (side,) = f.crashes_dir.glob("*.json")
    (txt,) = f.crashes_dir.glob("*.txt")
    assert want and json.loads(side.read_text())["stack_hash"] == want
    assert f"stack_hash:    {want}" in txt.read_text()
