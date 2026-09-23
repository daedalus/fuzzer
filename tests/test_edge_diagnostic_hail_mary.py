"""``edge_diagnostic.py matrix --hail-mary`` grow-step helpers.

The matrix sub-command can grow a corpus with a real ``fuzzer-tool fuzz
--hail-mary`` campaign before collecting shim edges.  Two things must hold
for that to stay correct:

1. ``_so_sibling`` must pick a dlopen-able, non-ASAN counterpart of a PIE
   target so the hail-mary in-process mode (which ``_apply_hail_mary``
   force-enables) can load the target instead of dying on
   ``cannot dynamically load position-independent executable``.
2. ``_hail_mary_grow`` must hand the campaign a *copy* of the corpus (the
   original is never mutated) and strip the StateStore sidecar files the
   fuzzer drops into its corpus dir (``state.json``, ``edge_tracker.json``,
   ...) before the matrix collection runs over the grown directory --
   otherwise those JSON files get executed as fuzz inputs.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "edge_diagnostic.py"


def _load():
    spec = importlib.util.spec_from_file_location("edge_diagnostic", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ED = _load()


def test_so_sibling_prefers_non_asan(tmp_path):
    pie = tmp_path / "fuzzgoat_read"
    pie.write_bytes(b"PIE")
    (tmp_path / "fuzzgoat_read_asan.so").write_bytes(b"asan")
    (tmp_path / "fuzzgoat_read_noasan.so").write_bytes(b"noasan")

    found = ED._so_sibling(pie)

    assert found == tmp_path / "fuzzgoat_read_noasan.so"


def test_so_sibling_returns_none_when_no_so(tmp_path):
    pie = tmp_path / "target"
    pie.write_bytes(b"PIE")

    assert ED._so_sibling(pie) is None


def test_so_sibling_so_target_is_rejected(tmp_path):
    so = tmp_path / "already.so"
    so.write_bytes(b"so")

    assert ED._so_sibling(so) is None


def test_hail_mary_grow_copies_corpus_and_strips_sidecars(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "input1").write_bytes(b"abc")
    (corpus / "input2").write_bytes(b"def")
    target = tmp_path / "fuzzgoat_read_noasan.so"
    target.write_bytes(b"so")

    def fake_run(cmd, **kwargs):
        grown = Path(cmd[cmd.index("-d") + 1])
        assert "--max-execs" in cmd, "campaign must budget by executions, not -n iterations"
        assert "-n" not in cmd
        for name in ("state.json", "edge_tracker.json", "markov.json"):
            (grown / name).write_bytes(b"{}")
        (grown / "mutated1").write_bytes(b"xyz")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fake_mkdtemp(prefix):
        p = tmp_path / "edge_diag_hm_x"
        p.mkdir()
        return str(p)

    monkeypatch.setattr(ED.subprocess, "run", fake_run)
    monkeypatch.setattr(ED.tempfile, "mkdtemp", fake_mkdtemp)

    grown = ED._hail_mary_grow(target, corpus, iters=50, inprocess_func="fuzz_shm_run")

    assert (grown / "input1").read_bytes() == b"abc"
    assert (grown / "input2").read_bytes() == b"def"
    assert (grown / "mutated1").read_bytes() == b"xyz"
    for name in ("state.json", "edge_tracker.json", "markov.json"):
        assert not (grown / name).exists()
    assert (corpus / "input1").read_bytes() == b"abc"
    assert not (corpus / "mutated1").exists()


def test_hail_mary_grow_fails_loudly_on_campaign_error(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "input1").write_bytes(b"abc")
    target = tmp_path / "fuzzgoat_read_noasan.so"
    target.write_bytes(b"so")

    monkeypatch.setattr(
        ED.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="boom", stderr=""),
    )

    def fake_mkdtemp(prefix):
        p = tmp_path / "edge_diag_hm_x"
        p.mkdir()
        return str(p)

    monkeypatch.setattr(ED.tempfile, "mkdtemp", fake_mkdtemp)

    with pytest.raises(SystemExit, match="campaign failed with rc=1"):
        ED._hail_mary_grow(target, corpus, iters=50, inprocess_func="fuzz_shm_run")
