"""Regression: every --resume aborted on a target with a Katz node channel.

`Fuzzer.__init__` ran `_load_corpus()` -> `load_state()` before it built
`_katz_channel`, so the resume-time coverage contract always read
`node_channel=False` against a saved `True` and raised; the saved Katz
state was never restored either. Found resuming an ffmpeg campaign in Docker.
"""

from __future__ import annotations

import pytest

_TARGET = "targets/test_target"
_KATZ_STATE = {"hits": [3, 1]}


class _FakeKatz:
    """Stands in for KatzChannel: no ELF, no SHM."""

    node_of: dict = {}
    n_nodes = 0

    def __init__(self):
        self.loaded = None

    def upload(self) -> bool:
        return True

    def state_dict(self) -> dict:
        return dict(_KATZ_STATE)

    def load_state_dict(self, state: dict) -> None:
        self.loaded = state


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")
    monkeypatch.setattr("fuzzer_tool.core.elf.detect_ctx_bits", lambda _t: 4)


def _with_katz(monkeypatch, katz: _FakeKatz | None) -> None:
    monkeypatch.setattr(
        "fuzzer_tool.services.katz_channel.KatzChannel.build",
        staticmethod(lambda *a, **k: katz),
    )


def _fuzzer(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir(parents=True, exist_ok=True)
    crashes.mkdir(parents=True, exist_ok=True)
    return Fuzzer(
        target=_TARGET, corpus_dir=str(corpus), crashes_dir=str(crashes), max_len=4096, **kw
    )


def _saved_with_katz(tmp_path, monkeypatch) -> None:
    _with_katz(monkeypatch, _FakeKatz())
    f = _fuzzer(tmp_path)
    assert f._katz_channel is not None
    f._save_state()


def test_regression_resume_restores_katz(tmp_path, monkeypatch):
    """Falsification: same target, same channel -- resume loads and restores Katz state."""
    _saved_with_katz(tmp_path, monkeypatch)

    katz = _FakeKatz()
    _with_katz(monkeypatch, katz)
    _fuzzer(tmp_path, resume=True)

    assert katz.loaded == _KATZ_STATE


def test_regression_resume_without_katz_refused(tmp_path, monkeypatch):
    """Adversarial: a resumed run that truly lacks the channel must still be refused."""
    _saved_with_katz(tmp_path, monkeypatch)

    _with_katz(monkeypatch, None)
    with pytest.raises(RuntimeError, match="node_channel"):
        _fuzzer(tmp_path, resume=True)
