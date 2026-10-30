"""`edge-dump -n/--max-seeds N`: replay at most N new seeds, in corpus order.

Known seeds (already in the outputs) do not count, so repeated runs walk the
corpus in chunks. N is the progress total: a streamed seeds.tar.gz, otherwise
uncountable, gets an ETA.
"""

import io
import logging
import sys
import tarfile

import pytest

from fuzzer_tool.adapters import seed_tar
from fuzzer_tool.cli import commands
from fuzzer_tool.services import edge_export
from tests.test_edge_export_incremental import _Corpus, _load

SEEDS = [bytes([0x61 + i, 0x62 + i]) for i in range(5)]


def _seeded(tmp_path, monkeypatch) -> _Corpus:
    c = _Corpus(tmp_path, monkeypatch)
    for i, d in enumerate(SEEDS):
        c.add(f"s{i}", d)
    return c


def _tar_only(tmp_path, monkeypatch) -> _Corpus:
    c = _Corpus(tmp_path, monkeypatch)
    with tarfile.open(c.root / seed_tar.TAR_NAME, "w:gz") as tf:
        for i, d in enumerate(SEEDS):
            info = tarfile.TarInfo(f"seeds/t{i}")
            info.size = len(d)
            tf.addfile(info, io.BytesIO(d))
    return c


def _dump(c: _Corpus, out, n: int) -> int:
    c.replayed.clear()
    return edge_export.export_edges("t", str(c.root), [str(out)], max_seeds=n)


def test_caps_replayed_seeds(tmp_path, monkeypatch):
    c = _seeded(tmp_path, monkeypatch)
    out = tmp_path / "e.pkl"
    assert _dump(c, out, 2) == 2
    assert len(c.replayed) == 2
    assert len(_load(out)) == 2


def test_known_seeds_do_not_count(tmp_path, monkeypatch):
    """Falsification: chunks advance; each run replays N seeds not seen before."""
    c = _seeded(tmp_path, monkeypatch)
    out = tmp_path / "e.sqlite"
    runs = []
    for _ in range(3):
        _dump(c, out, 2)
        runs.append(list(c.replayed))

    assert [len(r) for r in runs] == [2, 2, 1]
    assert sorted(d for r in runs for d in r) == sorted(SEEDS)


@pytest.mark.parametrize("n", [0, -1])
def test_zero_or_negative_means_all(n, tmp_path, monkeypatch):
    c = _seeded(tmp_path, monkeypatch)
    assert _dump(c, tmp_path / "e.pkl", n) == len(SEEDS)


def test_cap_above_corpus(tmp_path, monkeypatch):
    """Adversarial: N past the corpus ends cleanly with everything exported."""
    c = _seeded(tmp_path, monkeypatch)
    assert _dump(c, tmp_path / "e.pkl", 100) == len(SEEDS)


def test_tar_gets_eta_with_cap(tmp_path, monkeypatch, capsys):
    c = _tar_only(tmp_path, monkeypatch)
    _dump(c, tmp_path / "e.pkl", 3)
    line = capsys.readouterr().out.split("\r")[-1]
    assert "Replayed 3/3" in line and "ETA" in line


def test_tar_no_eta_without_cap(tmp_path, monkeypatch, capsys):
    """Falsification: without -n a streamed tar total stays unknown."""
    c = _tar_only(tmp_path, monkeypatch)
    _dump(c, tmp_path / "e.pkl", 0)
    line = capsys.readouterr().out.split("\r")[-1]
    assert "ETA" not in line


def test_tar_stopped_early_is_not_torn(tmp_path, monkeypatch, caplog):
    """Adversarial: leaving the stream at N must not log the archive as unreadable."""
    c = _tar_only(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING):
        _dump(c, tmp_path / "e.pkl", 2)
    assert len(c.replayed) == 2
    assert seed_tar.TAR_NAME not in caplog.text


def test_cli_flag(tmp_path, monkeypatch):
    c = _seeded(tmp_path, monkeypatch)
    out = tmp_path / "e.pkl"
    monkeypatch.setattr(commands, "_validate_target", lambda t: None)
    argv = ["fuzzer-tool", "edge-dump", "t", "-d", str(c.root), "-o", str(out), "-n", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    assert commands.main() == 0
    assert len(_load(out)) == 3
