"""`edge-dump` incremental: seed_ids already in an existing -o file are not replayed.

run 1: corpus {a, b}     -> replay a, b        -> out = {a, b}
run 2: corpus {a, b, c}  -> replay c only      -> out = {a, b, c}
"""

import pickle
import sqlite3

import numpy as np
import pytest

from fuzzer_tool.adapters import process
from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.services import edge_export
from tests.test_edge_export import _expected, _Shm

FORMATS = ["e.pkl", "e.sqlite"]


class _Corpus:
    """Corpus under seeds/ plus a fake replay that logs every input it runs."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "corpus"
        (self.root / "seeds").mkdir(parents=True)
        self.replayed: list[bytes] = []
        self.blind = False
        monkeypatch.setattr(edge_export, "ShmCoverage", _Shm)
        monkeypatch.setattr(process, "run_target_stdin", self._run)

    def _run(self, target, data, timeout, env=None, **_):
        self.replayed.append(data)
        _Shm.current = b"" if self.blind else data
        return 0, "", 0

    def add(self, name: str, data: bytes) -> None:
        (self.root / "seeds" / name).write_bytes(data)

    def dump(self, *outs) -> int:
        self.replayed.clear()
        return edge_export.export_edges("t", str(self.root), [str(o) for o in outs])


def _pkl_chunks(path) -> dict:
    """Merge every pickled batch in *path* (one dict per batch, appended)."""
    merged: dict = {}
    with open(path, "rb") as f:
        while True:
            try:
                merged.update(pickle.load(f))
            except EOFError:
                return merged


def _load(path) -> dict[str, list[int]]:
    """Seed -> edge list from either format, read without the code under test."""
    if path.suffix == ".pkl":
        return {k: v.tolist() for k, v in _pkl_chunks(path).items()}

    con = sqlite3.connect(path)
    try:
        got = {s: [] for (s,) in con.execute("SELECT seed_id FROM seeds")}
        for e, s in con.execute("SELECT edge_id, seed_id FROM edge_seeds ORDER BY edge_id"):
            got[s].append(e)
        return got
    finally:
        con.close()


@pytest.mark.parametrize("name", FORMATS)
def test_known_seeds_skipped(name, tmp_path, monkeypatch):
    c = _Corpus(tmp_path, monkeypatch)
    out = tmp_path / name
    c.add("a", b"ab")
    c.add("b", b"bc")
    assert c.dump(out) == 2

    c.add("c", b"cd")
    assert c.dump(out) == 3

    assert c.replayed == [b"cd"]
    assert _load(out) == {hash_data(d): _expected(d) for d in (b"ab", b"bc", b"cd")}


@pytest.mark.parametrize("name", FORMATS)
def test_no_existing_file_replays_all(name, tmp_path, monkeypatch):
    """Falsification: without an output on disk every seed runs."""
    c = _Corpus(tmp_path, monkeypatch)
    c.add("a", b"ab")
    c.add("b", b"bc")
    c.dump(tmp_path / name)
    assert sorted(c.replayed) == [b"ab", b"bc"]


@pytest.mark.parametrize("name", FORMATS)
def test_up_to_date_untouched(name, tmp_path, monkeypatch):
    """Nothing new: no replay, no rewrite, still success."""
    c = _Corpus(tmp_path, monkeypatch)
    out = tmp_path / name
    c.add("a", b"ab")
    c.dump(out)
    before = (out.read_bytes(), out.stat().st_mtime_ns)

    assert c.dump(out) == 1
    assert c.replayed == []
    assert (out.read_bytes(), out.stat().st_mtime_ns) == before


@pytest.mark.parametrize("name", FORMATS)
def test_edgeless_seed_remembered(name, tmp_path, monkeypatch):
    """Adversarial: an empty-coverage seed is a known negative, not re-run forever."""
    c = _Corpus(tmp_path, monkeypatch)
    out = tmp_path / name
    c.add("a", b"ab")
    c.add("e", b"")
    c.dump(out)

    c.add("c", b"cd")
    c.dump(out)
    assert c.replayed == [b"cd"]
    assert _load(out)[hash_data(b"")] == []


@pytest.mark.parametrize("name", FORMATS)
def test_corrupt_output_refused(name, tmp_path, monkeypatch, capsys):
    """Adversarial: an unreadable existing output is neither replayed over nor overwritten."""
    c = _Corpus(tmp_path, monkeypatch)
    out = tmp_path / name
    out.write_bytes(b"\x00garbage")
    c.add("a", b"ab")

    assert c.dump(out) == 0
    assert c.replayed == []
    assert out.read_bytes() == b"\x00garbage"
    assert "Cannot read" in capsys.readouterr().err


@pytest.mark.parametrize("name", FORMATS)
def test_new_blackout_keeps_prior(name, tmp_path, monkeypatch):
    """Adversarial: new seeds all edgeless (broken target) are not cached as negatives."""
    c = _Corpus(tmp_path, monkeypatch)
    out = tmp_path / name
    c.add("a", b"ab")
    c.dump(out)
    prior = _load(out)

    c.add("c", b"cd")
    c.blind = True
    assert c.dump(out) == 0
    assert _load(out) == prior

    c.blind = False
    c.dump(out)
    assert c.replayed == [b"cd"]


def test_skip_needs_every_output(tmp_path, monkeypatch):
    """A seed is skipped only when all outputs hold it; each output keeps its priors."""
    c = _Corpus(tmp_path, monkeypatch)
    pkl, db = tmp_path / "e.pkl", tmp_path / "e.sqlite"
    c.add("a", b"ab")
    c.dump(pkl)

    c.add("b", b"bc")
    c.dump(pkl, db)

    assert sorted(c.replayed) == [b"ab", b"bc"]
    want = {hash_data(d): _expected(d) for d in (b"ab", b"bc")}
    assert _load(pkl) == want
    assert _load(db) == want


def test_wrong_pickle_type_refused(tmp_path, monkeypatch, capsys):
    """Adversarial: a valid pickle that is not {str: ndarray} is not merged into."""
    c = _Corpus(tmp_path, monkeypatch)
    out = tmp_path / "e.pkl"
    out.write_bytes(pickle.dumps([1, 2, 3]))
    c.add("a", b"ab")

    assert c.dump(out) == 0
    assert c.replayed == []
    assert "Cannot read" in capsys.readouterr().err


def test_sqlite_seeds_table_pk(tmp_path, monkeypatch):
    c = _Corpus(tmp_path, monkeypatch)
    db = tmp_path / "e.sqlite"
    c.add("a", b"ab")
    c.dump(db)

    con = sqlite3.connect(db)
    try:
        rows = con.execute("PRAGMA table_info(seeds)").fetchall()
        n = dict(con.execute("SELECT seed_id, n_edges FROM seeds"))
    finally:
        con.close()
    assert [r[1] for r in rows if r[5]] == ["seed_id"]
    assert n == {hash_data(b"ab"): np.unique(list(b"ab")).size}
