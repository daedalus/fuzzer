"""`edge-dump` writes outputs in batches as edges arrive, not one dump at the end.

    replay s0 s1 | s2 s3 | s4      batch pinned to 2
    write        ^ batch ^ batch ^ tail (close)

A killed run keeps every written batch; the next run skips those seeds.
"""

import pickle
import sqlite3

import numpy as np
import pytest

from fuzzer_tool.adapters.edge_pickle import EdgePickleWriter, read_edges_pickle
from fuzzer_tool.adapters.edge_sqlite import EdgeSqliteWriter
from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.services import edge_export
from tests.test_edge_export import _expected
from tests.test_edge_export_incremental import FORMATS, _Corpus, _load

BATCH = 2
SEEDS = [bytes([0x61 + i, 0x62 + i]) for i in range(5)]  # b"ab", b"bc", ...


@pytest.fixture(autouse=True)
def _small_batches(monkeypatch):
    """Pin the adaptive batch to BATCH: bounds equal, so it cannot move."""
    for name in ("_BATCH_START", "_BATCH_MIN", "_BATCH_MAX"):
        monkeypatch.setattr(edge_export, name, BATCH)


def _arr(*ids: int) -> np.ndarray:
    return np.array(ids, dtype=np.uint32)


def _write(writer, items) -> None:
    writer.open()
    writer.write(list(items))
    writer.close()


def _on_disk(path) -> int:
    """Seeds readable from *path* right now (0 if absent)."""
    if not path.exists():
        return 0
    if path.suffix == ".pkl":
        return len(_load(path))
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    finally:
        con.close()


def _seeded(tmp_path, monkeypatch) -> _Corpus:
    c = _Corpus(tmp_path, monkeypatch)
    for i, d in enumerate(SEEDS):
        c.add(f"s{i}", d)
    return c


@pytest.mark.parametrize("name", FORMATS)
def test_written_during_replay(name, tmp_path, monkeypatch):
    """Falsification of dump-at-end: batches land on disk while replay runs."""
    c = _seeded(tmp_path, monkeypatch)
    out = tmp_path / name
    seen_on_disk: list[int] = []
    run = c._run

    def spy(target, data, timeout, env=None, **kw):
        seen_on_disk.append(_on_disk(out))
        return run(target, data, timeout, env, **kw)

    monkeypatch.setattr("fuzzer_tool.adapters.process.run_target_stdin", spy)
    assert c.dump(out) == len(SEEDS)

    # Before replaying seed i, every full batch of the first i seeds is on disk.
    assert seen_on_disk == [i // BATCH * BATCH for i in range(len(SEEDS))]
    assert _load(out) == {hash_data(d): _expected(d) for d in SEEDS}


@pytest.mark.parametrize("name", FORMATS)
def test_killed_run_resumes(name, tmp_path, monkeypatch):
    """Adversarial: a crash mid-replay keeps written batches; rerun replays the rest."""
    c = _seeded(tmp_path, monkeypatch)
    out = tmp_path / name
    run = c._run
    kill_at = SEEDS[3]

    def dies(target, data, timeout, env=None, **kw):
        if data == kill_at:
            raise KeyboardInterrupt
        return run(target, data, timeout, env, **kw)

    monkeypatch.setattr("fuzzer_tool.adapters.process.run_target_stdin", dies)
    with pytest.raises(KeyboardInterrupt):
        c.dump(out)
    survived = _on_disk(out)
    assert survived >= BATCH

    monkeypatch.setattr("fuzzer_tool.adapters.process.run_target_stdin", run)
    c.dump(out)
    assert len(c.replayed) == len(SEEDS) - survived
    assert _load(out) == {hash_data(d): _expected(d) for d in SEEDS}


def test_sqlite_counts_across_batches(tmp_path):
    """n_seeds accumulates across batch upserts, as one bulk write would."""
    db = tmp_path / "e.sqlite"
    edges = {f"s{i}": np.array(sorted({1, 2 + i % 2, 9}), dtype=np.uint32) for i in range(5)}

    w = EdgeSqliteWriter(db)
    assert w.open() == set()
    items = list(edges.items())
    for i in range(0, len(items), BATCH):
        w.write(items[i : i + BATCH])
    w.close()

    ref: dict[int, int] = {}
    for v in edges.values():
        for e in v.tolist():
            ref[e] = ref.get(e, 0) + 1
    con = sqlite3.connect(db)
    try:
        assert dict(con.execute("SELECT edge_id, n_seeds FROM edges")) == ref
    finally:
        con.close()


def test_writer_ignores_known_and_repeat(tmp_path):
    """Adversarial: a seed already in the file, or added twice, is written once."""
    db = tmp_path / "e.sqlite"
    _write(EdgeSqliteWriter(db), [("a", _arr(1)), ("a", _arr(1))])

    w = EdgeSqliteWriter(db)
    assert w.open() == {"a"}
    w.write([("a", _arr(1))])
    w.close()

    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT n_seeds FROM edges").fetchall() == [(1,)]
    finally:
        con.close()


@pytest.mark.parametrize("cls,name", [(EdgeSqliteWriter, "e.sqlite"), (EdgePickleWriter, "e.pkl")])
def test_no_file_without_writes(cls, name, tmp_path):
    """Writers create nothing until a non-empty batch is written."""
    writer = cls(tmp_path / name)
    writer.open()
    writer.write([])
    writer.close()
    assert not (tmp_path / name).exists()


def test_regression_pickle_single_load(tmp_path):
    """Several batches still make one pickle: one pickle.load returns every seed."""
    out = tmp_path / "e.pkl"
    w = EdgePickleWriter(out)
    w.open()
    for i in range(0, 5, BATCH):
        w.write([(f"s{j}", _arr(j)) for j in range(i, min(i + BATCH, 5))])

        # Complete after every batch, not only after close.
        with open(out, "rb") as f:
            assert len(pickle.load(f)) == min(i + BATCH, 5)
            assert f.read() == b""
    w.close()

    with open(out, "rb") as f:
        got = pickle.load(f)
    assert {k: v.tolist() for k, v in got.items()} == {f"s{i}": [i] for i in range(5)}
    assert all(v.dtype == np.uint32 for v in got.values())
    assert not any(v.flags.writeable for v in got.values())  # frombuffer: documented read-only


def test_pickle_reopen_appends_same_pickle(tmp_path):
    """Falsification: a second run extends the one dict, it does not add a pickle."""
    out = tmp_path / "e.pkl"
    _write(EdgePickleWriter(out), [("a", _arr(1))])
    _write(EdgePickleWriter(out), [("b", _arr(2))])

    with open(out, "rb") as f:
        assert set(pickle.load(f)) == {"a", "b"}
        assert f.read() == b""


@pytest.mark.parametrize("cut", [1, 5, 20])
def test_pickle_torn_tail_repaired(cut, tmp_path, caplog):
    """Adversarial: a kill mid-batch is cut back to the last complete batch."""
    out = tmp_path / "e.pkl"
    w = EdgePickleWriter(out)
    w.open()
    w.write([(f"s{i}", _arr(i)) for i in range(2)])
    w.write([(f"t{i}", _arr(i)) for i in range(2)])
    w.close()
    with open(out, "r+b") as f:
        f.truncate(out.stat().st_size - cut)  # STOP gone, maybe part of batch 2

    w = EdgePickleWriter(out)
    known = w.open()
    assert {"s0", "s1"} <= known <= {"s0", "s1", "t0", "t1"}
    assert "torn" in caplog.text.lower()
    w.write([("new", _arr(9))])
    w.close()

    with open(out, "rb") as f:
        got = pickle.load(f)
    assert set(got) == known | {"new"}
    assert {k: v.tolist() for k, v in got.items() if k.startswith("s")} == {"s0": [0], "s1": [1]}


def test_pickle_trailing_junk_cut(tmp_path, caplog):
    """Adversarial: bytes after the pickle's STOP are not left behind."""
    out = tmp_path / "e.pkl"
    _write(EdgePickleWriter(out), [("a", _arr(1))])
    with open(out, "ab") as f:
        f.write(b"junk")

    w = EdgePickleWriter(out)
    assert w.open() == {"a"}
    w.write([("b", _arr(2))])
    w.close()
    with open(out, "rb") as f:
        assert set(pickle.load(f)) == {"a", "b"}
        assert f.read() == b""


def test_pickle_open_does_not_load_arrays(tmp_path, monkeypatch):
    """Known ids come back without rebuilding a single ndarray."""
    out = tmp_path / "e.pkl"
    _write(EdgePickleWriter(out), [(f"s{i}", _arr(i, i + 1)) for i in range(4)])

    def no_arrays(*a, **kw):
        raise AssertionError("ndarray rebuilt")

    # Arrays are stored as numpy.frombuffer(bytes, dtype); unpickling resolves it by name.
    monkeypatch.setattr(np, "frombuffer", no_arrays)

    # Control: a full load does hit the patched rebuild, so the probe can fail.
    with pytest.raises(AssertionError, match="rebuilt"), open(out, "rb") as f:
        pickle.load(f)
    assert EdgePickleWriter(out).open() == {f"s{i}" for i in range(4)}


@pytest.mark.parametrize("batches", [1, 3])
def test_pickle_legacy_migrated(batches, tmp_path):
    """Old exports (one dict, or back-to-back batch pickles) become one pickle."""
    out = tmp_path / "e.pkl"
    with open(out, "wb") as f:
        for b in range(batches):
            pickle.dump({f"old{b}": np.array([b], dtype=np.uint32)}, f, protocol=5)

    w = EdgePickleWriter(out)
    assert w.open() == {f"old{b}" for b in range(batches)}
    w.write([("new", _arr(2))])
    w.close()

    with open(out, "rb") as f:
        got = pickle.load(f)
        assert f.read() == b""
    assert set(got) == {f"old{b}" for b in range(batches)} | {"new"}
    assert set(read_edges_pickle(out)) == set(got)


def test_pickle_2d_array_round_trips(tmp_path):
    """Adversarial: a non-1-D array falls back to numpy's reducer, shape intact."""
    out = tmp_path / "e.pkl"
    grid = np.arange(6, dtype=np.uint32).reshape(2, 3)
    _write(EdgePickleWriter(out), [("g", grid), ("v", _arr(4, 5))])
    with open(out, "rb") as f:
        got = pickle.load(f)
    assert got["g"].shape == (2, 3) and got["g"].tolist() == grid.tolist()
    assert got["v"].tolist() == [4, 5]


def test_pickle_not_an_export_refused(tmp_path):
    out = tmp_path / "e.pkl"
    out.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        EdgePickleWriter(out).open()
