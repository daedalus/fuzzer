"""`edge-dump --sqlite`: edge-keyed SQLite inverse of the per-seed export.

edges(edge_id PK, n_seeds)  1--*  edge_seeds(edge_id, seed_id)
"""

import sqlite3
import sys

import numpy as np
import pytest

from fuzzer_tool.adapters.edge_sqlite import write_edges_sqlite
from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.cli import commands
from fuzzer_tool.services import edge_export
from tests.test_edge_export import _expected, _setup


def _arr(*ids: int) -> np.ndarray:
    return np.array(sorted(ids), dtype=np.uint32)


def _inverse(edges: dict[str, np.ndarray]) -> dict[int, set[str]]:
    """Reference inverse, built with plain loops (independent of the adapter)."""
    inv: dict[int, set[str]] = {}
    for seed, ids in edges.items():
        for e in ids.tolist():
            inv.setdefault(e, set()).add(seed)
    return inv


def _read(path) -> tuple[dict[int, int], dict[int, set[str]]]:
    con = sqlite3.connect(path)
    try:
        counts = dict(con.execute("SELECT edge_id, n_seeds FROM edges"))
        owners: dict[int, set[str]] = {}
        for e, s in con.execute("SELECT edge_id, seed_id FROM edge_seeds"):
            owners.setdefault(e, set()).add(s)
        return counts, owners
    finally:
        con.close()


def _pk(path, table: str) -> list[str]:
    con = sqlite3.connect(path)
    try:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
        return [r[1] for r in sorted(rows, key=lambda r: r[5]) if r[5]]
    finally:
        con.close()


def test_edge_id_is_primary_key(tmp_path):
    db = tmp_path / "e.sqlite"
    write_edges_sqlite(db, {"s1": _arr(1, 2)})
    assert _pk(db, "edges") == ["edge_id"]


def test_inverse_matches_reference(tmp_path):
    edges = {"s1": _arr(1, 2, 3), "s2": _arr(2, 3), "s3": _arr(3, 7), "s4": _arr()}
    db = tmp_path / "e.sqlite"
    write_edges_sqlite(db, edges)

    counts, owners = _read(db)
    ref = _inverse(edges)
    assert owners == ref
    assert counts == {e: len(s) for e, s in ref.items()}


def test_shared_edge_one_row(tmp_path):
    """Falsification: an edge held by many seeds is one PK row, not one per seed."""
    db = tmp_path / "e.sqlite"
    write_edges_sqlite(db, {"a": _arr(5), "b": _arr(5), "c": _arr(5)})

    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1
    finally:
        con.close()


def test_existing_file_replaced(tmp_path):
    """Adversarial: a stale DB (or junk) at the path is replaced, not merged."""
    db = tmp_path / "e.sqlite"
    db.write_bytes(b"not a database")
    write_edges_sqlite(db, {"old": _arr(9)})
    write_edges_sqlite(db, {"new": _arr(1)})

    counts, owners = _read(db)
    assert owners == {1: {"new"}}
    assert not (tmp_path / "e.sqlite.tmp").exists()


def test_large_edge_ids(tmp_path):
    """Adversarial: uint32 max survives (no signed overflow)."""
    top = np.iinfo(np.uint32).max
    db = tmp_path / "e.sqlite"
    write_edges_sqlite(db, {"s": _arr(0, top)})
    assert set(_read(db)[0]) == {0, top}


def test_export_writes_sqlite_only(tmp_path, monkeypatch):
    """Service: sqlite alone (no pickle) is a valid output."""
    seeds = {"a": b"ab", "b": b"bc"}
    corpus, out = _setup(tmp_path, monkeypatch, seeds)
    db = tmp_path / "e.sqlite"

    n = edge_export.export_edges("t", str(corpus), [str(db)])

    assert n == len(seeds)
    assert not out.exists()
    ref = _inverse({hash_data(d): np.array(_expected(d), dtype=np.uint32) for d in seeds.values()})
    assert _read(db)[1] == ref


def test_blackout_writes_no_sqlite(tmp_path, monkeypatch):
    corpus, _ = _setup(tmp_path, monkeypatch, {"a": b"ab"}, blind=True)
    db = tmp_path / "e.sqlite"
    assert edge_export.export_edges("t", str(corpus), [str(db)]) == 0
    assert not db.exists()


def test_cli_both_outputs(tmp_path, monkeypatch):
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab"})
    db = tmp_path / "e.sqlite"
    monkeypatch.setattr(commands, "_validate_target", lambda t: None)
    argv = ["fuzzer-tool", "edge-dump", "t", "-d", str(corpus), "-o", str(out), "-o", str(db)]
    monkeypatch.setattr(sys, "argv", argv)

    assert commands.main() == 0
    assert out.exists()
    assert set(_read(db)[0]) == set(_expected(b"ab"))


def test_cli_needs_an_output(tmp_path, monkeypatch):
    corpus, _ = _setup(tmp_path, monkeypatch, {"a": b"ab"})
    monkeypatch.setattr(commands, "_validate_target", lambda t: None)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "edge-dump", "t", "-d", str(corpus)])

    with pytest.raises(SystemExit) as exc:
        commands.main()
    assert exc.value.code != 0


@pytest.mark.parametrize("name", ["e.json", "e", "e.sqlite.bak", "e.PKL.gz"])
def test_unknown_extension_rejected_before_replay(name, tmp_path, monkeypatch, capsys):
    """Adversarial: no sink for the suffix -> nothing replayed, nothing written."""
    corpus, _ = _setup(tmp_path, monkeypatch, {"a": b"ab"})
    replayed = []
    monkeypatch.setattr(edge_export, "_replay_all", lambda *a: replayed.append(a))
    out = tmp_path / name

    assert edge_export.export_edges("t", str(corpus), [str(out)]) == 0
    assert not replayed
    assert not out.exists()
    assert ".pkl" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", [".pkl", ".PKL", ".sqlite", ".SQLITE"])
def test_extension_picks_format(suffix, tmp_path, monkeypatch):
    """Falsification: .pkl is a pickle, .sqlite a database, case-insensitive."""
    corpus, _ = _setup(tmp_path, monkeypatch, {"a": b"ab"})
    out = tmp_path / f"e{suffix}"

    assert edge_export.export_edges("t", str(corpus), [str(out)]) == 1
    head = out.read_bytes()[:16]
    assert head.startswith(b"SQLite format 3") == (suffix.lower() == ".sqlite")
