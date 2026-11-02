"""`full-cover`: every edge in an edge dump, fewest seeds, verified.

The dump is the record of coverage (`edge-dump` output), not the corpus: a
corpus pruned last week still has its edges here, and a cover is only useful
if it reaches all of them.
"""

import io
import pickle
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.cli import commands
from fuzzer_tool.services import full_cover


def _seed(corpus: Path, data: bytes) -> Path:
    """A seed in the canonical layout: seeds/<hh>/id_<hash>."""
    sid = hash_data(data)
    d = corpus / "seeds" / sid[:2]
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"id_{sid}"
    p.write_bytes(data)
    return p


def _dump(path: Path, batches: list[dict[bytes, list[int]]]) -> Path:
    """Write *batches* the way EdgePickleWriter does: pickles back to back."""
    with open(path, "wb") as f:
        for batch in batches:
            pickle.dump({hash_data(k): np.array(v, dtype=np.uint32) for k, v in batch.items()}, f)
    return path


def test_redundant_seed_dropped_every_edge_kept(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2,3}
    _seed(corpus, b"b")  # {3}     redundant
    _seed(corpus, b"cc")  # {4,5}
    dump = _dump(
        tmp_path / "e.pkl",
        [{b"aaaa": [1, 2, 3], b"b": [3], b"cc": [4, 5]}],
    )

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")

    assert rep.kept == 2
    assert rep.union_edges == 5
    assert rep.reachable_edges == 5
    assert rep.unreachable_edges == 0
    assert len(list((tmp_path / "out").rglob("id_*"))) == 2


def test_unreachable_edge_refuses_to_write(tmp_path):
    """Falsification: a dump-only seed holds edge 9; pruning would lose it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"bb")  # {2,3}
    dump = _dump(
        tmp_path / "e.pkl",
        [{b"aaaa": [1, 2], b"bb": [2, 3], b"gone": [9]}],
    )

    with pytest.raises(full_cover.UnreachableEdges):
        full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_allow_unreachable_reports_and_proceeds(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"bb")  # {2,3}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"bb": [2, 3], b"gone": [9]}])

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out", allow_unreachable=True)

    assert rep.unreachable_edges == 1
    assert rep.gone_seeds == 1
    assert rep.reachable_edges == 3


def test_every_batch_of_a_multi_batch_dump_counts(tmp_path):
    """Adversarial: edge 4 lives only in batch 2; a first-pickle read misses it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"bb")  # {2,3,4}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2]}, {b"bb": [2, 3, 4]}])

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")

    assert rep.union_edges == 4
    assert rep.reachable_edges == 4
    assert rep.kept == 2


def test_in_place_moves_redundant_seed_to_pruned(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    keep = _seed(corpus, b"aaaa")  # {1,2}
    drop = _seed(corpus, b"a")  # {1}  redundant
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"a": [1]}])

    full_cover.full_cover(dump, corpus)

    assert keep.exists()
    assert not drop.exists()
    assert (corpus / "pruned" / drop.relative_to(corpus)).exists()


def test_seed_absent_from_dump_is_unmeasured_and_kept(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    extra = _seed(corpus, b"zzz")  # never dumped
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2]}])

    rep = full_cover.full_cover(dump, corpus)

    assert rep.unmeasured == 1
    assert extra.exists()


def test_incomplete_cover_is_rejected(tmp_path, monkeypatch):
    """Falsification of the verifier: a cover that misses an edge must not ship."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"bb")  # {2,3}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"bb": [2, 3]}])
    monkeypatch.setattr(full_cover, "min_cover", lambda edges, sizes: [])

    with pytest.raises(full_cover.IncompleteCover):
        full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")


def test_flat_loose_seeds_kept_by_content_hash(tmp_path):
    """Flat files are seeds when their names key on content hash."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    def loose(data: bytes) -> Path:
        p = corpus / f"id_{hash_data(data)}"
        p.write_bytes(data)
        return p

    loose(b"aaaa")  # {1, 2}
    loose(b"bb")  # {2, 3}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"bb": [2, 3]}])

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")

    assert rep.kept == 2
    assert rep.reachable_edges == 3


def _tar(corpus: Path, seeds: dict[bytes, list[int]]) -> Path:
    """seeds.tar.gz with canonical ``seeds/<hh>/id_<hash>`` members."""
    path = corpus / "seeds.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for data in seeds:
            sid = hash_data(data)
            info = tarfile.TarInfo(f"seeds/{sid[:2]}/id_{sid}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _zip(corpus: Path, seeds: dict[bytes, list[int]]) -> Path:
    path = corpus / "seeds.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for data in seeds:
            sid = hash_data(data)
            zf.writestr(f"{sid[:2]}/id_{sid}", data)
    return path


def _written_hashes(out: Path) -> set[str]:
    """Content hashes of every seed written under *out* (seeds/<hh>/id_*)."""
    return {hash_data(p.read_bytes()) for p in out.rglob("id_*")}


def test_tarred_corpus_covers_and_copies_out(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _tar(corpus, {b"aaaa": [1, 2], b"a": [1], b"cc": [3]})
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"a": [1], b"cc": [3]}])
    out = tmp_path / "out"

    rep = full_cover.full_cover(dump, corpus, output_dir=out)

    assert rep.kept == 2
    assert rep.reachable_edges == 3
    assert _written_hashes(out) == {hash_data(b"aaaa"), hash_data(b"cc")}


def test_tarred_corpus_refuses_in_place(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _tar(corpus, {b"aaaa": [1, 2]})
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2]}])

    with pytest.raises(full_cover.ArchiveReadOnly):
        full_cover.full_cover(dump, corpus)


def test_zipped_corpus_covers_and_copies_out(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _zip(corpus, {b"aaaa": [1, 2], b"a": [1], b"cc": [3]})
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"a": [1], b"cc": [3]}])
    out = tmp_path / "out"

    rep = full_cover.full_cover(dump, corpus, output_dir=out)

    assert rep.kept == 2
    assert _written_hashes(out) == {hash_data(b"aaaa"), hash_data(b"cc")}


def test_cli_writes_cover(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"a")  # {1} redundant
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"a": [1]}])
    out = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        ["fuzzer-tool", "full-cover", "-d", str(corpus), "-e", str(dump), "-o", str(out)],
    )
    rc = commands.main()

    assert rc == 0
    assert len(list(out.iterdir())) == 1


def test_cli_unreachable_exits_nonzero(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"gone": [9]}])
    out = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        ["fuzzer-tool", "full-cover", "-d", str(corpus), "-e", str(dump), "-o", str(out)],
    )
    rc = commands.main()

    err = capsys.readouterr().err
    assert rc == 2
    assert "unreachable" in err
