"""seeds.tar.gz and seeds.zip read beside seeds/, as the same level.

    corpus/
    ├── seeds/ab/id_..     loose files
    ├── seeds.zip          read (existing behaviour)
    └── seeds.tar.gz       read-only: streamed, never extracted

Tar member layout follows seeds.zip: optional ``seeds/`` (or ``./seeds/``)
prefix, ``pruned/`` skipped, irreplaceable/crashing/timeouts protected.
"""

from __future__ import annotations

import io
import logging
import tarfile
import zipfile
from pathlib import Path

import pytest

from fuzzer_tool.adapters import seed_tar, seed_zip
from fuzzer_tool.adapters.filesystem import (
    discover_seed_files,
    hash_data,
    iter_seed_batches,
    load_corpus,
)


def _tar(corpus: Path, members: dict[str, bytes], extra=()) -> Path:
    """Write corpus/seeds.tar.gz with *members*; *extra* are prebuilt TarInfos."""
    path = corpus / seed_tar.TAR_NAME
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for info in extra:
            tf.addfile(info)
    return path


def _corpus(tmp_path: Path, loose: dict[str, bytes] | None = None) -> Path:
    corpus = tmp_path / "corpus"
    (corpus / "seeds").mkdir(parents=True)
    for name, data in (loose or {}).items():
        (corpus / "seeds" / name).write_bytes(data)
    return corpus


def _load(corpus: Path) -> tuple[set[bytes], set[str]]:
    seeds, _, protected = load_corpus(corpus, add_default=False)
    return set(seeds), protected


@pytest.mark.parametrize("prefix", ["", "seeds/", "./seeds/", "./"])
def test_load_unions_files_and_tar(prefix, tmp_path):
    corpus = _corpus(tmp_path, {"loose": b"LOOSE"})
    _tar(corpus, {f"{prefix}ab/one": b"ONE", f"{prefix}two": b"TWO"})

    seeds, _ = _load(corpus)
    assert seeds == {b"LOOSE", b"ONE", b"TWO"}


def test_no_tar_unchanged(tmp_path):
    """Falsification: without the archive only loose seeds load."""
    corpus = _corpus(tmp_path, {"loose": b"LOOSE"})
    assert _load(corpus)[0] == {b"LOOSE"}


def test_tar_protected_trees_marked(tmp_path):
    corpus = _corpus(tmp_path)
    trees = ("irreplaceable", "crashing", "timeouts")
    _tar(corpus, {f"seeds/{t}/x": t.encode() for t in trees} | {"main": b"MAIN"})

    seeds, protected = _load(corpus)
    assert seeds == {b"MAIN", *(t.encode() for t in trees)}
    assert protected == {hash_data(t.encode()) for t in trees}


def test_tar_pruned_skipped(tmp_path):
    """Falsification: pruned members (file or zip-tombstone spelling) stay pruned."""
    corpus = _corpus(tmp_path)
    _tar(
        corpus, {"seeds/pruned/ab/x": b"GONE", "pruned/y": b"GONE2", ".pruned/z": b"", "k": b"KEEP"}
    )
    assert _load(corpus)[0] == {b"KEEP"}


def test_same_seed_loose_and_tar_loads_once(tmp_path):
    corpus = _corpus(tmp_path, {"a": b"SAME"})
    _tar(corpus, {"a": b"SAME"})
    seeds, _, _ = load_corpus(corpus, add_default=False)
    assert seeds == [b"SAME"]


def test_tar_and_zip_together(tmp_path):
    corpus = _corpus(tmp_path)
    _tar(corpus, {"t": b"FROM_TAR"})
    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME, "w") as zf:
        zf.writestr("z", b"FROM_ZIP")
    assert _load(corpus)[0] == {b"FROM_TAR", b"FROM_ZIP"}


def test_hostile_tar_members_ignored(tmp_path):
    """Adversarial: traversal, absolute, links, devices and dirs are never seeds."""
    corpus = _corpus(tmp_path)
    link = tarfile.TarInfo("seeds/link")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    hard = tarfile.TarInfo("seeds/hard")
    hard.type, hard.linkname = tarfile.LNKTYPE, "seeds/ok"
    fifo = tarfile.TarInfo("seeds/fifo")
    fifo.type = tarfile.FIFOTYPE
    d = tarfile.TarInfo("seeds/dir")
    d.type = tarfile.DIRTYPE
    _tar(
        corpus,
        {"../evil": b"E1", "/abs": b"E2", "seeds/../../evil": b"E3", "seeds/ok": b"OK"},
        extra=(link, hard, fifo, d),
    )

    assert _load(corpus)[0] == {b"OK"}
    assert not (tmp_path / "evil").exists()


def test_truncated_tar_keeps_loose_and_warns(tmp_path, caplog):
    """Adversarial: a torn archive yields what precedes the tear, never raises."""
    corpus = _corpus(tmp_path, {"loose": b"LOOSE"})
    path = _tar(corpus, {f"m{i}": bytes([i]) * 4096 for i in range(64)})
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])

    with caplog.at_level(logging.WARNING):
        seeds, _ = _load(corpus)
    assert b"LOOSE" in seeds
    assert seed_tar.TAR_NAME in caplog.text


def test_not_a_tar_warns(tmp_path, caplog):
    corpus = _corpus(tmp_path, {"loose": b"LOOSE"})
    (corpus / seed_tar.TAR_NAME).write_bytes(b"garbage")
    with caplog.at_level(logging.WARNING):
        assert _load(corpus)[0] == {b"LOOSE"}
    assert seed_tar.TAR_NAME in caplog.text


def test_iter_seed_batches_reads_tar(tmp_path):
    corpus = _corpus(tmp_path, {"loose": b"LOOSE"})
    _tar(corpus, {"t": b"TAR"})
    got = {d for batch in iter_seed_batches(corpus, 10) for d in batch}
    assert got == {b"LOOSE", b"TAR"}


@pytest.mark.parametrize("name", [seed_zip.ZIP_NAME, seed_tar.TAR_NAME])
def test_regression_flat_corpus_archives_not_seed_files(name, tmp_path):
    """A flat corpus (no seeds/) must not list its own archive as a seed file."""
    corpus = tmp_path / "flat"
    corpus.mkdir()
    (corpus / "seed").write_bytes(b"S")
    (corpus / name).write_bytes(b"archive bytes")
    (corpus / "other.zip").write_bytes(b"a zip-format seed")

    names = {p.name for p in discover_seed_files(corpus)}
    assert names == {"seed", "other.zip"}


def test_live_count_matches_live(tmp_path):
    corpus = _corpus(tmp_path)
    _tar(corpus, {"a": b"A", "b": b"B", "pruned/c": b"C", "seeds/crashing/d": b"D"})
    assert seed_tar.count(corpus) == sum(1 for _, prot in seed_tar.live(corpus) if not prot)
