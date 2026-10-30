"""seeds.zip / seeds.tar.gz are read in place, one member at a time, in one pass.

No pre-pass that decompresses an archive to count it, and no read-only zip
path that inflates every foreign member before yielding the first.
"""

from __future__ import annotations

import gzip
import io
import tarfile
import zipfile

import pytest

from fuzzer_tool.adapters import process, seed_tar, seed_zip
from fuzzer_tool.adapters.filesystem import count_archive_seeds, hash_data
from fuzzer_tool.services import edge_export
from fuzzer_tool.services.minimize import _print_progress
from tests.test_edge_export import _Shm

N = 8


@pytest.fixture
def zip_reads(monkeypatch):
    """Count ZipFile.read calls (one per member inflated)."""
    calls: list[str] = []
    real = zipfile.ZipFile.read

    def read(self, name, pwd=None):
        calls.append(name if isinstance(name, str) else name.filename)
        return real(self, name, pwd)

    monkeypatch.setattr(zipfile.ZipFile, "read", read)
    return calls


@pytest.fixture
def tar_opens(monkeypatch):
    """Count gzip stream opens of seeds.tar.gz."""
    calls: list[str] = []
    real = gzip.open

    def opener(path, *a, **kw):
        calls.append(str(path))
        return real(path, *a, **kw)

    monkeypatch.setattr(seed_tar.gzip, "open", opener)
    return calls


def _zip(corpus, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _tar(corpus, members: dict[str, bytes]) -> None:
    with tarfile.open(corpus / seed_tar.TAR_NAME, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _canonical(data: bytes) -> str:
    h = hash_data(data)
    return f"{h[:2]}/id_{h}"


def _corpus(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "seeds").mkdir(parents=True)
    return corpus


def test_readonly_foreign_members_streamed(tmp_path, zip_reads):
    """Falsification target: the first foreign seed arrives after one inflate, not N."""
    corpus = _corpus(tmp_path)
    _zip(corpus, {f"f{i}.bin": bytes([i]) * 64 for i in range(N)})

    live = seed_zip.open_readonly(corpus).live()
    first, protected = next(live)

    assert len(zip_reads) == 1
    assert first == bytes([0]) * 64 and not protected
    assert len(list(live)) == N - 1


def test_readonly_foreign_dedup_kept(tmp_path, zip_reads):
    """Adversarial: streaming keeps dedup -- same bytes twice, and a copy of a main member."""
    corpus = _corpus(tmp_path)
    _zip(corpus, {_canonical(b"MAIN"): b"MAIN", "dup_of_main": b"MAIN", "a": b"X", "b": b"X"})

    got = [d for d, _ in seed_zip.open_readonly(corpus).live()]
    assert sorted(got) == [b"MAIN", b"X"]


def test_zip_count_reads_index_only(tmp_path, zip_reads):
    corpus = _corpus(tmp_path)
    _zip(corpus, {_canonical(bytes([i])): bytes([i]) for i in range(N)} | {"foreign": b"F"})

    assert count_archive_seeds(corpus) == N + 1
    assert zip_reads == []


def test_edge_dump_one_pass_per_archive(tmp_path, monkeypatch, zip_reads, tar_opens):
    """Each zip member inflated once; the tar stream opened once."""
    corpus = _corpus(tmp_path)
    _zip(corpus, {f"z{i}": b"z" + bytes([i]) for i in range(N)})
    _tar(corpus, {f"t{i}": b"t" + bytes([i]) for i in range(N)})
    monkeypatch.setattr(edge_export, "ShmCoverage", _Shm)

    def run(target, data, timeout, env=None, **_):
        _Shm.current = data
        return 0, "", 0

    monkeypatch.setattr(process, "run_target_stdin", run)

    assert edge_export.export_edges("t", str(corpus), [str(tmp_path / "e.pkl")]) == 2 * N
    assert len(zip_reads) == N
    assert len(tar_opens) == 1


def test_progress_unknown_total(capsys):
    """Tar seeds are uncounted: total 0 prints done/rate without % or ETA."""
    _print_progress(5, 0, 0.0, 3, 1024)
    line = capsys.readouterr().out
    assert "Replayed 5 " in line
    assert "%" not in line and "ETA" not in line
    assert "seeds/s" in line and "edges: 3" in line


def test_empty_tar_reports_no_seeds(tmp_path, monkeypatch, capsys):
    """Adversarial: unknown total resolves to zero after the stream: no seeds, nothing written."""
    corpus = _corpus(tmp_path)
    _tar(corpus, {"seeds/pruned/x": b"P"})
    monkeypatch.setattr(edge_export, "ShmCoverage", _Shm)
    out = tmp_path / "e.pkl"

    assert edge_export.export_edges("t", str(corpus), [str(out)]) == 0
    assert not out.exists()
    assert "No seeds to replay" in capsys.readouterr().err
