"""`edge-dump` replays seeds.zip and seeds.tar.gz members beside seeds/.

Live, unprotected members only (protected trees are excluded as their
seeds/ directories are); a seed loose and archived replays once.
"""

import io
import tarfile
import zipfile

from fuzzer_tool.adapters import process, seed_tar, seed_zip
from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.services import edge_export
from tests.test_edge_export import _expected, _Shm
from tests.test_edge_export_incremental import _load


class _Corpus:
    """Loose seeds plus archives; fake replay logs every input."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "corpus"
        (self.root / "seeds").mkdir(parents=True)
        self.replayed: list[bytes] = []
        monkeypatch.setattr(edge_export, "ShmCoverage", _Shm)
        monkeypatch.setattr(process, "run_target_stdin", self._run)

    def _run(self, target, data, timeout, env=None, **_):
        self.replayed.append(data)
        _Shm.current = data
        return 0, "", 0

    def loose(self, name: str, data: bytes) -> None:
        (self.root / "seeds" / name).write_bytes(data)

    def zip(self, members: dict[str, bytes]) -> None:
        with zipfile.ZipFile(self.root / seed_zip.ZIP_NAME, "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)

    def tar(self, members: dict[str, bytes]) -> None:
        with tarfile.open(self.root / seed_tar.TAR_NAME, "w:gz") as tf:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    def dump(self, out, **kw) -> int:
        self.replayed.clear()
        return edge_export.export_edges("t", str(self.root), [str(out)], **kw)


def _want(*seeds: bytes) -> dict[str, list[int]]:
    return {hash_data(d): _expected(d) for d in seeds}


def test_loose_zip_tar_all_exported(tmp_path, monkeypatch):
    c = _Corpus(tmp_path, monkeypatch)
    out = tmp_path / "e.pkl"
    c.loose("l", b"ab")
    c.zip({"z": b"cd"})
    c.tar({"seeds/t": b"ef"})

    assert c.dump(out) == 3
    assert _load(out) == _want(b"ab", b"cd", b"ef")


def test_archive_only_corpus(tmp_path, monkeypatch):
    """Adversarial: no loose files at all is not an empty corpus."""
    c = _Corpus(tmp_path, monkeypatch)
    out = tmp_path / "e.pkl"
    c.tar({"t": b"ef"})

    assert c.dump(out) == 1
    assert _load(out) == _want(b"ef")


def test_protected_and_pruned_members_excluded(tmp_path, monkeypatch):
    """Falsification: archive trees follow the seeds/ exclusions."""
    c = _Corpus(tmp_path, monkeypatch)
    out = tmp_path / "e.pkl"
    ir = hash_data(b"IR")
    c.zip({"z": b"cd", f"irreplaceable/{ir[:2]}/id_{ir}": b"IR"})
    c.tar({"t": b"ef", "seeds/crashing/x": b"CR", "seeds/pruned/x": b"PR"})

    c.dump(out)
    assert sorted(c.replayed) == [b"cd", b"ef"]


def test_loose_and_archived_replayed_once(tmp_path, monkeypatch):
    c = _Corpus(tmp_path, monkeypatch)
    c.loose("l", b"ab")
    c.zip({"z": b"ab"})
    c.tar({"t": b"ab"})

    c.dump(tmp_path / "e.pkl")
    assert c.replayed == [b"ab"]


def test_archive_seeds_incremental(tmp_path, monkeypatch):
    c = _Corpus(tmp_path, monkeypatch)
    out = tmp_path / "e.sqlite"
    c.tar({"t": b"ef"})
    c.dump(out)

    c.zip({"z": b"cd"})
    c.dump(out)
    assert c.replayed == [b"cd"]
    assert _load(out) == _want(b"cd", b"ef")


def test_oversized_archive_seed_skipped(tmp_path, monkeypatch, capsys):
    c = _Corpus(tmp_path, monkeypatch)
    c.loose("l", b"ab")
    c.tar({"big": b"x" * 64})

    c.dump(tmp_path / "e.pkl", max_seed_bytes=8)
    assert c.replayed == [b"ab"]
    assert "Skipped: 1 seeds over 8 bytes" in capsys.readouterr().out
